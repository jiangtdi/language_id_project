"""Train OWSM v4 + Conv-Adapter PEFT language identification model."""

import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio_dataset import AudioLanguageDataset, create_audio_windows, load_and_resample, scan_audio_files
from src.logit_calibration import fit_logit_bias
from src.owsm_adapter_lid_model import MODEL_VERSION, OWSMAdapterLanguageClassifier
from src.owsm_local_model import MODEL_DISPLAY_NAME, OWSM_MODEL_DIR, resolve_owsm_model_dir

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
VOXLINGUA_MANIFEST_PATH = PROJECT_ROOT / "data" / "voxlingua_domain" / "voxlingua_split.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_owsm_adapter_lid.pt"
LAST_TRAINING_STATE_PATH = CHECKPOINT_DIR / "last_owsm_adapter_training_state.pt"
LABEL_MAP_PATH = CHECKPOINT_DIR / "label_map.json"
TRAINING_CURVE_PATH = OUTPUT_DIR / "owsm_adapter_training_curve.png"

RANDOM_SEED = 42

# PyCharm-friendly training knobs.
DEVICE = "cuda"  # "cuda" / "cpu" / "auto"
EPOCHS = 8
BATCH_SIZE = 1
DURATION = 10.0
TRAIN_WINDOW_STRIDE = 5.0
VAL_WINDOW_STRIDE = 5.0
MAX_VAL_WINDOWS = 4

ADAPTER_LR = 1e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.08
GRAD_ACCUM_STEPS = 8
MAX_GRAD_NORM = 1.0
LR_FACTOR = 0.5
LR_PATIENCE = 1
EARLY_STOP_PATIENCE = 2
MIN_EPOCHS_BEFORE_EARLY_STOP = 4

# OWSM v4 is a 1B-scale encoder. Keep a practical cap for classroom training.
# Set to None only if you want to run the full dataset overnight.
MAX_TRAIN_SAMPLES_PER_CLASS = 150
MAX_DOMAIN_TRAIN_SAMPLES_PER_CLASS = 200
MAX_DOMAIN_VAL_SAMPLES_PER_CLASS = 100
USE_VOXLINGUA_DOMAIN_ADAPTATION = True
RESUME_TRAINING = False
DETERMINISTIC_TRAINING = True


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC_TRAINING:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device_arg: str = DEVICE) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 DEVICE='cuda'，但当前环境没有可用 GPU。")
        return torch.device("cuda")
    return torch.device("cpu")


def split_samples(samples: List[Tuple[Path, str]]):
    labels = [label for _, label in samples]
    stratify = labels if len(set(labels)) > 1 and min(labels.count(x) for x in set(labels)) >= 2 else None
    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    temp_labels = [label for _, label in temp_samples]
    temp_stratify = (
        temp_labels
        if len(set(temp_labels)) > 1 and min(temp_labels.count(x) for x in set(temp_labels)) >= 2
        else None
    )
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_stratify,
    )
    return train_samples, val_samples, test_samples


def limit_per_class(samples: List[Tuple[Path, str]], max_per_class: int = None):
    if max_per_class is None:
        return samples
    selected = []
    counts = Counter()
    for audio_path, label in samples:
        if counts[label] < max_per_class:
            selected.append((audio_path, label))
            counts[label] += 1
    return selected


def load_manifest_split(manifest_path: Path, split_name: str, allowed_labels: set) -> List[Tuple[Path, str]]:
    if not manifest_path.exists():
        return []
    samples: List[Tuple[Path, str]] = []
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split_name:
                continue
            label = row.get("label", "")
            audio_path = Path(row.get("path", ""))
            if label in allowed_labels and audio_path.exists():
                samples.append((audio_path, label))
    return samples


def filter_readable_samples(samples: List[Tuple[Path, str]], name: str):
    valid_samples = []
    bad_samples = []

    for audio_path, label in tqdm(samples, desc=f"check {name}", leave=False):
        try:
            audio = load_and_resample(audio_path)
            if audio is None or len(audio) == 0:
                bad_samples.append(audio_path)
                print(f"跳过坏音频: {audio_path}，原因: 空音频")
                continue

            if not np.isfinite(audio).all():
                bad_samples.append(audio_path)
                print(f"跳过坏音频: {audio_path}，原因: 包含 NaN/Inf")
                continue

            valid_samples.append((audio_path, label))
        except Exception as exc:
            bad_samples.append(audio_path)
            print(f"跳过坏音频: {audio_path}，原因: {exc}")

    print(f"{name}: 可用 {len(valid_samples)} 条，坏音频 {len(bad_samples)} 条")
    return valid_samples


def compute_class_weight_tensor(train_samples: List[Tuple[Path, str]], label_to_idx: Dict[str, int]) -> torch.Tensor:
    counts = Counter(label for _, label in train_samples)
    total = len(train_samples)
    num_classes = len(label_to_idx)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for label, idx in label_to_idx.items():
        weights[idx] = total / (num_classes * max(1, counts[label]))
    return weights


def build_model(num_classes: int, device: torch.device) -> OWSMAdapterLanguageClassifier:
    model = OWSMAdapterLanguageClassifier(
        num_classes=num_classes,
        model_dir=resolve_owsm_model_dir(OWSM_MODEL_DIR),
        owsm_device=str(device),
        adapter_bottleneck=128,
        adapter_layers=2,
        embedding_dim=256,
    ).to(device)
    model.initialize_head(device=device, sample_seconds=3.0)
    return model


def run_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (waveforms, labels) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attention_mask = torch.ones_like(waveforms, dtype=torch.long, device=device)

        logits = model(waveforms, attention_mask=attention_mask, labels=labels, apply_margin=True)
        loss = criterion(logits, labels) / GRAD_ACCUM_STEPS
        loss.backward()

        if step % GRAD_ACCUM_STEPS == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * GRAD_ACCUM_STEPS * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate_samples(model, samples, label_to_idx, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    all_logits, all_labels = [], []

    for audio_path, label_name in tqdm(samples, desc="val", leave=False):
        audio = load_and_resample(audio_path)
        windows = create_audio_windows(
            audio,
            duration=DURATION,
            stride=VAL_WINDOW_STRIDE,
            max_windows=MAX_VAL_WINDOWS,
        )
        waveforms = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
        labels = torch.full((waveforms.size(0),), label_to_idx[label_name], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(waveforms, dtype=torch.long, device=device)
        logits_per_window = model(waveforms, attention_mask=attention_mask, labels=None, apply_margin=False)
        logits = logits_per_window.mean(dim=0, keepdim=True)
        label = torch.tensor([label_to_idx[label_name]], dtype=torch.long, device=device)
        loss = criterion(logits, label)

        total_loss += float(loss.item())
        y_true.append(int(label.item()))
        y_pred.append(int(logits.argmax(dim=1).item()))
        all_logits.append(logits.cpu())
        all_labels.append(label.cpu())

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    avg_loss = total_loss / max(1, len(samples))
    return avg_loss, acc, torch.cat(all_logits), torch.cat(all_labels)


def save_training_curve(history):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    axes[1].plot(epochs, history["val_acc"], label="val_acc")
    axes[1].legend()
    axes[1].set_title("Accuracy")
    fig.tight_layout()
    fig.savefig(TRAINING_CURVE_PATH, dpi=160)
    plt.close(fig)


def get_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python_random": random.getstate(),
        "numpy_random": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "pos": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return

    python_state = state.get("python_random")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy_random")
    if numpy_state is not None:
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.array(numpy_state["keys"], dtype=np.uint32),
                numpy_state["pos"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )

    torch_cpu_state = state.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(torch_cpu_state.cpu())

    torch_cuda_state = state.get("torch_cuda")
    if torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_state)


def save_training_resume_state(
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    patience: int,
    history: dict,
    label_to_idx: Dict[str, int],
):
    state = {
        "model_version": MODEL_VERSION,
        "next_epoch": epoch + 1,
        "best_acc": best_acc,
        "patience": patience,
        "history": history,
        "label_to_idx": label_to_idx,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if not key.startswith("owsm.")
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": get_rng_state(),
    }
    torch.save(state, LAST_TRAINING_STATE_PATH)


def load_training_resume_state(model, optimizer, scheduler, label_to_idx: Dict[str, int], device: torch.device):
    if not RESUME_TRAINING or not LAST_TRAINING_STATE_PATH.exists():
        return 1, -1.0, 0, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    state = torch.load(LAST_TRAINING_STATE_PATH, map_location=device, weights_only=False)
    if state.get("model_version") != MODEL_VERSION:
        print("发现旧训练断点，但模型版本不一致，忽略断点重新训练。")
        return 1, -1.0, 0, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if state.get("label_to_idx") != label_to_idx:
        print("发现旧训练断点，但类别映射不一致，忽略断点重新训练。")
        return 1, -1.0, 0, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    model.load_state_dict(state["model_state_dict"], strict=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    set_rng_state(state.get("rng_state"))

    next_epoch = int(state.get("next_epoch", 1))
    best_acc = float(state.get("best_acc", -1.0))
    patience = int(state.get("patience", 0))
    history = state.get("history") or {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"已从训练断点恢复: next_epoch={next_epoch}, best_val_acc={best_acc:.4f}")
    return next_epoch, best_acc, patience, history


def main():
    set_seed()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    resolve_owsm_model_dir(OWSM_MODEL_DIR)
    samples, label_names = scan_audio_files(RAW_DATA_DIR)
    if not samples:
        print(f"未在 {RAW_DATA_DIR} 发现训练音频。")
        return

    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    train_samples, val_samples, test_samples = split_samples(samples)
    train_samples = limit_per_class(train_samples, MAX_TRAIN_SAMPLES_PER_CLASS)

    if USE_VOXLINGUA_DOMAIN_ADAPTATION:
        domain_train = limit_per_class(
            load_manifest_split(VOXLINGUA_MANIFEST_PATH, "train", set(label_names)),
            MAX_DOMAIN_TRAIN_SAMPLES_PER_CLASS,
        )
        domain_val = limit_per_class(
            load_manifest_split(VOXLINGUA_MANIFEST_PATH, "val", set(label_names)),
            MAX_DOMAIN_VAL_SAMPLES_PER_CLASS,
        )
        train_samples.extend(domain_train)
        val_samples.extend(domain_val)

    train_samples = filter_readable_samples(train_samples, "train")
    val_samples = filter_readable_samples(val_samples, "val")
    test_samples = filter_readable_samples(test_samples, "test")

    if not train_samples:
        raise RuntimeError("过滤坏音频后训练集为空，请检查数据目录。")
    if not val_samples:
        raise RuntimeError("过滤坏音频后验证集为空，请检查数据目录。")

    print(f"模型: {MODEL_DISPLAY_NAME}")
    print(f"类别: {label_names}")
    print(f"训练集: {len(train_samples)}，验证集: {len(val_samples)}，测试集: {len(test_samples)}")

    device = get_device()
    print(f"使用设备: {device}")
    model = build_model(num_classes=len(label_to_idx), device=device)

    train_dataset = AudioLanguageDataset(
        train_samples,
        label_to_idx=label_to_idx,
        duration=DURATION,
        random_crop=True,
        augment=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    class_weights = compute_class_weight_tensor(train_samples, label_to_idx).to(device)
    print(f"类别权重: {[round(float(x), 4) for x in class_weights.cpu()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.trainable_parameters(), lr=ADAPTER_LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE)

    best_state = None
    start_epoch, best_acc, patience, history = load_training_resume_state(
        model,
        optimizer,
        scheduler,
        label_to_idx,
        device,
    )

    if start_epoch > EPOCHS:
        print(f"训练断点显示已完成 {EPOCHS} 个 epoch，无需继续训练。")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_logits, val_labels = evaluate_samples(model, val_samples, label_to_idx, criterion, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={[group['lr'] for group in optimizer.param_groups]}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            logit_bias = fit_logit_bias(val_logits, val_labels, num_classes=len(label_to_idx), steps=200)
            best_state = {
                "model_version": MODEL_VERSION,
                "model_name": MODEL_DISPLAY_NAME,
                "model_state_dict": model.state_dict(),
                "num_classes": len(label_to_idx),
                "label_to_idx": label_to_idx,
                "duration": DURATION,
                "sample_rate": 16000,
                "encoder_dim": model.encoder_dim,
                "adapter_bottleneck": model.adapter_bottleneck,
                "adapter_layers": model.adapter_layers_count,
                "embedding_dim": model.embedding_dim,
                "logit_bias": logit_bias,
            }
            torch.save(best_state, BEST_MODEL_PATH)
            with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
                json.dump(label_to_idx, f, ensure_ascii=False, indent=2)
            print(f"已保存最佳模型: {BEST_MODEL_PATH}")
        else:
            patience += 1

        save_training_resume_state(
            model,
            optimizer,
            scheduler,
            epoch,
            best_acc,
            patience,
            history,
            label_to_idx,
        )
        print(f"已保存训练断点: epoch={epoch}, 下次从 epoch={epoch + 1} 继续")

        if epoch >= MIN_EPOCHS_BEFORE_EARLY_STOP and patience >= EARLY_STOP_PATIENCE:
            print(f"触发 Early Stopping：连续 {patience} 个 epoch 验证集未提升。")
            break

    save_training_curve(history)
    print(f"训练曲线已保存: {TRAINING_CURVE_PATH}")
    print(f"最佳验证准确率: {best_acc:.4f}")
    if best_state is None and not BEST_MODEL_PATH.exists():
        print("训练未产生可用 checkpoint，请检查数据和模型加载。")
    else:
        print("训练完成。可继续运行 evaluate_owsm_adapter_lid.py 进行评估。")


if __name__ == "__main__":
    main()
