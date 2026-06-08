"""Evaluate the trained OWSM v4 adapter LID model on the held-out FLEURS split."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.audio_dataset import create_audio_windows, load_and_resample, scan_audio_files
from src.logit_calibration import apply_logit_bias
from src.owsm_adapter_lid_model import MODEL_VERSION, OWSMAdapterLanguageClassifier
from src.owsm_local_model import OWSM_MODEL_DIR, resolve_owsm_model_dir

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_owsm_adapter_lid.pt"
LABEL_MAP_PATH = CHECKPOINT_DIR / "label_map.json"
REPORT_PATH = OUTPUT_DIR / "owsm_adapter_report.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "owsm_adapter_confusion_matrix.png"

RANDOM_SEED = 42
DEVICE = "cuda"
EVAL_WINDOW_STRIDE = 5.0
MAX_EVAL_WINDOWS = 6


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 DEVICE='cuda'，但当前环境没有可用 GPU。")
        return torch.device("cuda")
    return torch.device("cpu")


def build_test_split(samples):
    labels = [label for _, label in samples]
    stratify = labels if len(set(labels)) > 1 and min(labels.count(x) for x in set(labels)) >= 2 else None
    _, temp_samples = train_test_split(samples, test_size=0.3, random_state=RANDOM_SEED, stratify=stratify)
    temp_labels = [label for _, label in temp_samples]
    temp_stratify = (
        temp_labels if len(set(temp_labels)) > 1 and min(temp_labels.count(x) for x in set(temp_labels)) >= 2 else None
    )
    _, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_stratify)
    return test_samples


def load_trained_model(device: torch.device):
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)
    idx_to_label = {int(idx): label for label, idx in label_to_idx.items()}

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise RuntimeError("当前 checkpoint 不是 OWSM v4 adapter 版本，请重新训练。")

    model = OWSMAdapterLanguageClassifier(
        num_classes=checkpoint.get("num_classes", len(label_to_idx)),
        model_dir=resolve_owsm_model_dir(OWSM_MODEL_DIR),
        owsm_device=str(device),
        encoder_dim=checkpoint["encoder_dim"],
        adapter_bottleneck=checkpoint.get("adapter_bottleneck", 128),
        adapter_layers=checkpoint.get("adapter_layers", 2),
        embedding_dim=checkpoint.get("embedding_dim", 256),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, label_to_idx, idx_to_label, checkpoint


@torch.no_grad()
def main():
    if not BEST_MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
        print("未找到 OWSM adapter 模型，请先运行 python train_owsm_adapter_lid.py")
        return

    samples, _ = scan_audio_files(RAW_DATA_DIR)
    if not samples:
        print("未发现可用音频数据。")
        return

    device = get_device(DEVICE)
    model, label_to_idx, idx_to_label, checkpoint = load_trained_model(device)
    ordered_labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    test_samples = build_test_split(samples)
    duration = checkpoint.get("duration", 10.0)
    sample_rate = checkpoint.get("sample_rate", 16000)
    logit_bias = checkpoint.get("logit_bias")

    y_true, y_pred = [], []
    for audio_path, label_name in test_samples:
        audio = load_and_resample(audio_path, target_sr=sample_rate)
        windows = create_audio_windows(
            audio,
            duration=duration,
            sr=sample_rate,
            stride=EVAL_WINDOW_STRIDE,
            max_windows=MAX_EVAL_WINDOWS,
        )
        waveforms = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
        attention_mask = torch.ones_like(waveforms, dtype=torch.long, device=device)
        logits = model(waveforms, attention_mask=attention_mask).mean(dim=0, keepdim=True)
        logits = apply_logit_bias(logits, logit_bias)
        y_true.append(label_to_idx[label_name])
        y_pred.append(int(logits.argmax(dim=1).item()))

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=ordered_labels, digits=4, zero_division=0)

    print(f"使用设备: {device}")
    print("当前使用 OWSM v4 + Conv-Adapter PEFT + Temporal Attention Pooling + Angular Margin 分类头。")
    print(f"Test Accuracy: {acc:.4f}")
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=ordered_labels).plot(
        cmap="Blues",
        xticks_rotation=30,
        ax=ax,
        colorbar=True,
    )
    ax.set_title("Confusion Matrix (OWSM v4 Adapter LID)")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
