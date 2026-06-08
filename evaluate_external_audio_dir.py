"""Evaluate OWSM v4 adapter model on external VoxLingua107 audio."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.audio_dataset import AUDIO_EXTENSIONS, create_audio_windows, load_and_resample
from src.logit_calibration import apply_logit_bias
from src.owsm_adapter_lid_model import MODEL_VERSION, OWSMAdapterLanguageClassifier
from src.owsm_local_model import OWSM_MODEL_DIR, resolve_owsm_model_dir

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_owsm_adapter_lid.pt"
LABEL_MAP_PATH = CHECKPOINT_DIR / "label_map.json"
REPORT_PATH = OUTPUT_DIR / "external_voxlingua_report.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "external_voxlingua_confusion_matrix.png"
PREDICTIONS_CSV_PATH = OUTPUT_DIR / "external_voxlingua_predictions.csv"

DEVICE = "cuda"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external_test" / "VoxLingua107"
VOXLINGUA_MANIFEST_PATH = PROJECT_ROOT / "data" / "voxlingua_domain" / "voxlingua_split.csv"
USE_MANIFEST_TEST_SPLIT = True
EXTERNAL_SPLIT_NAME = "test"
PREDICT_WINDOW_STRIDE = 5.0
MAX_PREDICT_WINDOWS = 8
PRINT_EACH_SAMPLE = False
EMPTY_CACHE_EVERY_SAMPLE = True
RESUME_EXTERNAL_EVAL = True
PREDICTION_FIELDS = ["path", "true_label", "pred_label", "confidence", "correct"]


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 DEVICE='cuda'，但当前环境没有可用 GPU。")
        return torch.device("cuda")
    return torch.device("cpu")


def scan_external_samples(data_dir: Path):
    samples = []
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for audio_path in sorted(label_dir.rglob("*")):
            if audio_path.is_file() and audio_path.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append((audio_path, label_dir.name))
    return samples


def load_manifest_samples(manifest_path: Path, split_name: str):
    if not manifest_path.exists():
        return []
    samples = []
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split_name:
                continue
            audio_path = Path(row.get("path", ""))
            label = row.get("label", "")
            if audio_path.exists() and label:
                samples.append((audio_path, label))
    return samples


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


def predict_windows_low_memory(model, windows, device):
    logits_list = []

    for window in windows:
        waveform = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(waveform, dtype=torch.long, device=device)
        logits = model(waveform, attention_mask=attention_mask)
        logits_list.append(logits.detach().cpu())

        del waveform, attention_mask, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(logits_list, dim=0).mean(dim=0, keepdim=True).to(device)


def load_existing_predictions(label_to_idx):
    if not RESUME_EXTERNAL_EVAL or not PREDICTIONS_CSV_PATH.exists():
        return [], [], [], set()

    rows = []
    y_true = []
    y_pred = []
    completed_paths = set()

    with open(PREDICTIONS_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("path", "")
            true_label = row.get("true_label", "")
            pred_label = row.get("pred_label", "")
            if not path or true_label not in label_to_idx or pred_label not in label_to_idx:
                continue

            rows.append(row)
            completed_paths.add(path)
            y_true.append(label_to_idx[true_label])
            y_pred.append(label_to_idx[pred_label])

    if completed_paths:
        print(f"已加载外部评估断点: {len(completed_paths)} 条已完成样本")
    return y_true, y_pred, rows, completed_paths


def append_prediction_row(row):
    if not RESUME_EXTERNAL_EVAL:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = PREDICTIONS_CSV_PATH.exists() and PREDICTIONS_CSV_PATH.stat().st_size > 0
    with open(PREDICTIONS_CSV_PATH, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def main():
    if not BEST_MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
        print("未找到模型文件或 label_map.json，请先运行 train_owsm_adapter_lid.py。")
        return
    if not EXTERNAL_DATA_DIR.exists():
        print(f"外部测试目录不存在: {EXTERNAL_DATA_DIR}")
        print("请先运行 download_voxlingua_samples.py。")
        return

    samples = []
    if USE_MANIFEST_TEST_SPLIT:
        samples = load_manifest_samples(VOXLINGUA_MANIFEST_PATH, EXTERNAL_SPLIT_NAME)
        if samples:
            print(f"使用 VoxLingua107 manifest 的 {EXTERNAL_SPLIT_NAME} split 进行外部评估。")
        else:
            print("未找到有效 manifest test split，将回退为扫描整个外部目录。")
    if not samples:
        samples = scan_external_samples(EXTERNAL_DATA_DIR)
    if not samples:
        print(f"外部测试目录中没有音频: {EXTERNAL_DATA_DIR}")
        return

    device = get_device(DEVICE)
    model, label_to_idx, idx_to_label, checkpoint = load_trained_model(device)
    ordered_labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    duration = checkpoint.get("duration", 10.0)
    sample_rate = checkpoint.get("sample_rate", 16000)
    logit_bias = checkpoint.get("logit_bias")

    unknown_labels = sorted({label for _, label in samples if label not in label_to_idx})
    if unknown_labels:
        raise RuntimeError(f"外部测试目录包含训练集中不存在的标签: {unknown_labels}")

    y_true, y_pred, rows, completed_paths = load_existing_predictions(label_to_idx)
    print(f"使用设备: {device}")
    print(f"外部测试目录: {EXTERNAL_DATA_DIR}")
    print(f"外部样本数: {len(samples)}")

    skipped = 0
    for audio_path, label_name in tqdm(samples, desc="external eval"):
        audio_key = str(audio_path)
        if audio_key in completed_paths:
            continue

        try:
            audio = load_and_resample(audio_path, target_sr=sample_rate)
            windows = create_audio_windows(
                audio,
                duration=duration,
                sr=sample_rate,
                stride=PREDICT_WINDOW_STRIDE,
                max_windows=MAX_PREDICT_WINDOWS,
            )
            logits = predict_windows_low_memory(model, windows, device)
            logits = apply_logit_bias(logits, logit_bias)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_idx = int(probs.argmax().item())
            true_idx = label_to_idx[label_name]
            confidence = float(probs[pred_idx])
        except RuntimeError as exc:
            skipped += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"跳过评估样本: {audio_path}，原因: {exc}")
            continue
        except Exception as exc:
            skipped += 1
            print(f"跳过评估样本: {audio_path}，原因: {exc}")
            continue

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        row = {
            "path": audio_key,
            "true_label": label_name,
            "pred_label": idx_to_label[pred_idx],
            "confidence": confidence,
            "correct": pred_idx == true_idx,
        }
        rows.append(row)
        completed_paths.add(audio_key)
        append_prediction_row(row)
        if PRINT_EACH_SAMPLE:
            print(f"{audio_path.name:>8} | true={label_name:<8} pred={idx_to_label[pred_idx]:<8} conf={confidence:.4f}")
        if EMPTY_CACHE_EVERY_SAMPLE and device.type == "cuda":
            torch.cuda.empty_cache()

    if skipped:
        print(f"外部评估跳过样本数: {skipped}")

    if not y_true:
        raise RuntimeError("没有可用的外部评估样本，请检查音频文件或显存设置。")

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=ordered_labels, digits=4, zero_division=0)
    print(f"\nExternal Test Accuracy: {acc:.4f}")
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"External Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
    with open(PREDICTIONS_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=ordered_labels).plot(
        cmap="Blues",
        xticks_rotation=30,
        ax=ax,
        colorbar=True,
    )
    ax.set_title("External Confusion Matrix (OWSM v4 Adapter)")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close(fig)

    print(f"报告已保存: {REPORT_PATH}")
    print(f"混淆矩阵已保存: {CONFUSION_MATRIX_PATH}")
    print(f"逐条预测已保存: {PREDICTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()
