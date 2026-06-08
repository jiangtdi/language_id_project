"""Predict one audio file with the trained OWSM v4 adapter LID model."""

import json
from pathlib import Path

import numpy as np
import torch

from src.audio_dataset import create_audio_windows, load_and_resample
from src.logit_calibration import apply_logit_bias
from src.owsm_adapter_lid_model import MODEL_VERSION, OWSMAdapterLanguageClassifier
from src.owsm_local_model import OWSM_MODEL_DIR, resolve_owsm_model_dir

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_owsm_adapter_lid.pt"
LABEL_MAP_PATH = CHECKPOINT_DIR / "label_map.json"

DEVICE = "cuda"
AUDIO_PATH =""
PREDICT_WINDOW_STRIDE = 5.0
MAX_PREDICT_WINDOWS = 6


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 DEVICE='cuda'，但当前环境没有可用 GPU。")
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_model(device: torch.device):
    if not BEST_MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
        raise FileNotFoundError("请先运行 train_owsm_adapter_lid.py 训练模型。")

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
    return model, idx_to_label, checkpoint


@torch.no_grad()
def predict_audio(audio_path: Path, device: torch.device):
    model, idx_to_label, checkpoint = load_trained_model(device)
    duration = checkpoint.get("duration", 10.0)
    sample_rate = checkpoint.get("sample_rate", 16000)
    logit_bias = checkpoint.get("logit_bias")

    audio = load_and_resample(audio_path, target_sr=sample_rate)
    windows = create_audio_windows(
        audio,
        duration=duration,
        sr=sample_rate,
        stride=PREDICT_WINDOW_STRIDE,
        max_windows=MAX_PREDICT_WINDOWS,
    )
    waveforms = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
    attention_mask = torch.ones_like(waveforms, dtype=torch.long, device=device)
    logits = model(waveforms, attention_mask=attention_mask).mean(dim=0, keepdim=True)
    logits = apply_logit_bias(logits, logit_bias)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return {idx_to_label[i]: float(probs[i]) for i in range(len(probs))}


def main():
    if not AUDIO_PATH:
        print("请先在 predict_owsm_adapter_lid.py 里设置 AUDIO_PATH。")
        return
    audio_path = Path(AUDIO_PATH)
    if not audio_path.exists():
        print(f"音频文件不存在: {audio_path}")
        return
    device = get_device(DEVICE)
    probabilities = predict_audio(audio_path, device)
    sorted_items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    print(f"使用设备: {device}")
    print(f"预测语种: {sorted_items[0][0]}")
    print(f"置信度: {sorted_items[0][1]:.4f}")
    for label, prob in sorted_items:
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
