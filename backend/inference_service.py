"""Inference service for the trained OWSM adapter language ID model."""

import json
import threading
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.audio_dataset import create_audio_windows, load_and_resample
from src.logit_calibration import apply_logit_bias
from src.owsm_adapter_lid_model import MODEL_VERSION, OWSMAdapterLanguageClassifier
from src.owsm_local_model import MODEL_DISPLAY_NAME, OWSM_MODEL_DIR, resolve_owsm_model_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_owsm_adapter_lid.pt"
LABEL_MAP_PATH = CHECKPOINT_DIR / "label_map.json"

DEVICE = "cuda"
PREDICT_WINDOW_STRIDE = 5.0
MAX_PREDICT_WINDOWS = 2


class LanguageIdService:
    def __init__(self):
        self.device = self._get_device(DEVICE)
        self.model = None
        self.idx_to_label: Dict[int, str] = {}
        self.checkpoint = {}
        self._load_lock = threading.Lock()

    @staticmethod
    def _get_device(device_arg: str) -> torch.device:
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_arg == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            if not BEST_MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
                raise FileNotFoundError("请先训练模型，确保 checkpoints/best_owsm_adapter_lid.pt 和 label_map.json 存在。")

            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
                label_to_idx = json.load(file)
            self.idx_to_label = {int(idx): label for label, idx in label_to_idx.items()}

            self.checkpoint = torch.load(BEST_MODEL_PATH, map_location=self.device)
            if self.checkpoint.get("model_version") != MODEL_VERSION:
                raise RuntimeError("当前 checkpoint 不是 OWSM v4 adapter 版本，请重新训练。")

            model = OWSMAdapterLanguageClassifier(
                num_classes=self.checkpoint.get("num_classes", len(label_to_idx)),
                model_dir=resolve_owsm_model_dir(OWSM_MODEL_DIR),
                owsm_device=str(self.device),
                encoder_dim=self.checkpoint["encoder_dim"],
                adapter_bottleneck=self.checkpoint.get("adapter_bottleneck", 128),
                adapter_layers=self.checkpoint.get("adapter_layers", 2),
                embedding_dim=self.checkpoint.get("embedding_dim", 256),
            ).to(self.device)
            model.load_state_dict(self.checkpoint["model_state_dict"])
            model.eval()
            self.model = model

    def model_info(self) -> Dict[str, object]:
        if not self.idx_to_label:
            if LABEL_MAP_PATH.exists():
                with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
                    label_to_idx = json.load(file)
                self.idx_to_label = {int(idx): label for label, idx in label_to_idx.items()}
            else:
                self.idx_to_label = {index: label for index, label in enumerate(["Chinese", "English", "French", "Japanese", "Korean"])}
        return {
            "model_name": MODEL_DISPLAY_NAME,
            "supported_languages": [self.idx_to_label[index] for index in sorted(self.idx_to_label)],
            "device": str(self.device),
            "model_loaded": self.model is not None,
        }

    @torch.no_grad()
    def predict_path(self, audio_path: Path) -> Dict[str, object]:
        self.load()
        duration = self.checkpoint.get("duration", 10.0)
        sample_rate = self.checkpoint.get("sample_rate", 16000)
        logit_bias = self.checkpoint.get("logit_bias")

        audio = load_and_resample(audio_path, target_sr=sample_rate)
        windows = create_audio_windows(
            audio,
            duration=duration,
            sr=sample_rate,
            stride=PREDICT_WINDOW_STRIDE,
            max_windows=MAX_PREDICT_WINDOWS,
        )

        logits_list: List[torch.Tensor] = []
        for window in windows:
            waveform = torch.tensor(window[None, :], dtype=torch.float32, device=self.device)
            attention_mask = torch.ones_like(waveform, dtype=torch.long, device=self.device)
            logits = self.model(waveform, attention_mask=attention_mask)
            logits_list.append(logits.detach().cpu())
            del waveform, attention_mask, logits
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        logits = torch.cat(logits_list, dim=0).mean(dim=0, keepdim=True).to(self.device)
        logits = apply_logit_bias(logits, logit_bias)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        items = [
            {"label": self.idx_to_label[index], "probability": float(probabilities[index])}
            for index in range(len(probabilities))
        ]
        items.sort(key=lambda item: item["probability"], reverse=True)
        return {
            "predicted_language": items[0]["label"],
            "confidence": items[0]["probability"],
            "probabilities": items,
        }

    def predict_bytes(self, content: bytes, suffix: str = ".wav") -> Dict[str, object]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
            file.write(content)
            temp_path = Path(file.name)
        try:
            return self.predict_path(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


language_id_service = LanguageIdService()
