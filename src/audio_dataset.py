"""Audio loading, scanning, windowing, and lightweight augmentation utilities."""

from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


def scan_audio_files(raw_data_dir: Path) -> Tuple[List[Tuple[Path, str]], List[str]]:
    raw_data_dir = Path(raw_data_dir)
    samples: List[Tuple[Path, str]] = []
    if not raw_data_dir.exists():
        return samples, []

    label_names = sorted(path.name for path in raw_data_dir.iterdir() if path.is_dir())
    for label in label_names:
        for audio_path in sorted((raw_data_dir / label).rglob("*")):
            if audio_path.is_file() and audio_path.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append((audio_path, label))
    return samples, label_names


def load_and_resample(audio_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio


def fix_audio_length(audio: np.ndarray, duration: float, sr: int = SAMPLE_RATE, random_crop: bool = False) -> np.ndarray:
    target_length = int(duration * sr)
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), mode="constant").astype(np.float32)
    if len(audio) > target_length:
        start = 0
        if random_crop:
            start = int(np.random.randint(0, len(audio) - target_length + 1))
        return audio[start : start + target_length].astype(np.float32)
    return audio.astype(np.float32)


def create_audio_windows(
    audio: np.ndarray,
    duration: float,
    sr: int = SAMPLE_RATE,
    stride: float = 5.0,
    max_windows: int = 6,
) -> List[np.ndarray]:
    target_length = int(duration * sr)
    if len(audio) <= target_length:
        return [fix_audio_length(audio, duration=duration, sr=sr, random_crop=False)]

    step = max(1, int(stride * sr))
    starts = list(range(0, len(audio) - target_length + 1, step))
    last_start = len(audio) - target_length
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    if max_windows and len(starts) > max_windows:
        indices = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in indices]

    return [audio[start : start + target_length].astype(np.float32) for start in starts]


def augment_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    augmented = audio.astype(np.float32, copy=True)
    augmented *= np.random.uniform(0.85, 1.15)

    if np.random.rand() < 0.5 and augmented.size > 1:
        max_shift = max(1, int(0.1 * sr))
        shift = int(np.random.randint(-max_shift, max_shift + 1))
        if shift > 0:
            augmented = np.pad(augmented[:-shift], (shift, 0), mode="constant")
        elif shift < 0:
            augmented = np.pad(augmented[-shift:], (0, -shift), mode="constant")

    if np.random.rand() < 0.35:
        rms = float(np.sqrt(np.mean(augmented ** 2))) if augmented.size else 0.0
        if rms > 0:
            snr_db = np.random.uniform(18.0, 30.0)
            noise_rms = rms / (10 ** (snr_db / 20.0))
            augmented += np.random.normal(0.0, noise_rms, size=augmented.shape).astype(np.float32)

    peak = float(np.max(np.abs(augmented))) if augmented.size else 0.0
    if peak > 1.0:
        augmented = augmented / peak
    return augmented.astype(np.float32)


class AudioLanguageDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, str]],
        label_to_idx: Dict[str, int],
        duration: float,
        random_crop: bool = False,
        augment: bool = False,
    ):
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.duration = duration
        self.random_crop = random_crop
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        audio_path, label_name = self.samples[index]
        audio = load_and_resample(audio_path)
        audio = fix_audio_length(audio, duration=self.duration, random_crop=self.random_crop)
        if self.augment:
            audio = augment_audio(audio)
        return torch.from_numpy(audio).float(), torch.tensor(self.label_to_idx[label_name], dtype=torch.long)
