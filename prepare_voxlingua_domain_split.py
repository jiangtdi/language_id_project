"""
Create a reproducible VoxLingua107 domain-adaptation split.

This script does not copy audio files. It writes a CSV manifest that points to
the already extracted files under data/external_test/VoxLingua107.

Run order in PyCharm:
1) download_voxlingua_samples.py
2) prepare_voxlingua_domain_split.py
3) train_owsm_adapter_lid.py
4) evaluate_external_audio_dir.py
"""

import csv
import random
from pathlib import Path

from src.audio_dataset import AUDIO_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "data" / "external_test" / "VoxLingua107"
OUTPUT_DIR = PROJECT_ROOT / "data" / "voxlingua_domain"
MANIFEST_PATH = OUTPUT_DIR / "voxlingua_split.csv"

# ========== PyCharm 里可以直接改这里 ==========
RANDOM_SEED = 42
TRAIN_PER_CLASS = 5000
VAL_PER_CLASS = 1000
TEST_PER_CLASS = 2000
LABELS = ["Chinese", "English", "French", "Japanese", "Korean"]
# =============================================


def scan_label_files(label_dir: Path):
    return sorted(
        path
        for path in label_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"VoxLingua107 目录不存在，请先运行 download_voxlingua_samples.py: {SOURCE_DIR}")

    rng = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"读取 VoxLingua107: {SOURCE_DIR}")
    print(f"输出划分清单: {MANIFEST_PATH}")

    for label in LABELS:
        files = scan_label_files(SOURCE_DIR / label)
        required = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
        if len(files) < required:
            raise RuntimeError(
                f"{label} 音频不足：当前 {len(files)} 条，需要至少 {required} 条。"
                "请减少 TRAIN_PER_CLASS / VAL_PER_CLASS / TEST_PER_CLASS。"
            )

        rng.shuffle(files)
        split_specs = [
            ("train", files[:TRAIN_PER_CLASS]),
            ("val", files[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VAL_PER_CLASS]),
            (
                "test",
                files[
                    TRAIN_PER_CLASS + VAL_PER_CLASS :
                    TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
                ],
            ),
        ]

        for split_name, split_files in split_specs:
            for audio_path in split_files:
                rows.append(
                    {
                        "path": str(audio_path),
                        "label": label,
                        "split": split_name,
                    }
                )
            print(f"{label:<8} {split_name:<5}: {len(split_files)}")

    with open(MANIFEST_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print("完成：后续 train_owsm_adapter_lid.py 会自动读取该清单进行域适配训练。")


if __name__ == "__main__":
    main()
