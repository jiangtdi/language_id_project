"""
FLEURS 五语种数据集下载脚本。

PyCharm 直接运行即可。
如果想改下载数量，只需要修改下面“可直接修改的参数”区域。
"""

import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

# ========== 可直接修改的参数（PyCharm 友好） ==========
# 为了让 OWSM v4 + Conv-Adapter PEFT 更稳定，建议每类至少 800~1500 条。
TARGET_SAMPLES_PER_LANG = 1000

# 使用 FLEURS 的多个 split 扩充数据量。
FLEURS_SPLITS: Tuple[str, ...] = ("train", "validation", "test")

# True：允许本脚本联网下载 FLEURS 数据集。
# 注意：这只影响数据集下载，不影响本地加载 owsm_ctc_v4_1B 模型。
ALLOW_HF_DOWNLOAD = True

# True：删除 data/raw/ 下已有对应语种目录后重新下载。
# False：保留已有 wav，只补足到 TARGET_SAMPLES_PER_LANG。
RESET_DATA_DIR = False
# =====================================================

LANGUAGE_CONFIGS: Dict[str, str] = {
    "Chinese": "cmn_hans_cn",
    "English": "en_us",
    "Japanese": "ja_jp",
    "Korean": "ko_kr",
    "French": "fr_fr",
}

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def save_audio_sample(sample: dict, output_path: Path) -> bool:
    """保存单条 FLEURS 样本为 wav 文件。"""
    if "audio" not in sample or sample["audio"] is None:
        return False

    audio = sample["audio"]
    audio_array = audio.get("array")
    sampling_rate = audio.get("sampling_rate")
    if audio_array is None or sampling_rate is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, np.asarray(audio_array, dtype=np.float32), sampling_rate)
    return True


def existing_wav_count(output_dir: Path) -> int:
    return len(list(output_dir.glob("*.wav"))) if output_dir.exists() else 0


def prepare_output_dir(output_dir: Path) -> int:
    if RESET_DATA_DIR and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return existing_wav_count(output_dir)


def load_fleurs_split(fleurs_config: str, split: str):
    dataset = load_dataset(
        "google/fleurs",
        fleurs_config,
        split=split,
        streaming=True,
    )
    return dataset.cast_column("audio", Audio())


def download_language(language_name: str, fleurs_config: str, splits: Iterable[str]) -> int:
    """下载并导出一个语种的多个 FLEURS split，直到达到目标样本数。"""
    if not ALLOW_HF_DOWNLOAD:
        raise RuntimeError("ALLOW_HF_DOWNLOAD=False，当前脚本禁止联网下载。")

    output_dir = RAW_DATA_DIR / language_name
    existing_count = prepare_output_dir(output_dir)
    target_new_count = max(0, TARGET_SAMPLES_PER_LANG - existing_count)

    if target_new_count == 0:
        print(f"{language_name}: 已有 {existing_count} 条，达到目标 {TARGET_SAMPLES_PER_LANG} 条，跳过。")
        return existing_count

    exported_count = existing_count
    saved_this_run = 0
    skipped_count = 0
    seen_before_this_run = 0

    progress = tqdm(
        total=target_new_count,
        desc=f"{language_name} ({fleurs_config})",
        unit="file",
    )

    try:
        for split in splits:
            if exported_count >= TARGET_SAMPLES_PER_LANG:
                break

            dataset = load_fleurs_split(fleurs_config, split)
            for sample_index, sample in enumerate(dataset):
                if exported_count >= TARGET_SAMPLES_PER_LANG:
                    break

                if seen_before_this_run < existing_count:
                    seen_before_this_run += 1
                    continue

                output_path = output_dir / f"{fleurs_config}_{exported_count:04d}.wav"
                try:
                    if save_audio_sample(sample, output_path):
                        exported_count += 1
                        saved_this_run += 1
                        progress.update(1)
                    else:
                        skipped_count += 1
                    progress.set_postfix(split=split, saved=saved_this_run, skipped=skipped_count)
                except Exception as exc:
                    skipped_count += 1
                    progress.set_postfix(split=split, saved=saved_this_run, skipped=skipped_count)
                    print(f"\n跳过异常音频样本：{language_name}/{split}/{sample_index}，原因：{exc}")
    finally:
        progress.close()

    return exported_count


def main() -> None:
    print("开始下载并整理 FLEURS 五语种数据集")
    print(f"目标：每个语种 {TARGET_SAMPLES_PER_LANG} 条 wav")
    print(f"使用 split: {', '.join(FLEURS_SPLITS)}")
    print(f"是否重置旧数据目录: {RESET_DATA_DIR}")
    print("=" * 60)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    exported_summary: Dict[str, int] = {}

    for language_name, fleurs_config in LANGUAGE_CONFIGS.items():
        try:
            exported_summary[language_name] = download_language(language_name, fleurs_config, FLEURS_SPLITS)
        except Exception as exc:
            exported_summary[language_name] = existing_wav_count(RAW_DATA_DIR / language_name)
            print(f"\n下载失败：{language_name} ({fleurs_config})，错误：{exc}")

    print("=" * 60)
    print("导出统计：")
    for language_name, count in exported_summary.items():
        print(f"{language_name}: {count} 条 wav 音频")

    insufficient_langs = [
        language_name
        for language_name, count in exported_summary.items()
        if count < TARGET_SAMPLES_PER_LANG
    ]
    print("=" * 60)
    if insufficient_langs:
        print(f"以下语种未达到目标数量：{', '.join(insufficient_langs)}")
        print("请检查网络/HuggingFace 数据集访问，或降低 TARGET_SAMPLES_PER_LANG。")
    else:
        print("数据集扩充完成，可以重新运行 train_owsm_adapter_lid.py。")


if __name__ == "__main__":
    main()
