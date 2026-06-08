import csv
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_DIR = PROJECT_ROOT / "data"
TARGET_DATA_DIR = PROJECT_ROOT / "data1"

LANGUAGES = ["Chinese", "English", "French", "Japanese", "Korean"]

BASE_SAMPLES_PER_CLASS = 300
VOXLINGUA_SAMPLES_PER_CLASS = 300
RANDOM_SEED = 42

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

VOXLINGUA_DIR = SOURCE_DATA_DIR / "voxlingua_domain"
VOXLINGUA_MANIFEST = VOXLINGUA_DIR / "voxlingua_split.csv"

EXCLUDE_DIR_NAMES = {
    "data1",
    "external_test",
    "_downloads",
    "voxlingua_domain",
}


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES


def reset_target_dir() -> None:
    if TARGET_DATA_DIR.exists():
        shutil.rmtree(TARGET_DATA_DIR)
    TARGET_DATA_DIR.mkdir(parents=True, exist_ok=True)


def collect_base_files(language: str) -> list[Path]:
    files: list[Path] = []
    for path in SOURCE_DATA_DIR.rglob("*"):
        if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        if is_audio_file(path) and language in path.parts:
            files.append(path)
    return sorted(files)


def detect_manifest_columns(fieldnames: list[str]) -> tuple[str, str, str | None]:
    path_candidates = ["path", "audio_path", "file", "filepath", "wav", "audio"]
    label_candidates = ["label", "language", "lang", "class", "category"]
    split_candidates = ["split", "subset"]

    path_column = next((name for name in path_candidates if name in fieldnames), None)
    label_column = next((name for name in label_candidates if name in fieldnames), None)
    split_column = next((name for name in split_candidates if name in fieldnames), None)

    if path_column is None or label_column is None:
        raise RuntimeError(
            "无法识别 voxlingua_split.csv 的列名，至少需要音频路径列和标签列。"
        )

    return path_column, label_column, split_column


def resolve_audio_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    first_try = PROJECT_ROOT / path
    if first_try.exists():
        return first_try
    return VOXLINGUA_DIR / path


def collect_voxlingua_train_files(language: str) -> list[Path]:
    if not VOXLINGUA_MANIFEST.exists():
        print(f"未找到 VoxLingua 清单，跳过: {VOXLINGUA_MANIFEST}")
        return []

    with VOXLINGUA_MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise RuntimeError(f"清单为空: {VOXLINGUA_MANIFEST}")

        path_column, label_column, split_column = detect_manifest_columns(reader.fieldnames)
        files: list[Path] = []

        for row in reader:
            if row.get(label_column) != language:
                continue
            if split_column is not None and row.get(split_column) != "train":
                continue

            audio_path = resolve_audio_path(row[path_column])
            if is_audio_file(audio_path):
                files.append(audio_path)

    return sorted(files)


def copy_files(group_name: str, language: str, selected_files: list[Path]) -> None:
    target_lang_dir = TARGET_DATA_DIR / group_name / language
    target_lang_dir.mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(selected_files, start=1):
        target_path = target_lang_dir / f"{language}_{index:04d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, target_path)


def sample_files(files: list[Path], count: int, name: str) -> list[Path]:
    if len(files) < count:
        raise RuntimeError(f"{name} 可用音频不足 {count} 条，当前只有 {len(files)} 条。")
    return random.sample(files, count)


def main() -> None:
    if not SOURCE_DATA_DIR.exists():
        raise FileNotFoundError(f"源数据目录不存在: {SOURCE_DATA_DIR}")

    random.seed(RANDOM_SEED)
    reset_target_dir()

    print(f"源数据目录: {SOURCE_DATA_DIR}")
    print(f"目标目录: {TARGET_DATA_DIR}")
    print(f"原始数据集每类抽取: {BASE_SAMPLES_PER_CLASS}")
    print(f"VoxLingua 每类抽取: {VOXLINGUA_SAMPLES_PER_CLASS}")
    print("=" * 70)

    total = 0
    for language in LANGUAGES:
        base_files = collect_base_files(language)
        base_selected = sample_files(
            base_files,
            BASE_SAMPLES_PER_CLASS,
            f"base/{language}",
        )
        copy_files("base", language, base_selected)
        total += len(base_selected)
        print(f"base/{language}: 找到 {len(base_files)} 条，复制 {len(base_selected)} 条")

        voxlingua_files = collect_voxlingua_train_files(language)
        voxlingua_selected = sample_files(
            voxlingua_files,
            VOXLINGUA_SAMPLES_PER_CLASS,
            f"voxlingua_domain/{language}",
        )
        copy_files("voxlingua_domain", language, voxlingua_selected)
        total += len(voxlingua_selected)
        print(
            f"voxlingua_domain/{language}: 找到 {len(voxlingua_files)} 条，"
            f"复制 {len(voxlingua_selected)} 条"
        )

    print("=" * 70)
    print(f"完成：共复制 {total} 条音频到 {TARGET_DATA_DIR}")
    print("上传云服务器后，可把 data1 目录解压到项目根目录。")


if __name__ == "__main__":
    main()
