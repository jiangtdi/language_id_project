"""
下载并全量解压 VoxLingua107 五语种外部测试音频。

PyCharm 直接运行即可。
脚本会下载 Chinese / English / French / Japanese / Korean 对应 zip，
并把压缩包中的全部 wav 解压到 data/external_test/VoxLingua107/语言名/。

数据来源：https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/
"""

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "external_test" / "VoxLingua107"
CACHE_DIR = PROJECT_ROOT / "data" / "external_test" / "_downloads" / "VoxLingua107"
ZIP_BASE_URL = "https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107"

# ========== 可直接修改的参数 ==========
# True：删除旧的外部测试目录后重新全量解压。
# False：保留已解压 wav；如果某个语种目录已有 wav，就跳过该语种解压。
RESET_EXTERNAL_TEST_DIR = False

# True：下载后保留 zip，后续重新解压不用再下载。
# False：解压完成后删除 zip，节省磁盘空间。
KEEP_DOWNLOADED_ZIP = True
# =====================================

LANGUAGE_ZIPS = {
    "Chinese": "zh.zip",
    "English": "en.zip",
    "French": "fr.zip",
    "Japanese": "ja.zip",
    "Korean": "ko.zip",
}


def wav_count(output_dir: Path) -> int:
    return len(list(output_dir.glob("*.wav"))) if output_dir.exists() else 0


def prepare_language_dir(output_dir: Path):
    if RESET_EXTERNAL_TEST_DIR and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def download_zip(zip_filename: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / zip_filename
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        print(f"已存在有效 zip，跳过下载: {zip_path}")
        return zip_path
    if zip_path.exists():
        print(f"发现损坏或未完成 zip，删除后重新下载: {zip_path}")
        zip_path.unlink()

    url = f"{ZIP_BASE_URL}/{zip_filename}"
    temp_path = zip_path.with_suffix(zip_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    print(f"下载: {url}")
    urlretrieve(url, temp_path)
    if not zipfile.is_zipfile(temp_path):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"下载完成但文件不是有效 zip，请检查网络或下载地址: {url}")

    temp_path.replace(zip_path)
    return zip_path


def extract_all_wavs(zip_path: Path, output_dir: Path, prefix: str) -> int:
    existing_count = wav_count(output_dir)
    if existing_count > 0 and not RESET_EXTERNAL_TEST_DIR:
        print(f"  已有 {existing_count} 条 wav，跳过解压。")
        return existing_count

    saved_count = 0
    with zipfile.ZipFile(zip_path) as archive:
        wav_members = [
            member
            for member in archive.namelist()
            if member.lower().endswith(".wav") and not member.endswith("/")
        ]
        total = len(wav_members)
        print(f"  zip 内共有 {total} 条 wav，开始全量解压...")

        for index, member in enumerate(wav_members):
            output_path = output_dir / f"{prefix}_{index:05d}.wav"
            if output_path.exists():
                continue
            with archive.open(member) as source, open(output_path, "wb") as target:
                shutil.copyfileobj(source, target)
            saved_count += 1

            if saved_count % 500 == 0:
                print(f"  已解压 {saved_count}/{total} 条...")

    total_count = wav_count(output_dir)
    print(f"  解压完成：新增 {saved_count} 条，当前共 {total_count} 条。")
    return total_count


def download_language(label: str, zip_filename: str) -> int:
    output_dir = OUTPUT_DIR / label
    prepare_language_dir(output_dir)

    print("=" * 60)
    print(f"{label}: 全量下载并解压 {zip_filename}")

    zip_path = download_zip(zip_filename)
    count = extract_all_wavs(zip_path, output_dir, prefix=Path(zip_filename).stem)

    if not KEEP_DOWNLOADED_ZIP and zip_path.exists():
        zip_path.unlink()
        print(f"  已删除 zip: {zip_path}")

    return count


def main():
    print("开始下载并全量解压 VoxLingua107 外部测试音频...")
    print(f"是否重置旧目录: {RESET_EXTERNAL_TEST_DIR}")
    print(f"是否保留 zip: {KEEP_DOWNLOADED_ZIP}")
    print("提示：五个语种 zip 总体积较大，首次运行会花较长时间并占用较多磁盘。")

    summary = {}
    for label, zip_filename in LANGUAGE_ZIPS.items():
        try:
            summary[label] = download_language(label, zip_filename)
        except Exception as exc:
            summary[label] = wav_count(OUTPUT_DIR / label)
            print(f"{label}: 下载或解压失败，当前已有 {summary[label]} 条，原因: {exc}")

    print("=" * 60)
    print("外部测试音频统计:")
    for label, count in summary.items():
        print(f"{label}: {count} 条")

    print("全量外部测试数据准备完成。可运行 evaluate_external_audio_dir.py。")


if __name__ == "__main__":
    main()
