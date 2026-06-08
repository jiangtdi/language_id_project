"""Download OWSM-CTC v4 1B to ./owsm_ctc_v4_1B with HuggingFace Hub."""

from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_REPO = "espnet/owsm_ctc_v4_1B"
LOCAL_DIR = PROJECT_ROOT / "owsm_ctc_v4_1B"


def main():
    print(f"开始下载: {MODEL_REPO}")
    print(f"保存目录: {LOCAL_DIR}")
    snapshot_download(
        repo_id=MODEL_REPO,
        repo_type="model",
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("下载完成。目录应包含 README.md、meta.yaml、data/、exp/。")


if __name__ == "__main__":
    main()
