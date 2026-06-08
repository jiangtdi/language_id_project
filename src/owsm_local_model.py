"""
Local OWSM v4 model configuration.

The project uses a manually downloaded HuggingFace snapshot:
    espnet/owsm_ctc_v4_1B -> ./owsm_ctc_v4_1B

Keep the model local so training and inference do not depend on network access.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OWSM_MODEL_DIR = PROJECT_ROOT / "owsm_ctc_v4_1B"
OWSM_MODEL_NAME = "espnet/owsm_ctc_v4_1B"
MODEL_DISPLAY_NAME = "OWSM-CTC v4 + Conv-Adapter PEFT"


def resolve_owsm_model_dir(model_dir: Path = OWSM_MODEL_DIR) -> Path:
    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"本地 OWSM v4 模型目录不存在: {model_dir}\n"
            "请先下载 espnet/owsm_ctc_v4_1B 到项目根目录下的 owsm_ctc_v4_1B/。"
        )
    if not (model_dir / "meta.yaml").exists():
        raise FileNotFoundError(
            f"未找到 OWSM v4 的 meta.yaml: {model_dir / 'meta.yaml'}\n"
            "请确认目录结构完整，应该包含 README.md、meta.yaml、data/、exp/。"
        )
    print(f"使用本地 OWSM v4 模型: {model_dir}")
    return model_dir
