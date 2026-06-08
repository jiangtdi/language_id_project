"""OWSM v4 + Conv-Adapter PEFT + Temporal Attention Pooling + Angular Margin head."""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.owsm_local_model import OWSM_MODEL_DIR, resolve_owsm_model_dir

MODEL_VERSION = "owsm_v4_conv_adapter_tap_aam_v1"


class AdditiveAngularMarginLinear(nn.Module):
    """Angular-margin classifier for discriminative language embeddings."""

    def __init__(self, in_features: int, out_features: int, scale: float = 18.0, margin: float = 0.20):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scale = scale
        self.margin = margin
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None, apply_margin: bool = False):
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weights).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if apply_margin and labels is not None:
            theta = torch.acos(cosine)
            target_logits = torch.cos(theta + self.margin)
            one_hot = F.one_hot(labels, num_classes=cosine.size(1)).to(dtype=cosine.dtype, device=cosine.device)
            cosine = cosine * (1.0 - one_hot) + target_logits * one_hot
        return cosine * self.scale


class ConvAdapterBlock(nn.Module):
    """Convolution-enhanced adapter: a small trainable module on top of frozen OWSM features."""

    def __init__(self, channels: int, bottleneck: int = 128, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(channels)
        self.down = nn.Linear(channels, bottleneck)
        self.conv = nn.Conv1d(bottleneck, bottleneck, kernel_size=kernel_size, padding=padding)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x


class TemporalAttentionPooling(nn.Module):
    """Pool frame-level features into one utterance embedding using learned temporal attention."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(1, channels // 2)
        self.attention = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        if lengths is not None:
            lengths = lengths.clamp(min=1, max=x.size(1))
            time_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            scores = scores.masked_fill(time_ids >= lengths.unsqueeze(1), -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = (x * weights).sum(dim=1)
        second_moment = ((x ** 2) * weights).sum(dim=1)
        std = torch.sqrt(torch.clamp(second_moment - mean ** 2, min=0.0) + 1e-8)
        return torch.cat([mean, std], dim=1)


class OWSMFeatureExtractor(nn.Module):
    """Local wrapper around ESPnet OWSM-CTC v4 encoder."""

    def __init__(self, model_dir: Path = OWSM_MODEL_DIR, device: str = "cpu", dtype: str = "float32"):
        super().__init__()
        self.model_dir = resolve_owsm_model_dir(model_dir)
        self.device_name = device
        self.dtype = dtype
        self.speech2text = self._load_speech2text()
        self.s2t_model = getattr(self.speech2text, "s2t_model", None) or getattr(self.speech2text, "model", None)
        if self.s2t_model is None:
            raise RuntimeError("无法从 ESPnet Speech2TextGreedySearch 中取得 s2t_model。")
        self.s2t_model.eval()
        for param in self.s2t_model.parameters():
            param.requires_grad = False

    def _find_espnet_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        meta_path = self.model_dir / "meta.yaml"
        if meta_path.exists():
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            config_rel = meta.get("yaml_files", {}).get("s2t_train_config")
            model_rel = meta.get("files", {}).get("s2t_model_file")
            config_path = self.model_dir / config_rel if config_rel else None
            model_path = self.model_dir / model_rel if model_rel else None
            if config_path and model_path and config_path.exists() and model_path.exists():
                return config_path, model_path

        config_candidates = list(self.model_dir.rglob("config.yaml")) + list(self.model_dir.rglob("*config*.yaml"))
        model_candidates = [
            path
            for path in list(self.model_dir.rglob("*.pth")) + list(self.model_dir.rglob("*.pt"))
            if "optim" not in path.name.lower() and "scheduler" not in path.name.lower()
        ]
        config = config_candidates[0] if config_candidates else None
        model = model_candidates[0] if model_candidates else None
        return config, model

    def _load_speech2text(self):
        try:
            from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
        except Exception as exc:
            raise ImportError(
                "未安装 ESPnet 依赖。请运行: pip install espnet espnet_model_zoo typeguard humanfriendly"
            ) from exc

        config, model = self._find_espnet_files()
        if not config or not model:
            raise RuntimeError(
                f"OWSM v4 本地模型文件不完整，未找到 config.yaml 或 .pth 权重: {self.model_dir}"
            )

        previous_cwd = Path.cwd()
        try:
            # ESPnet configs may contain relative paths such as
            # exp/s2t_stats_raw_bpe50000/train/feats_stats.npz.
            # Build the model from the local model directory so these paths
            # resolve inside ./owsm_ctc_v4_1B instead of the project root.
            os.chdir(self.model_dir)
            return Speech2TextGreedySearch(
                s2t_train_config=str(config),
                s2t_model_file=str(model),
                device=self.device_name,
                dtype=self.dtype,
            )
        except Exception as exc:
            raise RuntimeError(
                "OWSM v4 本地模型加载失败。请确认模型目录完整，并已安装 espnet、sentencepiece、typeguard、humanfriendly。"
            ) from exc
        finally:
            os.chdir(previous_cwd)

    @torch.no_grad()
    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        input_values = input_values.to(self.device_name)
        if attention_mask is None:
            lengths = torch.full((input_values.size(0),), input_values.size(1), dtype=torch.long, device=input_values.device)
        else:
            lengths = attention_mask.to(input_values.device).sum(dim=1).long()

        batch_size = input_values.size(0)
        lang_id = self.speech2text.converter.token2id[self.speech2text.lang_sym]
        task_id = self.speech2text.converter.token2id[self.speech2text.task_sym]

        text_prev = torch.full(
            (batch_size, 1),
            fill_value=self.s2t_model.na,
            dtype=torch.long,
            device=input_values.device,
        )
        text_prev_lengths = torch.ones(batch_size, dtype=torch.long, device=input_values.device)
        prefix = torch.tensor([lang_id, task_id], dtype=torch.long, device=input_values.device).unsqueeze(0)
        prefix = prefix.repeat(batch_size, 1)
        prefix_lengths = torch.full((batch_size,), 2, dtype=torch.long, device=input_values.device)

        try:
            output = self.s2t_model.encode(
                speech=input_values.to(getattr(torch, self.dtype)),
                speech_lengths=lengths,
                text_prev=text_prev,
                text_prev_lengths=text_prev_lengths,
                prefix=prefix,
                prefix_lengths=prefix_lengths,
            )
            if isinstance(output, tuple):
                features = output[0]
                feature_lengths = output[1] if len(output) > 1 and torch.is_tensor(output[1]) else None
            else:
                features = output
                feature_lengths = None
            if isinstance(features, tuple):
                features = features[0]
            if features.dim() != 3:
                raise RuntimeError(f"OWSM encoder 输出维度异常: {tuple(features.shape)}")
            return features.detach(), feature_lengths
        except Exception as exc:
            raise RuntimeError(f"OWSM encoder 特征提取失败: {exc}") from exc


class OWSMAdapterLanguageClassifier(nn.Module):
    """Frozen OWSM v4 encoder with trainable Conv-Adapter PEFT language classifier."""

    def __init__(
        self,
        num_classes: int,
        model_dir: Path = OWSM_MODEL_DIR,
        owsm_device: str = "cpu",
        encoder_dim: Optional[int] = None,
        adapter_bottleneck: int = 128,
        adapter_layers: int = 2,
        embedding_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.owsm = OWSMFeatureExtractor(model_dir=model_dir, device=owsm_device)
        self.encoder_dim = encoder_dim
        self.adapter_bottleneck = adapter_bottleneck
        self.adapter_layers_count = adapter_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.adapters: Optional[nn.ModuleList] = None
        self.pooling: Optional[TemporalAttentionPooling] = None
        self.embedding_head: Optional[nn.Sequential] = None
        self.classifier: Optional[AdditiveAngularMarginLinear] = None
        if encoder_dim is not None:
            self._build_head(encoder_dim)

    def _build_head(self, encoder_dim: int):
        self.encoder_dim = int(encoder_dim)
        self.adapters = nn.ModuleList(
            [
                ConvAdapterBlock(
                    channels=self.encoder_dim,
                    bottleneck=self.adapter_bottleneck,
                    dropout=self.dropout,
                )
                for _ in range(self.adapter_layers_count)
            ]
        )
        self.pooling = TemporalAttentionPooling(self.encoder_dim)
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(self.encoder_dim * 2),
            nn.Linear(self.encoder_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.classifier = AdditiveAngularMarginLinear(self.embedding_dim, self.num_classes)

    def ensure_head(self, features: torch.Tensor):
        if self.adapters is None:
            self._build_head(features.size(-1))
            self.adapters.to(features.device)
            self.pooling.to(features.device)
            self.embedding_head.to(features.device)
            self.classifier.to(features.device)

    def trainable_parameters(self):
        return [param for name, param in self.named_parameters() if not name.startswith("owsm.") and param.requires_grad]

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        apply_margin: bool = False,
    ) -> torch.Tensor:
        features, feature_lengths = self.owsm(input_values, attention_mask=attention_mask)
        self.ensure_head(features)
        features = features.to(next(self.classifier.parameters()).device)
        if feature_lengths is not None:
            feature_lengths = feature_lengths.to(features.device).long()
        x = features
        for adapter in self.adapters:
            x = adapter(x)
        pooled = self.pooling(x, lengths=feature_lengths)
        embedding = self.embedding_head(pooled)
        return self.classifier(embedding, labels=labels, apply_margin=apply_margin)

    def initialize_head(self, device: torch.device, sample_seconds: float = 3.0, sample_rate: int = 16000):
        dummy = torch.zeros(1, int(sample_seconds * sample_rate), device=device)
        mask = torch.ones_like(dummy, dtype=torch.long)
        self.eval()
        with torch.no_grad():
            _ = self(dummy, attention_mask=mask)
