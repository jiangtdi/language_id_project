import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_owsm_adapter_lid as train_script
from src.audio_dataset import augment_audio, create_audio_windows
from src.owsm_adapter_lid_model import AdditiveAngularMarginLinear, ConvAdapterBlock, TemporalAttentionPooling


def test_compute_class_weight_tensor_inverse_frequency():
    label_to_idx = {"A": 0, "B": 1}
    train_samples = [
        ("a1.wav", "A"),
        ("a2.wav", "A"),
        ("a3.wav", "A"),
        ("b1.wav", "B"),
    ]
    weights = train_script.compute_class_weight_tensor(train_samples, label_to_idx)
    assert weights.shape[0] == 2
    assert torch.isclose(weights[0], torch.tensor(4 / (2 * 3)), atol=1e-6)
    assert torch.isclose(weights[1], torch.tensor(4 / (2 * 1)), atol=1e-6)
    assert weights[1] > weights[0]


def test_conv_adapter_preserves_frame_shape():
    block = ConvAdapterBlock(channels=8, bottleneck=4)
    x = torch.randn(2, 5, 8)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_temporal_attention_pooling_returns_mean_std_embedding():
    pooling = TemporalAttentionPooling(channels=4)
    x = torch.ones(3, 6, 4)
    output = pooling(x)
    assert output.shape == (3, 8)
    assert torch.isfinite(output).all()


def test_angular_margin_head_shape():
    head = AdditiveAngularMarginLinear(in_features=8, out_features=3)
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 1])
    logits = head(embeddings, labels=labels, apply_margin=True)
    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()


def test_create_audio_windows_covers_beginning_middle_and_end():
    audio = torch.arange(10, dtype=torch.float32).numpy()
    windows = create_audio_windows(audio, duration=4, sr=1, stride=2, max_windows=3)
    assert [window.tolist() for window in windows] == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0, 9.0],
    ]


def test_augment_audio_preserves_shape_and_finite_values():
    audio = torch.linspace(-0.5, 0.5, steps=16000).numpy()
    augmented = augment_audio(audio, sr=16000)
    assert augmented.shape == audio.shape
    assert torch.isfinite(torch.from_numpy(augmented)).all()
