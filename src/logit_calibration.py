from typing import Optional

import torch
from torch import nn


def apply_logit_bias(logits: torch.Tensor, logit_bias: Optional[torch.Tensor]) -> torch.Tensor:
    if logit_bias is None:
        return logits
    return logits + logit_bias.to(device=logits.device, dtype=logits.dtype)


def fit_logit_bias(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    steps: int = 300,
    lr: float = 0.1,
    l2: float = 1e-3,
) -> torch.Tensor:
    """Fit a small validation-set bias to reduce systematic class under-prediction."""
    logits = logits.detach().float()
    labels = labels.detach().long()
    bias = torch.zeros(num_classes, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        centered_bias = bias - bias.mean()
        loss = criterion(logits + centered_bias, labels) + l2 * centered_bias.pow(2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        bias -= bias.mean()
    return bias.detach()
