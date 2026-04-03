from __future__ import annotations

import timm
import torch
from torch import nn


def build_model(
    num_classes: int,
    model_name: str = "mobilenetv3_large_100",
) -> nn.Module:
    return timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
    )


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
