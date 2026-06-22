# -*- coding: utf-8 -*-
"""秘密解码器 D_s —— (可能被攻击的)图 -> L 个比特概率。

两种 backbone:
  * cnn       —— 轻量卷积栈 + 自适应池化 + 全连接(StegaStamp / VINE Decoder 思路),CPU 友好;
  * convnext  —— torchvision ConvNeXt-Base + Linear + Sigmoid(VINE CustomConvNeXt 路线,更强但更重)。
输入图取值约定为 [-1,1];convnext 路线内部转成 ImageNet 归一化。
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CNNDecoder", "ConvNeXtDecoder", "build_secret_decoder"]


def _down(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 2, 1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


class CNNDecoder(nn.Module):
    """轻量解码器:6 次 stride-2 下采样 -> 自适应平均池化 -> 全连接 -> sigmoid。"""

    def __init__(self, secret_bits: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            _down(3, 32), _down(32, 64), _down(64, 64),
            _down(64, 128), _down(128, 128), _down(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 256), nn.SiLU(inplace=True),
            nn.Linear(256, secret_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h)
        return torch.sigmoid(self.head(h))


class ConvNeXtDecoder(nn.Module):
    """VINE 路线:ConvNeXt-Base 主干 + Linear(1000->L) + Sigmoid。"""

    def __init__(self, secret_bits: int = 100, pretrained: bool = False):
        super().__init__()
        try:
            from torchvision import models
        except Exception as e:  # noqa: BLE001
            raise ImportError("ConvNeXt 解码器(--decoder convnext)需要 torchvision;"
                              "请 `pip install torchvision`,或改用 --decoder cnn(仅需 torch)。") from e
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        self.convnext = models.convnext_base(weights=weights)
        self.convnext.classifier.append(nn.Linear(1000, secret_bits))
        self.convnext.classifier.append(nn.Sigmoid())
        # ImageNet 归一化常数([-1,1] 图先转 [0,1] 再标准化)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(-1, 1) + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return self.convnext(x)


def build_secret_decoder(kind: str = "cnn", secret_bits: int = 100, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind == "cnn":
        return CNNDecoder(secret_bits)
    if kind == "convnext":
        return ConvNeXtDecoder(secret_bits, **kwargs)
    raise ValueError(f"未知解码器: {kind}")
