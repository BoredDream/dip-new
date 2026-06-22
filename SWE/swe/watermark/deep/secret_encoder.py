# -*- coding: utf-8 -*-
"""秘密编码器 E_s —— 消息 -> 潜空间残差 Δz(RoSteALS / VINE ConditionAdaptor 思路)。

把 L 位消息投影并上采样成与潜变量 z 同形的残差 Δz,嵌入时 z_w = z + Δz。
最后一层用**较小**的初始化(默认 std=0.1)使初始残差幅度受控、训练更稳。

说明:VINE 把 skip 卷积初始化为 1e-5(近零)是因为它用的是**冻结的预训练 VAE**,
近零起步即保证输出≈原图;本模块默认配合**从零联合训练的 tiny VAE**,故取稍大的 0.1
以保证梯度通畅、消息路径能学起来。换冻结 SD-VAE 时可调小 init_scale 走"近零起步"路线。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SecretEncoder"]


class SecretEncoder(nn.Module):
    def __init__(self, secret_bits: int = 100, latent_channels: int = 4,
                 latent_size: int = 32, hidden: int = 64, init_scale: float = 0.1):
        super().__init__()
        self.secret_bits = secret_bits
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.seed_size = 16                          # 消息先投影到 16×16,再上采样到 latent_size

        self.proj = nn.Linear(secret_bits, self.seed_size * self.seed_size)
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden, 3, 1, 1), nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden, latent_channels, 3, 1, 1)
        # 较小初始化:控制初始残差幅度、训练更稳(冻结 VAE 时可调更小走近零起步)
        nn.init.normal_(self.out.weight, std=init_scale)
        nn.init.zeros_(self.out.bias)

    def forward(self, secret: torch.Tensor) -> torch.Tensor:
        """secret: (B, L) in {0,1} -> Δz: (B, C, latent_size, latent_size)。"""
        b = secret.shape[0]
        s = 2.0 * secret - 1.0                       # {0,1} -> {-1,1}
        h = self.proj(s).view(b, 1, self.seed_size, self.seed_size)
        h = F.interpolate(h, size=(self.latent_size, self.latent_size), mode="bilinear",
                          align_corners=False)
        h = self.conv(h)
        return self.out(h)
