# -*- coding: utf-8 -*-
"""可微攻击层 N —— 训练时模拟失真/编辑(实施方案 4.3 噪声层 + VINE 代理攻击)。

核心思想(VINE):扩散编辑昂贵且不可微,但其对图像的破坏在频域上近似"模糊";
于是训练时用一组**便宜可微**的失真去逼近它,逼解码器学会抗编辑。失真池含:
    恒等 / 高斯噪声 / 高斯模糊 / 降采样-升采样 / **VAE 往返**
其中 VAE 往返(vae.decode(vae.encode(x)))是"扩散再生成的廉价可微代理",
是抗 AI 再生成鲁棒性的关键(实施方案 4.3:N 含 VAE 往返)。

课程式训练:失真强度与触发概率随 global_step 线性上升(ramp),先易后难。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AttackLayer", "gaussian_blur"]


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    k = max(3, int(2 * round(3 * sigma) + 1))
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)                          # (k,k)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """对 (B,3,H,W) 做可微高斯模糊(depthwise 卷积,反射填充)。"""
    if sigma <= 0:
        return x
    ker = _gaussian_kernel(sigma, x.device, x.dtype)
    k = ker.shape[0]
    ker = ker.expand(x.shape[1], 1, k, k)
    x = F.pad(x, (k // 2,) * 4, mode="reflect")
    return F.conv2d(x, ker, groups=x.shape[1])


class AttackLayer(nn.Module):
    def __init__(self, vae: Optional[nn.Module] = None, curriculum_steps: int = 2000,
                 p_max: float = 0.9, seed: int = 2026):
        super().__init__()
        self.vae = vae
        self.curriculum_steps = max(1, curriculum_steps)
        self.p_max = p_max
        self._distortions = ["noise", "blur", "resize", "vae"]

    def ramp(self, step: int) -> float:
        return min(1.0, step / self.curriculum_steps)

    def forward(self, x: torch.Tensor, step: int = 10 ** 9,
                eval_distortion: Optional[str] = None) -> torch.Tensor:
        """x: (B,3,H,W) in [-1,1]。按课程随机施加一种失真;eval_distortion 指定时强制该失真。"""
        r = self.ramp(step)
        if eval_distortion is None:
            if torch.rand(()) > self.p_max * (0.3 + 0.7 * r):     # 一定概率恒等
                return x
            choice = self._distortions[int(torch.randint(0, len(self._distortions), ()))]
        else:
            choice = eval_distortion
            r = 1.0

        if choice == "noise":
            sigma = (0.02 + 0.10 * r) * torch.rand(())
            return (x + torch.randn_like(x) * sigma).clamp(-1, 1)

        if choice == "blur":
            sigma = 0.4 + (1.6 * r) * float(torch.rand(()))
            return gaussian_blur(x, sigma).clamp(-1, 1)

        if choice == "resize":
            factor = 1.0 - (0.6 * r) * float(torch.rand(()))
            factor = max(0.35, factor)
            H, W = x.shape[-2:]
            h, w = max(8, int(H * factor)), max(8, int(W * factor))
            down = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            return F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False).clamp(-1, 1)

        if choice == "vae" and self.vae is not None:
            # VAE 往返:扩散再生成的廉价可微代理。frozen SD-VAE 时梯度仍可回传(参数冻结但可微)。
            z = self.vae.encode(x)
            return self.vae.decode(z).clamp(-1, 1)

        return x
