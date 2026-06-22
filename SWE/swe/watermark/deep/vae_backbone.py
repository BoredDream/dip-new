# -*- coding: utf-8 -*-
"""VAE 主干 —— 深度潜空间水印的"载体"。

实施方案 4.3 / RoSteALS:水印藏进 VAE 潜变量 z。扩散 img2img 的必经路径是
`图 -> VAE编码 -> 潜空间 -> VAE解码 -> 新图`,把水印与潜空间内容纠缠,去噪难以单独抹除。

提供统一接口 VAEBackbone:
    encode(x)  : (B,3,H,W) in [-1,1]  ->  z (B, C, H/f, W/f)
    decode(z)  : z                    ->  (B,3,H,W) in [-1,1]

两种实现:
  * SDVAEBackbone  —— 冻结的 Stable Diffusion VAE(diffusers,C=4,f=8),RoSteALS 正路,出真实结果;
  * TinyVAE        —— 自带小型(可训练)自编码器(C=4,f=8),纯 torch、免下载,
                      用于 CPU 冒烟测试与无 diffusers 环境;此时 VAE 与 E_s/D_s 联合训练。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VAEBackbone", "TinyVAE", "SDVAEBackbone", "build_vae_backbone"]


class VAEBackbone(nn.Module):
    latent_channels: int = 4
    downscale: int = 8
    trainable: bool = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


class TinyVAE(VAEBackbone):
    """轻量(可训练)自编码器,latent 4×(H/8)×(W/8)。仅用于冒烟测试/无 diffusers 环境。"""

    trainable = True

    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.latent_channels = latent_channels
        self.downscale = 8
        self.enc = nn.Sequential(
            _block(3, 32), _block(32, 64, 2), _block(64, 128, 2), _block(128, 128, 2),
            nn.Conv2d(128, latent_channels, 3, 1, 1),
        )
        self.dec_in = nn.Conv2d(latent_channels, 128, 3, 1, 1)
        self.dec = nn.Sequential(
            _block(128, 128), _block(128, 64), _block(64, 32),
        )
        self.dec_out = nn.Conv2d(32, 3, 3, 1, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z)
        for layer in self.dec:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = layer(h)
        return torch.tanh(self.dec_out(h))


class SDVAEBackbone(VAEBackbone):
    """冻结的 Stable Diffusion VAE(diffusers AutoencoderKL)。RoSteALS 正路。"""

    trainable = False

    def __init__(self, model: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except Exception as e:  # noqa: BLE001
            raise ImportError("SD-VAE 主干需要 diffusers,请 `pip install -r requirements-deep.txt`,"
                              "或在配置里把 deep.vae 设为 'tiny'。") from e
        self.vae = AutoencoderKL.from_pretrained(model)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.latent_channels = self.vae.config.latent_channels
        self.downscale = 8
        self._sf = self.vae.config.scaling_factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(x).latent_dist.sample()
        return z * self._sf

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self._sf).sample.clamp(-1, 1)


def build_vae_backbone(name: str = "tiny", device: Optional[str] = None,
                       **kwargs) -> VAEBackbone:
    name = name.lower()
    if name == "tiny":
        vae = TinyVAE(**kwargs)
    elif name in ("sd", "sd-vae", "stable-diffusion"):
        vae = SDVAEBackbone(**kwargs)
    else:
        raise ValueError(f"未知 VAE 主干: {name}")
    if device:
        vae = vae.to(device)
    return vae
