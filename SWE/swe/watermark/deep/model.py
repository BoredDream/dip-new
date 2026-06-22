# -*- coding: utf-8 -*-
"""LatentWatermark —— 深度潜空间水印的封装(嵌入/提取 + 任意分辨率技巧)。

嵌入:z = Φ(x); Δz = E_s(m); x_w = Ψ(z + Δz)。
提取:m' = D_s(x')。
推理用 VINE 的"残差缩放"技巧实现任意分辨率(watermark_encoding.py 思路):
模型只在 image_size(如 256)工作,先在 256 上算含水印残差,再把残差上采样回原分辨率
叠加到原图 —— 既省显存又保高清。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae_backbone import build_vae_backbone
from .secret_encoder import SecretEncoder
from .secret_decoder import build_secret_decoder

__all__ = ["LatentWatermark", "build_latent_watermark", "load_latent_watermark"]


def _to_tensor(img_uint8: np.ndarray, device) -> torch.Tensor:
    """RGB uint8 (H,W,3) -> (1,3,H,W) in [-1,1]。灰度自动堆成 3 通道。"""
    arr = img_uint8
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)[None]
    return (t / 127.5 - 1.0).to(device)


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) in [-1,1] -> RGB uint8 (H,W,3)。"""
    x = ((x.clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


class LatentWatermark(nn.Module):
    def __init__(self, vae: nn.Module, secret_encoder: nn.Module, secret_decoder: nn.Module,
                 image_size: int = 256, secret_bits: int = 100):
        super().__init__()
        self.vae = vae
        self.secret_encoder = secret_encoder
        self.secret_decoder = secret_decoder
        self.image_size = image_size
        self.secret_bits = secret_bits
        #: 架构描述({vae, decoder, image_size, secret_bits}),由 build_latent_watermark 填入,
        #  随 checkpoint 一同保存,供 load_latent_watermark 自动按正确结构重建。
        self.config = None

    # ---------------- 训练用(张量,可微) ---------------- #
    def embed_tensor(self, x: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(x)
        dz = self.secret_encoder(secret)
        return self.vae.decode(z + dz)

    def extract_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.secret_decoder(x)

    def trainable_parameters(self):
        params = list(self.secret_encoder.parameters()) + list(self.secret_decoder.parameters())
        if getattr(self.vae, "trainable", False):
            params += list(self.vae.parameters())
        return params

    # ---------------- 推理用(numpy 图,任意分辨率) ---------------- #
    @torch.no_grad()
    def embed(self, image: np.ndarray, bits: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        self.eval()
        orig = _to_tensor(image, device)
        H, W = orig.shape[-2:]
        secret = torch.from_numpy(np.asarray(bits, dtype=np.float32))[None].to(device)
        x_small = F.interpolate(orig, size=(self.image_size, self.image_size),
                                mode="bilinear", align_corners=False)
        x_w_small = self.embed_tensor(x_small, secret)
        residual = x_w_small - x_small                       # 在 image_size 上算残差
        residual = F.interpolate(residual, size=(H, W), mode="bilinear", align_corners=False)
        x_w = (orig + residual).clamp(-1, 1)                 # 残差放大回原分辨率再叠加
        return _to_uint8(x_w)

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        self.eval()
        x = _to_tensor(image, device)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        probs = self.extract_tensor(x)[0].cpu().numpy()
        return (probs > 0.5).astype(np.int64)

    @torch.no_grad()
    def extract_probs(self, image: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        self.eval()
        x = _to_tensor(image, device)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        return self.extract_tensor(x)[0].cpu().numpy()

    # ---------------- 存取 ---------------- #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "secret_encoder": self.secret_encoder.state_dict(),
            "secret_decoder": self.secret_decoder.state_dict(),
            "vae": self.vae.state_dict() if getattr(self.vae, "trainable", False) else None,
            "image_size": self.image_size,
            "secret_bits": self.secret_bits,
            "config": self.config,          # 架构(vae/decoder/...),供 load_latent_watermark 还原
        }, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or "cpu")
        self.secret_encoder.load_state_dict(ckpt["secret_encoder"])
        self.secret_decoder.load_state_dict(ckpt["secret_decoder"])
        if ckpt.get("vae") is not None and getattr(self.vae, "trainable", False):
            self.vae.load_state_dict(ckpt["vae"])


def build_latent_watermark(cfg: Optional[dict] = None, device: str = "cpu") -> LatentWatermark:
    """按配置(默认取 config.DEEP)构造 LatentWatermark。"""
    if cfg is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))))
        from config import DEEP as cfg  # type: ignore

    image_size = cfg.get("image_size", 256)
    secret_bits = cfg.get("secret_bits", 100)
    vae_kind = cfg.get("vae", "tiny")
    dec_kind = cfg.get("decoder", "cnn")
    vae = build_vae_backbone(vae_kind, device=device)
    latent_size = image_size // vae.downscale
    enc = SecretEncoder(secret_bits, vae.latent_channels, latent_size)
    dec = build_secret_decoder(dec_kind, secret_bits)
    model = LatentWatermark(vae, enc, dec, image_size, secret_bits).to(device)
    model.config = {"vae": vae_kind, "decoder": dec_kind,
                    "image_size": image_size, "secret_bits": secret_bits}
    return model


def load_latent_watermark(path: str, device: str = "cpu") -> LatentWatermark:
    """从 checkpoint 还原 LatentWatermark —— 按保存的架构 config 自动重建 vae/decoder。

    旧版 checkpoint(无 config)回退到 tiny + cnn,并沿用其中的 image_size/secret_bits。
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config") or {
        "vae": "tiny", "decoder": "cnn",
        "image_size": ckpt.get("image_size", 256),
        "secret_bits": ckpt.get("secret_bits", 100),
    }
    model = build_latent_watermark(cfg, device=device)
    model.load(path, map_location=device)
    return model
