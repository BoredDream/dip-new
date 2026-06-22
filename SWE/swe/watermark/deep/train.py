# -*- coding: utf-8 -*-
"""深度潜空间水印的训练(实施方案 4.3 + VINE 课程式训练)。

损失:
    L = λ_secret·BCE(D_s(N(x_w)), m)        # 水印可读(经攻击层 N 后仍能解出)
      + w_img·( λ_mse·MSE(x_w, x) + λ_lpips·LPIPS(x_w, x) )   # 不可见性
课程(RoSteALS 经验):
    * 先 warmup:不加攻击、图像损失权重为 0,让秘密恢复损失收敛(先保证水印读得出);
    * 之后逐步引入攻击层 N(强度随 step 上升)并线性提升图像损失权重 w_img。
N 含 VAE 往返(扩散再生成的廉价可微代理),是抗 AI 编辑鲁棒性的关键。

LPIPS 为可选依赖:未安装 `lpips` 时自动跳过,仅用 MSE。
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .attack_layer import AttackLayer
from ..utils import random_bits
from ...data.datasets import load_image

__all__ = ["train_deep_watermark", "sample_batch"]


def _load_pool(image_paths: List[str], image_size: int) -> List[np.ndarray]:
    return [load_image(p, size=image_size) for p in image_paths]


def sample_batch(pool: List[np.ndarray], batch_size: int, device, seed_iter) -> torch.Tensor:
    rng = np.random.RandomState(next(seed_iter))
    idx = rng.randint(0, len(pool), size=batch_size)
    imgs = []
    for i in idx:
        a = pool[i].astype(np.float32)
        if rng.rand() < 0.5:                       # 随机水平翻转增广
            a = a[:, ::-1, :].copy()
        imgs.append(a)
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2) / 127.5 - 1.0
    return x.to(device)


def _psnr01(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss((x + 1) / 2, (y + 1) / 2).item()
    return 99.0 if mse <= 1e-12 else 10 * np.log10(1.0 / mse)


def _bit_acc(probs: torch.Tensor, secret: torch.Tensor) -> float:
    return ((probs > 0.5).float() == secret).float().mean().item()


def train_deep_watermark(model, image_paths: List[str], steps: int = 1500,
                         batch_size: int = 4, lr: float = 1e-4, warmup: int = 200,
                         curriculum_steps: int = 1000, lambda_secret: float = 1.0,
                         lambda_mse: float = 1.5, lambda_lpips: float = 1.0,
                         lambda_recon: float = 1.0,
                         device: str = "cpu", log_every: int = 50,
                         seed: int = 2026, verbose: bool = True):
    """在 image_paths 上训练 model(就地更新),返回训练日志列表。"""
    model.train()
    pool = _load_pool(image_paths, model.image_size)
    attack = AttackLayer(vae=model.vae, curriculum_steps=curriculum_steps)
    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)

    # 可选 LPIPS
    lpips_fn = None
    if lambda_lpips > 0:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").to(device)
            for p in lpips_fn.parameters():
                p.requires_grad_(False)
        except Exception:
            if verbose:
                print("[train] 未找到 lpips 库,改用纯 MSE 不可见性损失。")

    seed_iter = iter(range(seed, seed + steps * 7 + 10))
    logs = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x = sample_batch(pool, batch_size, device, seed_iter)
        secret = torch.from_numpy(
            np.stack([random_bits(model.secret_bits, seed=next(seed_iter))
                      for _ in range(batch_size)]).astype(np.float32)).to(device)

        # 共享 encode:x_w 用 z+Δz;可训练 VAE 另算重建 Ψ(Φ(x)) 做始终在线的重建损失
        z = model.vae.encode(x)
        dz = model.secret_encoder(secret)
        x_w = model.vae.decode(z + dz)
        x_att = x_w if step <= warmup else attack(x_w, step - warmup)
        probs = model.extract_tensor(x_att)

        loss_secret = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), secret)
        loss_mse = F.mse_loss(x_w, x)
        loss_lpips = lpips_fn(x_w, x).mean() if lpips_fn is not None else torch.zeros((), device=device)

        # 可训练 tiny VAE 需要始终在线的重建信号(否则随机 VAE 的 tanh 饱和会切断梯度);
        # 冻结 SD-VAE 时此项无意义,跳过。
        vae_trainable = getattr(model.vae, "trainable", False)
        loss_recon = F.mse_loss(model.vae.decode(z), x) if vae_trainable else torch.zeros((), device=device)

        w_img = 0.0 if step <= warmup else min(1.0, (step - warmup) / max(1, curriculum_steps))
        loss = (lambda_secret * loss_secret
                + (lambda_recon * loss_recon if vae_trainable else 0.0)
                + w_img * (lambda_mse * loss_mse + lambda_lpips * loss_lpips))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (step % log_every == 0 or step == 1):
            rec = {"step": step, "loss": loss.item(), "bit_acc": _bit_acc(probs, secret),
                   "psnr": _psnr01(x_w, x), "ramp": attack.ramp(max(0, step - warmup))}
            logs.append(rec)
            print(f"[{step:5d}/{steps}] loss={rec['loss']:.4f} bit_acc={rec['bit_acc']:.3f} "
                  f"PSNR={rec['psnr']:.1f}dB attack_ramp={rec['ramp']:.2f} "
                  f"({(time.time()-t0)/step*1000:.0f}ms/it)")
    return logs
