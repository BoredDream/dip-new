# -*- coding: utf-8 -*-
"""评估指标 —— 模块五之一。

不可见性(攻击前,含水印图 vs 原图):PSNR / SSIM / LPIPS
鲁棒性(攻击后):比特准确率 = 1 - BER(头号指标,0.5=随机=水印死)
篡改定位:IoU / F1 / precision / recall
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["psnr", "ssim", "lpips_distance", "bit_accuracy", "ber",
           "iou", "f1_score", "precision_recall"]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """峰值信噪比(dB),>40 视为不可见。"""
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """结构相似性 ∈[0,1],>0.95 较好。优先 skimage,缺失则用简化全局实现。"""
    try:
        from skimage.metrics import structural_similarity as _ssim
        if a.ndim == 3:
            return float(_ssim(a, b, channel_axis=-1, data_range=255))
        return float(_ssim(a, b, data_range=255))
    except Exception:
        a = a.astype(np.float64); b = b.astype(np.float64)
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                     ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


_LPIPS_FN = None


def lpips_distance(a: np.ndarray, b: np.ndarray, net: str = "alex") -> float:
    """LPIPS 感知距离(越低越好)。需要 torch + lpips;缺失返回 NaN。"""
    global _LPIPS_FN
    try:
        import torch
        import lpips
    except Exception:
        return float("nan")
    if _LPIPS_FN is None:
        _LPIPS_FN = lpips.LPIPS(net=net)
        _LPIPS_FN.eval()

    def _t(x):
        x = x.astype(np.float32)
        if x.ndim == 2:
            x = np.stack([x] * 3, -1)
        return torch.from_numpy(x).permute(2, 0, 1)[None] / 127.5 - 1.0

    with torch.no_grad():
        return float(_LPIPS_FN(_t(a), _t(b)).item())


def bit_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    """两段比特的逐位一致率。0.5 = 随机猜测基线。"""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.mean(np.asarray(a[:n]) == np.asarray(b[:n])))


def ber(a: np.ndarray, b: np.ndarray) -> float:
    """误码率 = 1 - 比特准确率。"""
    return 1.0 - bit_accuracy(a, b)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = np.asarray(pred, dtype=bool); gt = np.asarray(gt, dtype=bool)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """篡改定位 IoU(交并比)。"""
    pred = np.asarray(pred, dtype=bool); gt = np.asarray(gt, dtype=bool)
    inter = int(np.sum(pred & gt))
    union = int(np.sum(pred | gt))
    return inter / union if union else (1.0 if inter == 0 else 0.0)


def f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """篡改定位 F1。"""
    prec, rec = precision_recall(pred, gt)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
