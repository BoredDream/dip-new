# -*- coding: utf-8 -*-
"""经典攻击(对照组)—— 模块四之一。

实施方案 4.4:对每种攻击做多强度扫描以画衰减曲线。所有攻击:
  输入 uint8/float 图(灰度 2D 或 RGB 3D)+ 强度参数 -> 输出 **同尺寸** uint8 图。

  * gaussian_noise(σ)        高斯噪声(对应第 3 章)
  * salt_pepper_noise(amount)椒盐噪声
  * gaussian_blur(σ)         高斯模糊
  * median_filter(size)      中值滤波
  * jpeg_recompress(quality) JPEG 再压缩(复用模块一自实现编解码器)
  * center_crop(keep)        中心裁剪后缩放回原尺寸
  * rescale(factor)          降采样再升采样(重采样攻击)
  * rotate(degrees)          旋转(破坏分块对齐,几何攻击代表)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter as _median, rotate as _rotate
from skimage.transform import resize as _resize

from ..data.datasets import to_uint8

__all__ = [
    "gaussian_noise", "salt_pepper_noise", "gaussian_blur", "median_filter",
    "jpeg_recompress", "center_crop", "rescale", "rotate", "CLASSIC_ATTACKS",
]


def _per_channel(img: np.ndarray, fn) -> np.ndarray:
    """对灰度直接作用,对 RGB 逐通道作用。"""
    if img.ndim == 2:
        return fn(img)
    return np.stack([fn(img[..., c]) for c in range(img.shape[-1])], axis=-1)


def gaussian_noise(img: np.ndarray, sigma: float = 5.0, seed: int = 0) -> np.ndarray:
    if sigma <= 0:
        return to_uint8(img)
    rng = np.random.RandomState(seed)
    return to_uint8(img.astype(np.float64) + rng.normal(0, sigma, img.shape))


def salt_pepper_noise(img: np.ndarray, amount: float = 0.02, seed: int = 0) -> np.ndarray:
    if amount <= 0:
        return to_uint8(img)
    rng = np.random.RandomState(seed)
    out = img.astype(np.float64).copy()
    mask = rng.rand(*img.shape[:2])
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return to_uint8(out)


def gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return to_uint8(img)
    return to_uint8(_per_channel(img.astype(np.float64), lambda c: gaussian_filter(c, sigma)))


def median_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
    if size <= 1:
        return to_uint8(img)
    return to_uint8(_per_channel(img, lambda c: _median(c, size=size)))


def jpeg_recompress(img: np.ndarray, quality: float = 50) -> np.ndarray:
    """复用模块一自实现 JPEG 编解码器做再压缩攻击。"""
    from ..codec.jpeg import JPEGCodec
    rec, _ = JPEGCodec(quality=quality, subsample=True).compress_decompress(to_uint8(img))
    return rec


def center_crop(img: np.ndarray, keep: float = 0.8) -> np.ndarray:
    """中心保留 keep 比例区域,再缩放回原尺寸(裁剪+缩放攻击)。"""
    if keep >= 1.0:
        return to_uint8(img)
    H, W = img.shape[:2]
    h, w = int(H * keep), int(W * keep)
    top, left = (H - h) // 2, (W - w) // 2
    crop = img[top:top + h, left:left + w]
    out = _resize(crop.astype(np.float64), (H, W), order=1, anti_aliasing=True,
                  preserve_range=True)
    return to_uint8(out)


def rescale(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """以 factor 降采样再升采样回原尺寸(重采样攻击)。"""
    if factor >= 1.0:
        return to_uint8(img)
    H, W = img.shape[:2]
    h, w = max(1, int(H * factor)), max(1, int(W * factor))
    down = _resize(img.astype(np.float64), (h, w), order=1, anti_aliasing=True, preserve_range=True)
    up = _resize(down, (H, W), order=1, anti_aliasing=True, preserve_range=True)
    return to_uint8(up)


def rotate(img: np.ndarray, degrees: float = 5.0) -> np.ndarray:
    """旋转 degrees 度(reshape=False,保持原尺寸,边缘反射填充)。"""
    if degrees == 0:
        return to_uint8(img)
    out = _per_channel(img.astype(np.float64),
                       lambda c: _rotate(c, degrees, reshape=False, order=1, mode="reflect"))
    return to_uint8(out)


#: 攻击注册表:名称 -> (函数, 参数名, 默认强度网格)。供评估器扫描。
CLASSIC_ATTACKS: Dict[str, Tuple[Callable, str, List]] = {
    "gaussian_noise": (gaussian_noise, "sigma", [0, 2, 5, 10, 15, 20]),
    "salt_pepper": (salt_pepper_noise, "amount", [0.0, 0.01, 0.02, 0.05, 0.1]),
    "gaussian_blur": (gaussian_blur, "sigma", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
    "median": (median_filter, "size", [1, 3, 5, 7]),
    "jpeg": (jpeg_recompress, "quality", [90, 70, 50, 30, 10]),
    "crop": (center_crop, "keep", [1.0, 0.95, 0.9, 0.8, 0.7]),
    "rescale": (rescale, "factor", [1.0, 0.75, 0.5, 0.35, 0.25]),
    "rotate": (rotate, "degrees", [0, 1, 2, 5, 10]),
}
