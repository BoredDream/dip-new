# -*- coding: utf-8 -*-
"""色彩空间转换与色度下采样(JPEG 第 1、2 步)。

RGB <-> YCbCr 用 JPEG/BT.601 全量程定义(实施方案 4.1 第 1 步):
    Y  = 0.299 R + 0.587 G + 0.114 B
    Cb = 128 - 0.168736 R - 0.331264 G + 0.5 B
    Cr = 128 + 0.5 R - 0.418688 G - 0.081312 B
逆变换:
    R = Y + 1.402 (Cr-128)
    G = Y - 0.344136 (Cb-128) - 0.714136 (Cr-128)
    B = Y + 1.772 (Cb-128)

色度下采样 4:2:0:Cb、Cr 在 2×2 邻域取平均,水平/垂直各降一半(人眼对色度不敏感)。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rgb_to_ycbcr",
    "ycbcr_to_rgb",
    "subsample_420",
    "upsample_420",
]

_RGB2YCC = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312],
], dtype=np.float64)

_YCC2RGB = np.array([
    [1.0, 0.0, 1.402],
    [1.0, -0.344136, -0.714136],
    [1.0, 1.772, 0.0],
], dtype=np.float64)


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """RGB(uint8 或 float, H×W×3) -> YCbCr float(H×W×3)。Y∈[0,255], Cb/Cr 居中于 128。"""
    rgb = rgb.astype(np.float64)
    ycc = rgb @ _RGB2YCC.T
    ycc[..., 1] += 128.0
    ycc[..., 2] += 128.0
    return ycc


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    """YCbCr float -> RGB float(未裁剪;调用方负责 clip 到 [0,255])。"""
    ycc = ycc.astype(np.float64).copy()
    ycc[..., 1] -= 128.0
    ycc[..., 2] -= 128.0
    return ycc @ _YCC2RGB.T


def subsample_420(chroma: np.ndarray) -> np.ndarray:
    """对单个色度分量做 4:2:0 下采样(2×2 box 平均)。要求宽高为偶数。"""
    h, w = chroma.shape
    h2, w2 = h - h % 2, w - w % 2
    c = chroma[:h2, :w2]
    return (c[0::2, 0::2] + c[0::2, 1::2] + c[1::2, 0::2] + c[1::2, 1::2]) / 4.0


def upsample_420(chroma: np.ndarray, out_shape) -> np.ndarray:
    """4:2:0 上采样:每个色度像素复制成 2×2,再裁/补到目标尺寸。"""
    up = np.repeat(np.repeat(chroma, 2, axis=0), 2, axis=1)
    H, W = out_shape
    out = np.zeros((H, W), dtype=np.float64)
    h = min(H, up.shape[0])
    w = min(W, up.shape[1])
    out[:h, :w] = up[:h, :w]
    # 右/下边若有余数,用最近边复制填充
    if h < H:
        out[h:H, :w] = out[h - 1:h, :w]
    if w < W:
        out[:, w:W] = out[:, w - 1:w]
    return out
