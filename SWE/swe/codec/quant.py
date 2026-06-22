# -*- coding: utf-8 -*-
"""标准 JPEG 量化表与质量因子缩放。

量化表取 JPEG 标准附录 K(Annex K.1 亮度 / K.2 色度),与 libjpeg / guetzli
默认基表一致。质量因子缩放用 libjpeg 经典公式(亦见实施方案 4.1 第 5 步):

    S = 5000/Q            (Q < 50)
    S = 200 - 2Q          (Q >= 50)
    Q'(u,v) = clip( floor((Q_base(u,v)·S + 50) / 100), 1, 255 )
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "LUMA_QUANT_TABLE",
    "CHROMA_QUANT_TABLE",
    "quality_to_qtable",
    "scale_from_quality",
]

# 标准亮度量化表(Annex K.1) — 对应实施方案中列出的 Q50 亮度表
LUMA_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

# 标准色度量化表(Annex K.2)
CHROMA_QUANT_TABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


def scale_from_quality(quality: float) -> float:
    """质量因子 Q -> 缩放系数 S(libjpeg 约定)。"""
    q = float(quality)
    q = max(1.0, min(100.0, q))
    if q < 50:
        return 5000.0 / q
    return 200.0 - 2.0 * q


def quality_to_qtable(quality: float, chroma: bool = False) -> np.ndarray:
    """按质量因子缩放标准基表,返回截断到 [1,255] 的整数量化表。"""
    base = CHROMA_QUANT_TABLE if chroma else LUMA_QUANT_TABLE
    S = scale_from_quality(quality)
    q = np.floor((base * S + 50.0) / 100.0)
    return np.clip(q, 1, 255).astype(np.float64)
