# -*- coding: utf-8 -*-
"""Watson DCT 感知模型(JND, Just-Noticeable-Difference)—— 改进③的可见性约束。

出处:Watson 1993,"DCT quantization matrices visually optimized for individual
images"。本模块只取其**结构**为各 DCT 频带分配"可藏多少扰动而不被察觉"的预算:

    JND(i,j, block) = t_basic(i,j) · luminance_mask(block)

  * t_basic(i,j):亮度频率敏感度表(8×8)。人眼对中频最敏感(阈值小=能藏的少),
    对极低频与高频不敏感(阈值大=能藏的多)——这是"对比敏感度函数 CSF"的离散版。
  * luminance_mask:亮度掩蔽。块越亮,人眼越不易察觉同等扰动 → 阈值随块亮度抬高。

诚实声明:Watson 表原为"原图 8×8 DCT"标定;本项目嵌在 DWT-LL 子带的 DCT 上,
故借用其**频率权重形状**,绝对尺度由 P2 改进版方法(`ImprovedDwtDct`)的合成步长 δ0 校准
(见 `swe/watermark/improved_dwtdct.py`)。

每一步都可在含水印图上重算(常数表 + 块均值),不依赖被嵌系数本身,
保证嵌入端/提取端推出的步长一致(QIM 自适应步长的必要条件)。
"""
from __future__ import annotations

import numpy as np

from ..codec.zigzag import ZIGZAG_ORDER

__all__ = ["WATSON_LUMINANCE_TABLE", "watson_band_weights", "luminance_mask"]

# Watson 1993 亮度频率敏感度表(8×8,行=垂直频率,列=水平频率)。
# 值越小 = 人眼越敏感 = 该频带能藏的扰动越少。左上(低频)小、右下(高频)大。
WATSON_LUMINANCE_TABLE = np.array([
    [1.40, 1.01, 1.16, 1.66, 2.40, 3.43, 4.79, 6.56],
    [1.01, 1.45, 1.32, 1.52, 2.00, 2.71, 3.67, 4.93],
    [1.16, 1.32, 2.24, 2.59, 2.98, 3.64, 4.60, 5.88],
    [1.66, 1.52, 2.59, 3.77, 4.55, 5.30, 6.28, 7.60],
    [2.40, 2.00, 2.98, 4.55, 6.15, 7.46, 8.71, 10.17],
    [3.43, 2.71, 3.64, 5.30, 7.46, 9.62, 11.58, 13.51],
    [4.79, 3.67, 4.60, 6.28, 8.71, 11.58, 14.50, 17.29],
    [6.56, 4.93, 5.88, 7.60, 10.17, 13.51, 17.29, 21.15],
], dtype=np.float64)

#: 把 8×8 表按 zigzag(低频在前)展平成 64 维,便于按频带编号索引。
_WATSON_ZIGZAG = WATSON_LUMINANCE_TABLE.reshape(-1)[ZIGZAG_ORDER]


def watson_band_weights(bands) -> np.ndarray:
    """返回所选 zigzag 频带的 Watson 频率敏感度阈值(相对量纲,均值归一)。

    归一化使所选频带权重均值为 1:这样 P2 改进版的基础步长 δ0 仍是主旋钮,
    JND 只负责"在各频带间按可见性重新分配预算"(敏感频带少嵌、不敏感频带多嵌)。
    """
    bands = np.asarray(list(bands), dtype=np.int64)
    t = _WATSON_ZIGZAG[bands]
    return t / (t.mean() + 1e-12)


def luminance_mask(block_mean: float, a_t: float = 0.649, mid_gray: float = 128.0) -> float:
    """亮度掩蔽因子:块越亮,可藏的扰动越多(阈值抬高)。

    factor = (block_mean / mid_gray) ** a_t,Watson 取 a_t≈0.649。
    用块的**像素均值**(嵌入前后稳定、可在含水印图上重算),不用易自指的 DC 系数。
    """
    base = max(block_mean, 1.0) / mid_gray
    return float(base ** a_t)
