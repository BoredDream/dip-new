# -*- coding: utf-8 -*-
"""模块二/三:水印方法。

classic.py        —— 经典信号处理水印(LSB / DCT / DWT / SVD / DWT-SVD / DWT-DCT / DFT / 扩频)
improved_dwtdct   —— 改进版 DwtDct(创新②:多低频带 + 纹理自适应 + Watson JND)
dct_qim_jpeg      —— 在 JPEG 编码流水线内部嵌入的 DCT-QIM 水印(核心基线)
fragile.py        —— 脆弱/半脆弱水印,用于篡改定位
deep/             —— 学习型潜空间水印(RoSteALS 蓝本 + VINE 训练技巧)
"""
from .base import BaseWatermark
from .improved_dwtdct import ImprovedDwtDct

__all__ = ["BaseWatermark", "ImprovedDwtDct"]
