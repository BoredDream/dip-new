# -*- coding: utf-8 -*-
"""模块二/三:水印方法。

classic.py   —— 经典信号处理水印(LSB / DCT / DWT / SVD / DWT-SVD / DWT-DCT / DFT / 扩频)
dct_qim_jpeg —— 在 JPEG 编码流水线内部嵌入的 DCT-QIM 水印(核心基线)
fragile.py   —— 脆弱/半脆弱水印,用于篡改定位
deep/        —— 学习型潜空间水印(RoSteALS 蓝本 + VINE 训练技巧)
"""
from .base import BaseWatermark

__all__ = ["BaseWatermark"]
