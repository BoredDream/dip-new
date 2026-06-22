# -*- coding: utf-8 -*-
"""模块六:纠错编码(ECC)。无第三方依赖的 RS / 汉明 / 重复码实现。"""
from .codec import (
    ECC,
    RepetitionECC,
    HammingECC,
    ReedSolomonECC,
    make_ecc,
)

__all__ = ["ECC", "RepetitionECC", "HammingECC", "ReedSolomonECC", "make_ecc"]
