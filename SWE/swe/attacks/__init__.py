# -*- coding: utf-8 -*-
"""模块四:攻击套件。classic.py(经典失真) + ai.py(AI 再生成,按需依赖)。"""
from .classic import (
    gaussian_noise,
    salt_pepper_noise,
    gaussian_blur,
    median_filter,
    jpeg_recompress,
    center_crop,
    rescale,
    rotate,
    CLASSIC_ATTACKS,
)

__all__ = [
    "gaussian_noise",
    "salt_pepper_noise",
    "gaussian_blur",
    "median_filter",
    "jpeg_recompress",
    "center_crop",
    "rescale",
    "rotate",
    "CLASSIC_ATTACKS",
]
