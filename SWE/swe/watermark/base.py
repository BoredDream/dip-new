# -*- coding: utf-8 -*-
"""水印统一接口。

所有"在单通道(灰度/亮度 Y)上嵌入比特"的方法都实现该接口,从而被评估器与
攻击套件以同一方式驱动(沿用 en-water 的设计约定):

    embed(channel_float, bits)  -> 含水印通道(float, 已 clip 到 [0,255])
    extract(channel_float, n)   -> 提取出的 n 个比特(int 数组)

约定 channel 为浮点的单通道数组(取值 0..255)。彩色场景由上层在 YCbCr 的 Y 通道
上调用、其余通道原样保留。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseWatermark(ABC):
    """单通道比特水印基类。"""

    #: 是否为盲提取(无需原图)。本项目方法均为盲提取。
    blind: bool = True

    @abstractmethod
    def embed(self, channel: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """把 bits 嵌入 channel,返回含水印通道。"""

    @abstractmethod
    def extract(self, channel: np.ndarray, n_bits: int) -> np.ndarray:
        """从(可能被攻击的)channel 提取 n_bits 个比特。"""

    def capacity(self, shape) -> int:
        """该方法在给定尺寸下可嵌入的最大比特数(默认未知,返回 -1)。"""
        return -1

    @property
    def name(self) -> str:
        return type(self).__name__
