# -*- coding: utf-8 -*-
"""水印通用工具:文本<->比特、消息打包、QIM 量化索引调制、随机比特。

QIM(Quantization Index Modulation)是除 LSB / 扩频外所有变换域方法共享的嵌入原语:
在某个"载体系数"上量化到不同格点来携带 1 比特。改鲁棒性主要靠调步长 delta
(越大越鲁棒、PSNR 越低)。本文件与 en-water/traditional_watermark.py 的工具保持一致。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "text_to_bits", "bits_to_text", "pack_message", "unpack_message",
    "qim_embed", "qim_extract", "qim_embed_vec", "qim_extract_vec",
    "random_bits",
]


# ----------------------------- 文本 <-> 比特 ----------------------------- #
def text_to_bits(text: str) -> np.ndarray:
    """UTF-8 文本 -> 0/1 比特数组(大端,每字符 8 位)。"""
    data = text.encode("utf-8")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.int64)


def bits_to_text(bits: np.ndarray) -> str:
    """0/1 比特数组 -> UTF-8 文本(长度截到 8 的倍数)。"""
    bits = np.asarray(bits, dtype=np.uint8)
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n]).tobytes().decode("utf-8", errors="replace")


def pack_message(text: str) -> np.ndarray:
    """文本 -> [32 位长度头 | payload]。长度头记录 payload 比特数。"""
    payload = text_to_bits(text)
    header = np.array([int(b) for b in format(len(payload), "032b")], dtype=np.int64)
    return np.concatenate([header, payload])


def unpack_message(bits: np.ndarray) -> str:
    """[32 位长度头 | payload] -> 文本。"""
    bits = np.asarray(bits, dtype=np.int64)
    if len(bits) < 32:
        return ""
    length = int("".join(str(int(b)) for b in bits[:32]), 2)
    length = max(0, min(length, len(bits) - 32))
    return bits_to_text(bits[32:32 + length])


# ----------------------------- QIM ----------------------------- #
def qim_embed(value: float, bit: int, delta: float) -> float:
    """标量 QIM:bit=0 量化到 delta 整数格点,bit=1 偏移半格。"""
    if bit == 0:
        return round(value / delta) * delta
    return round((value - delta / 2) / delta) * delta + delta / 2


def qim_extract(value: float, delta: float) -> int:
    """标量 QIM 解调:看 value 离哪组格点更近。"""
    q0 = round(value / delta) * delta
    q1 = round((value - delta / 2) / delta) * delta + delta / 2
    return 0 if abs(value - q0) <= abs(value - q1) else 1


def qim_embed_vec(values: np.ndarray, bits: np.ndarray, delta: float) -> np.ndarray:
    """向量化 QIM 嵌入。values、bits 等长。"""
    values = np.asarray(values, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.int64)
    q0 = np.round(values / delta) * delta
    q1 = np.round((values - delta / 2) / delta) * delta + delta / 2
    return np.where(bits == 0, q0, q1)


def qim_extract_vec(values: np.ndarray, delta: float) -> np.ndarray:
    """向量化 QIM 解调。"""
    values = np.asarray(values, dtype=np.float64)
    q0 = np.round(values / delta) * delta
    q1 = np.round((values - delta / 2) / delta) * delta + delta / 2
    return (np.abs(values - q0) > np.abs(values - q1)).astype(np.int64)


def random_bits(n: int, seed: int = 2026) -> np.ndarray:
    """生成 n 个伪随机 0/1 比特(固定种子,便于复现)。"""
    return np.random.RandomState(seed).binomial(1, 0.5, n).astype(np.int64)
