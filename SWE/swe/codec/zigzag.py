# -*- coding: utf-8 -*-
"""Zigzag 扫描、DC 的 DPCM、AC 的游程编码(JPEG 第 6、7 步)。

  * Zigzag:把 8×8 量化系数按"低频在前"的之字形序列化成 64 维向量;
  * DC-DPCM:每块 DC 只编码与上一块 DC 的差值;
  * AC-RLE:把 AC 系数编成 (前导零个数, 非零值) 对,长零串用 ZRL(15,0),块尾用 EOB。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = [
    "ZIGZAG_ORDER",
    "zigzag",
    "izigzag",
    "dc_to_dpcm",
    "dpcm_to_dc",
    "rle_encode_ac",
    "rle_decode_ac",
    "EOB",
    "ZRL",
]


def _build_zigzag_order(n: int = 8) -> np.ndarray:
    """生成 n×n 的 zigzag 顺序(返回长度 n² 的行优先索引数组)。"""
    coords: List[Tuple[int, int]] = []
    for s in range(2 * n - 1):           # s = 反对角线编号 (= i + j)
        diag = [(i, s - i) for i in range(n) if 0 <= s - i < n]
        if s % 2 == 0:
            diag.reverse()               # 偶数对角线自下而上
        coords.extend(diag)
    return np.array([r * n + c for r, c in coords], dtype=np.int64)


ZIGZAG_ORDER = _build_zigzag_order(8)
_INV_ZIGZAG = np.argsort(ZIGZAG_ORDER)

# AC 游程编码的特殊符号
EOB = (0, 0)     # End-of-Block:本块剩余 AC 全为 0
ZRL = (15, 0)    # Zero-Run-Length:16 个连续 0


def zigzag(block: np.ndarray) -> np.ndarray:
    """8×8 -> 64 维 zigzag 向量。"""
    return block.reshape(-1)[ZIGZAG_ORDER]


def izigzag(vec: np.ndarray) -> np.ndarray:
    """64 维 zigzag 向量 -> 8×8。"""
    return np.asarray(vec).reshape(-1)[_INV_ZIGZAG].reshape(8, 8)


def dc_to_dpcm(dc_values: np.ndarray) -> np.ndarray:
    """DC 序列 -> 差分(第一个保留原值)。"""
    dc = np.asarray(dc_values, dtype=np.int64)
    diff = np.empty_like(dc)
    diff[0] = dc[0]
    diff[1:] = dc[1:] - dc[:-1]
    return diff


def dpcm_to_dc(diff: np.ndarray) -> np.ndarray:
    """差分 -> DC 序列(dc_to_dpcm 的逆)。"""
    return np.cumsum(np.asarray(diff, dtype=np.int64))


def rle_encode_ac(ac: np.ndarray) -> List[Tuple[int, int]]:
    """63 维 AC 向量 -> [(run, value), ...],以 EOB 结束(全零则仅 EOB)。"""
    ac = np.asarray(ac, dtype=np.int64)
    symbols: List[Tuple[int, int]] = []
    run = 0
    for v in ac:
        if v == 0:
            run += 1
            continue
        while run > 15:                  # 超过 15 个零先发 ZRL
            symbols.append(ZRL)
            run -= 16
        symbols.append((run, int(v)))
        run = 0
    # 仅当最后一个非零系数之后仍有零(run>0,含整块全零)才发 EOB。
    # 若第 63 个 AC 系数非零(无尾随零),按 JPEG 约定不写 EOB,否则解码端会多读一个符号而错位。
    if run > 0:
        symbols.append(EOB)
    return symbols


def rle_decode_ac(symbols: List[Tuple[int, int]]) -> np.ndarray:
    """[(run, value), ...] -> 63 维 AC 向量。"""
    ac: List[int] = []
    for run, value in symbols:
        if (run, value) == EOB:
            break
        ac.extend([0] * run)
        if (run, value) == ZRL:
            continue
        ac.append(value)
    ac.extend([0] * (63 - len(ac)))
    return np.array(ac[:63], dtype=np.int64)
