# -*- coding: utf-8 -*-
"""二维 DCT-II / IDCT。

实现两份:
  * 通用矩阵版 dct2/idct2(任意 N×N,用于教学与小波-DCT 复合方法);
  * 8×8 专用版 dct_8x8/idct_8x8(缓存 8×8 正交基,JPEG 流水线高频调用)。

二维 DCT 可写成 X = D · f · Dᵀ(行列分离),D 为一维正交 DCT-II 矩阵。
guetzli 的 `fdct.cc` 是"带缩放的快速整数 DCT"(输出按 16 缩放、用整数余弦常数近似,
分两趟做可分离变换以求速度);本文件计算的是**同一种变换 DCT-II**,但改用浮点正交矩阵
乘法(更易读、便于教学与小波-DCT 复合方法),与 guetzli 的整数缩放版在数值尺度上不同。

JPEG 标准的 8×8 前向 DCT 定义(课件第 4 章 4.4):
    F(u,v) = (1/4) C(u) C(v) ΣΣ f(x,y) cos[(2x+1)uπ/16] cos[(2y+1)vπ/16]
其中 C(0)=1/√2, C(k≠0)=1。本文件的正交归一化与之等价(常数并入矩阵)。
"""
from __future__ import annotations

import numpy as np

__all__ = ["dct2", "idct2", "dct_8x8", "idct_8x8", "dct_matrix"]


def dct_matrix(n: int) -> np.ndarray:
    """正交 DCT-II 矩阵 D,使 dct(x) = D @ x、idct(X) = D.T @ X。"""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    D = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    D *= np.sqrt(2.0 / n)
    D[0, :] *= np.sqrt(0.5)
    return D


def dct2(block: np.ndarray) -> np.ndarray:
    """二维 DCT:X = Dr @ block @ Dc.T(支持非方形)。"""
    Dr = dct_matrix(block.shape[0])
    Dc = dct_matrix(block.shape[1])
    return Dr @ block @ Dc.T


def idct2(coef: np.ndarray) -> np.ndarray:
    """二维 IDCT:block = Dr.T @ X @ Dc(dct2 的逆)。"""
    Dr = dct_matrix(coef.shape[0])
    Dc = dct_matrix(coef.shape[1])
    return Dr.T @ coef @ Dc


# 8×8 基缓存(JPEG 主路径) ---------------------------------------------------- #
_D8 = dct_matrix(8)
_D8T = _D8.T


def dct_8x8(block: np.ndarray) -> np.ndarray:
    """单个 8×8 块的二维 DCT。block 已电平移位(像素−128)。"""
    return _D8 @ block @ _D8T


def idct_8x8(coef: np.ndarray) -> np.ndarray:
    """单个 8×8 块的二维 IDCT。"""
    return _D8T @ coef @ _D8


def dct_8x8_batch(blocks: np.ndarray) -> np.ndarray:
    """批量 8×8 DCT。blocks 形状 (..., 8, 8)。

    用 einsum 做 D @ X @ Dᵀ,向量化所有块,等价 guetzli 对每块独立做可分离 DCT。
    """
    return np.einsum("ij,...jk,lk->...il", _D8, blocks, _D8, optimize=True)


def idct_8x8_batch(coefs: np.ndarray) -> np.ndarray:
    """批量 8×8 IDCT。coefs 形状 (..., 8, 8)。"""
    return np.einsum("ji,...jk,kl->...il", _D8, coefs, _D8, optimize=True)
