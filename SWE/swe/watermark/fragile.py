# -*- coding: utf-8 -*-
"""脆弱 / 半脆弱水印 —— 篡改定位(实施方案 4.2.3,创新点 3)。

与鲁棒水印目标相反:脆弱水印要"一改就碎",从而**定位**图像被 AI 重绘/篡改的区域。
两者双水印共存(脆弱水印嵌 LSB / 半脆弱嵌 DCT,互不干扰鲁棒水印的中频载体)。

  * FragileWatermark      —— Wong 式块级脆弱水印:块 MSB 的哈希写入该块 LSB,
                             任何改动(含 LSB)都会令哈希校验失败 -> 块级定位。
  * SemiFragileWatermark  —— 把块的低频内容签名(粗量化,抗轻度 JPEG)用 QIM 嵌入中频,
                             设容差阈值,使其容忍 benign 压缩但对语义重绘敏感(AIGC 溯源推荐)。

定位输出:块分辨率的布尔篡改图 tamper_map(True=判定被篡改),以及上采样到像素的热图。
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np

from ..codec.dct import dct2, idct2
from .utils import qim_embed, qim_extract

__all__ = ["FragileWatermark", "SemiFragileWatermark", "tamper_heatmap"]


def _hash_bits(key: int, by: int, bx: int, msb_bytes: bytes, n: int) -> np.ndarray:
    """由 (密钥, 块坐标, 块 MSB 内容) 生成 n 个伪随机校验比特。

    把块坐标并入哈希,可抵御"整块搬运/拼贴"(vector quantization)攻击。
    """
    h = hashlib.sha256()
    h.update(key.to_bytes(8, "little", signed=False))
    h.update(by.to_bytes(4, "little"))
    h.update(bx.to_bytes(4, "little"))
    h.update(msb_bytes)
    digest = h.digest()
    need = (n + 7) // 8
    while len(digest) < need:                       # 需要更多比特则继续摘要
        h.update(digest)
        digest += h.digest()
    return np.unpackbits(np.frombuffer(digest[:need], dtype=np.uint8))[:n].astype(np.int64)


class FragileWatermark:
    """Wong 式块级脆弱水印(精确定位,任何像素改动即破碎)。"""

    def __init__(self, block: int = 8, key: int = 2026):
        self.bs = block
        self.key = key

    def embed(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.uint8).copy()
        out = g & 0xFE                               # 清零 LSB 平面
        H, W = g.shape
        bs = self.bs
        for by in range(H // bs):
            for bx in range(W // bs):
                r, c = by * bs, bx * bs
                blk = out[r:r + bs, c:c + bs]
                bits = _hash_bits(self.key, by, bx, blk.tobytes(), bs * bs)
                out[r:r + bs, c:c + bs] = blk | bits.reshape(bs, bs).astype(np.uint8)
        return out.astype(np.float64)

    def verify(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.uint8)
        H, W = g.shape
        bs = self.bs
        tamper = np.zeros((H // bs, W // bs), dtype=bool)
        for by in range(H // bs):
            for bx in range(W // bs):
                r, c = by * bs, bx * bs
                blk = g[r:r + bs, c:c + bs]
                msb = blk & 0xFE
                expect = _hash_bits(self.key, by, bx, msb.tobytes(), bs * bs)
                actual = (blk & 1).reshape(-1)
                tamper[by, bx] = not np.array_equal(actual, expect)
        return tamper


class SemiFragileWatermark:
    """半脆弱水印:容忍轻度 JPEG、对(重)绘篡改敏感。

    思路(标准半脆弱认证):给每个块嵌入一段**与密钥+块坐标绑定**的伪随机认证码,
    经 QIM 写入若干**中频** DCT 系数。验证时提取并与"期望认证码"比对,Hamming 距离 > tol
    判篡改。鲁棒性由 QIM 步长 Δ 控制:
      * Δ 取得较大 -> benign JPEG 的量化扰动不翻转 QIM 比特 -> 不误报;
      * 任何重绘/强编辑都会改变中频系数 -> 认证码破碎 -> 定位到该块。
    认证码不依赖块内容本身,故攻击者无法靠"伪造自洽块"(如整块涂成常数)绕过检测。
    """

    def __init__(self, block: int = 8, sig_bits: int = 6, delta: float = 40.0,
                 tol: int = 1, key: int = 2026):
        self.bs = block
        self.sig_bits = sig_bits
        self.delta = delta                           # QIM 步长:越大越能扛 benign JPEG
        self.tol = tol                               # 容差:Hamming 距离 > tol 才判篡改
        self.key = key
        self._carrier = [(3, 2), (2, 3), (4, 1), (1, 4), (3, 3), (4, 2)][:sig_bits]

    def _expected(self, by: int, bx: int) -> np.ndarray:
        """块的期望认证码(由密钥与块坐标决定,与块内容无关)。"""
        return _hash_bits(self.key, by, bx, b"", self.sig_bits)

    def embed(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float64).copy()
        H, W = g.shape
        bs = self.bs
        for by in range(H // bs):
            for bx in range(W // bs):
                r, c = by * bs, bx * bs
                X = dct2(g[r:r + bs, c:c + bs])
                sig = self._expected(by, bx)
                for k, (u, v) in enumerate(self._carrier):
                    X[u, v] = qim_embed(X[u, v], int(sig[k]), self.delta)
                g[r:r + bs, c:c + bs] = idct2(X)
        return np.clip(g, 0, 255)

    def verify(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float64)
        H, W = g.shape
        bs = self.bs
        tamper = np.zeros((H // bs, W // bs), dtype=bool)
        for by in range(H // bs):
            for bx in range(W // bs):
                r, c = by * bs, bx * bs
                X = dct2(g[r:r + bs, c:c + bs])
                expect = self._expected(by, bx)
                actual = np.array([qim_extract(X[u, v], self.delta) for (u, v) in self._carrier],
                                  dtype=np.int64)
                if int(np.sum(expect != actual)) > self.tol:
                    tamper[by, bx] = True
        return tamper


def tamper_heatmap(tamper_map: np.ndarray, image_shape: Tuple[int, int],
                   block: int) -> np.ndarray:
    """把块级布尔篡改图上采样成像素级热图(0=正常, 255=篡改)。"""
    heat = np.repeat(np.repeat(tamper_map.astype(np.uint8) * 255, block, axis=0), block, axis=1)
    H, W = image_shape
    out = np.zeros((H, W), dtype=np.uint8)
    h = min(H, heat.shape[0]); w = min(W, heat.shape[1])
    out[:h, :w] = heat[:h, :w]
    return out
