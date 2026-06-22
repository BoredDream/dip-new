# -*- coding: utf-8 -*-
"""经典(信号处理类)不可见水印合集 —— 模块二基线。

移植自 en-water/traditional_watermark.py,统一为 BaseWatermark 接口,DCT 复用
swe.codec.dct。包含:
    LSB / DCT / DFT / DWT(Haar) / SVD / DWT-SVD / DWT-DCT / 扩频
变换域方法统一用 QIM 在"载体系数"上嵌 1 比特/单元,差异只在选哪个系数。

与 en-water 的差异:
  * DFT 重写为"不做 fftshift、在原始频谱上取共轭对 (u,v) 与 ((H-u)%H,(W-v)%W)",
    修正了原实现因 fftshift 导致共轭配对错位、干净图也只有 ~0.46 比特准确率的缺陷。
"""
from __future__ import annotations

import numpy as np

from .base import BaseWatermark
from .utils import qim_embed, qim_extract, qim_embed_vec, qim_extract_vec
from ..codec.dct import dct2, idct2

__all__ = [
    "LSBWatermark", "DCTWatermark", "DFTWatermark", "DWTWatermark",
    "SVDWatermark", "DWTSVDWatermark", "DWTDCTWatermark", "SpreadSpectrumWatermark",
    "haar_dwt2", "haar_idwt2", "CLASSIC_METHODS",
]


# ----------------------------- Haar 小波(自实现) ----------------------------- #
def haar_dwt2(img: np.ndarray):
    """一级二维 Haar 小波分解 -> (LL, LH, HL, HH)。要求长宽为偶数。"""
    x = img.astype(np.float64)
    a = (x[:, 0::2] + x[:, 1::2]) / np.sqrt(2)
    d = (x[:, 0::2] - x[:, 1::2]) / np.sqrt(2)
    LL = (a[0::2, :] + a[1::2, :]) / np.sqrt(2)
    HL = (a[0::2, :] - a[1::2, :]) / np.sqrt(2)
    LH = (d[0::2, :] + d[1::2, :]) / np.sqrt(2)
    HH = (d[0::2, :] - d[1::2, :]) / np.sqrt(2)
    return LL, LH, HL, HH


def haar_idwt2(LL, LH, HL, HH) -> np.ndarray:
    """一级二维 Haar 小波重构(haar_dwt2 的逆)。"""
    a = np.zeros((LL.shape[0] * 2, LL.shape[1]))
    d = np.zeros((LL.shape[0] * 2, LL.shape[1]))
    a[0::2, :] = (LL + HL) / np.sqrt(2)
    a[1::2, :] = (LL - HL) / np.sqrt(2)
    d[0::2, :] = (LH + HH) / np.sqrt(2)
    d[1::2, :] = (LH - HH) / np.sqrt(2)
    x = np.zeros((a.shape[0], a.shape[1] * 2))
    x[:, 0::2] = (a + d) / np.sqrt(2)
    x[:, 1::2] = (a - d) / np.sqrt(2)
    return x


def _iter_blocks(h: int, w: int, bs: int):
    for r in range(0, h - bs + 1, bs):
        for c in range(0, w - bs + 1, bs):
            yield r, c


def _fit(bits: np.ndarray, n: int) -> np.ndarray:
    """保证提取结果恰为 n 个比特:超出截断,不足补 0。

    当图像过小、容量 < 请求比特数时(如极小图的 DWT-DCT),补 0 使返回长度一致,
    避免下游因长度不匹配而静默错位。容量充足时此操作无副作用。
    """
    bits = np.asarray(bits, dtype=np.int64)
    if len(bits) >= n:
        return bits[:n]
    return np.concatenate([bits, np.zeros(n - len(bits), dtype=np.int64)])


def _crop_even(ch: np.ndarray) -> np.ndarray:
    h, w = ch.shape
    return ch[:h - h % 2, :w - w % 2]


# ----------------------------- 方法实现 ----------------------------- #
class LSBWatermark(BaseWatermark):
    """最低有效位:把比特写进像素最低位。容量大,但极不鲁棒(下下限基线)。"""

    def embed(self, ch, bits):
        out = ch.astype(np.uint8).copy().ravel()
        n = min(len(bits), len(out))
        out[:n] = (out[:n] & 0xFE) | np.asarray(bits[:n], dtype=np.uint8)
        return out.reshape(ch.shape).astype(np.float64)

    def extract(self, ch, n_bits):
        return _fit((ch.astype(np.uint8).ravel()[:n_bits] & 1).astype(np.int64), n_bits)

    def capacity(self, shape):
        return int(np.prod(shape))


class DCTWatermark(BaseWatermark):
    """8×8 分块 DCT,在中频系数上 QIM 嵌 1 比特/块。抗 JPEG 压缩。"""

    def __init__(self, block=8, coef=(3, 2), delta=18.0):
        self.bs, self.coef, self.delta = block, coef, delta

    def embed(self, ch, bits):
        out = ch.astype(np.float64).copy()
        u, v = self.coef
        i = 0
        for r, c in _iter_blocks(*ch.shape, self.bs):
            if i >= len(bits):
                break
            X = dct2(out[r:r + self.bs, c:c + self.bs])
            X[u, v] = qim_embed(X[u, v], int(bits[i]), self.delta)
            out[r:r + self.bs, c:c + self.bs] = idct2(X)
            i += 1
        return np.clip(out, 0, 255)

    def extract(self, ch, n_bits):
        u, v = self.coef
        bits, i = [], 0
        for r, c in _iter_blocks(*ch.shape, self.bs):
            if i >= n_bits:
                break
            X = dct2(ch[r:r + self.bs, c:c + self.bs].astype(np.float64))
            bits.append(qim_extract(X[u, v], self.delta))
            i += 1
        return _fit(np.array(bits, dtype=np.int64), n_bits)

    def capacity(self, shape):
        h, w = shape
        return (h // self.bs) * (w // self.bs)


class DWTWatermark(BaseWatermark):
    """一级 Haar 小波,在 HL 子带系数上 QIM 嵌入。抗压缩/缩放。"""

    def __init__(self, delta=24.0):
        self.delta = delta

    def embed(self, ch, bits):
        ch = _crop_even(ch).astype(np.float64)
        LL, LH, HL, HH = haar_dwt2(ch)
        flat = HL.ravel()
        n = min(len(bits), len(flat))
        flat[:n] = qim_embed_vec(flat[:n], bits[:n], self.delta)
        return np.clip(haar_idwt2(LL, LH, flat.reshape(HL.shape), HH), 0, 255)

    def extract(self, ch, n_bits):
        ch = _crop_even(ch).astype(np.float64)
        _, _, HL, _ = haar_dwt2(ch)
        flat = HL.ravel()
        n = min(n_bits, len(flat))
        return _fit(qim_extract_vec(flat[:n], self.delta), n_bits)


class SVDWatermark(BaseWatermark):
    """8×8 分块 SVD,对每块最大奇异值 QIM 嵌 1 比特/块。数值稳定。"""

    def __init__(self, block=8, delta=30.0):
        self.bs, self.delta = block, delta

    def embed(self, ch, bits):
        out = ch.astype(np.float64).copy()
        i = 0
        for r, c in _iter_blocks(*ch.shape, self.bs):
            if i >= len(bits):
                break
            blk = out[r:r + self.bs, c:c + self.bs]
            U, S, Vt = np.linalg.svd(blk)
            S[0] = qim_embed(S[0], int(bits[i]), self.delta)
            out[r:r + self.bs, c:c + self.bs] = (U * S) @ Vt
            i += 1
        return np.clip(out, 0, 255)

    def extract(self, ch, n_bits):
        bits, i = [], 0
        for r, c in _iter_blocks(*ch.shape, self.bs):
            if i >= n_bits:
                break
            S = np.linalg.svd(ch[r:r + self.bs, c:c + self.bs].astype(np.float64),
                              compute_uv=False)
            bits.append(qim_extract(S[0], self.delta))
            i += 1
        return _fit(np.array(bits, dtype=np.int64), n_bits)

    def capacity(self, shape):
        h, w = shape
        return (h // self.bs) * (w // self.bs)


class DWTSVDWatermark(BaseWatermark):
    """先一级 Haar 小波,再对 LL 子带分块 SVD,奇异值上 QIM。经典鲁棒组合。"""

    def __init__(self, block=4, delta=30.0):
        self.bs, self.delta = block, delta

    def embed(self, ch, bits):
        ch = _crop_even(ch).astype(np.float64)
        LL, LH, HL, HH = haar_dwt2(ch)
        i = 0
        for r, c in _iter_blocks(*LL.shape, self.bs):
            if i >= len(bits):
                break
            blk = LL[r:r + self.bs, c:c + self.bs]
            U, S, Vt = np.linalg.svd(blk)
            S[0] = qim_embed(S[0], int(bits[i]), self.delta)
            LL[r:r + self.bs, c:c + self.bs] = (U * S) @ Vt
            i += 1
        return np.clip(haar_idwt2(LL, LH, HL, HH), 0, 255)

    def extract(self, ch, n_bits):
        ch = _crop_even(ch).astype(np.float64)
        LL, _, _, _ = haar_dwt2(ch)
        bits, i = [], 0
        for r, c in _iter_blocks(*LL.shape, self.bs):
            if i >= n_bits:
                break
            S = np.linalg.svd(LL[r:r + self.bs, c:c + self.bs], compute_uv=False)
            bits.append(qim_extract(S[0], self.delta))
            i += 1
        return _fit(np.array(bits, dtype=np.int64), n_bits)


class DWTDCTWatermark(BaseWatermark):
    """先一级 Haar 小波,再对 LL 子带 8×8 分块 DCT,中频系数 QIM。
    对应 VINE 论文里的 DwtDct 基线思路。"""

    def __init__(self, block=8, coef=(3, 2), delta=20.0):
        self.bs, self.coef, self.delta = block, coef, delta

    def embed(self, ch, bits):
        ch = _crop_even(ch).astype(np.float64)
        LL, LH, HL, HH = haar_dwt2(ch)
        u, v = self.coef
        i = 0
        for r, c in _iter_blocks(*LL.shape, self.bs):
            if i >= len(bits):
                break
            X = dct2(LL[r:r + self.bs, c:c + self.bs])
            X[u, v] = qim_embed(X[u, v], int(bits[i]), self.delta)
            LL[r:r + self.bs, c:c + self.bs] = idct2(X)
            i += 1
        return np.clip(haar_idwt2(LL, LH, HL, HH), 0, 255)

    def extract(self, ch, n_bits):
        ch = _crop_even(ch).astype(np.float64)
        LL, _, _, _ = haar_dwt2(ch)
        u, v = self.coef
        bits, i = [], 0
        for r, c in _iter_blocks(*LL.shape, self.bs):
            if i >= n_bits:
                break
            X = dct2(LL[r:r + self.bs, c:c + self.bs])
            bits.append(qim_extract(X[u, v], self.delta))
            i += 1
        return _fit(np.array(bits, dtype=np.int64), n_bits)


class DFTWatermark(BaseWatermark):
    """离散傅里叶幅度谱水印:在中频环上取共轭对称的系数对,用 QIM 改幅度。
    天然抗平移(平移只改相位不改幅度)。

    实现要点(修正 en-water 版):不做 fftshift,直接在 np.fft.fft2 的原始频谱上工作,
    此时 (u,v) 的共轭伙伴恰为 ((H-u)%H, (W-v)%W),配对自洽,干净图可达 ~1.0 准确率。
    """

    def __init__(self, delta=1200.0, radius_ratio=0.25, seed=2026):
        self.delta, self.radius_ratio, self.seed = delta, radius_ratio, seed

    def _positions(self, shape, n_bits):
        """在中频环(以 DC=(0,0) 为中心、按环绕频率计距离)挑选互不共轭的 (u,v)。"""
        h, w = shape
        rng = np.random.RandomState(self.seed)
        radius = min(h, w) * self.radius_ratio
        # 环绕频率坐标:fu∈[-h/2,h/2)
        fu = np.minimum(np.arange(h), h - np.arange(h))
        fv = np.minimum(np.arange(w), w - np.arange(w))
        dist = np.hypot(fu[:, None], fv[None, :])
        cand = list(zip(*np.where(np.abs(dist - radius) < 1.5)))
        rng.shuffle(cand)
        chosen, used = [], set()
        for u, v in cand:
            conj = ((h - u) % h, (w - v) % w)
            if (u, v) in used or conj in used or (u, v) == conj:
                continue
            used.add((u, v)); used.add(conj)
            chosen.append((u, v))
            if len(chosen) >= n_bits:
                break
        return chosen

    def embed(self, ch, bits):
        h, w = ch.shape
        F = np.fft.fft2(ch.astype(np.float64))
        mag, phase = np.abs(F), np.angle(F)
        for i, (u, v) in enumerate(self._positions(ch.shape, len(bits))):
            m = qim_embed(mag[u, v], int(bits[i]), self.delta)
            cu, cv = (h - u) % h, (w - v) % w
            mag[u, v] = mag[cu, cv] = m
        out = np.real(np.fft.ifft2(mag * np.exp(1j * phase)))
        return np.clip(out, 0, 255)

    def extract(self, ch, n_bits):
        mag = np.abs(np.fft.fft2(ch.astype(np.float64)))
        pos = self._positions(ch.shape, n_bits)
        return _fit(np.array([qim_extract(mag[u, v], self.delta) for u, v in pos], dtype=np.int64), n_bits)


class SpreadSpectrumWatermark(BaseWatermark):
    """扩频:每个比特对应一段伪随机 ±1 序列,叠加到整图 DCT 中频系数;
    解码用相关性检测(符号判 0/1)。抗高斯噪声的经典范式。"""

    def __init__(self, alpha=6.0, n_coef=4096, seed=2026):
        self.alpha, self.n_coef, self.seed = alpha, n_coef, seed

    def _mid_freq_index(self, shape):
        h, w = shape
        order = np.argsort(np.add.outer(np.arange(h), np.arange(w)), axis=None)
        lo = int(len(order) * 0.1)
        return order[lo:lo + self.n_coef]

    def _codes(self, idx_len, n_bits):
        rng = np.random.RandomState(self.seed)
        return rng.choice([-1.0, 1.0], size=(n_bits, idx_len))

    def embed(self, ch, bits):
        X = dct2(ch.astype(np.float64))
        flat = X.ravel()
        idx = self._mid_freq_index(ch.shape)
        codes = self._codes(len(idx), len(bits))
        for i, b in enumerate(bits):
            flat[idx] += self.alpha * (1.0 if b == 1 else -1.0) * codes[i]
        return np.clip(idct2(flat.reshape(X.shape)), 0, 255)

    def extract(self, ch, n_bits):
        X = dct2(ch.astype(np.float64)).ravel()
        idx = self._mid_freq_index(ch.shape)
        codes = self._codes(len(idx), n_bits)
        return _fit((codes @ X[idx] > 0).astype(np.int64), n_bits)


#: 名称 -> 构造器,供 demo / 评估器统一遍历
CLASSIC_METHODS = {
    "LSB": LSBWatermark,
    "DCT": DCTWatermark,
    "DFT": DFTWatermark,
    "DWT": DWTWatermark,
    "SVD": SVDWatermark,
    "DWT-SVD": DWTSVDWatermark,
    "DWT-DCT": DWTDCTWatermark,
    "SpreadSpec": SpreadSpectrumWatermark,
}
