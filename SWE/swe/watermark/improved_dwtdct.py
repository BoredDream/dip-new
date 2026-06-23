# -*- coding: utf-8 -*-
"""改进版 DwtDct 水印(ImprovedDwtDct)—— 实施方案创新点②(P2 主方法)。

在经典 ``DWTDCTWatermark`` 骨架(Y→一级 Haar DWT 取 LL→8×8 分块 DCT→载体系数 QIM)上,
做三处**可解释、可独立开关**的改进(支持消融实验,见方案 §4.4 / 图4):

  改进①(位置/多频带)  multiband:嵌入位置由默认中频 #18 → P1 诊断选出的**低频幸存带集合**
                       (默认 zigzag #1–7,方案 §4.3 结论),并跨**多频带 + 多块重复**做多数表决降 BER。
  改进②(强度/纹理)    texture:基础步长按块纹理活动度缩放——纹理强的块容忍更大 Δ(藏得住),
                       平坦块用小 Δ(避免块效应)。对应"纹理掩蔽"。
  改进③(可见性/JND)   jnd:Watson DCT JND = t_basic(频带) · 亮度掩蔽(块),给每个频带一个
                       "恰可察觉"上限,把低频嵌入的可见性压住(有 HVS 理论支撑,不靠调参)。

合成步长:``Δ_eff(块,带) = δ0 · texture_gain(块) · jnd_weight(带) · luminance_mask(块均值)``。

**盲提取一致性铁律**(方案风险 #2):自适应步长只用**嵌入后稳定、可在含水印图上重算**的统计量推出,
绝不自指被嵌的载体系数——
  * ``luminance_mask`` 用块均值,而块均值 = DCT 的 DC 系数(#0),本方法从不改 DC → 精确稳定;
  * ``texture_gain`` 用**非载体高频带**(载体集合之外、被嵌入完全不触碰的带)的能量 → 精确稳定。
故嵌入端与提取端推出的 Δ_eff 逐块逐带一致,干净图比特准确率 = 1.0。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import BaseWatermark
from .utils import qim_embed, qim_extract
from .jnd import watson_band_weights, luminance_mask
from .classic import haar_dwt2, haar_idwt2, _iter_blocks, _fit, _crop_even
from ..codec.dct import dct2, idct2
from ..codec.zigzag import zigzag, izigzag

__all__ = ["ImprovedDwtDct", "DEFAULT_LOW_BANDS"]

#: P1 频带诊断推荐的低频幸存带(zigzag 频带号;0=DC 已排除,见方案 §4.3)。
DEFAULT_LOW_BANDS: List[int] = [1, 2, 3, 4, 5, 6, 7]

# 纹理增益的固定常数(LL-DCT 交流能量量纲;只用于把"块纹理活动度"映射成 [GMIN,GMAX] 的步长倍率)。
_TEX_REF = 6.0
_TEX_GMIN = 0.6
_TEX_GMAX = 2.0
# 纹理统计取用的"非载体"频带上界(zigzag idx),与 DC、载体带都不重叠 → 嵌入不触碰、可精确重算。
_TEX_HI = 28


class ImprovedDwtDct(BaseWatermark):
    """改进版 DwtDct:多低频带 QIM + 纹理自适应 + Watson JND,盲提取。

    三个布尔开关可独立关闭以做消融:
        multiband=False → 退化为单带(只用 bands[0]),不做多频带重复;
        texture=False   → texture_gain 恒为 1;
        jnd=False        → watson 频带权重与亮度掩蔽均恒为 1(步长只剩 δ0)。
    """

    def __init__(self, block: int = 8, bands: Optional[Sequence[int]] = None,
                 delta: float = 20.0, multiband: bool = True,
                 texture: bool = True, jnd: bool = True, repeat: int = 4):
        self.bs = block
        self.bands = list(bands) if bands is not None else list(DEFAULT_LOW_BANDS)
        self.delta = float(delta)
        self.multiband = multiband
        self.texture = texture
        self.jnd = jnd
        self.repeat = max(1, int(repeat))
        # 实际承载的频带:multiband 用全部低频带,否则只用第一个。
        self._carriers = self.bands if self.multiband else self.bands[:1]
        # 频带权重(均值归一;关 jnd 时为全 1)。索引与 self._carriers 对齐。
        if self.jnd:
            self._band_w = watson_band_weights(self._carriers)
        else:
            self._band_w = np.ones(len(self._carriers), dtype=np.float64)
        # 纹理统计取用的非载体高频带:载体集合之外、(0, _TEX_HI) 区间内的带。
        cset = set(self._carriers)
        self._tex_bands = [b for b in range(1, _TEX_HI) if b not in cset]

    # ----------------------------- 自适应步长零件 ----------------------------- #
    def _texture_gain(self, zz: np.ndarray) -> float:
        """由块的非载体高频带能量估计纹理活动度 → 步长倍率(纹理强→倍率大)。

        只读 ``self._tex_bands``(嵌入从不修改),故嵌入端/提取端结果精确一致。
        """
        if not self.texture or not self._tex_bands:
            return 1.0
        energy = float(np.sqrt(np.mean(zz[self._tex_bands] ** 2)))
        gain = np.sqrt((energy + 1e-6) / _TEX_REF)
        return float(np.clip(gain, _TEX_GMIN, _TEX_GMAX))

    def _luminance(self, ll_block: np.ndarray) -> float:
        """块亮度掩蔽因子。用 LL 块均值换算回像素亮度(一级 Haar LL≈2×局部均值)。

        块均值只由 DC(#0)决定,本方法不改 DC → 嵌入前后精确稳定。
        """
        if not self.jnd:
            return 1.0
        pixel_mean = float(np.mean(ll_block)) / 2.0
        return luminance_mask(pixel_mean)

    def _block_deltas(self, ll_block: np.ndarray, zz: np.ndarray) -> np.ndarray:
        """该块各载体带的合成步长 Δ_eff(块,带)。"""
        tex = self._texture_gain(zz)
        lum = self._luminance(ll_block)
        return self.delta * tex * lum * self._band_w

    # ----------------------------- 嵌入 / 提取 ----------------------------- #
    def _plan(self, n_blocks: int, n_units: int) -> int:
        """每条消息比特重复嵌 repeat 个块(轮询交错,跨空间分散以抗局部攻击)。
        返回实际占用的块数 total;块 j 承载比特 (j % n_units)。"""
        return min(n_blocks, n_units * self.repeat)

    def embed(self, ch, bits):
        ch = _crop_even(ch).astype(np.float64)
        LL, LH, HL, HH = haar_dwt2(ch)
        blocks = list(_iter_blocks(*LL.shape, self.bs))
        n_units = len(bits)
        total = self._plan(len(blocks), n_units)
        for j in range(total):
            r, c = blocks[j]
            bit = int(bits[j % n_units])
            blk = LL[r:r + self.bs, c:c + self.bs]
            X = dct2(blk)
            zz = zigzag(X)
            deltas = self._block_deltas(blk, zz)
            for k, band in enumerate(self._carriers):
                zz[band] = qim_embed(zz[band], bit, float(deltas[k]))
            LL[r:r + self.bs, c:c + self.bs] = idct2(izigzag(zz))
        return np.clip(haar_idwt2(LL, LH, HL, HH), 0, 255)

    def extract(self, ch, n_bits):
        ch = _crop_even(ch).astype(np.float64)
        LL, _, _, _ = haar_dwt2(ch)
        blocks = list(_iter_blocks(*LL.shape, self.bs))
        total = self._plan(len(blocks), n_bits)
        # 每条消息比特累计所有副本(repeat 块 × 各载体带)的 0/1 票。
        votes = np.zeros((n_bits, 2), dtype=np.int64)
        for j in range(total):
            r, c = blocks[j]
            blk = LL[r:r + self.bs, c:c + self.bs]
            zz = zigzag(dct2(blk))
            deltas = self._block_deltas(blk, zz)
            bi = j % n_bits
            for k, band in enumerate(self._carriers):
                votes[bi, qim_extract(zz[band], float(deltas[k]))] += 1
        # 多数表决;无票(容量不足)的比特记 0,交由 _fit 处理。
        out = (votes[:, 1] > votes[:, 0]).astype(np.int64)
        out[votes.sum(axis=1) == 0] = 0
        return _fit(out, n_bits)

    def capacity(self, shape):
        h, w = _crop_even(np.empty(shape)).shape
        n_blocks = (h // 2 // self.bs) * (w // 2 // self.bs)  # LL 是半尺寸
        return n_blocks // self.repeat
