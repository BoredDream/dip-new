# -*- coding: utf-8 -*-
"""频带存活诊断(P1)—— 改进 DwtDct"往哪嵌"的客观依据。

回答一个可答辩的问题:
    在低-strength img2img(或其纯 CPU 代理)下,DwtDct 的嵌入域
    —— 亮度 Y 的 DWT-LL 子带、再做 8×8 DCT —— 的哪些频带基本不被破坏?
    不被破坏(扰动小)的频带 = 水印应该改嵌的位置。

整条链路每一步都映射到课程概念,不引入任何黑盒:
    RGB→Y 亮度(BT.601) → 一级 Haar DWT 取 LL(多分辨率分析)
    → 8×8 DCT(频域) → zigzag(低频在前的频带排序)
    → 逐频带统计"攻击前/后系数变化"。

对每个 zigzag 频带 k(0=DC … 63=最高频)、每个攻击强度 s:
    abs[k,s] = mean_{块,图} |D_attacked[k] − D_clean[k]|         绝对扰动(决定 QIM 步长下限)
    rel[k,s] = abs[k,s] / (mean_{块,图} |D_clean[k]| + ε)        相对扰动(越小越稳→越该嵌)

诊断默认走纯 CPU 代理攻击(regeneration_surrogate),保证无 GPU 也能"落地"出图;
有 GPU+diffusers 时把 attack_fn 换成 diffusion_img2img 即可,接口一致。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

from ..codec.color import rgb_to_ycbcr
from ..codec.dct import dct2
from ..codec.zigzag import zigzag, ZIGZAG_ORDER
from ..watermark.classic import haar_dwt2
from ..data.datasets import to_uint8

__all__ = [
    "FrequencyDiagnosis",
    "diagnose_frequency_survival",
    "default_embed_band",
    "recommend_bands",
]


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _luma(img: np.ndarray) -> np.ndarray:
    """取亮度 Y(BT.601);灰度图原样返回。DwtDct 嵌在亮度上,故诊断也在亮度上。"""
    if img.ndim == 2:
        return img.astype(np.float64)
    return rgb_to_ycbcr(img)[..., 0]


def _match_size(att: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """攻击若改变了尺寸,用 bicubic 拉回参考尺寸,保证与 clean 的 8×8 分块逐块对齐。"""
    if att.shape[:2] == ref.shape[:2]:
        return att
    from PIL import Image
    H, W = ref.shape[:2]
    return np.asarray(Image.fromarray(to_uint8(att)).resize((W, H), Image.BICUBIC))


def _blocks_zigzag(LL: np.ndarray, block: int) -> np.ndarray:
    """LL 子带按 block×block 无重叠分块,每块 DCT 后 zigzag 成 64 维。

    返回 (n_blocks, 64);遍历顺序固定(行优先),保证 clean 与 attacked 的块一一对应。
    """
    h = (LL.shape[0] // block) * block
    w = (LL.shape[1] // block) * block
    out: List[np.ndarray] = []
    for r in range(0, h, block):
        for c in range(0, w, block):
            out.append(zigzag(dct2(LL[r:r + block, c:c + block])))
    return np.asarray(out) if out else np.zeros((0, block * block))


def _call_attack(fn: Callable[..., np.ndarray], img: np.ndarray, strength: float, seed: int) -> np.ndarray:
    """统一调用攻击函数:带 seed 形参的(代理攻击)逐图换种子,不带的(img2img)直接调。"""
    params = inspect.signature(fn).parameters
    if "seed" in params:
        return fn(img, strength=strength, seed=seed)
    return fn(img, strength=strength)


def default_embed_band(coef=(3, 2)) -> int:
    """DwtDct 默认嵌入系数 (row,col) 对应的 zigzag 频带编号(0=DC … 63=最高频)。

    与 codec 的 ZIGZAG_ORDER 保持一致(不硬编码),(3,2) 在标准 JPEG 之字形里 = 18,属中频。
    """
    flat = coef[0] * 8 + coef[1]
    return int(np.where(ZIGZAG_ORDER == flat)[0][0])


def recommend_bands(rel_at_strength: np.ndarray, k: int = 6, lo: int = 1, hi: int = 40) -> List[int]:
    """在 [lo, hi) 频带范围内挑相对扰动最小的 k 个频带。

    lo=1 跳过 DC(嵌 DC 改的是整体亮度,极可见);hi=40 跳过几乎全被重写的高频。
    注意:本函数只按"稳定性"选,可见性约束留给 P2 的 JND(最稳的往往是低频,需 JND 压可见性)。
    """
    band = np.arange(len(rel_at_strength))
    mask = (band >= lo) & (band < hi)
    cand = band[mask]
    order = cand[np.argsort(rel_at_strength[mask])]
    return sorted(int(b) for b in order[:k])


# --------------------------------------------------------------------------- #
# 诊断结果容器
# --------------------------------------------------------------------------- #
@dataclass
class FrequencyDiagnosis:
    strengths: np.ndarray      # (S,)   攻击强度网格
    abs_disturb: np.ndarray    # (64,S) 绝对扰动 mean|ΔDCT|
    rel_disturb: np.ndarray    # (64,S) 相对扰动 = abs / clean_energy
    clean_energy: np.ndarray   # (64,)  各频带 clean 平均 |系数|(也是可见性预算的参考)
    default_band: int          # DwtDct 默认嵌入频带(zigzag idx)

    def recommend(self, strength_idx: int = 0, k: int = 6, lo: int = 1, hi: int = 40) -> List[int]:
        """按最低(最温和、最具威胁)强度挑稳定频带。"""
        return recommend_bands(self.rel_disturb[:, strength_idx], k=k, lo=lo, hi=hi)

    def default_rank(self, strength_idx: int = 0) -> int:
        """默认频带在"相对扰动从小到大"里的排名(1=最稳)。用于量化'默认位置有多差'。"""
        order = np.argsort(self.rel_disturb[:, strength_idx])
        return int(np.where(order == self.default_band)[0][0]) + 1


# --------------------------------------------------------------------------- #
# 主诊断
# --------------------------------------------------------------------------- #
def diagnose_frequency_survival(
    images: Sequence[np.ndarray],
    attack_fn: Callable[..., np.ndarray],
    strengths: Sequence[float],
    block: int = 8,
    seed: int = 2026,
    coef=(3, 2),
    verbose: bool = True,
) -> FrequencyDiagnosis:
    """对一批图,统计 img2img(代理)在各强度下对 DwtDct 嵌入域各频带的破坏。

    images:    uint8 RGB(或灰度)图列表;建议 256×256、尺寸为 16 的倍数。
    attack_fn: regeneration_surrogate(默认,CPU)或 diffusion_img2img(GPU)。
    strengths: 强度网格,建议密集覆盖低段,如 [0.05,0.1,0.15,0.2,0.25,0.3]。
    """
    strengths = list(strengths)
    S = len(strengths)
    n_coef = block * block

    # 1) 预计算每张图的 clean LL-DCT zigzag 块(与强度无关,只算一次)
    clean_blocks: List[np.ndarray] = []
    for img in images:
        clean_blocks.append(_blocks_zigzag(haar_dwt2(_luma(img))[0], block))
    clean_all = np.concatenate(clean_blocks, axis=0) if clean_blocks else np.zeros((0, n_coef))
    clean_energy = np.abs(clean_all).mean(axis=0) if len(clean_all) else np.zeros(n_coef)

    # 2) 逐强度:攻击 -> 取同样的块网格 -> 累积 |ΔDCT|
    abs_disturb = np.zeros((n_coef, S))
    for si, s in enumerate(strengths):
        diffs: List[np.ndarray] = []
        for ii, img in enumerate(images):
            att = _match_size(_call_attack(attack_fn, img, s, seed + ii), img)
            z1 = _blocks_zigzag(haar_dwt2(_luma(att))[0], block)
            diffs.append(np.abs(z1 - clean_blocks[ii]))
        abs_disturb[:, si] = np.concatenate(diffs, axis=0).mean(axis=0)
        if verbose:
            print(f"  strength={s:<5}  mean|dDCT|(all bands)={abs_disturb[:, si].mean():.3f}")

    rel_disturb = abs_disturb / (clean_energy[:, None] + 1e-6)
    return FrequencyDiagnosis(
        strengths=np.asarray(strengths, dtype=np.float64),
        abs_disturb=abs_disturb,
        rel_disturb=rel_disturb,
        clean_energy=clean_energy,
        default_band=default_embed_band(coef),
    )
