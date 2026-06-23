# -*- coding: utf-8 -*-
"""改进版 DwtDct(创新②,P2 主方法)的单元测试。

验收要点(方案风险 #2):自适应步长不自指被嵌系数 → 干净图比特准确率 = 1.0,
且三个改进开关可独立关闭仍能干净解码。
"""
import numpy as np
import pytest

from swe.watermark.improved_dwtdct import ImprovedDwtDct, DEFAULT_LOW_BANDS
from swe.watermark.utils import random_bits
from swe.eval.metrics import psnr


@pytest.fixture
def gray():
    rng = np.random.RandomState(0)
    # 带结构的图(非纯噪声),便于变换域方法(与 test_watermark_classic 一致)
    x, y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
    img = (128 + 80 * np.sin(6 * x) * np.cos(5 * y) + rng.normal(0, 5, (128, 128)))
    return np.clip(img, 0, 255)


# 四档:默认低频多频带 / 关 multiband / 关 texture / 关 jnd / 全开
_VARIANTS = {
    "full":        dict(multiband=True,  texture=True,  jnd=True),
    "no_multi":    dict(multiband=False, texture=True,  jnd=True),
    "no_texture":  dict(multiband=True,  texture=False, jnd=True),
    "no_jnd":      dict(multiband=True,  texture=True,  jnd=False),
    "plain":       dict(multiband=True,  texture=False, jnd=False),
}


@pytest.mark.parametrize("name,kw", list(_VARIANTS.items()))
def test_clean_extract_each_variant(name, kw, gray):
    """干净图:每种开关组合都应精确还原(≥0.99)。"""
    m = ImprovedDwtDct(repeat=4, **kw)
    bits = random_bits(16)
    wm = m.embed(gray.copy(), bits)
    ext = m.extract(wm, len(bits))
    assert np.mean(ext == bits) >= 0.99, f"{name} 干净提取应≈1.0"


def test_default_bands_are_low_freq():
    m = ImprovedDwtDct()
    assert m.bands == DEFAULT_LOW_BANDS
    assert max(m.bands) <= 12, "默认嵌入应落在低频区(方案 §4.3 诊断结论)"


def test_multiband_off_uses_single_carrier(gray):
    m = ImprovedDwtDct(multiband=False)
    assert len(m._carriers) == 1
    bits = random_bits(16)
    wm = m.embed(gray.copy(), bits)
    assert np.mean(m.extract(wm, 16) == bits) >= 0.99


def test_invisible(gray):
    """嵌入应隐形(PSNR 足够高)。"""
    bits = random_bits(16)
    wm = ImprovedDwtDct().embed(gray.copy(), bits)
    assert psnr(gray, wm) > 35


def test_jnd_improves_or_keeps_invisibility(gray):
    """开启 JND(频带权重 + 亮度掩蔽)不应显著恶化可见性。

    JND 在敏感频带少嵌、不敏感频带多嵌(均值归一),PSNR 与关闭时应相当量级。
    """
    bits = random_bits(16)
    p_on = psnr(gray, ImprovedDwtDct(jnd=True).embed(gray.copy(), bits))
    p_off = psnr(gray, ImprovedDwtDct(jnd=False).embed(gray.copy(), bits))
    assert p_on >= p_off - 3.0, f"JND 不应明显恶化 PSNR(on={p_on:.1f} off={p_off:.1f})"


def test_majority_vote_helps_under_noise(gray):
    """多块重复 + 多数表决:加噪后 repeat 大的应不差于 repeat=1。"""
    rng = np.random.RandomState(3)
    bits = random_bits(8)
    noisy_acc = []
    for rep in (1, 6):
        m = ImprovedDwtDct(repeat=rep)
        wm = m.embed(gray.copy(), bits)
        atk = np.clip(wm + rng.normal(0, 8, wm.shape), 0, 255)
        noisy_acc.append(np.mean(m.extract(atk, 8) == bits))
    assert noisy_acc[1] >= noisy_acc[0] - 1e-9, f"repeat 大应更鲁棒: {noisy_acc}"
