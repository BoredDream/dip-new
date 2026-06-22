# -*- coding: utf-8 -*-
"""频带存活诊断(P1)的单元测试。

用确定性合成攻击(噪声强度 ∝ strength)验证:
  * 默认嵌入频带的 zigzag 映射正确((0,0)=DC=0,(3,2)=中频=18);
  * 输出矩阵形状正确、相对扰动非负;
  * 攻击越强,总扰动越大(整体单调);
  * 推荐频带落在指定频段区间内、默认频带排名合法。
"""
import numpy as np

from swe.eval.diagnose import (
    diagnose_frequency_survival, default_embed_band, recommend_bands,
)


def _structured_rgb(seed: int, n: int = 128) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    base = 128 + 70 * np.sin(7 * x + seed) * np.cos(6 * y) + rng.normal(0, 4, (n, n))
    rgb = np.stack([base, np.roll(base, 3, 0), np.roll(base, 3, 1)], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _noise_attack(img, strength=0.3, seed=0):
    """确定性合成攻击:加 std ∝ strength 的高斯噪声(供诊断测试用,快且可复现)。"""
    rng = np.random.RandomState(seed)
    out = img.astype(np.float64) + rng.normal(0, 40.0 * strength, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def test_default_embed_band_mapping():
    assert default_embed_band((0, 0)) == 0          # DC 永远是 zigzag 第 0 个
    assert default_embed_band((3, 2)) == 18         # 标准 JPEG 之字形:中频
    b = default_embed_band((3, 2))
    assert 8 <= b <= 30                             # 确属中频区间


def test_diagnosis_shapes_and_monotonic():
    imgs = [_structured_rgb(s) for s in (1, 2, 3)]
    strengths = [0.1, 0.3, 0.6]
    diag = diagnose_frequency_survival(imgs, _noise_attack, strengths, seed=7, verbose=False)

    assert diag.abs_disturb.shape == (64, 3)
    assert diag.rel_disturb.shape == (64, 3)
    assert diag.clean_energy.shape == (64,)
    assert np.all(diag.abs_disturb >= 0)
    assert np.all(diag.rel_disturb >= 0)

    # 攻击越强,各频带平均绝对扰动整体越大
    total = diag.abs_disturb.mean(axis=0)
    assert total[2] > total[1] > total[0]


def test_recommend_and_rank():
    imgs = [_structured_rgb(s) for s in (4, 5)]
    diag = diagnose_frequency_survival(imgs, _noise_attack, [0.1, 0.2], seed=11, verbose=False)

    rec = diag.recommend(strength_idx=0, k=6, lo=1, hi=40)
    assert len(rec) == 6
    assert all(1 <= b < 40 for b in rec)
    assert len(set(rec)) == 6                       # 不重复

    rank = diag.default_rank(strength_idx=0)
    assert 1 <= rank <= 64


def test_recommend_bands_picks_smallest():
    rel = np.arange(64, dtype=float)                # 频带 i 的相对扰动 = i,越大越不稳
    rec = recommend_bands(rel, k=3, lo=1, hi=40)
    assert rec == [1, 2, 3]                         # 应挑 [lo,hi) 内最小的三个
