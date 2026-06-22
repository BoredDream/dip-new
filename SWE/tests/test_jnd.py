# -*- coding: utf-8 -*-
"""Watson DCT JND 感知模型(改进③)的单元测试。"""
import numpy as np

from swe.watermark.jnd import (
    WATSON_LUMINANCE_TABLE, watson_band_weights, luminance_mask, _WATSON_ZIGZAG,
)


def test_watson_table_shape():
    assert WATSON_LUMINANCE_TABLE.shape == (8, 8)
    assert np.all(WATSON_LUMINANCE_TABLE > 0)
    # 低频(左上)阈值小=人眼更敏感;高频(右下)阈值大=更不敏感
    assert WATSON_LUMINANCE_TABLE[0, 0] < WATSON_LUMINANCE_TABLE[7, 7]


def test_band_weights_normalized():
    bands = [1, 2, 3, 4, 5, 7]
    w = watson_band_weights(bands)
    assert w.shape == (len(bands),)
    assert np.all(w > 0)
    # 均值归一:所选频带权重均值≈1(让 δ0 仍是主旋钮)
    assert abs(w.mean() - 1.0) < 1e-9


def test_zigzag_dc_maps_to_table_dc():
    # zigzag 频带 0 = DC,对应表的 (0,0)
    assert _WATSON_ZIGZAG[0] == WATSON_LUMINANCE_TABLE[0, 0]
    assert _WATSON_ZIGZAG.shape == (64,)


def test_luminance_mask_monotonic():
    # 块越亮,掩蔽因子越大(可藏越多扰动)
    assert luminance_mask(64) < luminance_mask(128) < luminance_mask(200)
    # 中灰(128)处因子=1
    assert abs(luminance_mask(128) - 1.0) < 1e-9
    # 极暗块也不会爆成 0(下限保护)
    assert luminance_mask(0) > 0
