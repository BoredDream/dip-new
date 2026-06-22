# -*- coding: utf-8 -*-
"""JPEG 编解码器与变换的单元测试。"""
import numpy as np
import pytest

from swe.codec.dct import dct_8x8, idct_8x8, dct2, idct2
from swe.codec.zigzag import zigzag, izigzag, rle_encode_ac, rle_decode_ac
from swe.codec.quant import quality_to_qtable
from swe.codec import JPEGCodec


def test_dct_inverse():
    rng = np.random.RandomState(0)
    blk = rng.uniform(-128, 127, (8, 8))
    assert np.allclose(idct_8x8(dct_8x8(blk)), blk, atol=1e-8)


def test_dct2_inverse_nonsquare():
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 255, (4, 8))
    assert np.allclose(idct2(dct2(x)), x, atol=1e-8)


def test_dct_dc_is_mean_times_8():
    blk = np.full((8, 8), 10.0)
    assert dct_8x8(blk)[0, 0] == pytest.approx(80.0, abs=1e-6)


def test_zigzag_inverse():
    blk = np.arange(64).reshape(8, 8)
    assert np.array_equal(izigzag(zigzag(blk)), blk)


def test_zigzag_low_freq_first():
    # DC 在最前,zigzag 第二项是 (0,1) 或 (1,0)
    blk = np.zeros((8, 8)); blk[0, 0] = 5
    assert zigzag(blk)[0] == 5


def test_rle_roundtrip_full_block():
    rng = np.random.RandomState(1)
    ac = rng.randint(-3, 4, 63)
    assert np.array_equal(rle_decode_ac(rle_encode_ac(ac)), ac)


def test_quality_scaling_monotonic():
    # 质量越高,量化步长越小
    q90 = quality_to_qtable(90).mean()
    q50 = quality_to_qtable(50).mean()
    q10 = quality_to_qtable(10).mean()
    assert q90 < q50 < q10


def test_codec_roundtrip_gray():
    rng = np.random.RandomState(2)
    img = rng.randint(0, 256, (64, 64), dtype=np.uint8)
    rec, bpp = JPEGCodec(quality=90).compress_decompress(img)
    assert rec.shape == img.shape and rec.dtype == np.uint8
    # 高质量下 PSNR 应较高
    mse = np.mean((rec.astype(float) - img) ** 2)
    assert 10 * np.log10(255 ** 2 / mse) > 28


def test_codec_roundtrip_rgb():
    rng = np.random.RandomState(3)
    img = rng.randint(0, 256, (32, 48, 3), dtype=np.uint8)
    rec, _ = JPEGCodec(quality=80).compress_decompress(img)
    assert rec.shape == img.shape


def test_codec_compression_ratio():
    # 低质量 bpp 应小于高质量
    rng = np.random.RandomState(4)
    img = rng.randint(0, 256, (64, 64), dtype=np.uint8)
    _, bpp10 = JPEGCodec(quality=10).compress_decompress(img)
    _, bpp90 = JPEGCodec(quality=90).compress_decompress(img)
    assert bpp10 < bpp90
