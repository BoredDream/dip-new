# -*- coding: utf-8 -*-
"""经典水印与 JPEG 域水印的单元测试。"""
import numpy as np
import pytest

from swe.watermark.classic import CLASSIC_METHODS
from swe.watermark.dct_qim_jpeg import DCTQIMJPEGWatermark
from swe.watermark.utils import (random_bits, pack_message, unpack_message,
                                 qim_embed, qim_extract)
from swe.attacks.classic import jpeg_recompress


@pytest.fixture
def gray():
    rng = np.random.RandomState(0)
    # 带结构的图(非纯噪声),便于变换域方法
    x, y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
    img = (128 + 80 * np.sin(6 * x) * np.cos(5 * y) + rng.normal(0, 5, (128, 128)))
    return np.clip(img, 0, 255)


def test_qim_scalar():
    for b in (0, 1):
        v = qim_embed(123.4, b, 10.0)
        assert qim_extract(v, 10.0) == b


def test_message_pack_unpack():
    msg = "DIP-2026 水印"
    assert unpack_message(pack_message(msg)) == msg


@pytest.mark.parametrize("name", list(CLASSIC_METHODS.keys()))
def test_classic_clean_extract(name, gray):
    m = CLASSIC_METHODS[name]()
    bits = random_bits(48)
    wm = m.embed(gray.copy(), bits)
    ext = m.extract(wm, len(bits))
    assert np.mean(ext == bits) >= 0.99, f"{name} 干净提取应≈1.0"


def test_classic_invisible(gray):
    # PSNR 应较高(除扩频外)
    from swe.eval.metrics import psnr
    bits = random_bits(48)
    for name in ["LSB", "DCT", "DWT-SVD"]:
        wm = CLASSIC_METHODS[name]().embed(gray.copy(), bits)
        assert psnr(gray, wm) > 35


def test_dct_qim_jpeg_survives_recompression():
    rng = np.random.RandomState(1)
    x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    img = np.clip(128 + 60 * np.sin(8 * x) + rng.normal(0, 4, (256, 256)), 0, 255)
    bits = random_bits(100)
    m = DCTQIMJPEGWatermark(quality=50, repeat=4)
    wm = m.embed(img, bits)
    assert np.mean(m.extract(wm, 100) == bits) == 1.0
    # 同质量 JPEG 再压缩后仍应高准确率
    rec = jpeg_recompress(np.clip(np.round(wm), 0, 255).astype(np.uint8), 50)
    assert np.mean(m.extract(rec.astype(float), 100) == bits) >= 0.95


def test_dct_qim_jpeg_in_pipeline():
    rng = np.random.RandomState(2)
    img = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    bits = random_bits(64)
    m = DCTQIMJPEGWatermark(quality=50, repeat=4)
    wm_img, stream = m.embed_in_jpeg(img, bits)
    from swe.codec.color import rgb_to_ycbcr
    Y = rgb_to_ycbcr(wm_img)[..., 0]
    assert np.mean(m.extract(Y, 64) == bits) >= 0.98
    assert stream.bpp > 0


@pytest.mark.parametrize("H,W", [(256, 256), (250, 250), (248, 200)])
def test_dct_qim_jpeg_in_pipeline_arbitrary_size(H, W):
    """非 16 倍数尺寸也应可靠(embed_in_jpeg 会先裁到 16 的倍数,保证块网格一致)。"""
    from swe.codec.color import rgb_to_ycbcr
    rng = np.random.RandomState(H + W)
    img = rng.randint(0, 256, (H, W, 3), dtype=np.uint8)
    bits = random_bits(64)
    m = DCTQIMJPEGWatermark(quality=50, repeat=2)
    wm_img, _ = m.embed_in_jpeg(img, bits)
    Y = rgb_to_ycbcr(wm_img)[..., 0]
    assert np.mean(m.extract(Y, 64) == bits) >= 0.98
