# -*- coding: utf-8 -*-
"""评估指标与脆弱水印的单元测试。"""
import numpy as np

from swe.eval.metrics import psnr, ssim, bit_accuracy, ber, iou, f1_score
from swe.watermark.fragile import FragileWatermark, SemiFragileWatermark


def test_psnr_identical_inf():
    a = np.zeros((16, 16), dtype=np.uint8)
    assert psnr(a, a) == float("inf")


def test_ssim_identical_one():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 256, (32, 32), dtype=np.uint8)
    assert ssim(a, a) > 0.999


def test_bit_accuracy_and_ber():
    a = np.array([0, 1, 0, 1])
    b = np.array([0, 1, 1, 1])
    assert bit_accuracy(a, b) == 0.75
    assert abs(ber(a, b) - 0.25) < 1e-9


def test_iou_f1():
    gt = np.array([[1, 1], [0, 0]], dtype=bool)
    pred = np.array([[1, 0], [0, 0]], dtype=bool)
    assert iou(pred, gt) == 0.5
    assert abs(f1_score(pred, gt) - 2 / 3) < 1e-9


def test_fragile_no_false_positive():
    rng = np.random.RandomState(1)
    g = rng.randint(0, 256, (64, 64)).astype(np.float64)
    fw = FragileWatermark(block=8)
    wm = fw.embed(g)
    assert fw.verify(wm).sum() == 0


def test_fragile_localizes_tamper():
    rng = np.random.RandomState(2)
    g = rng.randint(0, 256, (64, 64)).astype(np.float64)
    fw = FragileWatermark(block=8)
    wm = fw.embed(g)
    att = wm.copy(); att[16:32, 16:48] = 100  # 篡改 2×4 块
    tmap = fw.verify(att)
    assert tmap.sum() >= 8  # 至少定位到这些块


def test_semifragile_tolerates_jpeg_detects_paint():
    import io
    from PIL import Image
    rng = np.random.RandomState(3)
    x, y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
    g = np.clip(128 + 60 * np.sin(6 * x) + rng.normal(0, 4, (128, 128)), 0, 255)
    sf = SemiFragileWatermark(block=8)
    wm = sf.embed(g)
    # benign JPEG80:误报应很少
    buf = io.BytesIO()
    Image.fromarray(np.clip(np.round(wm), 0, 255).astype(np.uint8), "L").save(buf, "JPEG", quality=80)
    jp = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("L"), dtype=np.float64)
    assert sf.verify(jp).mean() < 0.05
    # 篡改:应能定位
    att = wm.copy(); att[32:64, 48:80] = 120
    assert sf.verify(att).sum() >= 8
