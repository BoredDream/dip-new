# -*- coding: utf-8 -*-
"""脚本 01:自实现 JPEG 编解码器自测,并与 PIL/libjpeg 校准(实施方案第 5 节第 1 步)。

对样例图在多个质量因子下:用本项目 JPEGCodec 编解码,报告 PSNR 与 bpp(压缩率),
并与 PIL 的 JPEG 对比 PSNR,验证"自实现 JPEG 与标准一致"。
用法: python scripts/01_test_jpeg_codec.py [图片路径]
"""

import io
import sys

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from swe.codec import JPEGCodec
from swe.data.datasets import load_gray, load_image
from swe.eval.metrics import psnr, ssim
import config


def main(path=config.DEFAULT_SAMPLE):
    print(f"输入: {path}\n")
    print("== 灰度,质量扫描:本项目 vs PIL ==")
    g = load_gray(path, size=256, multiple_of=8)
    print(f"{'Q':>4}{'ours PSNR':>11}{'ours bpp':>10}{'PIL PSNR':>10}{'ΔPSNR':>8}")
    gray_rows = []
    for q in (95, 90, 70, 50, 30, 10):
        rec, bpp = JPEGCodec(quality=q).compress_decompress(g)
        buf = io.BytesIO(); Image.fromarray(g, "L").save(buf, format="JPEG", quality=q)
        pil = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("L"))
        po, pp = psnr(g, rec), psnr(g, pil)
        delta = po - pp
        gray_rows.append((q, po, bpp, pp, delta))
        print(f"{q:>4}{po:>11.2f}{bpp:>10.3f}{pp:>10.2f}{delta:>8.2f}")

    print("\n== 彩色 (4:2:0),质量扫描 ==")
    img = load_image(path, size=256, multiple_of=16)
    print(f"{'Q':>4}{'ours PSNR':>11}{'ours SSIM':>11}{'ours bpp':>10}")
    color_rows = []
    for q in (90, 70, 50, 30):
        rec, bpp = JPEGCodec(quality=q, subsample=True).compress_decompress(img)
        p = psnr(img, rec)
        s = ssim(img, rec)
        color_rows.append((q, p, s, bpp))
        print(f"{q:>4}{p:>11.2f}{s:>11.3f}{bpp:>10.3f}")

    # 真实校准判断,而不是固定打印预设结论。
    delta_threshold = 0.25
    gray_delta_ok = all(abs(delta) <= delta_threshold for q, _, _, _, delta in gray_rows if q <= 90)
    gray_bpp_ok = all(gray_rows[i][2] >= gray_rows[i + 1][2] for i in range(len(gray_rows) - 1))
    color_bpp_ok = all(color_rows[i][3] >= color_rows[i + 1][3] for i in range(len(color_rows) - 1))
    color_quality_ok = all(color_rows[i][1] >= color_rows[i + 1][1] for i in range(len(color_rows) - 1))

    print("\n== 自动校准判断 ==")
    print(f"灰度 ΔPSNR 校准: {'PASS' if gray_delta_ok else 'FAIL'} "
          f"(要求 Q<=90 时 |ΔPSNR| <= {delta_threshold:.2f} dB)")
    print(f"灰度 bpp 单调性: {'PASS' if gray_bpp_ok else 'FAIL'} "
          "(要求 Q 降低时 bpp 不上升)")
    print(f"彩色 bpp 单调性: {'PASS' if color_bpp_ok else 'FAIL'} "
          "(要求 Q 降低时 bpp 不上升)")
    print(f"彩色 PSNR 单调性: {'PASS' if color_quality_ok else 'FAIL'} "
          "(要求 Q 降低时 PSNR 不上升)")

    ok = gray_delta_ok and gray_bpp_ok and color_bpp_ok and color_quality_ok
    if ok:
        print("\n结论:校准通过。本项目 JPEG 与 PIL/libjpeg 在灰度 PSNR 上基本一致,"
              "且压缩率/画质随 Q 的变化符合预期。")
    else:
        print("\n结论:校准未通过。请检查上方 FAIL 项对应的 JPEG 编解码或质量因子逻辑。")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main(sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_SAMPLE) else 1)
