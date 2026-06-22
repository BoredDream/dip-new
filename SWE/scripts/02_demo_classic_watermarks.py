# -*- coding: utf-8 -*-
"""脚本 02:经典水印合集 demo(模块二基线)。

对样例图嵌入**自定义文本水印**,报告 PSNR / 干净比特准确率 / 文本是否完整还原,以及
JPEG50、高斯噪声、高斯模糊三种攻击后的比特准确率,并保存含水印图与"残差×20"可视化。

用法:
  python scripts/02_demo_classic_watermarks.py                       # 默认样例图与默认水印文本
  python scripts/02_demo_classic_watermarks.py --message "版权©张三"  # 自定义水印内容
  python scripts/02_demo_classic_watermarks.py path/to.png --message "ID:2026"
"""
import argparse
import os

import _bootstrap  # noqa: F401
import numpy as np

from swe.watermark.classic import CLASSIC_METHODS
from swe.watermark.utils import pack_message, unpack_message
from swe.data.datasets import load_gray, save_image, to_uint8
from swe.attacks.classic import jpeg_recompress, gaussian_noise, gaussian_blur
from swe.eval.metrics import psnr, bit_accuracy
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=config.DEFAULT_SAMPLE, help="输入图片路径")
    ap.add_argument("--message", default="DIP-SWE水印", help="要嵌入的水印文本(支持中文)")
    args = ap.parse_args()

    out_dir = os.path.join(config.RESULTS_DIR, "classic")
    os.makedirs(out_dir, exist_ok=True)
    g = load_gray(args.image, size=256, multiple_of=8).astype(np.float64)
    bits = pack_message(args.message)          # [32位长度头 | UTF-8 payload]
    n_bits = len(bits)
    save_image(os.path.join(out_dir, "00_original.png"), g)

    print(f"输入: {args.image}\n水印文本: {args.message!r}  ->  {n_bits} bits\n")
    print(f"{'method':<12}{'PSNR':>8}{'clean':>8}{'JPEG50':>8}{'noise10':>9}{'blur1.5':>9}   文本还原")
    print("-" * 70)
    for name, ctor in CLASSIC_METHODS.items():
        m = ctor()
        wm = m.embed(g.copy(), bits)
        wm_u8 = to_uint8(wm)
        save_image(os.path.join(out_dir, f"{name}_watermarked.png"), wm_u8)
        save_image(os.path.join(out_dir, f"{name}_residual_x20.png"),
                   np.clip(128 + (wm - g) * 20, 0, 255))

        ext = m.extract(wm, n_bits)
        clean = bit_accuracy(bits, ext)
        text = unpack_message(ext)
        jp = bit_accuracy(bits, m.extract(jpeg_recompress(wm_u8, 50).astype(np.float64), n_bits))
        ns = bit_accuracy(bits, m.extract(gaussian_noise(wm_u8, 10).astype(np.float64), n_bits))
        bl = bit_accuracy(bits, m.extract(gaussian_blur(wm_u8, 1.5).astype(np.float64), n_bits))
        ok = ("OK  " + text) if text == args.message else ("X  " + repr(text))
        print(f"{name:<12}{psnr(g, wm):>8.1f}{clean:>8.2f}{jp:>8.2f}{ns:>9.2f}{bl:>9.2f}   {ok}")

    print(f"\n图像已保存到: {out_dir}")
    print("结论:LSB/DCT/DFT 抗攻击弱(下限基线),SVD/DWT-SVD/扩频较鲁棒;"
          "扩频以低 PSNR 换强鲁棒。干净时各法均能完整还原文本。")


if __name__ == "__main__":
    main()
