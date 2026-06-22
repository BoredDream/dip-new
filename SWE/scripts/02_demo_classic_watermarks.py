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
    R = {}
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
        text_ok = (text == args.message)
        R[name] = dict(psnr=psnr(g, wm), clean=clean, jp=jp, ns=ns, bl=bl,
                       avg_atk=(jp + ns + bl) / 3.0, text_ok=text_ok)
        ok = ("OK  " + text) if text_ok else ("X  " + repr(text))
        print(f"{name:<12}{R[name]['psnr']:>8.1f}{clean:>8.2f}{jp:>8.2f}{ns:>9.2f}{bl:>9.2f}   {ok}")

    print(f"\n图像已保存到: {out_dir}")

    # —— 自动判定:下面每条结论都由上表实测值按明确判据算出,不写死预设结论 ——
    # 判据依据:攻击后比特准确率 0.5=随机猜=水印已死;取三攻击(JPEG50/噪声σ10/模糊1.5)
    # 的均值衡量综合鲁棒性。<0.65 视为"接近随机=弱基线";>=0.85 视为"较鲁棒"。
    WEAK_TH, ROBUST_TH = 0.65, 0.85
    text_recovered = [n for n, r in R.items() if r["text_ok"]]
    weak = sorted(n for n, r in R.items() if r["avg_atk"] < WEAK_TH)
    robust = sorted(n for n, r in R.items() if r["avg_atk"] >= ROBUST_TH)
    min_psnr_method = min(R, key=lambda n: R[n]["psnr"])
    ss = R.get("SpreadSpec")
    ss_tradeoff = ss is not None and min_psnr_method == "SpreadSpec" and ss["avg_atk"] >= ROBUST_TH

    print("\n== 自动判定(依据上表实测值,非预设结论)==")
    print(f"[1] 干净可逆: {len(text_recovered)}/{len(R)} 种方法完整还原水印文本"
          + ("(全部通过)" if len(text_recovered) == len(R)
             else "(未全过,成功的: " + ", ".join(text_recovered) + ")"))
    print(f"[2] 抗攻击弱(三攻击均值<{WEAK_TH},近随机=下限基线): {weak or '无'}")
    print(f"[3] 抗攻击强(三攻击均值>={ROBUST_TH}): {robust or '无'}")
    print(f"[4] 扩频'低PSNR换强鲁棒'判定: {'成立' if ss_tradeoff else '不成立'}"
          + (f" (SpreadSpec PSNR={ss['psnr']:.1f}dB 全场最低={min_psnr_method == 'SpreadSpec'},"
             f"抗攻击均值={ss['avg_atk']:.2f}>={ROBUST_TH}={ss['avg_atk'] >= ROBUST_TH})" if ss else ""))


if __name__ == "__main__":
    main()
