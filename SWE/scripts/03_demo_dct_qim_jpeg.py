# -*- coding: utf-8 -*-
"""脚本 03:DCT-QIM 水印嵌入 JPEG 流水线(核心基线)+ 纠错码(ECC)。

演示两件事:
  A. "压缩同时嵌入":用 JPEGCodec 的 coeff_hook 在量化前把水印写进 Y 中频系数,
     输出真实 JPEG 解码图与码流 bpp,并验证经同质量 JPEG 再压缩后比特准确率。
  B. ECC 信息恢复:把用户 ID 先 RS 编码再嵌入,经攻击后比对"有/无 ECC"的完整恢复率。
用法: python scripts/03_demo_dct_qim_jpeg.py [图片路径]
"""
import sys

import _bootstrap  # noqa: F401
import numpy as np

from swe.watermark.dct_qim_jpeg import DCTQIMJPEGWatermark
from swe.watermark.utils import random_bits
from swe.ecc.codec import ReedSolomonECC
from swe.data.datasets import load_image, load_gray, to_uint8
from swe.codec.color import rgb_to_ycbcr
from swe.attacks.classic import jpeg_recompress, gaussian_noise
from swe.eval.metrics import psnr, bit_accuracy
import config


def main(path=config.DEFAULT_SAMPLE):
    print("== A. DCT-QIM 嵌入 JPEG 压缩流水线 ==")
    img = load_image(path, size=256, multiple_of=16)
    bits = random_bits(100)
    for q in (50, 30):
        wm = DCTQIMJPEGWatermark(quality=q, repeat=4)
        wm_img, stream = wm.embed_in_jpeg(img, bits)
        Y = rgb_to_ycbcr(wm_img)[..., 0]
        clean = bit_accuracy(bits, wm.extract(Y, 100))
        # 再过一次同质量 JPEG(模块一)
        Yj = rgb_to_ycbcr(jpeg_recompress(wm_img, q))[..., 0]
        rej = bit_accuracy(bits, wm.extract(Yj, 100))
        print(f"  Q={q}: bpp={stream.bpp:.3f} PSNR={psnr(img, wm_img):.1f}dB "
              f"Δ={wm.delta:.0f} | clean={clean:.2f} JPEG{q}再压={rej:.2f}")

    print("\n== B. ECC 信息恢复率(96-bit 用户 ID,攻击=JPEG35+噪声σ8,两路同 repeat=2) ==")
    g = load_gray(path, size=256, multiple_of=8).astype(np.float64)
    user_id = random_bits(96, seed=123)
    rs = ReedSolomonECC(nsym=24)            # 每块纠 12 字节错
    coded = rs.encode_bits(user_id)
    print(f"  ID={len(user_id)}bit -> RS 编码 {len(coded)}bit(冗余 {len(coded)/len(user_id):.1f}×)")

    def pipeline(payload, repeat=2):
        m = DCTQIMJPEGWatermark(quality=50, repeat=repeat)
        wm = to_uint8(m.embed(g.copy(), payload))
        att = gaussian_noise(jpeg_recompress(wm, 35), 8)   # 中等强度攻击
        return m.extract(att.astype(np.float64), len(payload))

    # 无 ECC:直接嵌 96-bit ID(repeat=2)
    raw = pipeline(user_id)
    raw_ok = np.array_equal(raw, user_id)
    raw_acc = bit_accuracy(user_id, raw)
    # 有 ECC:嵌 RS 编码后比特(repeat=2),提取后纠错
    ext_coded = pipeline(coded)
    rec = rs.decode_bits(ext_coded, len(user_id))
    ecc_ok = np.array_equal(rec, user_id)
    print(f"  无 ECC: 提取比特准确率={raw_acc:.3f}  ->  ID 完整恢复={raw_ok}")
    print(f"  有 ECC: 提取比特准确率={bit_accuracy(coded, ext_coded):.3f}  "
          f"-> RS 纠错后 ID 完整恢复={ecc_ok}")
    print("\n结论:DCT-QIM 嵌在 JPEG 流水线内可扛同质量再压缩;ECC 把'差几位'纠回,"
          "把比特准确率转化为'信息完整恢复'。")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_SAMPLE)
