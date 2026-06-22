# -*- coding: utf-8 -*-
"""脚本 06:深度潜空间水印 embed/extract demo + 鲁棒性快测。

加载 checkpoint(无则现场快速训练一个 tiny 模型),对样例图嵌入随机消息,
报告 PSNR 与干净/JPEG/模糊/再生成代理攻击后的比特准确率,保存含水印图与残差图。
用法: python scripts/06_demo_deep_watermark.py [--ckpt path] [图片]
"""
import argparse
import os

import _bootstrap  # noqa: F401
import numpy as np

from swe.watermark.deep.model import build_latent_watermark, load_latent_watermark
from swe.watermark.deep.train import train_deep_watermark
from swe.watermark.utils import random_bits, text_to_bits, bits_to_text
from swe.data.datasets import load_image, save_image, list_images, to_uint8
from swe.watermark.classic import DWTSVDWatermark
from swe.attacks.classic import jpeg_recompress, gaussian_blur, gaussian_noise
from swe.attacks.ai import regeneration_surrogate
from swe.eval.metrics import psnr, bit_accuracy
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(config.CHECKPOINTS_DIR, "deep_wm.pth"))
    ap.add_argument("--image", default=config.DEFAULT_SAMPLE)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--secret-bits", type=int, default=32)
    ap.add_argument("--message", default="", help="水印文本(留空=随机比特;超出容量会截断)")
    args = ap.parse_args()

    if os.path.exists(args.ckpt):
        model = load_latent_watermark(args.ckpt, device="cpu")   # 按 checkpoint 内的架构自动重建
        args.secret_bits, args.image_size = model.secret_bits, model.image_size
        print(f"已加载 checkpoint: {args.ckpt}  "
              f"(vae={model.config['vae']} decoder={model.config['decoder']} "
              f"size={model.image_size} bits={model.secret_bits})")
    else:
        cfg = dict(vae="tiny", decoder="cnn", image_size=args.image_size, secret_bits=args.secret_bits)
        model = build_latent_watermark(cfg, device="cpu")
        print("未找到 checkpoint,现场快速训练 tiny 模型(约 1-2 分钟)...")
        train_deep_watermark(model, list_images(config.SAMPLES_DIR), steps=400,
                             batch_size=4, lr=3e-4, warmup=60, curriculum_steps=160,
                             lambda_lpips=0.0, device="cpu", log_every=100)

    out_dir = os.path.join(config.RESULTS_DIR, "deep")
    os.makedirs(out_dir, exist_ok=True)
    img = load_image(args.image, size=192)              # 任意分辨率(残差缩放技巧)
    if args.message:
        mb = text_to_bits(args.message)
        if len(mb) > args.secret_bits:
            print(f"[警告] 文本 {len(mb)} bits 超过容量 {args.secret_bits},已截断。")
        bits = np.zeros(args.secret_bits, dtype=np.int64)
        bits[:min(len(mb), args.secret_bits)] = mb[:args.secret_bits]
    else:
        bits = random_bits(args.secret_bits, seed=7)
    wm = model.embed(img, bits)
    if args.message:
        nbytes = (min(len(text_to_bits(args.message)), args.secret_bits) // 8) * 8
        rec = bits_to_text(model.extract(wm)[:nbytes])
        note = "" if rec == args.message else "  (注:bit_acc<1.0 时文本会乱码属正常,需更充分训练/SD-VAE/或加 ECC)"
        print(f"水印文本(干净提取): {rec!r}{note}")
    save_image(os.path.join(out_dir, "deep_watermarked.png"), wm)
    save_image(os.path.join(out_dir, "deep_residual_x10.png"),
               np.clip(128 + (wm.astype(float) - img) * 10, 0, 255))

    # 经典基线:DWT-SVD 把同一组 bits 嵌到绿色通道,同图 / 同 payload / 同攻击 → 公平对比
    classic = DWTSVDWatermark()
    cimg = img.astype(np.float64).copy()
    cimg[..., 1] = classic.embed(cimg[..., 1], bits)
    wm_c = to_uint8(cimg)

    print(f"\nPSNR(含水印 vs 原图): 深度={psnr(img, wm):.1f}dB  DWT-SVD基线={psnr(img, wm_c):.1f}dB")
    print(f"{'attack':<22}{'deep':>8}{'DWT-SVD':>9}")
    print("-" * 39)
    tests = [("none", lambda x: x),
             ("JPEG q50", lambda x: jpeg_recompress(x, 50)),
             ("JPEG q30", lambda x: jpeg_recompress(x, 30)),
             ("gaussian_blur 1.5", lambda x: gaussian_blur(x, 1.5)),
             ("gaussian_noise 10", lambda x: gaussian_noise(x, 10)),
             ("regen_surrogate 0.3", lambda x: regeneration_surrogate(x, 0.3)),
             ("regen_surrogate 0.6", lambda x: regeneration_surrogate(x, 0.6))]
    rows = {}
    for name, fn in tests:
        a_d = bit_accuracy(bits, model.extract(fn(wm)))
        att_c = fn(wm_c)
        a_c = bit_accuracy(bits, classic.extract(att_c[..., 1].astype(np.float64), len(bits)))
        rows[name] = (a_d, a_c)
        print(f"{name:<22}{a_d:>8.3f}{a_c:>9.3f}")
    print(f"\n图像已保存到: {out_dir}")

    # —— 自动判定:对比结论由上表"深度 vs 经典 同攻击"实测算出,不写死预设结论 ——
    RANDOM, MARGIN = 0.5, 0.10
    regen_keys = [k for k in rows if k.startswith("regen")]
    deep_regen = float(np.mean([rows[k][0] for k in regen_keys]))
    cls_regen = float(np.mean([rows[k][1] for k in regen_keys]))
    deep_survives = deep_regen > RANDOM + MARGIN
    deep_beats = (deep_regen - cls_regen) >= MARGIN
    print("\n== 自动判定(依据上表 深度 vs DWT-SVD 同攻击实测,非预设结论)==")
    print(f"[1] 深度水印抗再生成代理: {'成立' if deep_survives else '不成立'} "
          f"(regen 留存均值={deep_regen:.2f}[需>{RANDOM + MARGIN:.2f}=随机线+裕度])")
    print(f"[2] 深度 > 经典基线(DWT-SVD): {'成立' if deep_beats else '不明显'} "
          f"(深度regen={deep_regen:.2f} − 经典regen={cls_regen:.2f} = {deep_regen - cls_regen:+.2f}[需>={MARGIN}])")
    if not deep_beats:
        print("    注:tiny VAE 仅冒烟训练,差距可能不显著;论文级对比需 --vae sd 充分训练(见 scripts 05/07)。")


if __name__ == "__main__":
    main()
