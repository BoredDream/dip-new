# -*- coding: utf-8 -*-
"""脚本 05:训练深度潜空间水印(模块三)。

  python scripts/05_train_deep_watermark.py --smoke                 # CPU 冒烟(tiny VAE,几分钟)
  python scripts/05_train_deep_watermark.py --data-dir path/to/imgs \
        --vae sd --decoder convnext --steps 20000 --image-size 256  # 正式训练(需 GPU + diffusers)

冻结 SD-VAE(--vae sd)是 RoSteALS 正路;自带 tiny VAE(--vae tiny)免下载、CPU 可跑,
此时 VAE 与 E_s/D_s 联合训练(适合验证流程,非论文级画质)。
"""
import argparse
import os
import sys

import _bootstrap  # noqa: F401

from swe.watermark.deep.model import build_latent_watermark
from swe.watermark.deep.train import train_deep_watermark
from swe.data.datasets import list_images
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="训练图目录;缺省用样例图")
    ap.add_argument("--vae", default="tiny", choices=["tiny", "sd"])
    ap.add_argument("--decoder", default="cnn", choices=["cnn", "convnext"])
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--secret-bits", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(config.CHECKPOINTS_DIR, "deep_wm.pth"))
    ap.add_argument("--smoke", action="store_true", help="快速冒烟:小尺寸、少步数")
    args = ap.parse_args()

    if args.smoke:
        args.vae, args.decoder = "tiny", "cnn"
        args.image_size, args.secret_bits = 128, 32
        args.steps, args.batch_size = 400, 4

    if args.data_dir:
        paths = list_images(args.data_dir)
        if not paths:
            sys.exit(f"目录无图片: {args.data_dir}")
    else:
        paths = list_images(config.SAMPLES_DIR)
        print(f"[警告] 未给 --data-dir,用 {len(paths)} 张样例图(仅供冒烟,会过拟合)。")

    cfg = dict(vae=args.vae, decoder=args.decoder, image_size=args.image_size,
               secret_bits=args.secret_bits)
    model = build_latent_watermark(cfg, device=args.device)
    n = sum(p.numel() for p in model.trainable_parameters())
    print(f"模型: vae={args.vae} decoder={args.decoder} size={args.image_size} "
          f"bits={args.secret_bits} 可训练参数={n/1e6:.2f}M\n")

    warmup = max(40, args.steps // 8)
    curriculum = max(80, args.steps // 3)
    train_deep_watermark(model, paths, steps=args.steps, batch_size=args.batch_size,
                         lr=args.lr, warmup=warmup, curriculum_steps=curriculum,
                         lambda_lpips=0.0 if args.vae == "tiny" else 1.0,
                         device=args.device)
    model.save(args.out)
    print(f"\n[OK] 已保存 checkpoint: {args.out}")


if __name__ == "__main__":
    main()
