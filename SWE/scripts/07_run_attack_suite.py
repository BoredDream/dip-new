# -*- coding: utf-8 -*-
"""脚本 07:跑"方法 × 攻击 × 强度"全套实验(模块四 + 五),保存 CSV。

默认对比 LSB / DCT-QIM-JPEG / DWT-SVD / 扩频 四种经典方法;若存在深度水印 checkpoint
(--include-deep),一并纳入对比。攻击含经典失真 + 再生成代理。结果(均值±标准差)
写入 results/attack_suite.csv,供脚本 08 出图。
用法:
  python scripts/07_run_attack_suite.py [--data-dir imgs] [--include-deep] [--quick]
"""
import argparse
import os

import _bootstrap  # noqa: F401

from swe.watermark.classic import LSBWatermark, DWTSVDWatermark, SpreadSpectrumWatermark
from swe.watermark.dct_qim_jpeg import DCTQIMJPEGWatermark
from swe.data.datasets import list_images, load_image
from swe.attacks.classic import CLASSIC_ATTACKS
from swe.attacks.ai import AI_ATTACKS
from swe.eval.runner import ChannelAdapter, ImageAdapter, run_experiments
from swe.eval.plots import save_records_csv
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--n-bits", type=int, default=64)
    ap.add_argument("--include-deep", action="store_true")
    ap.add_argument("--deep-ckpt", default=os.path.join(config.CHECKPOINTS_DIR, "deep_wm.pth"))
    ap.add_argument("--quick", action="store_true", help="更小的强度网格,跑得更快")
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "attack_suite.csv"))
    args = ap.parse_args()

    # 测试图
    src = list_images(args.data_dir) if args.data_dir else list_images(config.SAMPLES_DIR)
    if not src:
        raise SystemExit("无测试图。")
    images = [load_image(src[i % len(src)], size=256, multiple_of=16)
              for i in range(args.n_images)]
    print(f"测试图 {len(images)} 张(来源 {len(src)} 张)\n")

    # 方法适配器
    nb = args.n_bits
    adapters = {
        "LSB": ChannelAdapter("LSB", LSBWatermark(), nb),
        "DCT-QIM-JPEG": ChannelAdapter("DCT-QIM-JPEG", DCTQIMJPEGWatermark(quality=50, repeat=4), nb),
        "DWT-SVD": ChannelAdapter("DWT-SVD", DWTSVDWatermark(), nb),
        "SpreadSpec": ChannelAdapter("SpreadSpec", SpreadSpectrumWatermark(), nb),
    }
    if args.include_deep and os.path.exists(args.deep_ckpt):
        from swe.watermark.deep.model import load_latent_watermark
        model = load_latent_watermark(args.deep_ckpt, device="cpu")  # 按 checkpoint 架构自动重建
        adapters["Deep-Latent"] = ImageAdapter("Deep-Latent", model)
        print(f"已纳入深度水印对比(vae={model.config['vae']} decoder={model.config['decoder']})。\n")

    # 攻击集合(经典 + 再生成代理)
    attacks = {k: CLASSIC_ATTACKS[k] for k in ["gaussian_noise", "gaussian_blur", "jpeg", "rescale"]}
    rs = AI_ATTACKS["regen_surrogate"]
    attacks["regen_surrogate"] = (rs[0], rs[1], rs[2])
    if args.quick:
        attacks = {k: (v[0], v[1], v[2][::2]) for k, v in attacks.items()}

    records = run_experiments(adapters, images, attacks, with_lpips=False, verbose=True)
    save_records_csv(records, args.out)
    print(f"\n[OK] 结果已写入: {args.out}  (共 {len(records)} 条记录)")
    print("下一步:python scripts/08_make_report_figures.py 出三张核心图。")


if __name__ == "__main__":
    main()
