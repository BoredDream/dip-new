# -*- coding: utf-8 -*-
"""脚本 07:跑"方法 × 攻击 × 强度"全套实验(模块四 + 五),保存 CSV。

按实施方案 §3.2/§六 的对比阶梯组织(下界 < 改进前 < 改进版 < 上界)+ 改进版消融四档:
  * SpreadSpec        —— FF 式整图 DCT 加性扩频(**下界基线**);
  * DwtDct-default    —— 经典 DwtDct 默认中频 #18(**改进前基线** = 消融档0);
  * Imp+multiband     —— 改进版:仅开多低频带(消融档1);
  * Imp+texture       —— 改进版:多频带 + 纹理自适应(消融档2);
  * Imp+JND           —— 改进版:多频带 + 纹理 + Watson JND(**主角** = 消融档3);
  * DCT-QIM-JPEG      —— 我们在 JPEG 流水线内嵌的核心基线;
  * Deep-Latent       —— 深度潜空间水印(**上界基线**,需 --include-deep + checkpoint)。
攻击含经典失真 + 再生成代理。结果(均值±标准差)写入 results/attack_suite.csv,供脚本 08 出图。
(LSB/DCT/DWT/SVD/DWT-SVD/DFT 八种经典方法的广度对照见脚本 02。)
用法:
  python scripts/07_run_attack_suite.py [--data-dir imgs] [--include-deep] [--quick] [--n-images 10]
"""
import argparse
import os

import _bootstrap  # noqa: F401

from swe.watermark.classic import DWTDCTWatermark, SpreadSpectrumWatermark
from swe.watermark.improved_dwtdct import ImprovedDwtDct
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
    ap.add_argument("--ai-img2img", action="store_true",
                    help="加入真扩散 img2img 攻击(需 torch+diffusers+GPU,首次会下载 SD 权重)")
    ap.add_argument("--ai-grid", default="0.0,0.1,0.2,0.3",
                    help="diffusion_img2img 的 strength 网格(逗号分隔,小规模建议低段)")
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

    # 方法适配器:对比阶梯 + 改进版消融四档(档名见 swe.eval.plots.ABLATION_TIERS)
    nb = args.n_bits
    adapters = {
        "SpreadSpec":     ChannelAdapter("SpreadSpec", SpreadSpectrumWatermark(), nb),            # 下界
        "DwtDct-default": ChannelAdapter("DwtDct-default", DWTDCTWatermark(), nb),                # 改进前 = 消融档0
        "Imp+multiband":  ChannelAdapter("Imp+multiband",
                                         ImprovedDwtDct(multiband=True, texture=False, jnd=False), nb),
        "Imp+texture":    ChannelAdapter("Imp+texture",
                                         ImprovedDwtDct(multiband=True, texture=True, jnd=False), nb),
        "Imp+JND":        ChannelAdapter("Imp+JND",
                                         ImprovedDwtDct(multiband=True, texture=True, jnd=True), nb),  # 主角
        "DCT-QIM-JPEG":   ChannelAdapter("DCT-QIM-JPEG", DCTQIMJPEGWatermark(quality=50, repeat=4), nb),
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
    # 真扩散 img2img(GPU):用单独的低段强度网格,与 regen_surrogate 同图对照(代理 vs 真实)
    if args.ai_img2img:
        di = AI_ATTACKS["diffusion_img2img"]
        grid = [float(s) for s in args.ai_grid.split(",") if s.strip() != ""]
        attacks["diffusion_img2img"] = (di[0], di[1], grid)
        print(f"已加入真扩散 img2img 攻击(strength={grid})。首次运行将下载 SD 权重。\n")

    records = run_experiments(adapters, images, attacks, with_lpips=False, verbose=True)
    save_records_csv(records, args.out)
    print(f"\n[OK] 结果已写入: {args.out}  (共 {len(records)} 条记录)")
    print("下一步:python scripts/08_make_report_figures.py 出三张核心图。")


if __name__ == "__main__":
    main()
