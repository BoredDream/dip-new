# -*- coding: utf-8 -*-
"""脚本 09:频带存活诊断(P1)—— 决定改进版 DwtDct"往哪个频带嵌"。

对一批图测量 img2img(默认纯 CPU 代理 regeneration_surrogate;--use-img2img 换真扩散)
在不同强度下,对 DwtDct 嵌入域(亮度 Y 的 DWT-LL 再 8×8 DCT)各频带的破坏程度,
产出:
  * results/diagnose/freq_survival.png   频带×强度热图 + 稳定性剖面(报告核心图)
  * results/diagnose/freq_survival.csv   原始矩阵(band, strength, abs, rel)
并在控制台打印:DwtDct 默认频带(#18)在稳定性里的排名 + 推荐改嵌的频带。

用法:
  python scripts/09_diagnose_frequency_survival.py                      # CPU 代理,样例图
  python scripts/09_diagnose_frequency_survival.py --data-dir D:/coco --n-images 100
  python scripts/09_diagnose_frequency_survival.py --use-img2img        # 需 GPU + diffusers
"""
import argparse
import csv
import os

import _bootstrap  # noqa: F401

from swe.data.datasets import list_images, load_image
from swe.attacks.ai import regeneration_surrogate, diffusion_img2img
from swe.eval.diagnose import diagnose_frequency_survival
from swe.eval.plots import plot_frequency_survival
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="图像目录(默认用 data/samples)")
    ap.add_argument("--n-images", type=int, default=4, help="参与诊断的图数(真实结论建议 100)")
    ap.add_argument("--strengths", default="0.05,0.1,0.15,0.2,0.25,0.3",
                    help="强度网格,逗号分隔;低段要密")
    ap.add_argument("--use-img2img", action="store_true",
                    help="用真扩散 img2img 代替 CPU 代理(需 GPU + diffusers)")
    ap.add_argument("--prompt", default="a photo, high quality, detailed",
                    help="img2img 的中性 prompt(仅 --use-img2img 时生效)")
    ap.add_argument("--out-dir", default=os.path.join(config.RESULTS_DIR, "diagnose"))
    args = ap.parse_args()

    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    src = list_images(args.data_dir) if args.data_dir else list_images(config.SAMPLES_DIR)
    if not src:
        raise SystemExit("无测试图。请用 --data-dir 指定图像目录。")
    images = [load_image(src[i % len(src)], size=256, multiple_of=16)
              for i in range(args.n_images)]
    print(f"诊断图 {len(images)} 张(来源 {len(src)} 张),强度网格 {strengths}")

    if args.use_img2img:
        print("攻击:真扩散 img2img(需 GPU + diffusers)")
        attack = lambda img, strength: diffusion_img2img(img, strength=strength, prompt=args.prompt)
    else:
        print("攻击:regeneration_surrogate(纯 CPU 代理;有 GPU 可加 --use-img2img 换真扩散)")
        attack = regeneration_surrogate

    diag = diagnose_frequency_survival(images, attack, strengths, verbose=True)

    rec = diag.recommend(strength_idx=0, k=6)
    rank = diag.default_rank(strength_idx=0)
    n_band = diag.rel_disturb.shape[0]

    # 出图
    png = os.path.join(args.out_dir, "freq_survival.png")
    plot_frequency_survival(strengths, diag.rel_disturb, diag.default_band, png,
                            recommended=rec, strength_label=f"strength={strengths[0]:g}")

    # 存原始矩阵
    csv_path = os.path.join(args.out_dir, "freq_survival.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band", "strength", "abs_disturb", "rel_disturb"])
        for k in range(n_band):
            for si, s in enumerate(strengths):
                w.writerow([k, s, f"{diag.abs_disturb[k, si]:.6f}", f"{diag.rel_disturb[k, si]:.6f}"])

    # —— 诊断结论:下面每条都由实测的 rel_disturb 排名 / 推荐频带位置算出,不写死 ——
    print("\n===== 诊断结论(由实测推导,最温和强度 = 最具威胁的攻击)=====")
    default_off = rank > len(rec)            # 默认频带不在最稳的前 k 名 -> 选位偏差
    print(f"[1] DwtDct 默认嵌入频带 = #{diag.default_band}(中频),稳定性排名第 {rank}/{n_band}"
          f"(1=最稳)-> 判定: 默认位置{'选得偏(非最稳区,值得改嵌)' if default_off else '已接近最稳区'}。")
    print(f"[2] 推荐改嵌的稳定频带(相对扰动最小的前 {len(rec)} 个)= {rec}")
    # 推荐频带是否集中在低频区(zigzag 越小越低频)-> 决定会不会撞"可见性的墙"
    LOWFREQ_MAX = 12                         # zigzag<=12 视为低频区(8x8 共 64 个频带)
    rec_max = max(rec) if rec else -1
    lowfreq = rec_max <= LOWFREQ_MAX
    print(f"[3] 可见性判定: 推荐频带最大 zigzag={rec_max} {'<=' if lowfreq else '>'} {LOWFREQ_MAX} -> "
          + ("确属低频区,该频带幅值大、直接线性嵌入会偏可见 -> P2 必须用 JND/纹理自适应压可见性"
             if lowfreq else "未集中于低频,可见性压力较小"))
    print(f"\n[OK] 图: {png}")
    print(f"[OK] 数据: {csv_path}")
    print("下一步 P2:用上面推荐频带 + 纹理自适应强度 + JND 重写 DwtDct(ImprovedDwtDct)。")


if __name__ == "__main__":
    main()
