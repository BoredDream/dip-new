# -*- coding: utf-8 -*-
"""可视化 —— 模块五之三:三张核心图表 + 篡改定位热图。

实施方案 4.5 报告必放:
  1. 鲁棒性总表(行=方法,列=攻击,格=比特准确率)         plot_robustness_table
  2. 强度扫描曲线(比特准确率随攻击强度衰减,多方法同图)   plot_strength_sweep
  3. 攻防权衡曲线(横轴=画面改变量,纵轴=BER,标危险区)     plot_tradeoff
篡改定位:plot_tamper_panel(原图 / 篡改图 / 预测热图 / 真值)

注:图中文字用英文,避免 matplotlib 缺中文字体时出现"豆腐块"。
"""
from __future__ import annotations

import csv
from typing import List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

__all__ = [
    "save_records_csv", "plot_robustness_table", "plot_strength_sweep",
    "plot_tradeoff", "plot_tamper_panel", "plot_frequency_survival",
    "plot_ablation", "ABLATION_TIERS",
]

#: 改进版 DwtDct 的消融四档,按 默认 → +多频带 → +纹理 → +JND 递进(方案 §4.8 图4)。
#: 档名须与 scripts/07 的 adapters 键一致。
ABLATION_TIERS = ["DwtDct-default", "Imp+multiband", "Imp+texture", "Imp+JND"]


def save_records_csv(records, path: str) -> None:
    fields = ["method", "attack", "strength", "bit_acc_mean", "bit_acc_std",
              "psnr", "ssim", "lpips", "att_ssim", "att_psnr"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in records:
            w.writerow([getattr(r, k) for k in fields])


def plot_robustness_table(methods: List[str], attacks: List[str], matrix: np.ndarray,
                          path: str, title: str = "Bit accuracy: method x attack") -> None:
    """比特准确率热力表。颜色:0.5(随机=死)红 -> 1.0(完好)绿。"""
    fig, ax = plt.subplots(figsize=(1.2 * len(attacks) + 3, 0.6 * len(methods) + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(attacks))); ax.set_xticklabels(attacks, rotation=40, ha="right")
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(attacks)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="bit accuracy (0.5 = random)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_strength_sweep(records, attack: str, path: str,
                        methods: Optional[List[str]] = None) -> None:
    """某攻击下:比特准确率 vs 强度,多方法同图,标 0.5 随机基线。"""
    methods = methods or sorted({r.method for r in records})
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        rs = sorted([r for r in records if r.method == m and r.attack == attack],
                    key=lambda r: r.strength)
        if not rs:
            continue
        xs = [r.strength for r in rs]
        ys = [r.bit_acc_mean for r in rs]
        es = [r.bit_acc_std for r in rs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=m)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="random (0.5)")
    ax.set_xlabel(f"{attack} strength"); ax.set_ylabel("bit accuracy")
    ax.set_ylim(0.4, 1.02); ax.set_title(f"Robustness sweep: {attack}")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_tradeoff(records, path: str, methods: Optional[List[str]] = None) -> None:
    """攻防权衡:x = 画面改变量(1 - SSIM(原图, 攻击图)),y = BER。

    左上角(画面几乎没变、但 BER 高)= 危险区:水印死了图却还像原图。
    """
    methods = methods or sorted({r.method for r in records})
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        pts = [(1 - r.att_ssim, 1 - r.bit_acc_mean) for r in records
               if r.method == m and r.attack != "none" and not np.isnan(r.att_ssim)]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=18, alpha=0.7, label=m)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.axhspan(0.35, 0.55, xmin=0, xmax=0.3, color="red", alpha=0.08)
    ax.text(0.01, 0.45, "danger zone\n(watermark dead, image intact)",
            fontsize=8, color="darkred")
    ax.set_xlabel("image change  (1 - SSIM vs original)")
    ax.set_ylabel("watermark damage  (BER)")
    ax.set_title("Attack–quality trade-off")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_frequency_survival(strengths: Sequence[float], rel_disturb: np.ndarray,
                            default_band: int, path: str,
                            recommended: Optional[Sequence[int]] = None,
                            strength_label: Optional[str] = None) -> None:
    """P1 诊断图(改进 DwtDct 的"选频段"依据)。

    左:DCT 频带(zigzag, 0=DC 在下 -> 63=高频在上) × img2img 强度 的相对扰动热图
        (颜色越亮=越被破坏);叠一条虚线标 DwtDct 默认嵌入频带。
    右:最温和(=最具威胁)强度下,相对扰动 vs 频带的剖面;标默认位(红)与推荐位(绿)。
    读图结论:亮带=水印一嵌就死的频段;暗带=幸存频段=应改嵌的位置。
    """
    strengths = np.asarray(strengths, dtype=float)
    n_band = rel_disturb.shape[0]
    disp = np.clip(rel_disturb, 1e-3, None)        # LogNorm 需正值
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- 左:热图 ----
    im = axL.imshow(disp, origin="lower", aspect="auto", cmap="magma",
                    norm=LogNorm(vmin=disp.min(), vmax=disp.max()),
                    extent=[strengths.min(), strengths.max(), 0, n_band - 1])
    axL.axhline(default_band, color="cyan", ls="--", lw=1.5,
                label=f"default DwtDct band #{default_band}")
    if recommended:
        for b in recommended:
            axL.axhline(b, color="lime", ls=":", lw=1.0)
    axL.set_xlabel("img2img strength")
    axL.set_ylabel("DCT band (zigzag): 0 = DC  ->  high freq")
    axL.set_title("Per-band disturbance under regeneration")
    axL.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=axL, label="relative disturbance |dDCT| / |DCT_clean|")

    # ---- 右:最低强度下的稳定性剖面 ----
    bands = np.arange(n_band)
    prof = rel_disturb[:, 0]
    lab = strength_label or f"strength={strengths[0]:g}"
    axR.semilogy(bands, np.clip(prof, 1e-3, None), color="0.4", lw=1.2)
    axR.axvline(default_band, color="red", ls="--", lw=1.3,
                label=f"default #{default_band} (rel={prof[default_band]:.3f})")
    if recommended:
        axR.scatter(list(recommended), np.clip(prof[list(recommended)], 1e-3, None),
                    color="green", s=40, zorder=5, label="recommended (most stable)")
    axR.set_xlabel("DCT band (zigzag)")
    axR.set_ylabel("relative disturbance (log)")
    axR.set_title(f"Stability profile @ {lab} (mildest = most realistic attack)")
    axR.legend(fontsize=8); axR.grid(alpha=0.3, which="both")

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_ablation(records, path: str, tiers: Optional[Sequence[str]] = None,
                  focus_attack: str = "regen_surrogate") -> None:
    """消融曲线(方案 §4.8 图4):默认 DwtDct → +多频带 → +纹理 → +JND 的逐档增益。

    x = 四档(按 ABLATION_TIERS 顺序);y = 平均比特准确率。两条线:
      * "all attacks"   —— 所有攻击×强度的均值(综合鲁棒性);
      * focus_attack    —— 主战场(默认再生成代理 regen_surrogate)的均值。
    只画 records 中真实存在的档,缺档跳过(不报错)。
    """
    tiers = list(tiers or ABLATION_TIERS)
    present = [t for t in tiers if any(r.method == t for r in records)]
    if len(present) < 2:
        return  # 档数不足,无意义

    def _mean(tier, attack=None):
        vals = [r.bit_acc_mean for r in records if r.method == tier
                and r.attack != "none" and (attack is None or r.attack == attack)]
        return float(np.mean(vals)) if vals else np.nan

    x = np.arange(len(present))
    y_all = [_mean(t) for t in present]
    y_focus = [_mean(t, focus_attack) for t in present]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y_all, marker="o", lw=2, label="all attacks (mean)")
    if not all(np.isnan(y_focus)):
        ax.plot(x, y_focus, marker="s", lw=2, ls="--",
                label=f"{focus_attack} (mean)")
    for xi, yv in zip(x, y_all):
        if not np.isnan(yv):
            ax.annotate(f"{yv:.3f}", (xi, yv), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="random (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=15, ha="right")
    ax.set_ylabel("bit accuracy (mean)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Ablation: default DwtDct -> +multiband -> +texture -> +JND")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_tamper_panel(original: np.ndarray, tampered: np.ndarray, heatmap: np.ndarray,
                      path: str, gt_mask: Optional[np.ndarray] = None,
                      iou: Optional[float] = None, f1: Optional[float] = None) -> None:
    """篡改定位四联图:原图 / 篡改图 / 预测热图(红=篡改) / 真值。"""
    n = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    cmap = "gray"
    axes[0].imshow(original, cmap=cmap); axes[0].set_title("original (watermarked)")
    axes[1].imshow(tampered, cmap=cmap); axes[1].set_title("tampered")
    axes[2].imshow(tampered, cmap=cmap)
    axes[2].imshow(heatmap, cmap="Reds", alpha=0.5)
    t = "predicted tamper map"
    if iou is not None:
        t += f"\nIoU={iou:.2f}" + (f" F1={f1:.2f}" if f1 is not None else "")
    axes[2].set_title(t)
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap="Reds"); axes[3].set_title("ground truth")
    for a in axes:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
