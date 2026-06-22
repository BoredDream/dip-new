# -*- coding: utf-8 -*-
"""脚本 08:由实验 CSV 生成三张核心图表(模块五)。

读取 results/attack_suite.csv,生成:
  1. results/figures/robustness_table.png   鲁棒性总表(方法 × 攻击)
  2. results/figures/sweep_<attack>.png      各攻击的强度扫描曲线
  3. results/figures/tradeoff.png            攻防权衡曲线(危险区)
用法: python scripts/08_make_report_figures.py [csv路径]
"""
import csv
import os
import sys

import _bootstrap  # noqa: F401

from swe.eval.runner import Record, records_to_table
from swe.eval.plots import plot_robustness_table, plot_strength_sweep, plot_tradeoff
import config


def _load_records(path):
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def fv(k):
                v = row.get(k, "")
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return float("nan")
            s = row["strength"]
            try:
                s = float(s)
            except ValueError:
                pass
            records.append(Record(row["method"], row["attack"], s,
                                  fv("bit_acc_mean"), fv("bit_acc_std"),
                                  fv("psnr"), fv("ssim"), fv("lpips"),
                                  fv("att_ssim"), fv("att_psnr")))
    return records


def main(csv_path=None):
    csv_path = csv_path or os.path.join(config.RESULTS_DIR, "attack_suite.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"未找到 {csv_path},请先运行 scripts/07_run_attack_suite.py")
    records = _load_records(csv_path)
    fig_dir = config.FIGURES_DIR

    attacks = [a for a in dict.fromkeys(r.attack for r in records) if a != "none"]
    methods, atk, mat = records_to_table(records, attacks, strength_pick="mid")
    p1 = os.path.join(fig_dir, "robustness_table.png")
    plot_robustness_table(methods, atk, mat, p1)
    print(f"[OK] 鲁棒性总表: {p1}")

    for a in attacks:
        p = os.path.join(fig_dir, f"sweep_{a}.png")
        plot_strength_sweep(records, a, p)
        print(f"[OK] 强度扫描曲线 [{a}]: {p}")

    p3 = os.path.join(fig_dir, "tradeoff.png")
    plot_tradeoff(records, p3)
    print(f"[OK] 攻防权衡曲线: {p3}")
    print("\n三张核心图表已生成于 results/figures/。")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
