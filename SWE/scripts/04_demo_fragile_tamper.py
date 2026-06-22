# -*- coding: utf-8 -*-
"""脚本 04:脆弱 / 半脆弱水印篡改定位(创新点 3)。

嵌入脆弱水印 -> 在一块矩形区域伪造篡改(模拟 AI 重绘)-> 验证并定位 -> 计算 IoU/F1,
保存四联图(原图/篡改图/预测热图/真值)到 results/tamper/。
同时演示半脆弱水印对 benign JPEG 的容忍度(不误报)与对篡改的敏感性。
用法: python scripts/04_demo_fragile_tamper.py [图片路径]
"""
import os
import sys

import _bootstrap  # noqa: F401
import numpy as np

from swe.watermark.fragile import FragileWatermark, SemiFragileWatermark, tamper_heatmap
from swe.data.datasets import load_gray, to_uint8
from swe.attacks.classic import jpeg_recompress
from swe.eval.metrics import iou, f1_score
from swe.eval.plots import plot_tamper_panel
import config


def _make_tamper(img, box=(64, 128, 96, 176)):
    """在 box=(r0,r1,c0,c1) 内伪造篡改(局部压暗+偏移),返回 (篡改图, 块级真值掩码)。"""
    r0, r1, c0, c1 = box
    att = img.copy()
    att[r0:r1, c0:c1] = np.clip(img[r0:r1, c0:c1] * 0.4 + 90, 0, 255)
    return att, (r0, r1, c0, c1)


def main(path=config.DEFAULT_SAMPLE):
    out_dir = os.path.join(config.RESULTS_DIR, "tamper")
    os.makedirs(out_dir, exist_ok=True)
    g = load_gray(path, size=256, multiple_of=8).astype(np.float64)
    bs = 8

    M = {}
    for cls, label in [(FragileWatermark, "fragile"), (SemiFragileWatermark, "semifragile")]:
        wm_obj = cls(block=bs)
        wm = to_uint8(wm_obj.embed(g))
        att, box = _make_tamper(wm)
        # 半脆弱:篡改后再过一次 benign JPEG(更真实的 AIGC + 再压缩)
        att_in = jpeg_recompress(att, 80) if label == "semifragile" else att

        tmap = wm_obj.verify(att_in.astype(np.float64))
        heat = tamper_heatmap(tmap, g.shape, bs)

        # 块级真值掩码
        gt = np.zeros_like(tmap, dtype=bool)
        r0, r1, c0, c1 = box
        gt[r0 // bs:r1 // bs, c0 // bs:c1 // bs] = True
        i, f = iou(tmap, gt), f1_score(tmap, gt)

        clean_fp = int(wm_obj.verify(wm.astype(np.float64)).sum())
        rec = dict(clean_fp=clean_fp, iou=i, f1=f, benign_fp_rate=None)
        print(f"[{label}] clean误报={clean_fp}块  篡改后 IoU={i:.2f} F1={f:.2f}")
        if label == "semifragile":
            jp = wm_obj.verify(jpeg_recompress(wm, 80).astype(np.float64))
            rec["benign_fp_rate"] = float(jp.mean())
            print(f"           benign JPEG80 误报率={100 * rec['benign_fp_rate']:.1f}%")
        M[label] = rec

        gt_heat = tamper_heatmap(gt, g.shape, bs)
        plot_tamper_panel(wm, att_in, heat, os.path.join(out_dir, f"{label}_panel.png"),
                          gt_mask=gt_heat, iou=i, f1=f)

    print(f"\n四联图已保存到: {out_dir}")

    # —— 自动判定:下面结论均由上方实测的 误报数/IoU/F1 按明确判据算出,不写死 ——
    LOC_TH = 0.30          # 块级篡改定位 F1 达标线(含 JPEG 扰动,>0.3 视为有效定位)
    FP_TH = 0.05           # benign JPEG 误报率上限(<=5% 视为"容忍良性压缩、不误报")
    fr, sf = M["fragile"], M["semifragile"]
    fragile_ok = (fr["clean_fp"] == 0 and fr["f1"] >= LOC_TH)
    sf_tolerant = (sf["benign_fp_rate"] is not None and sf["benign_fp_rate"] <= FP_TH)
    sf_locates = (sf["f1"] >= LOC_TH)
    print("\n== 自动判定(依据上方实测值,非预设结论)==")
    print(f"[1] 脆弱水印精确定位: {'成立' if fragile_ok else '不成立'} "
          f"(clean误报={fr['clean_fp']}块[需=0],篡改F1={fr['f1']:.2f}[需>={LOC_TH}])")
    print(f"[2] 半脆弱容忍benign JPEG: {'成立' if sf_tolerant else '不成立'} "
          f"(JPEG80误报率={100 * (sf['benign_fp_rate'] or 0):.1f}%[需<={100 * FP_TH:.0f}%])")
    print(f"[3] 半脆弱仍能定位重绘区: {'成立' if sf_locates else '不成立'} "
          f"(篡改F1={sf['f1']:.2f}[需>={LOC_TH}])")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_SAMPLE)
