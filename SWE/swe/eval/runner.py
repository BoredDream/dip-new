# -*- coding: utf-8 -*-
"""实验运行器 —— 模块五之二:跑"方法 × 攻击 × 强度"全套实验。

用适配器统一驱动两类方法:
  * ChannelAdapter —— 经典/JPEG 域水印,嵌入 RGB 图的**绿色通道**(亮度主成分,其余通道保留;
    刻意不走 YCbCr,避免颜色空间取整破坏 LSB);
  * ImageAdapter   —— 深度潜空间水印,直接作用于整张 RGB 图。
对每张测试图:嵌入 -> 量不可见性(PSNR/SSIM/LPIPS) -> 各攻击各强度提取 -> 比特准确率。
报告均值 ± 标准差(实施方案 4.5 评估纪律)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..data.datasets import to_uint8
from .metrics import psnr, ssim, lpips_distance, bit_accuracy

__all__ = ["MethodAdapter", "ChannelAdapter", "ImageAdapter", "run_experiments"]


class MethodAdapter:
    def __init__(self, name: str, n_bits: int):
        self.name = name
        self.n_bits = n_bits
        self.bits = np.zeros(n_bits, dtype=np.int64)

    def set_message(self, bits: np.ndarray) -> None:
        self.bits = np.asarray(bits, dtype=np.int64)[: self.n_bits]

    def embed(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def extract(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ChannelAdapter(MethodAdapter):
    """经典单通道水印 -> 作用于 RGB 图的绿色通道(亮度主成分)。

    刻意不走 RGB↔YCbCr:uint8 颜色空间往返的取整噪声会破坏 LSB(使其干净准确率虚低);
    直接在绿色通道嵌入/提取既无此损失,又保持"单通道水印"的语义(green 是亮度最大分量)。
    """

    def __init__(self, name: str, watermark, n_bits: int, channel: int = 1):
        super().__init__(name, n_bits)
        self.wm = watermark
        self.channel = channel

    def embed(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return to_uint8(self.wm.embed(image.astype(np.float64), self.bits))
        out = image.astype(np.float64).copy()
        out[..., self.channel] = self.wm.embed(out[..., self.channel], self.bits)
        return to_uint8(out)

    def extract(self, image: np.ndarray) -> np.ndarray:
        ch = image.astype(np.float64) if image.ndim == 2 else image[..., self.channel].astype(np.float64)
        return self.wm.extract(ch, self.n_bits)


class ImageAdapter(MethodAdapter):
    """深度潜空间水印 -> 作用于整张 RGB 图。"""

    def __init__(self, name: str, model, n_bits: Optional[int] = None):
        super().__init__(name, n_bits or model.secret_bits)
        self.model = model

    def embed(self, image: np.ndarray) -> np.ndarray:
        return self.model.embed(image, self.bits)

    def extract(self, image: np.ndarray) -> np.ndarray:
        return self.model.extract(image)


@dataclass
class Record:
    method: str
    attack: str
    strength: object
    bit_acc_mean: float
    bit_acc_std: float
    psnr: float = float("nan")      # 不可见性(含水印图 vs 原图)— 仅 "none" 行有意义
    ssim: float = float("nan")
    lpips: float = float("nan")
    att_ssim: float = float("nan")  # 攻击后图 vs 原图的 SSIM(攻击造成的画面改变量)
    att_psnr: float = float("nan")


def run_experiments(adapters: Dict[str, MethodAdapter], images: List[np.ndarray],
                    attacks: Dict[str, tuple], seed: int = 2026,
                    with_lpips: bool = False, verbose: bool = True) -> List[Record]:
    """返回 Record 列表。attacks: name -> (fn, param_name, [strengths])。"""
    rng = np.random.RandomState(seed)
    records: List[Record] = []

    for mname, adapter in adapters.items():
        adapter.set_message(rng.binomial(1, 0.5, adapter.n_bits).astype(np.int64))
        # 嵌入 + 不可见性
        wms, psnrs, ssims, lps = [], [], [], []
        for img in images:
            wm = adapter.embed(img)
            wms.append(wm)
            psnrs.append(psnr(img, wm)); ssims.append(ssim(img, wm))
            if with_lpips:
                lps.append(lpips_distance(img, wm))
        inv = dict(psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)),
                   lpips=float(np.nanmean(lps)) if lps else float("nan"))

        # 干净(无攻击)
        clean = [bit_accuracy(adapter.bits, adapter.extract(wm)) for wm in wms]
        records.append(Record(mname, "none", 0, float(np.mean(clean)), float(np.std(clean)), **inv))
        if verbose:
            print(f"[{mname}] PSNR={inv['psnr']:.1f} SSIM={inv['ssim']:.3f} clean_acc={np.mean(clean):.3f}")

        # 各攻击各强度
        for aname, (fn, param, grid) in attacks.items():
            for s in grid:
                accs, dssim, dpsnr = [], [], []
                for img, wm in zip(images, wms):
                    try:
                        att = fn(wm, **{param: s})
                        accs.append(bit_accuracy(adapter.bits, adapter.extract(att)))
                        dssim.append(ssim(img, att)); dpsnr.append(psnr(img, att))
                    except Exception as e:  # noqa: BLE001
                        if verbose:
                            print(f"    !{aname}({s}) 跳过: {e}")
                        accs = []
                        break
                if accs:
                    records.append(Record(mname, aname, s, float(np.mean(accs)),
                                          float(np.std(accs)),
                                          att_ssim=float(np.mean(dssim)),
                                          att_psnr=float(np.mean(dpsnr))))
            if verbose and accs:
                last = [r for r in records if r.method == mname and r.attack == aname]
                print(f"    {aname:16s} acc@strengths: "
                      + " ".join(f"{r.bit_acc_mean:.2f}" for r in last))
    return records


def records_to_table(records: List[Record], attacks: List[str],
                     strength_pick: str = "mid") -> "tuple":
    """把 Record 列表整理成 (methods, attacks, 矩阵) 便于画总表。

    每个攻击挑一个代表强度(mid=中位档,last=最强档)。
    """
    methods = sorted({r.method for r in records})
    mat = np.full((len(methods), len(attacks)), np.nan)
    for j, a in enumerate(attacks):
        rs = sorted({r.strength for r in records if r.attack == a})
        if not rs:
            continue
        pick = rs[len(rs) // 2] if strength_pick == "mid" else rs[-1]
        for i, m in enumerate(methods):
            hit = [r for r in records if r.method == m and r.attack == a and r.strength == pick]
            if hit:
                mat[i, j] = hit[0].bit_acc_mean
    return methods, attacks, mat
