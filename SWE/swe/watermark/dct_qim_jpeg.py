# -*- coding: utf-8 -*-
"""DCT-QIM 频域水印 —— 嵌入 JPEG 流水线内部(模块二核心基线)。

实施方案 4.2.2:在 JPEG 编码的"DCT 之后、量化前"把比特用 QIM 嵌进中频系数。
设计三要点:
  1. 步长 Δ 取 >= 目标质量下该位置的 JPEG 量化步长 Q(u,v),水印即可扛 JPEG 再压缩;
  2. 重复编码 R + 多数表决:每个信息位嵌到 R 个块,提取时多数表决,显著降 BER;
  3. 密钥控制:用伪随机密钥决定承载哪些块,增强安全性。

两种用法,共享同一套"块->比特"承载计划与 QIM 逻辑:
  * embed/extract(channel, bits)   —— BaseWatermark 接口,供评估器与其它方法同台对比;
  * embed_in_jpeg(image, bits)     —— 真正走 JPEGCodec 的 coeff_hook 在压缩过程中嵌入,
                                      返回 (含水印图, JPEGStream),演示"压缩同时嵌入"并给出真实 bpp。

提取始终从(可能被攻击的)像素重新 8×8 分块 DCT 读 QIM —— 即盲提取、不需码流。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import BaseWatermark
from .utils import qim_embed_vec, qim_extract_vec
from ..codec.dct import dct_8x8_batch, idct_8x8_batch
from ..codec.quant import quality_to_qtable

__all__ = ["DCTQIMJPEGWatermark"]


class DCTQIMJPEGWatermark(BaseWatermark):
    def __init__(self, quality: float = 50, coef: Tuple[int, int] = (3, 2),
                 delta: Optional[float] = None, repeat: int = 4, key: int = 2026):
        self.quality = float(quality)
        self.coef = coef
        self.repeat = int(repeat)
        self.key = int(key)
        qstep = float(quality_to_qtable(quality, chroma=False)[coef])
        # Δ 默认取 3×量化步长:留足余量抗一次 JPEG 再压缩(QIM 抗量化经验法则 Δ>2Q)。
        self.delta = float(delta) if delta is not None else 3.0 * qstep
        self.qstep = qstep

    # --------------- 承载计划:把 (bit, repeat) 槽位映射到不同的块 --------------- #
    def _plan(self, grid: Tuple[int, int], n_bits: int) -> np.ndarray:
        """返回形状 (n_bits, repeat) 的块线性下标(密钥伪随机、互不重复)。"""
        nby, nbx = grid
        n_blocks = nby * nbx
        need = n_bits * self.repeat
        if need > n_blocks:
            raise ValueError(f"容量不足:需要 {need} 个块,仅有 {n_blocks} 个。"
                             f"减少比特数或 repeat,或增大图像。")
        order = np.random.RandomState(self.key).permutation(n_blocks)[:need]
        return order.reshape(n_bits, self.repeat)

    # --------------- BaseWatermark 接口 --------------- #
    def embed(self, ch: np.ndarray, bits: np.ndarray) -> np.ndarray:
        ch = ch.astype(np.float64)
        H, W = ch.shape
        Hc, Wc = H - H % 8, W - W % 8
        crop = ch[:Hc, :Wc]
        blocks = (crop.reshape(Hc // 8, 8, Wc // 8, 8).transpose(0, 2, 1, 3)).copy()
        coeffs = dct_8x8_batch(blocks - 128.0).reshape(-1, 8, 8)
        plan = self._plan((Hc // 8, Wc // 8), len(bits))
        u, v = self.coef
        for i, bit in enumerate(bits):
            idx = plan[i]
            coeffs[idx, u, v] = qim_embed_vec(coeffs[idx, u, v],
                                              np.full(len(idx), int(bit)), self.delta)
        out_blocks = idct_8x8_batch(coeffs.reshape(blocks.shape)) + 128.0
        out = (out_blocks.transpose(0, 2, 1, 3).reshape(Hc, Wc))
        res = ch.copy()
        res[:Hc, :Wc] = out
        return np.clip(res, 0, 255)

    def extract(self, ch: np.ndarray, n_bits: int) -> np.ndarray:
        ch = ch.astype(np.float64)
        H, W = ch.shape
        Hc, Wc = H - H % 8, W - W % 8
        crop = ch[:Hc, :Wc]
        blocks = crop.reshape(Hc // 8, 8, Wc // 8, 8).transpose(0, 2, 1, 3)
        coeffs = dct_8x8_batch(blocks - 128.0).reshape(-1, 8, 8)
        plan = self._plan((Hc // 8, Wc // 8), n_bits)
        u, v = self.coef
        votes = qim_extract_vec(coeffs[plan.ravel(), u, v], self.delta).reshape(n_bits, self.repeat)
        # 多数表决(repeat 为偶数时平票判 1,保持确定性)
        return (votes.sum(axis=1) * 2 >= self.repeat).astype(np.int64)

    def capacity(self, shape) -> int:
        h, w = shape
        return ((h // 8) * (w // 8)) // self.repeat

    # --------------- 真正在 JPEG 压缩过程中嵌入 --------------- #
    def embed_in_jpeg(self, image: np.ndarray, bits: np.ndarray):
        """走 JPEGCodec 的 coeff_hook,在量化前对 Y 的中频系数做 QIM。

        返回 (watermarked_uint8_image, JPEGStream)。watermarked 图是真实 JPEG 解码输出,
        其 Y 通道用 extract() 即可盲提取。

        注:先把图裁到 16 的倍数,使编码端(4:2:0 会补到 16 的倍数)不发生填充,
        从而保证"嵌入时的块网格"与"提取时(解码图裁回原尺寸后)的块网格"一致 ——
        否则非 16 倍数尺寸会因网格错位导致提取失败。
        """
        from ..codec.jpeg import JPEGCodec  # 局部导入避免循环依赖

        H, W = image.shape[:2]
        image = image[:H - H % 16, :W - W % 16]
        u, v = self.coef
        n_bits = len(bits)

        def hook(y_coeffs: np.ndarray, qtable: np.ndarray) -> np.ndarray:
            grid = y_coeffs.shape[:2]
            plan = self._plan(grid, n_bits)
            flat = y_coeffs.reshape(-1, 8, 8).copy()
            for i, bit in enumerate(bits):
                idx = plan[i]
                flat[idx, u, v] = qim_embed_vec(flat[idx, u, v],
                                                np.full(len(idx), int(bit)), self.delta)
            return flat.reshape(y_coeffs.shape)

        codec = JPEGCodec(quality=self.quality, subsample=True)
        stream = codec.encode(image, coeff_hook=hook)
        return codec.decode(stream), stream
