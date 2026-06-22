# -*- coding: utf-8 -*-
"""自实现 JPEG 编解码器(九步)—— 模块一主入口。

九步(实施方案 4.1):
  1. RGB -> YCbCr          (color.rgb_to_ycbcr)
  2. 色度 4:2:0 下采样       (color.subsample_420)
  3. 8×8 分块 + 电平移位 -128
  4. 二维 DCT               (dct.dct_8x8_batch)
  5. 量化                   (quant.quality_to_qtable)
  6. Zigzag 扫描            (zigzag.zigzag)
  7. DC 差分(DPCM)/AC 游程(RLE)
  8. Huffman 熵编码         (huffman 标准表)
  9. 解码 = 上述逆过程

水印挂钩:encode(..., coeff_hook) 在"第 4 步 DCT 之后、第 5 步量化之前"对亮度 Y 的
未量化 DCT 系数块调用 coeff_hook(y_blocks, luma_qtable) -> 修改后的 y_blocks,
这正是"在 JPEG 压缩过程中嵌入水印"的落点(供 dct_qim_jpeg 水印使用)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import color
from .dct import dct_8x8_batch, idct_8x8_batch
from .quant import quality_to_qtable
from .zigzag import zigzag, izigzag, rle_encode_ac, EOB, ZRL
from .huffman import (
    ENC_TABLES,
    DEC_TABLES,
    BitWriter,
    BitReader,
    magnitude_category,
    value_to_bits,
    bits_to_value,
)

CoeffHook = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class _CompSpec:
    name: str               # 'Y' / 'Cb' / 'Cr'
    padded_shape: Tuple[int, int]
    grid: Tuple[int, int]   # (块行数, 块列数)
    chroma: bool


@dataclass
class JPEGStream:
    """自有容器的压缩码流(非 JFIF 文件)。"""
    mode: str               # 'gray' | 'rgb'
    orig_shape: Tuple[int, int]
    quality: float
    subsample: bool
    comps: List[_CompSpec]
    entropy: bytes
    total_bits: int = 0

    @property
    def num_pixels(self) -> int:
        h, w = self.orig_shape
        return h * w

    @property
    def bpp(self) -> float:
        """每像素比特数(熵编码后),用于度量压缩率。"""
        return self.total_bits / max(1, self.num_pixels)

    @property
    def compressed_bytes(self) -> int:
        return len(self.entropy)


def _pad_to_multiple(arr: np.ndarray, m: int) -> np.ndarray:
    h, w = arr.shape
    ph, pw = (-h) % m, (-w) % m
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((0, ph), (0, pw)), mode="edge")


def _split_blocks(channel: np.ndarray) -> np.ndarray:
    """(H,W) -> (nby, nbx, 8, 8),H、W 已是 8 的倍数。"""
    h, w = channel.shape
    return (channel.reshape(h // 8, 8, w // 8, 8)
                   .transpose(0, 2, 1, 3))


def _merge_blocks(blocks: np.ndarray) -> np.ndarray:
    """(nby, nbx, 8, 8) -> (H,W)。"""
    nby, nbx = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(nby * 8, nbx * 8)


class JPEGCodec:
    """标准基线 JPEG 编解码器。支持灰度(2D)与彩色(3D, RGB)输入。"""

    def __init__(self, quality: float = 50, subsample: bool = True):
        self.quality = float(quality)
        self.subsample = subsample

    # ---------------- 前向:像素 -> 量化系数块 ---------------- #
    def _forward_component(self, channel: np.ndarray, qtable: np.ndarray,
                           hook: Optional[CoeffHook] = None) -> np.ndarray:
        blocks = _split_blocks(channel) - 128.0          # 第 3 步:电平移位
        coeffs = dct_8x8_batch(blocks)                    # 第 4 步:DCT
        if hook is not None:                              # 水印嵌入点(仅 Y 传入 hook)
            coeffs = hook(coeffs, qtable)
        quant = np.round(coeffs / qtable).astype(np.int64)  # 第 5 步:量化
        return quant

    def _inverse_component(self, quant: np.ndarray, qtable: np.ndarray) -> np.ndarray:
        coeffs = quant.astype(np.float64) * qtable        # 反量化
        blocks = idct_8x8_batch(coeffs) + 128.0           # IDCT + 反电平移位
        return _merge_blocks(blocks)

    # ---------------- 熵编码 / 解码 ---------------- #
    @staticmethod
    def _entropy_encode(quant: np.ndarray, chroma: bool, writer: BitWriter) -> None:
        dc_enc = ENC_TABLES["dc_chroma" if chroma else "dc_luma"]
        ac_enc = ENC_TABLES["ac_chroma" if chroma else "ac_luma"]
        nby, nbx = quant.shape[:2]
        prev_dc = 0
        for by in range(nby):
            for bx in range(nbx):
                zz = zigzag(quant[by, bx])
                # --- DC:DPCM + 幅度分类 ---
                dc = int(zz[0])
                diff = dc - prev_dc
                prev_dc = dc
                size = magnitude_category(diff)
                code, length = dc_enc[size]
                writer.write_bits(code, length)
                if size:
                    writer.write_bits(value_to_bits(diff, size), size)
                # --- AC:RLE + (run,size) Huffman + 附加比特 ---
                for run, value in rle_encode_ac(zz[1:]):
                    if (run, value) == EOB:
                        c, l = ac_enc[0x00]
                        writer.write_bits(c, l)
                        break
                    if (run, value) == ZRL:
                        c, l = ac_enc[0xF0]
                        writer.write_bits(c, l)
                        continue
                    vsize = magnitude_category(value)
                    sym = (run << 4) | vsize
                    c, l = ac_enc[sym]
                    writer.write_bits(c, l)
                    writer.write_bits(value_to_bits(value, vsize), vsize)

    @staticmethod
    def _entropy_decode(reader: BitReader, grid: Tuple[int, int], chroma: bool) -> np.ndarray:
        dc_dec = DEC_TABLES["dc_chroma" if chroma else "dc_luma"]
        ac_dec = DEC_TABLES["ac_chroma" if chroma else "ac_luma"]
        nby, nbx = grid
        out = np.zeros((nby, nbx, 8, 8), dtype=np.int64)
        prev_dc = 0
        for by in range(nby):
            for bx in range(nbx):
                size = reader.read_huffman(dc_dec)
                diff = bits_to_value(reader.read_bits(size), size) if size else 0
                dc = prev_dc + diff
                prev_dc = dc
                ac = np.zeros(63, dtype=np.int64)
                k = 0
                while k < 63:
                    sym = reader.read_huffman(ac_dec)
                    if sym == 0x00:           # EOB
                        break
                    run, vsize = sym >> 4, sym & 0xF
                    if vsize == 0:            # ZRL(run==15)
                        k += 16
                        continue
                    k += run
                    if k >= 63:
                        break
                    ac[k] = bits_to_value(reader.read_bits(vsize), vsize)
                    k += 1
                zz = np.concatenate([[dc], ac])
                out[by, bx] = izigzag(zz)
        return out

    # ---------------- 公共 API ---------------- #
    def encode(self, image: np.ndarray, coeff_hook: Optional[CoeffHook] = None) -> JPEGStream:
        """编码为 JPEGStream。coeff_hook 仅作用于亮度 Y(水印嵌入点)。"""
        mode = "gray" if image.ndim == 2 else "rgb"
        H, W = image.shape[:2]
        qY = quality_to_qtable(self.quality, chroma=False)
        writer = BitWriter()
        comps: List[_CompSpec] = []

        if mode == "gray":
            m = 8
            Y = _pad_to_multiple(image.astype(np.float64), m)
            qY_blocks = self._forward_component(Y, qY, hook=coeff_hook)
            comps.append(_CompSpec("Y", Y.shape, qY_blocks.shape[:2], False))
            self._entropy_encode(qY_blocks, False, writer)
        else:
            ycc = color.rgb_to_ycbcr(image)
            m = 16 if self.subsample else 8
            ycc = np.stack([_pad_to_multiple(ycc[..., c], m) for c in range(3)], axis=-1)
            Y = ycc[..., 0]
            qC = quality_to_qtable(self.quality, chroma=True)
            # Y(全分辨率,带水印 hook)
            qY_blocks = self._forward_component(Y, qY, hook=coeff_hook)
            comps.append(_CompSpec("Y", Y.shape, qY_blocks.shape[:2], False))
            self._entropy_encode(qY_blocks, False, writer)
            # Cb/Cr
            for ci, name in ((1, "Cb"), (2, "Cr")):
                ch = ycc[..., ci]
                ch = color.subsample_420(ch) if self.subsample else ch
                qb = self._forward_component(ch, qC, hook=None)
                comps.append(_CompSpec(name, ch.shape, qb.shape[:2], True))
                self._entropy_encode(qb, True, writer)

        return JPEGStream(mode=mode, orig_shape=(H, W), quality=self.quality,
                          subsample=self.subsample, comps=comps,
                          entropy=writer.getvalue(), total_bits=writer.total_bits)

    def decode(self, stream: JPEGStream) -> np.ndarray:
        """JPEGStream -> uint8 图像。"""
        qY = quality_to_qtable(stream.quality, chroma=False)
        qC = quality_to_qtable(stream.quality, chroma=True)
        reader = BitReader(stream.entropy)
        H, W = stream.orig_shape

        if stream.mode == "gray":
            spec = stream.comps[0]
            quant = self._entropy_decode(reader, spec.grid, False)
            Y = self._inverse_component(quant, qY)
            return np.clip(np.round(Y[:H, :W]), 0, 255).astype(np.uint8)

        planes = {}
        for spec in stream.comps:
            qtbl = qC if spec.chroma else qY
            quant = self._entropy_decode(reader, spec.grid, spec.chroma)
            planes[spec.name] = self._inverse_component(quant, qtbl)

        Hy, Wy = planes["Y"].shape
        Cb = color.upsample_420(planes["Cb"], (Hy, Wy)) if stream.subsample else planes["Cb"]
        Cr = color.upsample_420(planes["Cr"], (Hy, Wy)) if stream.subsample else planes["Cr"]
        ycc = np.stack([planes["Y"], Cb, Cr], axis=-1)
        rgb = color.ycbcr_to_rgb(ycc)
        return np.clip(np.round(rgb[:H, :W]), 0, 255).astype(np.uint8)

    # ---------------- 便捷封装 ---------------- #
    def compress_decompress(self, image: np.ndarray,
                            coeff_hook: Optional[CoeffHook] = None) -> Tuple[np.ndarray, float]:
        """一次"编码再解码",返回 (重建图, bpp)。JPEG 再压缩攻击复用它。"""
        stream = self.encode(image, coeff_hook=coeff_hook)
        return self.decode(stream), stream.bpp
