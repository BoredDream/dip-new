# -*- coding: utf-8 -*-
"""标准 JPEG 熵编码(Huffman)—— JPEG 第 8 步。

采用 JPEG 标准附录 K(K.3–K.6)的标准 Huffman 表(亮度/色度各一套 DC、AC),
与 libjpeg 默认表一致;按附录 C 的规范流程从 (BITS, HUFFVAL) 生成规范前缀码。

数值编码用"幅度分类 + 附加比特"(category/size + extra bits):
  * 对非零系数 v,size = |v| 的比特数;附加比特为 v(v>0)或 v+2^size-1(v<0)。
  * DC 编码 size 的 Huffman 码 + size 个附加比特;AC 编码 (run,size) 的 Huffman 码 + size 个附加比特。

注:本编码器输出自有容器的比特流(不写 JFIF 文件头/字节填充),用于度量真实压缩比
与跑通"DCT→量化→熵编码→熵解码→反量化→IDCT"完整链路;与 libjpeg 的对比校准在
scripts/01_test_jpeg_codec.py 中通过 PSNR/bpp 完成。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# 标准 Huffman 表规格 (BITS[1..16], HUFFVAL)
# --------------------------------------------------------------------------- #
_STD = {
    "dc_luma": {
        "bits": [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "vals": list(range(12)),
    },
    "dc_chroma": {
        "bits": [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "vals": list(range(12)),
    },
    "ac_luma": {
        "bits": [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
        "vals": [
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41,
            0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91,
            0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24,
            0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a,
            0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38,
            0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53,
            0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66,
            0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
            0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93,
            0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
            0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
            0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9,
            0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1,
            0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2,
            0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
        ],
    },
    "ac_chroma": {
        "bits": [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
        "vals": [
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12,
            0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14,
            0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15,
            0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17,
            0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a,
            0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
            0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
            0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a,
            0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
            0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5,
            0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
            0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9,
            0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2,
            0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
        ],
    },
}


def _build_canonical(bits: List[int], vals: List[int]) -> Dict[int, Tuple[int, int]]:
    """由 (BITS, HUFFVAL) 生成规范 Huffman 码:symbol -> (code, length)。

    流程同 JPEG 标准附录 C 图 C.2(逐长度递增分配码字)。
    """
    table: Dict[int, Tuple[int, int]] = {}
    code = 0
    k = 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            table[vals[k]] = (code, length)
            code += 1
            k += 1
        code <<= 1
    return table


# symbol -> (code, length) 的编码表
ENC_TABLES = {name: _build_canonical(spec["bits"], spec["vals"]) for name, spec in _STD.items()}
# (length, code) -> symbol 的解码表
DEC_TABLES = {
    name: {(length, code): sym for sym, (code, length) in tbl.items()}
    for name, tbl in ENC_TABLES.items()
}


# --------------------------------------------------------------------------- #
# 幅度分类(category / size + 附加比特)
# --------------------------------------------------------------------------- #
def magnitude_category(value: int) -> int:
    """非零系数 -> 幅度比特数(category/size);0 -> 0。"""
    return int(abs(value)).bit_length()


def value_to_bits(value: int, size: int) -> int:
    """value -> size 位附加比特(正数取本身,负数取 value+2^size-1)。"""
    if value >= 0:
        return value
    return value + (1 << size) - 1


def bits_to_value(bits: int, size: int) -> int:
    """size 位附加比特 -> 原始有符号值(value_to_bits 的逆)。"""
    if size == 0:
        return 0
    if bits >= (1 << (size - 1)):       # 高位为 1 -> 正数
        return bits
    return bits - (1 << size) + 1        # 高位为 0 -> 负数


# --------------------------------------------------------------------------- #
# 比特流读写(MSB 优先)
# --------------------------------------------------------------------------- #
class BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._cur = 0
        self._nbits = 0
        self.total_bits = 0

    def write_bits(self, value: int, length: int) -> None:
        for i in range(length - 1, -1, -1):
            bit = (value >> i) & 1
            self._cur = (self._cur << 1) | bit
            self._nbits += 1
            self.total_bits += 1
            if self._nbits == 8:
                self._bytes.append(self._cur)
                self._cur = 0
                self._nbits = 0

    def getvalue(self) -> bytes:
        if self._nbits > 0:
            return bytes(self._bytes) + bytes([self._cur << (8 - self._nbits)])
        return bytes(self._bytes)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_bit(self) -> int:
        byte = self._data[self._pos >> 3]
        bit = (byte >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit

    def read_bits(self, length: int) -> int:
        v = 0
        for _ in range(length):
            v = (v << 1) | self.read_bit()
        return v

    def read_huffman(self, dec_table: Dict[Tuple[int, int], int]) -> int:
        """逐位读取直到匹配一个规范码字,返回对应 symbol。"""
        code = 0
        for length in range(1, 17):
            code = (code << 1) | self.read_bit()
            sym = dec_table.get((length, code))
            if sym is not None:
                return sym
        raise ValueError("Huffman 解码失败:未匹配到合法码字")
