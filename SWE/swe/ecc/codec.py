# -*- coding: utf-8 -*-
"""纠错编码(ECC)—— 模块六。无第三方依赖的纯 Python 实现。

实施方案 4.6:把个人信息(如用户 ID)先用纠错码编码再嵌入,使其即便被攻击错几位
也能纠正、完整还原 -> 得到"信息恢复率"这一更贴近实用的指标。

提供三种码,统一接口 ECC.encode_bits / decode_bits(都在 0/1 比特数组上工作):
  * RepetitionECC   —— 每比特重复 R 次 + 多数表决(最简,纠随机错能力一般);
  * HammingECC      —— 汉明 (7,4),每 7 位纠 1 位(对应实施方案提到的分组纠错);
  * ReedSolomonECC  —— GF(2^8) 上的 RS 码,每块纠 nsym/2 个字节错(突发错友好,推荐)。

若安装了 `reedsolo` 库,ReedSolomonECC 会优先用它(更快);否则用内置实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

__all__ = ["ECC", "RepetitionECC", "HammingECC", "ReedSolomonECC", "make_ecc"]


# --------------------------------------------------------------------------- #
# 比特 <-> 字节
# --------------------------------------------------------------------------- #
def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.int64)


class ECC(ABC):
    """纠错码统一接口(在 0/1 比特数组上工作)。"""

    @abstractmethod
    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        """数据比特 -> 受保护比特(含冗余)。"""

    @abstractmethod
    def decode_bits(self, coded: np.ndarray, n_data_bits: int) -> np.ndarray:
        """受保护比特 -> 纠错后的前 n_data_bits 位数据。"""

    def encoded_len(self, n_data_bits: int) -> int:
        """给定数据比特数,返回编码后比特数(供水印分配容量)。"""
        return len(self.encode_bits(np.zeros(n_data_bits, dtype=np.int64)))

    @property
    def name(self) -> str:
        return type(self).__name__


# --------------------------------------------------------------------------- #
# 1) 重复码
# --------------------------------------------------------------------------- #
class RepetitionECC(ECC):
    def __init__(self, repeat: int = 3):
        assert repeat >= 1 and repeat % 2 == 1, "repeat 取奇数以便多数表决"
        self.repeat = repeat

    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(bits, dtype=np.int64), self.repeat)

    def decode_bits(self, coded: np.ndarray, n_data_bits: int) -> np.ndarray:
        coded = np.asarray(coded, dtype=np.int64)[: n_data_bits * self.repeat]
        groups = coded.reshape(n_data_bits, self.repeat)
        return (groups.sum(axis=1) * 2 > self.repeat).astype(np.int64)


# --------------------------------------------------------------------------- #
# 2) 汉明 (7,4)
# --------------------------------------------------------------------------- #
class HammingECC(ECC):
    """汉明 (7,4):4 数据位 -> 7 码字位,纠 1 位/块。"""

    # 生成矩阵 G (4x7) 与校验矩阵 H (3x7),systematic 形式 [I4 | P]
    _G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.int64)
    _H = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ], dtype=np.int64)

    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.int64)
        pad = (-len(bits)) % 4
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.int64)])
        data = bits.reshape(-1, 4)
        cw = (data @ self._G) % 2
        return cw.reshape(-1)

    def decode_bits(self, coded: np.ndarray, n_data_bits: int) -> np.ndarray:
        coded = np.asarray(coded, dtype=np.int64)
        nblk = (n_data_bits + 3) // 4
        coded = coded[: nblk * 7].reshape(nblk, 7).copy()
        # 校验子 -> 错误位置(syndrome 与 H 的某列相等则该位出错)
        synd = (coded @ self._H.T) % 2
        cols = self._H.T  # (7,3)
        for i in range(nblk):
            s = synd[i]
            if s.any():
                for j in range(7):
                    if np.array_equal(cols[j], s):
                        coded[i, j] ^= 1
                        break
        data = coded[:, :4].reshape(-1)
        return data[:n_data_bits]


# --------------------------------------------------------------------------- #
# 3) Reed–Solomon over GF(2^8)
# --------------------------------------------------------------------------- #
class _GF256:
    """GF(2^8) 算术(本原多项式 0x11d,生成元 alpha=2 —— 标准 RS 设定)。

    多项式系数约定:高次在前(canonical "Reed–Solomon for coders" 约定)。
    """

    def __init__(self, prim: int = 0x11d, generator: int = 2):
        self.exp = [0] * 512
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x = self._mul_noLUT(x, generator, prim)
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    @staticmethod
    def _mul_noLUT(x, y, prim):
        r = 0
        while y:
            if y & 1:
                r ^= x
            y >>= 1
            x <<= 1
            if x & 0x100:
                x ^= prim
        return r

    def mul(self, x, y):
        if x == 0 or y == 0:
            return 0
        return self.exp[self.log[x] + self.log[y]]

    def div(self, x, y):
        if y == 0:
            raise ZeroDivisionError
        if x == 0:
            return 0
        return self.exp[(self.log[x] + 255 - self.log[y]) % 255]

    def pow(self, x, p):
        return self.exp[(self.log[x] * p) % 255]

    def inverse(self, x):
        return self.exp[255 - self.log[x]]

    def poly_scale(self, p, s):
        return [self.mul(c, s) for c in p]

    def poly_add(self, p, q):
        r = [0] * max(len(p), len(q))
        for i in range(len(p)):
            r[i + len(r) - len(p)] = p[i]
        for i in range(len(q)):
            r[i + len(r) - len(q)] ^= q[i]
        return r

    def poly_mul(self, p, q):
        r = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            if a == 0:
                continue
            for j, b in enumerate(q):
                r[i + j] ^= self.mul(a, b)
        return r

    def poly_eval(self, p, x):
        y = p[0]
        for c in p[1:]:
            y = self.mul(y, x) ^ c
        return y

    def poly_div(self, dividend, divisor):
        """多项式带余除法,返回 (商, 余)。系数高次在前。"""
        out = list(dividend)
        for i in range(len(dividend) - (len(divisor) - 1)):
            coef = out[i]
            if coef != 0:
                for j in range(1, len(divisor)):
                    if divisor[j] != 0:
                        out[i + j] ^= self.mul(divisor[j], coef)
        sep = -(len(divisor) - 1)
        return out[:sep], out[sep:]


class ReedSolomonECC(ECC):
    """RS(k+nsym, k) over GF(2^8)。每个码块可纠 nsym//2 个字节错。

    数据比特 -> 字节 -> 按 k 字节分块 -> 每块加 nsym 个校验字节 -> 字节 -> 比特。
    解码用 Berlekamp–Massey 求错误定位多项式、Chien 搜索定位、Forney 算法求幅值。
    算法遵循 "Reed–Solomon codes for coders"(fcr=0, generator=2)的规范实现。
    """

    def __init__(self, nsym: int = 10, block_k: int = 223):
        assert 1 <= nsym <= 254
        assert 1 <= block_k <= 255 - nsym
        self.nsym = nsym
        self.block_k = block_k
        self._use_lib = False
        try:                                   # 有 reedsolo 库则优先(更快)
            import reedsolo
            self._rs = reedsolo.RSCodec(nsym)
            self._use_lib = True
        except Exception:
            self.gf = _GF256()
            self._gen = self._generator_poly(nsym)

    def _generator_poly(self, nsym):
        g = [1]
        for i in range(nsym):
            g = self.gf.poly_mul(g, [1, self.gf.pow(2, i)])
        return g

    def _encode_block(self, msg: List[int]) -> List[int]:
        _, remainder = self.gf.poly_div(list(msg) + [0] * (len(self._gen) - 1), self._gen)
        return list(msg) + remainder

    def _calc_syndromes(self, msg):
        return [self.gf.poly_eval(msg, self.gf.pow(2, i)) for i in range(self.nsym)]

    def _find_error_locator(self, synd):
        err_loc = [1]; old_loc = [1]
        for i in range(self.nsym):
            delta = synd[i]
            for j in range(1, len(err_loc)):
                delta ^= self.gf.mul(err_loc[-(j + 1)], synd[i - j])
            old_loc = old_loc + [0]
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = self.gf.poly_scale(old_loc, delta)
                    old_loc = self.gf.poly_scale(err_loc, self.gf.inverse(delta))
                    err_loc = new_loc
                err_loc = self.gf.poly_add(err_loc, self.gf.poly_scale(old_loc, delta))
        while len(err_loc) > 1 and err_loc[0] == 0:
            del err_loc[0]
        return err_loc

    def _find_errors(self, err_loc, nmess):
        errs = len(err_loc) - 1
        pos = []
        for i in range(nmess):
            if self.gf.poly_eval(err_loc, self.gf.pow(2, i)) == 0:
                pos.append(nmess - 1 - i)
        if len(pos) != errs:
            return None
        return pos

    def _correct_errata(self, msg, synd_full, err_pos):
        """Forney 算法求误差幅值并修正。synd_full = [0, S0, S1, ..., S_{nsym-1}]。"""
        # 误差定位多项式 Lambda(x) = Π (1 + X_i x),系数高次在前
        coef_pos = [len(msg) - 1 - p for p in err_pos]
        err_loc = [1]
        for i in coef_pos:
            err_loc = self.gf.poly_mul(err_loc, self.gf.poly_add([1], [self.gf.pow(2, i), 0]))
        # 误差求值多项式 Omega(x) = [S(x)·Lambda(x)] mod x^(nerr+1)
        nerr = len(err_loc) - 1
        _, remainder = self.gf.poly_div(self.gf.poly_mul(synd_full[::-1], err_loc),
                                        [1] + [0] * (nerr + 1))
        err_eval = remainder[::-1]
        # 错误位置 X_i = alpha^{coef_pos}
        X = [self.gf.pow(2, p) for p in coef_pos]
        E = [0] * len(msg)
        for i, Xi in enumerate(X):
            Xi_inv = self.gf.inverse(Xi)
            # 形式导数 Lambda'(Xi_inv) = Π_{j≠i}(1 + X_j·Xi_inv)
            err_loc_prime = 1
            for j, Xj in enumerate(X):
                if j != i:
                    err_loc_prime = self.gf.mul(err_loc_prime, 1 ^ self.gf.mul(Xi_inv, Xj))
            y = self.gf.poly_eval(err_eval[::-1], Xi_inv)
            y = self.gf.mul(self.gf.pow(Xi, 1), y)       # fcr=0 -> 乘 X_i^1
            if err_loc_prime == 0:
                continue
            E[err_pos[i]] = self.gf.div(y, err_loc_prime)
        return self.gf.poly_add(msg, E)

    def _correct_block(self, msg_in: List[int]) -> List[int]:
        msg = list(msg_in)
        synd = self._calc_syndromes(msg)            # [S0, ..., S_{nsym-1}]
        if max(synd) == 0:
            return msg                              # 无错
        err_loc = self._find_error_locator(synd)
        err_pos = self._find_errors(err_loc[::-1], len(msg))
        if err_pos is None:
            return msg                              # 错误过多,无法纠正
        return self._correct_errata(msg, [0] + synd, err_pos)

    def _block_data_lens(self, n_data_bytes: int) -> List[int]:
        """把 n_data_bytes 划分为每块最多 block_k 字节(最后一块可较短)。

        编码与解码共用此分块,保证码字边界一致(否则解码会读错帧)。
        """
        if n_data_bytes <= 0:
            return [0]
        lens, rem = [], n_data_bytes
        while rem > 0:
            lens.append(min(self.block_k, rem))
            rem -= self.block_k
        return lens

    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        data = bits_to_bytes(bits)
        out = bytearray()
        for i in range(0, len(data), self.block_k):
            block = list(data[i:i + self.block_k])
            if self._use_lib:
                out.extend(self._rs.encode(bytes(block)))
            else:
                out.extend(self._encode_block(block))
        return bytes_to_bits(bytes(out))

    def decode_bits(self, coded: np.ndarray, n_data_bits: int) -> np.ndarray:
        data = bits_to_bytes(coded)
        n_data_bytes = (n_data_bits + 7) // 8
        out = bytearray()
        pos = 0
        for L in self._block_data_lens(n_data_bytes):
            block_n = L + self.nsym
            chunk = list(data[pos:pos + block_n])
            pos += block_n
            if len(chunk) < block_n:
                chunk += [0] * (block_n - len(chunk))
            if self._use_lib:
                try:
                    dec = self._rs.decode(bytes(chunk))
                    dec = dec[0] if isinstance(dec, tuple) else dec
                    out.extend(bytes(dec)[:L])
                except Exception:
                    out.extend(bytes(chunk[:L]))
            else:
                out.extend(bytes(self._correct_block(chunk)[:L]))
        return bytes_to_bits(bytes(out))[:n_data_bits]


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def make_ecc(name: str = "rs", **kwargs) -> ECC:
    name = name.lower()
    if name in ("rep", "repetition"):
        return RepetitionECC(**kwargs)
    if name in ("hamming", "hamming74"):
        return HammingECC()
    if name in ("rs", "reedsolomon", "reed-solomon"):
        return ReedSolomonECC(**kwargs)
    raise ValueError(f"未知 ECC: {name}")
