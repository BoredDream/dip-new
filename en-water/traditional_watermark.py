# -*- coding: utf-8 -*-
"""
传统(信号处理类)隐式水印方法合集
====================================

本文件把经典的不可见水印方法集中实现在一起,仅依赖 numpy + Pillow,
不需要 opencv / pywt / scipy(DCT、Haar 小波、SVD、DFT 全部自己实现或用 numpy 内置)。

包含的方法
----------
1. LSBWatermark            最低有效位             —— 空域,容量大,鲁棒性极弱
2. DCTWatermark            分块离散余弦变换       —— 抗 JPEG 压缩
3. DFTWatermark            离散傅里叶幅度谱        —— 抗平移/几何
4. DWTWatermark            Haar 小波域            —— 抗压缩/缩放
5. SVDWatermark            分块奇异值             —— 稳定性好
6. DWTSVDWatermark         小波 + 奇异值          —— 经典组合,鲁棒
7. DWTDCTWatermark         小波 + 余弦            —— VINE 论文里的 DwtDct 基线思路
8. SpreadSpectrumWatermark 扩频(相关检测)       —— 抗噪声的经典范式

统一约定
--------
- 所有方法在图像的 **亮度(灰度/Y)通道** 上操作,彩色其余通道保持不变。
- embed(channel, bits) -> 含水印的通道;extract(channel) -> 提取出的比特。
- 变换域方法统一用 **QIM(量化索引调制)** 在某个系数上嵌入 1 比特/单元,
  这样所有方法共享同一套嵌入/提取逻辑,只是"载体系数"不同。
- 消息格式:32 位长度头(payload 比特数)+ payload,提取端据此还原。

运行
----
    python traditional_watermark.py            # 对 sample.jpg 跑全套 demo
    python traditional_watermark.py xxx.png    # 指定图片
"""

import sys
import numpy as np
from PIL import Image


# =============================================================================
# 一、通用工具:文本 <-> 比特、QIM、PSNR、颜色通道
# =============================================================================

def text_to_bits(text: str) -> np.ndarray:
    """UTF-8 文本 -> 0/1 比特数组(大端,每字符 8 位)。"""
    data = text.encode("utf-8")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.int64)


def bits_to_text(bits: np.ndarray) -> str:
    """0/1 比特数组 -> UTF-8 文本(长度补齐到 8 的倍数)。"""
    bits = np.asarray(bits, dtype=np.uint8)
    n = (len(bits) // 8) * 8
    data = np.packbits(bits[:n])
    return data.tobytes().decode("utf-8", errors="replace")


def pack_message(text: str) -> np.ndarray:
    """把文本打包成 [32 位长度头 | payload]。长度头记录 payload 的比特数。"""
    payload = text_to_bits(text)
    header = np.array([int(b) for b in format(len(payload), "032b")], dtype=np.int64)
    return np.concatenate([header, payload])


def unpack_message(bits: np.ndarray) -> str:
    """从 [32 位长度头 | payload] 还原文本。"""
    bits = np.asarray(bits, dtype=np.int64)
    if len(bits) < 32:
        return ""
    length = int("".join(str(int(b)) for b in bits[:32]), 2)
    length = max(0, min(length, len(bits) - 32))
    return bits_to_text(bits[32:32 + length])


def qim_embed(value: float, bit: int, delta: float) -> float:
    """QIM 量化:bit=0 量化到 delta 整数格点,bit=1 量化到偏移半格的格点。"""
    if bit == 0:
        return round(value / delta) * delta
    else:
        return round((value - delta / 2) / delta) * delta + delta / 2


def qim_extract(value: float, delta: float) -> int:
    """QIM 解调:看 value 离哪一组格点更近。"""
    q0 = round(value / delta) * delta
    q1 = round((value - delta / 2) / delta) * delta + delta / 2
    return 0 if abs(value - q0) <= abs(value - q1) else 1


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """峰值信噪比(衡量不可感知性,越高越好)。"""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def bit_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    """两段比特的逐位一致率。"""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.mean(np.asarray(a[:n]) == np.asarray(b[:n])))


# =============================================================================
# 二、自己实现 DCT 与 Haar 小波(不依赖 scipy / pywt)
# =============================================================================

def _dct_matrix(n: int) -> np.ndarray:
    """正交 DCT-II 矩阵 D,使得 dct(x) = D @ x。"""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    D = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    D *= np.sqrt(2.0 / n)
    D[0, :] *= np.sqrt(0.5)
    return D


def dct2(block: np.ndarray) -> np.ndarray:
    """二维 DCT:X = Dr @ block @ Dc^T(支持非方形矩阵)。"""
    Dr = _dct_matrix(block.shape[0])
    Dc = _dct_matrix(block.shape[1])
    return Dr @ block @ Dc.T


def idct2(coef: np.ndarray) -> np.ndarray:
    """二维逆 DCT:block = Dr^T @ X @ Dc(支持非方形矩阵)。"""
    Dr = _dct_matrix(coef.shape[0])
    Dc = _dct_matrix(coef.shape[1])
    return Dr.T @ coef @ Dc


def haar_dwt2(img: np.ndarray):
    """一级二维 Haar 小波分解,返回 (LL, LH, HL, HH)。要求长宽为偶数。"""
    x = img.astype(np.float64)
    # 先对列方向(行内相邻像素)做一维 Haar
    a = (x[:, 0::2] + x[:, 1::2]) / np.sqrt(2)
    d = (x[:, 0::2] - x[:, 1::2]) / np.sqrt(2)
    # 再对行方向做一维 Haar
    LL = (a[0::2, :] + a[1::2, :]) / np.sqrt(2)
    HL = (a[0::2, :] - a[1::2, :]) / np.sqrt(2)
    LH = (d[0::2, :] + d[1::2, :]) / np.sqrt(2)
    HH = (d[0::2, :] - d[1::2, :]) / np.sqrt(2)
    return LL, LH, HL, HH


def haar_idwt2(LL, LH, HL, HH) -> np.ndarray:
    """一级二维 Haar 小波重构,haar_dwt2 的逆。"""
    a = np.zeros((LL.shape[0] * 2, LL.shape[1]))
    d = np.zeros((LL.shape[0] * 2, LL.shape[1]))
    a[0::2, :] = (LL + HL) / np.sqrt(2)
    a[1::2, :] = (LL - HL) / np.sqrt(2)
    d[0::2, :] = (LH + HH) / np.sqrt(2)
    d[1::2, :] = (LH - HH) / np.sqrt(2)
    x = np.zeros((a.shape[0], a.shape[1] * 2))
    x[:, 0::2] = (a + d) / np.sqrt(2)
    x[:, 1::2] = (a - d) / np.sqrt(2)
    return x


def iter_blocks(h: int, w: int, bs: int):
    """生成所有不重叠 bs×bs 块的左上角坐标(裁掉右/下边的余数)。"""
    for r in range(0, h - bs + 1, bs):
        for c in range(0, w - bs + 1, bs):
            yield r, c


# =============================================================================
# 三、各方法实现(统一接口:embed / extract)
# =============================================================================

class LSBWatermark:
    """最低有效位:把比特直接写进像素最低位。容量 = 像素数,但极不鲁棒。"""

    def embed(self, ch: np.ndarray, bits: np.ndarray) -> np.ndarray:
        out = ch.astype(np.uint8).copy().ravel()
        n = min(len(bits), len(out))
        out[:n] = (out[:n] & 0xFE) | bits[:n].astype(np.uint8)
        return out.reshape(ch.shape)

    def extract(self, ch: np.ndarray, n_bits: int) -> np.ndarray:
        return (ch.astype(np.uint8).ravel()[:n_bits] & 1).astype(np.int64)


class DCTWatermark:
    """8×8 分块 DCT,在中频系数上用 QIM 嵌 1 比特/块。抗 JPEG 压缩。"""

    def __init__(self, block=8, coef=(3, 2), delta=18.0):
        self.bs, self.coef, self.delta = block, coef, delta

    def embed(self, ch, bits):
        out = ch.astype(np.float64).copy()
        u, v = self.coef
        i = 0
        for r, c in iter_blocks(*ch.shape, self.bs):
            if i >= len(bits):
                break
            blk = out[r:r + self.bs, c:c + self.bs]
            X = dct2(blk)
            X[u, v] = qim_embed(X[u, v], int(bits[i]), self.delta)
            out[r:r + self.bs, c:c + self.bs] = idct2(X)
            i += 1
        return np.clip(out, 0, 255)

    def extract(self, ch, n_bits):
        u, v = self.coef
        bits, i = [], 0
        for r, c in iter_blocks(*ch.shape, self.bs):
            if i >= n_bits:
                break
            X = dct2(ch[r:r + self.bs, c:c + self.bs].astype(np.float64))
            bits.append(qim_extract(X[u, v], self.delta))
            i += 1
        return np.array(bits, dtype=np.int64)


class DWTWatermark:
    """一级 Haar 小波,在 HL 子带系数上用 QIM 嵌入。抗压缩/缩放。"""

    def __init__(self, delta=24.0):
        self.delta = delta

    def _crop_even(self, ch):
        h, w = ch.shape
        return ch[:h - h % 2, :w - w % 2]

    def embed(self, ch, bits):
        ch = self._crop_even(ch).astype(np.float64)
        LL, LH, HL, HH = haar_dwt2(ch)
        flat = HL.ravel()
        n = min(len(bits), len(flat))
        for i in range(n):
            flat[i] = qim_embed(flat[i], int(bits[i]), self.delta)
        HL = flat.reshape(HL.shape)
        return np.clip(haar_idwt2(LL, LH, HL, HH), 0, 255)

    def extract(self, ch, n_bits):
        ch = self._crop_even(ch).astype(np.float64)
        _, _, HL, _ = haar_dwt2(ch)
        flat = HL.ravel()
        n = min(n_bits, len(flat))
        return np.array([qim_extract(flat[i], self.delta) for i in range(n)], dtype=np.int64)


class SVDWatermark:
    """8×8 分块 SVD,对每块最大奇异值用 QIM 嵌 1 比特/块。数值稳定。"""

    def __init__(self, block=8, delta=30.0):
        self.bs, self.delta = block, delta

    def embed(self, ch, bits):
        out = ch.astype(np.float64).copy()
        i = 0
        for r, c in iter_blocks(*ch.shape, self.bs):
            if i >= len(bits):
                break
            blk = out[r:r + self.bs, c:c + self.bs]
            U, S, Vt = np.linalg.svd(blk)
            S[0] = qim_embed(S[0], int(bits[i]), self.delta)
            out[r:r + self.bs, c:c + self.bs] = (U * S) @ Vt
            i += 1
        return np.clip(out, 0, 255)

    def extract(self, ch, n_bits):
        bits, i = [], 0
        for r, c in iter_blocks(*ch.shape, self.bs):
            if i >= n_bits:
                break
            S = np.linalg.svd(ch[r:r + self.bs, c:c + self.bs].astype(np.float64),
                              compute_uv=False)
            bits.append(qim_extract(S[0], self.delta))
            i += 1
        return np.array(bits, dtype=np.int64)


class DWTSVDWatermark:
    """先一级 Haar 小波,再对 LL 子带分块做 SVD,在奇异值上 QIM。经典鲁棒组合。"""

    def __init__(self, block=4, delta=30.0):
        self.bs, self.delta = block, delta

    def _crop_even(self, ch):
        h, w = ch.shape
        return ch[:h - h % 2, :w - w % 2]

    def embed(self, ch, bits):
        ch = self._crop_even(ch).astype(np.float64)
        LL, LH, HL, HH = haar_dwt2(ch)
        i = 0
        for r, c in iter_blocks(*LL.shape, self.bs):
            if i >= len(bits):
                break
            blk = LL[r:r + self.bs, c:c + self.bs]
            U, S, Vt = np.linalg.svd(blk)
            S[0] = qim_embed(S[0], int(bits[i]), self.delta)
            LL[r:r + self.bs, c:c + self.bs] = (U * S) @ Vt
            i += 1
        return np.clip(haar_idwt2(LL, LH, HL, HH), 0, 255)

    def extract(self, ch, n_bits):
        ch = self._crop_even(ch).astype(np.float64)
        LL, _, _, _ = haar_dwt2(ch)
        bits, i = [], 0
        for r, c in iter_blocks(*LL.shape, self.bs):
            if i >= n_bits:
                break
            S = np.linalg.svd(LL[r:r + self.bs, c:c + self.bs], compute_uv=False)
            bits.append(qim_extract(S[0], self.delta))
            i += 1
        return np.array(bits, dtype=np.int64)


class DWTDCTWatermark:
    """先一级 Haar 小波,再对 LL 子带 8×8 分块做 DCT,中频系数 QIM。
    对应 VINE 论文里的 DwtDct 基线思路。"""

    def __init__(self, block=8, coef=(3, 2), delta=20.0):
        self.bs, self.coef, self.delta = block, coef, delta

    def _crop_even(self, ch):
        h, w = ch.shape
        return ch[:h - h % 2, :w - w % 2]

    def embed(self, ch, bits):
        ch = self._crop_even(ch).astype(np.float64)
        LL, LH, HL, HH = haar_dwt2(ch)
        u, v = self.coef
        i = 0
        for r, c in iter_blocks(*LL.shape, self.bs):
            if i >= len(bits):
                break
            X = dct2(LL[r:r + self.bs, c:c + self.bs])
            X[u, v] = qim_embed(X[u, v], int(bits[i]), self.delta)
            LL[r:r + self.bs, c:c + self.bs] = idct2(X)
            i += 1
        return np.clip(haar_idwt2(LL, LH, HL, HH), 0, 255)

    def extract(self, ch, n_bits):
        ch = self._crop_even(ch).astype(np.float64)
        LL, _, _, _ = haar_dwt2(ch)
        u, v = self.coef
        bits, i = [], 0
        for r, c in iter_blocks(*LL.shape, self.bs):
            if i >= n_bits:
                break
            X = dct2(LL[r:r + self.bs, c:c + self.bs])
            bits.append(qim_extract(X[u, v], self.delta))
            i += 1
        return np.array(bits, dtype=np.int64)


class DFTWatermark:
    """离散傅里叶幅度谱:在中频环上选共轭对称的系数对,用 QIM 改幅度。
    天然抗平移(平移只改相位不改幅度)。"""

    def __init__(self, delta=2000.0, radius_ratio=0.25, seed=2026):
        self.delta, self.radius_ratio, self.seed = delta, radius_ratio, seed

    def _positions(self, shape, n_bits):
        """在中频环上挑选若干 (u,v),并排除其共轭点,保证嵌入后仍是实图像。"""
        h, w = shape
        cy, cx = h // 2, w // 2
        rng = np.random.RandomState(self.seed)
        radius = int(min(h, w) * self.radius_ratio)
        cand = []
        for u in range(h):
            for v in range(w):
                d = np.hypot(u - cy, v - cx)
                if abs(d - radius) < 1.5:
                    cand.append((u, v))
        rng.shuffle(cand)
        chosen, used = [], set()
        for u, v in cand:
            conj = ((h - u) % h, (w - v) % w)
            if (u, v) in used or conj in used or (u, v) == conj:
                continue
            used.add((u, v)); used.add(conj)
            chosen.append((u, v))
            if len(chosen) >= n_bits:
                break
        return chosen

    def embed(self, ch, bits):
        h, w = ch.shape
        F = np.fft.fftshift(np.fft.fft2(ch.astype(np.float64)))
        mag, phase = np.abs(F), np.angle(F)
        for i, (u, v) in enumerate(self._positions(ch.shape, len(bits))):
            m = qim_embed(mag[u, v], int(bits[i]), self.delta)
            cu, cv = (h - u) % h, (w - v) % w
            mag[u, v] = mag[cu, cv] = m
        F2 = mag * np.exp(1j * phase)
        out = np.real(np.fft.ifft2(np.fft.ifftshift(F2)))
        return np.clip(out, 0, 255)

    def extract(self, ch, n_bits):
        F = np.fft.fftshift(np.fft.fft2(ch.astype(np.float64)))
        mag = np.abs(F)
        pos = self._positions(ch.shape, n_bits)
        return np.array([qim_extract(mag[u, v], self.delta) for u, v in pos], dtype=np.int64)


class SpreadSpectrumWatermark:
    """扩频:每个比特对应一个伪随机 ±1 序列,叠加到 DCT 中频系数;
    解码用相关性检测(符号判 0/1)。抗高斯噪声的经典范式。"""

    def __init__(self, alpha=6.0, n_coef=4096, seed=2026):
        self.alpha, self.n_coef, self.seed = alpha, n_coef, seed

    def _mid_freq_index(self, shape):
        """选取整图 DCT 的中频系数下标(避开极低频与极高频)。"""
        h, w = shape
        X = np.add.outer(np.arange(h), np.arange(w))  # 反对角线序号 ~ 频率
        order = np.argsort(X, axis=None)
        lo, hi = int(len(order) * 0.1), int(len(order) * 0.1) + self.n_coef
        return order[lo:hi]

    def _codes(self, idx_len, n_bits):
        rng = np.random.RandomState(self.seed)
        return rng.choice([-1.0, 1.0], size=(n_bits, idx_len))

    def embed(self, ch, bits):
        X = dct2(ch.astype(np.float64))
        flat = X.ravel()
        idx = self._mid_freq_index(ch.shape)
        codes = self._codes(len(idx), len(bits))
        for i, b in enumerate(bits):
            sign = 1.0 if b == 1 else -1.0
            flat[idx] += self.alpha * sign * codes[i]
        return np.clip(idct2(flat.reshape(X.shape)), 0, 255)

    def extract(self, ch, n_bits):
        X = dct2(ch.astype(np.float64))
        flat = X.ravel()
        idx = self._mid_freq_index(ch.shape)
        codes = self._codes(len(idx), n_bits)
        vals = flat[idx]
        corr = codes @ vals  # 与每个比特的扩频序列做相关
        return (corr > 0).astype(np.int64)


# =============================================================================
# 四、便捷封装:在灰度通道上嵌入/提取文本水印
# =============================================================================
#
# 说明:这些算法的标准评测都在单通道(灰度或亮度 Y)上进行。直接在 uint8 灰度
# 数组上嵌入/提取,可避免 RGB<->YCbCr 来回转换的取整噪声(那会无谓地破坏 LSB)。
# 彩色场景下,把同一套逻辑作用在 YCbCr 的 Y 通道、其余通道原样保留即可。

def embed_gray(gray: np.ndarray, method, message: str):
    """在灰度 uint8 数组上嵌入文本水印,返回 (含水印 uint8 数组, 比特数)。"""
    bits = pack_message(message)
    wm = method.embed(gray.astype(np.float64), bits)
    return np.clip(np.round(wm), 0, 255).astype(np.uint8), len(bits)


def extract_gray(gray: np.ndarray, method, n_bits: int) -> str:
    """从灰度 uint8 数组提取比特并还原文本。"""
    bits = method.extract(gray.astype(np.float64), n_bits)
    return unpack_message(bits)


# =============================================================================
# 五、Demo:对一张图跑全套方法,报告 PSNR 与提取准确率
# =============================================================================

def _demo(path: str):
    message = "DIP-2026 水印"
    print(f"输入图像: {path}")
    print(f"嵌入消息: {message!r}\n")

    img = Image.open(path).convert("L")          # 灰度评测
    w, h = img.size
    img = img.resize((w - w % 8, h - h % 8))     # 尺寸取 8 的倍数,便于分块/小波
    gray = np.array(img, dtype=np.uint8)
    bits = pack_message(message)

    methods = [
        ("LSB",        LSBWatermark()),
        ("DCT",        DCTWatermark()),
        ("DFT",        DFTWatermark()),
        ("DWT",        DWTWatermark()),
        ("SVD",        SVDWatermark()),
        ("DWT-SVD",    DWTSVDWatermark()),
        ("DWT-DCT",    DWTDCTWatermark()),
        ("SpreadSpec", SpreadSpectrumWatermark()),
    ]

    # 准备输出目录,把处理后的图像存下来
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(gray, "L").save(os.path.join(out_dir, "00_original.png"))

    def save_residual(name, wm):
        """保存水印图,以及把(水印图-原图)的差异放大 20 倍的可视化图。"""
        Image.fromarray(wm, "L").save(os.path.join(out_dir, f"{name}_watermarked.png"))
        diff = wm.astype(np.float64) - gray.astype(np.float64)
        vis = np.clip(128 + diff * 20, 0, 255).astype(np.uint8)  # 放大 20 倍 + 灰底
        Image.fromarray(vis, "L").save(os.path.join(out_dir, f"{name}_residual_x20.png"))

    print(f"{'方法':<12}{'PSNR(dB)':>10}{'比特准确率':>10}   {'文本还原'}")
    print("-" * 60)
    wm_cache = {}
    for name, m in methods:
        wm, n_bits = embed_gray(gray, m, message)
        wm_cache[name] = (m, wm)
        save_residual(name, wm)
        ext_bits = m.extract(wm.astype(np.float64), len(bits))
        acc = bit_accuracy(bits, ext_bits)
        text = unpack_message(ext_bits)
        p = psnr(gray, wm)
        ok = text if text == message else f"{text!r}"
        print(f"{name:<12}{p:>10.2f}{acc:>10.3f}   {ok}")

    # 攻击鲁棒性对比:JPEG 压缩 与 高斯噪声
    import io
    print("\n[JPEG 质量 50 压缩后的比特准确率]")
    for name, (m, wm) in wm_cache.items():
        buf = io.BytesIO()
        Image.fromarray(wm, "L").save(buf, format="JPEG", quality=50)
        buf.seek(0)
        ay = np.array(Image.open(buf).convert("L"), dtype=np.float64)
        acc = bit_accuracy(bits, m.extract(ay, len(bits)))
        print(f"  {name:<12} {acc:.3f}")

    print("\n[高斯噪声 σ=5 后的比特准确率]")
    rng = np.random.RandomState(0)
    for name, (m, wm) in wm_cache.items():
        noisy = np.clip(wm.astype(np.float64) + rng.normal(0, 5, wm.shape), 0, 255)
        acc = bit_accuracy(bits, m.extract(noisy, len(bits)))
        print(f"  {name:<12} {acc:.3f}")

    print(f"\n图像已保存到: {out_dir}")
    print("  00_original.png            原始灰度图")
    print("  <方法>_watermarked.png     加水印后的图(肉眼应与原图几乎一致)")
    print("  <方法>_residual_x20.png    水印信号放大 20 倍的可视化(能看出藏在哪)")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    _demo(target)
