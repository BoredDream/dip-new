# -*- coding: utf-8 -*-
"""图像加载 / 划分工具(numpy + Pillow)。

约定:
  * 彩色图返回 uint8 数组,形状 (H, W, 3),RGB 顺序;
  * 灰度图返回 uint8 数组,形状 (H, W)。
深度水印需要的 [-1,1] / [0,1] 归一化在各自模块内部完成,这里只管原始像素。
"""
from __future__ import annotations

import os
import glob
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def _round_to_multiple(x: int, m: int) -> int:
    return x - (x % m)


def load_image(
    path: str,
    size: Optional[int | Tuple[int, int]] = None,
    multiple_of: int = 1,
) -> np.ndarray:
    """读入彩色图为 uint8 (H, W, 3)。

    size:        None=原尺寸;int=正方形 resize;(w,h)=指定 resize。
    multiple_of: 把最终宽高裁到该数的整数倍(JPEG 分块/小波要求 8 的倍数时用)。
    """
    img = Image.open(path).convert("RGB")
    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        img = img.resize(size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)
    if multiple_of > 1:
        h, w = arr.shape[:2]
        arr = arr[: _round_to_multiple(h, multiple_of), : _round_to_multiple(w, multiple_of)]
    return arr


def load_gray(
    path: str,
    size: Optional[int | Tuple[int, int]] = None,
    multiple_of: int = 1,
) -> np.ndarray:
    """读入灰度图为 uint8 (H, W)。经典水印的标准评测在单通道上进行。"""
    img = Image.open(path).convert("L")
    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        img = img.resize(size, Image.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8)
    if multiple_of > 1:
        h, w = arr.shape[:2]
        arr = arr[: _round_to_multiple(h, multiple_of), : _round_to_multiple(w, multiple_of)]
    return arr


def to_uint8(x: np.ndarray) -> np.ndarray:
    """把任意浮点像素裁剪并四舍五入到 uint8。"""
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def save_image(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = to_uint8(arr)
    mode = "L" if arr.ndim == 2 else "RGB"
    Image.fromarray(arr, mode).save(path)


def list_images(folder: str) -> List[str]:
    """列出目录下所有图片路径(排序,便于复现)。"""
    files: List[str] = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(files))


def split_train_val_test(
    files: List[str], ratios=(0.8, 0.1, 0.1), seed: int = 2026
):
    """按比例划分训练/验证/测试集(固定种子,保证测试集模型未见过)。"""
    rng = np.random.RandomState(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return files[:n_tr], files[n_tr:n_tr + n_va], files[n_tr + n_va:]


class ImageFolder:
    """极简图像数据集:按需读取 size×size 的 uint8 彩色图。

    深度水印训练用它做迭代器;无需 torch 也能用(返回 numpy)。
    """

    def __init__(self, folder: str, size: int = 256):
        self.files = list_images(folder)
        self.size = size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> np.ndarray:
        return load_image(self.files[idx], size=self.size)
