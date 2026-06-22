# -*- coding: utf-8 -*-
"""SWE 脚本 / 实验级配置。

说明(重要):本文件供 **脚本**(`scripts/`)使用,集中存放路径、随机种子、默认样例、
深度水印架构默认与攻击扫描网格。`swe/` 包内各算法类**自带与此处一致的默认参数**,
因此直接 `import swe` 使用时无需依赖本文件;两处默认值保持同步(下方数值即与库内一致)。
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# 路径(脚本广泛使用)
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

for _d in (RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_SAMPLE = os.path.join(SAMPLES_DIR, "sample.jpg")

# --------------------------------------------------------------------------- #
# 复现性
# --------------------------------------------------------------------------- #
SEED = 2026

# --------------------------------------------------------------------------- #
# 图像 / 消息规格
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 256            # 评估默认工作分辨率
JPEG_BLOCK = 8              # JPEG / DCT 分块边长
SECRET_BITS = 100          # 深度水印负载比特数(对齐 VINE 的 100bit)
DEFAULT_QUALITY = 50       # JPEG 默认质量因子

# --------------------------------------------------------------------------- #
# 经典 QIM 水印的默认量化步长 Δ —— 与各方法类构造函数默认值一致(参考/同步用)。
# Δ 越大越鲁棒、PSNR 越低。注:
#   * SpreadSpec 用 alpha(非 Δ),LSB 无参数,故不在此列;
#   * DCT-QIM-JPEG 的 Δ 不是定值,而是按目标质量自动取 = 3 × 该位置量化步长
#     (见 DCTQIMJPEGWatermark),Q50 时约为 66。
# --------------------------------------------------------------------------- #
DELTA = {
    "DCT": 18.0,
    "DWT": 24.0,
    "SVD": 30.0,
    "DWT-SVD": 30.0,
    "DWT-DCT": 20.0,
    "DFT": 1200.0,
}
DCT_QIM_JPEG_DELTA_FACTOR = 3.0   # DCT-QIM-JPEG:Δ = 该因子 × JPEG 量化步长

# --------------------------------------------------------------------------- #
# 深度水印默认。架构键(secret_bits/vae/decoder/image_size)由 build_latent_watermark
# 在未显式传入 cfg 时读取;训练超参(下方 train_* 项)是 train_deep_watermark 的默认值镜像。
# --------------------------------------------------------------------------- #
DEEP = {
    # —— 架构(build_latent_watermark 读取)——
    "secret_bits": SECRET_BITS,
    "vae": "tiny",          # "tiny"(自带小VAE,免下载,CPU可跑) | "sd"(冻结SD-VAE,需diffusers)
    "decoder": "cnn",       # "cnn"(轻量,CPU友好) | "convnext"(VINE CustomConvNeXt 路线)
    "image_size": IMAGE_SIZE,
    # —— 训练超参(镜像 train_deep_watermark 默认值)——
    "train_lr": 1e-4,
    "train_lambda_secret": 1.0,
    "train_lambda_mse": 1.5,
    "train_lambda_lpips": 1.0,    # 无 lpips 库则自动忽略
}

# --------------------------------------------------------------------------- #
# 攻击强度扫描参考网格。注:**实际生效的网格**内置在攻击注册表
# swe.attacks.classic.CLASSIC_ATTACKS 与 swe.attacks.ai.AI_ATTACKS 中;
# 此处仅作集中参考,脚本如需自定义扫描可据此覆盖。
# --------------------------------------------------------------------------- #
SWEEP = {
    "gaussian_noise_sigma": [0, 2, 5, 10, 15, 20],
    "gaussian_blur_sigma": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "jpeg_quality": [90, 70, 50, 30, 10],
    "diffusion_strength": [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
}
