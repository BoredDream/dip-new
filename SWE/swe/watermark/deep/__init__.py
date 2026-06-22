# -*- coding: utf-8 -*-
"""模块三:深度潜空间水印(RoSteALS 蓝本 + VINE 技巧)。

依赖分层:最小可用(tiny VAE + cnn 解码器)仅需 torch;convnext 解码器另需 torchvision;
SD-VAE 路线(--vae sd)另需 diffusers。仅在被导入时才加载 torch,故纯经典实验无需深度依赖。
"""
__all__ = ["LatentWatermark", "build_latent_watermark", "load_latent_watermark"]


def __getattr__(name):  # 延迟导入,避免无 torch 环境下 import swe.watermark 报错
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
