# -*- coding: utf-8 -*-
"""AI 再生成攻击(主战场)—— 模块四之二。

实施方案 4.4:
  * vae_roundtrip          SD-VAE 编码->解码(最简单最有效,扩散再生成的廉价代理)
  * diffusion_img2img      扩散 img2img + 去噪强度扫描(Zhao 等的再生成攻击)
  * superres_roundtrip     降采样 -> 超分放大(Real-ESRGAN;缺库则 bicubic 回退)
  * regeneration_surrogate 纯 CPU 可跑的"再生成代理":低通(模糊)+ 轻量重采样 + JPEG,
                           近似扩散重绘在频域上的低通效应(VINE 的核心洞察:编辑≈模糊)。

真扩散/真 SD-VAE 依赖 torch + diffusers(见 requirements-deep.txt);缺失时这些函数
会抛出带安装提示的异常,而 regeneration_surrogate / superres 回退路径在纯 CPU 即可运行。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..data.datasets import to_uint8

__all__ = [
    "vae_roundtrip", "diffusion_img2img", "superres_roundtrip",
    "regeneration_surrogate", "AI_ATTACKS",
]

_PIPE_CACHE: dict = {}


def _hint(pkg: str) -> str:
    return (f"AI 攻击需要 `{pkg}`,当前环境未安装。请 `pip install -r requirements-deep.txt`,"
            f"或改用 regeneration_surrogate(纯 CPU 代理攻击)。")


# --------------------------------------------------------------------------- #
# 1) SD-VAE 往返(真实潜空间再生成的最廉价形式)
# --------------------------------------------------------------------------- #
def vae_roundtrip(img: np.ndarray, model: str = "stabilityai/sd-vae-ft-mse",
                  device: Optional[str] = None) -> np.ndarray:
    """把图过一遍冻结 SD-VAE 的 encode->decode。需要 diffusers + torch。"""
    try:
        import torch
        from diffusers import AutoencoderKL
    except Exception as e:  # noqa: BLE001
        raise ImportError(_hint("diffusers")) from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = ("vae", model, device)
    if key not in _PIPE_CACHE:
        _PIPE_CACHE[key] = AutoencoderKL.from_pretrained(model).to(device).eval()
    vae = _PIPE_CACHE[key]

    arr = to_uint8(img)
    rgb = np.stack([arr] * 3, -1) if arr.ndim == 2 else arr
    x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 127.5 - 1.0
    with torch.no_grad():
        lat = vae.encode(x.to(device)).latent_dist.sample() * vae.config.scaling_factor
        rec = vae.decode(lat / vae.config.scaling_factor).sample
    rec = ((rec.clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
    out = to_uint8(rec)
    return out[..., 0] if img.ndim == 2 else out


# --------------------------------------------------------------------------- #
# 2) 扩散 img2img(去噪强度扫描)
# --------------------------------------------------------------------------- #
def diffusion_img2img(img: np.ndarray, strength: float = 0.3, prompt: str = "",
                      steps: int = 25, guidance: float = 7.5,
                      model: str = "runwayml/stable-diffusion-v1-5",
                      device: Optional[str] = None) -> np.ndarray:
    """扩散 img2img 再生成。strength = "重绘多少",0.1(轻)->0.8(重)。需要 diffusers。"""
    if strength <= 0:
        return to_uint8(img)
    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        raise ImportError(_hint("diffusers")) from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = ("img2img", model, device)
    if key not in _PIPE_CACHE:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None)
        _PIPE_CACHE[key] = pipe.to(device)
    pipe = _PIPE_CACHE[key]

    arr = to_uint8(img)
    rgb = np.stack([arr] * 3, -1) if arr.ndim == 2 else arr
    pil = Image.fromarray(rgb)
    out = pipe(prompt=prompt, image=pil, strength=strength, guidance_scale=guidance,
               num_inference_steps=steps, generator=torch.manual_seed(42)).images[0]
    out = np.asarray(out.resize((rgb.shape[1], rgb.shape[0])))
    return np.asarray(out)[..., 0] if img.ndim == 2 else out


# --------------------------------------------------------------------------- #
# 3) 超分往返(Real-ESRGAN;缺库则 bicubic 回退)
# --------------------------------------------------------------------------- #
def superres_roundtrip(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """降采样 1/scale 再放大 scale 倍(重采样攻击)。

    当前用 PIL bicubic 放大。如需真 Real-ESRGAN:`pip install realesrgan`,在标注处用
    RealESRGANer 推理替换 bicubic 放大(权重需另行下载)。
    """
    from PIL import Image
    arr = to_uint8(img)
    H, W = arr.shape[:2]
    small = np.asarray(Image.fromarray(arr).resize((max(1, W // scale), max(1, H // scale)),
                                                    Image.BICUBIC))
    up = np.asarray(Image.fromarray(small).resize((W, H), Image.BICUBIC))  # ← 可替换为 Real-ESRGAN
    return up


# --------------------------------------------------------------------------- #
# 4) 再生成代理(纯 CPU 可跑)—— VINE 洞察:扩散编辑≈低通/模糊
# --------------------------------------------------------------------------- #
def regeneration_surrogate(img: np.ndarray, strength: float = 0.3, seed: int = 0) -> np.ndarray:
    """用"模糊 + 轻量重采样 + JPEG + 少量噪声"近似一次扩散再生成的破坏。

    strength∈[0,1] 越大破坏越强:模糊 σ 与重采样比例随之增大。这是论文 VINE 用来
    替代昂贵不可微扩散编辑的可微/廉价代理攻击的离线版本,用于在无 GPU 时也能对比
    "经典水印 vs 深度水印"在类 AI 编辑下的留存率。
    """
    if strength <= 0:
        return to_uint8(img)
    from scipy.ndimage import gaussian_filter
    from .classic import rescale, jpeg_recompress

    sigma = 0.5 + 2.5 * strength                # 0.5 -> 3.0
    factor = max(0.25, 1.0 - 0.7 * strength)    # 1.0 -> 0.3
    quality = int(max(20, 85 - 60 * strength))  # 85 -> 25

    x = img.astype(np.float64)
    if x.ndim == 2:
        x = gaussian_filter(x, sigma)
    else:
        x = np.stack([gaussian_filter(x[..., c], sigma) for c in range(x.shape[-1])], -1)
    x = rescale(x, factor).astype(np.float64)
    rng = np.random.RandomState(seed)
    x = x + rng.normal(0, 2.0 * strength, x.shape)
    return jpeg_recompress(to_uint8(x), quality)


#: AI 攻击注册表:名称 -> (函数, 参数名, 默认强度网格, 是否需深度依赖)
AI_ATTACKS: Dict[str, Tuple[Callable, str, List, bool]] = {
    "regen_surrogate": (regeneration_surrogate, "strength", [0.0, 0.1, 0.2, 0.3, 0.5, 0.8], False),
    "superres": (superres_roundtrip, "scale", [1, 2, 4], False),
    "vae_roundtrip": (vae_roundtrip, "model", ["stabilityai/sd-vae-ft-mse"], True),
    "diffusion_img2img": (diffusion_img2img, "strength", [0.0, 0.1, 0.2, 0.3, 0.4, 0.6], True),
}
