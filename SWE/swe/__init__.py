# -*- coding: utf-8 -*-
"""SWE —— 面向 AIGC 时代的鲁棒图像水印系统。

融合两个来源:
  * guetzli (Google 的 DCT 感知 JPEG 编码器)  -> 自实现 JPEG 编解码器与 DCT 域水印
  * VINE    (ICLR 2025 潜空间扩散水印)        -> 深度潜空间水印 + 代理攻击训练

目标蓝本见仓库根目录 `鲁棒图像水印_实施方案.md`。
"""

__version__ = "0.1.0"
