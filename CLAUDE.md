# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 本仓库是一个数字图像处理(DIP)课程项目。课题目标(见 `statement.md`):**自行设计一种添加水印的方法,使图片经 AI 修图后仍能保持较高的水印留存率**,留存率通过解码端检测水印的比特准确率来衡量。仓库由两部分组成:`en-water/`(我们自己写的传统信号处理水印实现)与 `VINE/`(我们找到的同类参考项目,ICLR 2025 论文官方实现,作为对照基线与思路来源)。

## 仓库结构(重要)

- `en-water/` —— **本项目自己的代码**,主要改动应集中在这里。
- `VINE/` —— **嵌套的独立 git 仓库**(自带 `.git`,非 submodule),是外部参考实现。改动 VINE 内部逻辑前先读 `VINE/CLAUDE.md`,它有专门的环境/架构说明;通常只作参考、不修改。
- `statement.md` —— 课题陈述。

## en-water:传统水印方法合集

`en-water/traditional_watermark.py` 把 8 种经典不可见水印方法实现在**单个文件**里,仅依赖 `numpy + Pillow`(DCT、Haar 小波、SVD、DFT 均自行实现,刻意不引入 opencv / pywt / scipy)。

运行 demo(对一张图跑全套方法,输出 PSNR、比特准确率,以及 JPEG/高斯噪声攻击后的鲁棒性对比):

```shell
cd en-water
python traditional_watermark.py            # 默认对 sample.jpg
python traditional_watermark.py xxx.png    # 指定图片
```

结果图写入 `en-water/output/`:`00_original.png`、`<方法>_watermarked.png`(加水印图)、`<方法>_residual_x20.png`(水印信号放大 20 倍的可视化)。无测试套件,验证改动即靠跑这条 demo 看准确率/PSNR。

### 关键设计约定

- **统一接口**:每个方法是一个类,实现 `embed(channel, bits) -> 含水印通道` 与 `extract(channel, n_bits) -> 比特`。新增方法照此签名即可接入 demo 的方法列表。
- **统一嵌入逻辑 QIM**:除 LSB 和扩频外,变换域方法都用 QIM(量化索引调制)在某个"载体系数"上嵌 1 比特/单元——各方法的差异只在于选哪个系数(DCT 中频、HL 子带、最大奇异值、DFT 幅度谱中频环等)。改鲁棒性主要靠调各方法构造函数里的 `delta`(量化步长,越大越鲁棒但 PSNR 越低)。
- **单通道操作**:所有方法只在亮度/灰度通道上操作。彩色场景的约定是作用在 YCbCr 的 Y 通道、其余通道原样保留;demo 直接用灰度以避免 RGB↔YCbCr 取整噪声破坏 LSB。
- **消息格式**:`pack_message` 打包成 `[32 位长度头 | payload]`,`unpack_message` 据长度头还原文本;`text_to_bits`/`bits_to_text` 走 UTF-8。
- **分块前提**:变换域/分块方法要求边长为偶数或 8 的倍数,demo 里先把图 resize 到 8 的倍数(`_crop_even` 也会裁掉余数)。

## VINE(参考实现,详见 VINE/CLAUDE.md)

VINE 用扩散模型生成先验做水印嵌入,核心思路:**用"模糊"作为 AI 编辑的代理攻击**联合训练编码器+攻击层+解码器,使水印能扛过一次扩散重绘——这正是本课题想借鉴的方向。其完整环境(conda `vine`,固定 `torch==2.0.1` 等)、运行命令、架构(`VINE_Turbo` 编码器 / `CustomConvNeXt` 解码器)和被 vendored 的外部代码(`diffusers/`、`saicinpainting/`)说明都在 `VINE/CLAUDE.md` 与 `VINE/概要.md` 中,需要时再深入。
