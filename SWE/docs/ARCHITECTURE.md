# SWE 架构说明:实施方案 → 代码 的映射与设计决策

本文把 `鲁棒图像水印_实施方案.md` 的每个模块映射到具体代码,标注其**算法出处**
(guetzli / VINE / en-water),并记录关键工程决策。

---

## 0. 数据流总览

```
原图 x + 个人信息 m(经 ECC 编码)
   ├─[路线A 经典] JPEG 编码 DCT 域 QIM 嵌入 ──► 含水印 JPEG 流/图   (codec + dct_qim_jpeg)
   ├─[路线B 深度] VAE 潜空间嵌入再解码 ───────► 含水印图           (deep/model)
   └─[脆弱水印]  空域/低频块级校验 ────────────► 可篡改定位        (fragile)
        │  攻击套件(attacks/classic + attacks/ai)
        ▼
   提取器 D ─► m' ; 脆弱校验 ─► 篡改热图
        │  评估(eval/metrics + runner + plots)
        ▼  PSNR/SSIM/LPIPS(不可见性) + 比特准确率/BER(鲁棒性) + IoU/F1(定位)
```

---

## 1. 模块一:JPEG 编解码器(`swe/codec/`)—— 出处:guetzli

| 步骤 | 文件 | 说明 |
|---|---|---|
| ① RGB↔YCbCr,② 4:2:0 | `color.py` | BT.601 全量程定义;色度 2×2 box 平均 |
| ③ 分块 + 电平移位 −128 | `jpeg.py` | 子采样时整体补到 16 的倍数,保证色度网格是 8 的倍数 |
| ④ 8×8 DCT | `dct.py` | 矩阵版 `D·f·Dᵀ`(浮点正交 DCT-II);guetzli `fdct.cc` 为带缩放的快速整数 DCT,二者算同一种变换、数值尺度不同 |
| ⑤ 量化 | `quant.py` | 标准附录 K 量化表 + libjpeg 质量缩放 `S=(Q<50)?5000/Q:200−2Q` |
| ⑥ Zigzag | `zigzag.py` | 程序生成之字形序;⑦ DC-DPCM / AC-RLE(ZRL/EOB) |
| ⑧ Huffman | `huffman.py` | 标准附录 K.3–K.6 表,规范前缀码 + 幅度分类附加比特 |
| ⑨ 解码 | `jpeg.py` | 上述逆过程 |

**关键决策**
- **不写 JFIF 文件头/字节填充**,只产出自有容器的熵编码比特流 → 用于度量真实 bpp 与跑通
  完整链路;与 libjpeg 的"标准一致性"通过脚本 01 的 PSNR/bpp 对比校准(实施方案风险第 3 条)。
- **水印挂钩 `coeff_hook`**:在第 4 步 DCT 之后、第 5 步量化之前对**亮度 Y** 的未量化系数
  回调,这就是"在 JPEG 压缩过程中嵌入水印"的精确落点。
- **修复**:`rle_encode_ac` 仅在尾部有零时发 EOB —— 否则满块(第 63 个 AC 非零)会令解码端
  多读一个符号而整图错位(高质量下尤甚)。

## 2. 模块二:经典 + JPEG 域 + 脆弱水印(`swe/watermark/`)—— 出处:en-water + VINE

- `classic.py`:LSB / DCT / DFT / DWT(Haar)/ SVD / DWT-SVD / DWT-DCT / 扩频,统一 `BaseWatermark`
  接口,变换域方法共用 QIM。**修正 DFT**:不做 fftshift,在原始频谱上取共轭对
  `(u,v)` 与 `((H−u)%H,(W−v)%W)`,使干净图准确率从 ~0.46 → 1.00。
- `dct_qim_jpeg.py`:**核心基线**。Δ≈3× 量化步长抗再压缩;重复 R + 多数表决降 BER;
  密钥伪随机选块。两种入口:`embed/extract`(评估接口)与 `embed_in_jpeg`(真走流水线)。
- `fragile.py`:`FragileWatermark`(Wong 式,块 MSB 哈希写 LSB,任何改动即碎)与
  `SemiFragileWatermark`(**密钥绑定**的认证码经 QIM 嵌中频,Δ 大→容忍 benign JPEG、
  重绘即破碎)。后者刻意不依赖块内容,避免"伪造自洽块"绕过检测。

## 3. 模块三:深度潜空间水印(`swe/watermark/deep/`)—— 出处:VINE + RoSteALS 蓝本

| 组件 | 文件 | 对应 |
|---|---|---|
| VAE 主干(冻结 SD-VAE / 可训 tiny) | `vae_backbone.py` | RoSteALS 冻结 VAE;tiny 为免下载 CPU 版 |
| 秘密编码器 E_s(消息→潜残差 Δz) | `secret_encoder.py` | VINE ConditionAdaptor 思路;输出层较小初始化(默认 0.1;冻结 VAE 时可调更小近零起步,对应 VINE skip 1e-5) |
| 秘密解码器 D_s(图→比特) | `secret_decoder.py` | `cnn` 轻量 / `convnext`(VINE CustomConvNeXt) |
| 攻击层 N(模糊/噪声/重采样/**VAE 往返**) | `attack_layer.py` | VINE TransformNet 的灵魂:模糊作为编辑代理 + 课程式渐进 |
| 嵌入/提取 + 任意分辨率残差技巧 | `model.py` | VINE `watermark_encoding.py` 残差缩放 |
| 课程式训练(BCE+MSE+LPIPS) | `train.py` | RoSteALS 课程:先收敛比特、再加攻击与画质损失 |

**关键决策**
- 架构以**实施方案选定的 RoSteALS**(冻结 VAE + 小 E_s/D_s)为目标,用 **VINE 的具体技巧**
  (攻击层入环、VAE 往返代理、残差缩放、ConvNeXt 解码、近零初始化)填充细节 —— 即"算法对不上
  以两个项目为准"。
- **tiny VAE 需始终在线的重建损失**:随机初始化的 VAE 解码 `tanh` 一开始饱和会切断梯度,
  故对可训练 VAE 加 `MSE(Ψ(Φ(x)), x)`;冻结 SD-VAE 不需要(已预训练)。这是让 CPU 冒烟能学起来的关键。

## 4. 模块四:攻击套件(`swe/attacks/`)

- `classic.py`:高斯/椒盐噪声、高斯模糊、中值、**JPEG 再压缩(复用模块一)**、裁剪、缩放、旋转;
  每个带强度网格供扫描。
- `ai.py`:`vae_roundtrip`/`diffusion_img2img`(diffusers,缺失给安装提示)、`superres`(Real-ESRGAN→bicubic 回退)、
  `regeneration_surrogate`(纯 CPU 代理:模糊+重采样+JPEG+噪声,近似扩散重绘的低通效应,落地 VINE 洞察)。

## 5. 模块五:评估(`swe/eval/`)

- `metrics.py`:PSNR/SSIM/LPIPS(guarded)、比特准确率/BER、IoU/F1。
- `runner.py`:`ChannelAdapter`(经典→**绿色通道**,亮度主成分;不走 YCbCr 以免取整破坏 LSB)
  与 `ImageAdapter`(深度→RGB)统一驱动,
  跑方法×攻击×强度,记录均值±标准差及攻击后画面 SSIM/PSNR。
- `plots.py`:鲁棒性总表、强度扫描曲线、攻防权衡曲线(标"危险区")、篡改定位四联图。

## 6. 模块六:纠错码(`swe/ecc/codec.py`)

- 纯 Python:`ReedSolomonECC`(GF(2^8),Berlekamp–Massey + Chien + Forney,fcr=0/gen=2)、
  `HammingECC(7,4)`、`RepetitionECC`;有 `reedsolo` 库则优先。编码/解码用**一致分块**(末块可短)。

---

## 7. 与原始论文/项目的差异与诚实声明

- VINE 实际加载 `stabilityai/sd-turbo`(其摘要写 SDXL-Turbo);本项目深度水印取 RoSteALS 的
  **冻结 VAE** 路线(更轻、实施方案指定),非复刻 VINE 的一步扩散全微调。
- 真实扩散编辑不可微,故深度水印的抗编辑能力主要靠**可微代理攻击**(模糊/VAE 往返)在训练中习得,
  这正是"代理攻击"存在的根本原因(与 VINE 一致)。
- `tiny` VAE 仅供流程验证;论文级结果需冻结 SD-VAE + 较长训练 + 真实数据集(DIV2K/COCO)。
