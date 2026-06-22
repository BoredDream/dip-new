# SWE —— 面向 AIGC 时代的鲁棒图像水印系统

> 数字图像处理(DIP)大作业工程实现。把 **guetzli**(基于 DCT 的 JPEG 压缩)与
> **VINE**(ICLR 2025 潜空间扩散水印)两个项目融合,严格落地 `鲁棒图像水印_实施方案.md`
> 的六大模块:自实现 JPEG 编解码器 + 经典/JPEG 域水印 + 学习型潜空间深度水印 +
> 攻击套件(经典 + AI 再生成)+ 评估可视化 + 纠错码,并补齐篡改定位(脆弱水印)。

一句话定位:做一个 **"在 JPEG 压缩过程中嵌入隐形水印、并能扛住 AI 再生成攻击"** 的
图像版权保护系统,量化对比 **经典 DCT 水印(抗 JPEG 但遇扩散再生成即死)** 与
**深度潜空间水印(随内容穿过再生成而存活)**,并用脆弱水印定位 AI 篡改区域。

---

## 1. 这个项目融合了什么

| 来源项目 | 角色 | 在 SWE 中对应 |
|---|---|---|
| **guetzli**(Google,C++ 感知 JPEG 编码器) | DCT/JPEG 压缩参考 | 模块一 `swe/codec/`(自实现九步 JPEG)+ 模块二 DCT-QIM 嵌入流水线 |
| **VINE**(`dip/VINE-main/`,潜空间扩散水印) | 深度水印 + 代理攻击思路 | 模块三 `swe/watermark/deep/`(RoSteALS 架构 + VINE 的"攻击层入环"训练) |
| **en-water**(课题已有的传统水印实现) | 经典基线 | 模块二 `swe/watermark/classic.py`(8 种经典方法,并修正了 DFT 缺陷) |
| **实施方案.md** | 蓝本/目标 | 全部六模块的设计与实验依此落地 |

> 算法取舍原则(按需求约定):**以实施方案为目标蓝本;具体算法对不上时以两个真实项目为准;
> 缺失的实验自行补齐**(例如再生成代理攻击、纠错码信息恢复率、篡改定位 IoU/F1)。

---

## 2. 目录结构

```
SWE/
├── config.py                 全局配置(路径/尺寸/种子/默认超参/扫描网格)
├── requirements.txt          核心依赖(纯 CPU 可跑经典全链路)
├── requirements-deep.txt     深度水印 + 真 AI 攻击依赖(torch/diffusers/lpips...)
├── swe/                       主包
│   ├── codec/                模块一:自实现 JPEG 九步编解码器(参考 guetzli)
│   │   ├── color.py  dct.py  quant.py  zigzag.py  huffman.py  jpeg.py
│   ├── watermark/            模块二/三:水印方法
│   │   ├── classic.py        8 种经典水印(LSB/DCT/DFT/DWT/SVD/DWT-SVD/DWT-DCT/扩频)
│   │   ├── dct_qim_jpeg.py   DCT-QIM 嵌入 JPEG 流水线(核心基线)
│   │   ├── fragile.py        脆弱/半脆弱水印 -> 篡改定位
│   │   └── deep/             学习型潜空间水印(RoSteALS + VINE)
│   │       ├── vae_backbone.py  secret_encoder.py  secret_decoder.py
│   │       ├── attack_layer.py  model.py  train.py
│   ├── ecc/codec.py          模块六:RS / 汉明 / 重复码(无第三方依赖)
│   ├── attacks/              模块四:classic.py(经典失真) + ai.py(AI 再生成)
│   ├── eval/                 模块五:metrics.py  runner.py  plots.py
│   └── data/datasets.py      图像加载/划分
├── scripts/                  01..08 入口脚本(见下)
├── tests/                    pytest 单元测试(39 项)
├── data/samples/             样例图(sample.jpg, bees.png)
└── docs/                     ARCHITECTURE.md  EXPERIMENTS.md
```

---

## 3. 安装

```bash
# 经典全链路(JPEG 编解码 + 经典/JPEG 域水印 + 脆弱水印 + 经典攻击 + 评估 + ECC):纯 CPU
pip install -r requirements.txt

# 深度潜空间水印 + 真 AI 攻击(可选,按功能分层):
pip install -r requirements-deep.txt
```

> **深度依赖是分层的,按需安装**(详见 `requirements-deep.txt`):
> | 功能 | 需要的库 |
> |---|---|
> | 深度水印最小可用(`--vae tiny --decoder cnn`,CPU 冒烟) | **仅 torch** |
> | ConvNeXt 解码器(`--decoder convnext`) | + torchvision |
> | LPIPS 感知损失/指标(可选) | + lpips(缺失自动用 MSE) |
> | 冻结 SD-VAE(`--vae sd`)/ 真扩散·VAE 往返攻击 | + diffusers transformers accelerate |
>
> 即:**只跑 CPU 冒烟仅需 `torch`**;其余均为按功能可选,缺失时自动回退(SD-VAE→tiny VAE,
> 真扩散攻击→`regen_surrogate` 代理,LPIPS→MSE)。纠错码完全自带纯 Python 实现,无需任何库。
>
> ⚠️ **`torch` 是所有深度水印流程的必需依赖**(不是可选):未安装 torch 时,`scripts/05`、`scripts/06`、
> 以及 `scripts/07 --include-deep` 会在 `import torch` 处直接失败。而**经典链路完全不需要 torch**:
> `scripts/01–04`、`scripts/07`(不带 `--include-deep`)、`scripts/08` 仅靠 `requirements.txt` 即可运行。

---

## 4. 快速上手(8 个脚本)

```bash
cd SWE
python scripts/01_test_jpeg_codec.py          # ① 自实现 JPEG vs PIL 校准(PSNR/bpp)
python scripts/02_demo_classic_watermarks.py  # ② 8 种经典水印:PSNR + 干净/攻击后准确率
python scripts/03_demo_dct_qim_jpeg.py        # ③ DCT-QIM 嵌入 JPEG 流水线 + ECC 信息恢复
python scripts/04_demo_fragile_tamper.py      # ④ 脆弱/半脆弱水印篡改定位(出四联热图)
python scripts/05_train_deep_watermark.py --smoke   # ⑤ 训练深度水印(CPU 冒烟,~3 分钟)
python scripts/06_demo_deep_watermark.py      # ⑥ 深度水印 embed/extract + 鲁棒性快测
python scripts/07_run_attack_suite.py --include-deep # ⑦ 方法×攻击×强度全套实验 -> CSV
python scripts/08_make_report_figures.py      # ⑧ 由 CSV 生成三张核心图表
```

正式训练(需 GPU + diffusers,RoSteALS 正路):
```bash
python scripts/05_train_deep_watermark.py --data-dir /path/to/DIV2K \
       --vae sd --decoder convnext --image-size 256 --secret-bits 100 --steps 20000 --device cuda
```

---

## 5. 模块 → 实施方案 对应

| 实施方案模块 | 代码 | 状态(本机 CPU 已验证) |
|---|---|---|
| 4.1 JPEG 九步编解码器 | `swe/codec/` | ✅ 与 PIL/libjpeg PSNR 差 ≈0(Q≤90) |
| 4.2 经典水印(LSB/DCT-QIM/...) | `swe/watermark/classic.py`, `dct_qim_jpeg.py` | ✅ 干净 1.00;DCT-QIM 抗同质量再压缩 1.00 |
| 4.2.3 脆弱/半脆弱(篡改定位) | `swe/watermark/fragile.py` | ✅ 脆弱精确定位;半脆弱容忍 JPEG80 误报 0% |
| 4.3 深度潜空间水印(RoSteALS+VINE) | `swe/watermark/deep/` | ✅ tiny VAE CPU 冒烟:bit_acc→0.9+,抗再生成代理攻击 |
| 4.4 攻击套件(经典 + AI 再生成) | `swe/attacks/` | ✅ 经典 + 代理可跑;真扩散需 diffusers |
| 4.5 评估与可视化(三张核心图) | `swe/eval/` | ✅ 鲁棒性表/强度曲线/权衡曲线/篡改热图 |
| 4.6 纠错编码(BCH/RS) | `swe/ecc/` | ✅ RS 纠 5 字节错;汉明纠 1/块;重复码 |

---

## 6. 已验证的关键结果(本机 CPU)

- **JPEG 编解码器**:灰度 Q50 PSNR=32.81 dB / bpp=0.617,与 PIL **逐 Q 一致**(ΔPSNR≈0.01)。
- **经典水印**:8 种干净提取均 1.00;DFT 修正后由 ~0.46 提升到 1.00。
- **DCT-QIM 嵌入 JPEG 流水线**:两种口径要分清 —— **水印扰动本身** PSNR≈43.5 dB(`embed()` 含水印图 vs 原图、不含压缩);
  **`embed_in_jpeg()` 真实输出** PSNR≈30 dB(含 JPEG Q50 压缩损失,≈Q50 基线画质),bpp≈0.74;**同质量 JPEG 再压缩后比特准确率 1.00**。
- **半脆弱水印**:benign JPEG70–90 误报 0%,可定位平涂/自然重绘(~55/64 块)。
- **Reed–Solomon**:nsym=10 时纠正 ≤5 字节错 100%,6 字节失败(符合 t=5 理论上限)。
- **深度潜空间水印(tiny VAE,128px,~400 步)**:训练 bit_acc 由 ~0.5 升至 **0.93–1.0**(随种子/步数波动);
  推理对 JPEG / 再生成代理攻击留存率 **≈1.0**、对模糊/噪声 ≈0.97 —— 而经典法在再生成代理下跌向 ~0.5,
  复现 VINE "攻击层入环 → 抗 AI 编辑" 的核心结论。
- **单元测试**:`pytest` 39 项全过。

> 完整复现步骤与预期数值见 `docs/EXPERIMENTS.md`;设计/算法/出处映射见 `docs/ARCHITECTURE.md`。

---

## 7. 设计要点(从两个项目"抄"对的地方)

1. **JPEG 内嵌水印**:DCT-QIM 通过 `JPEGCodec` 的 `coeff_hook` 在量化前写入 Y 中频系数,
   步长 Δ≈3× 量化步长 → 天然抗同质量再压缩(实施方案 4.2.2)。
2. **模糊/VAE 往返作为 AI 编辑的代理攻击**:深度水印训练时把可微的模糊、重采样、
   **VAE 往返**塞进攻击层(VINE 的灵魂),逼解码器学会抗再生成(实施方案 4.3)。
3. **任意分辨率残差技巧**:深度水印在 256(或 128)算含水印残差,再上采样回原分辨率叠加
   (VINE `watermark_encoding.py` 思路),省显存保高清。
4. **纠错码**:像 VINE 的 BCH 那样给负载加冗余,把"差几位"纠回成"信息完整恢复"。

---

## 8. 注意事项

- `tiny` VAE 是**联合训练**的轻量自编码器,用于无 GPU/无 diffusers 时跑通全流程与冒烟;
  论文级画质/鲁棒性请用 `--vae sd`(冻结 SD-VAE,RoSteALS 正路)。
- 真扩散 img2img / SD-VAE 往返攻击需 `diffusers`,缺失时这些攻击会给出安装提示并可用
  `regen_surrogate` 替代。
- 评估为"已知消息"式比特准确率(可追溯水印),与 VINE/经典基线一致。
```
