# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> ⚠️ **分支差异**:本文件描述的是 `test` / `test1` 分支的工程结构(主代码在 `SWE/`,二者结构一致;
> 当前工作分支通常是 `test1`)。`master` 分支是另一套较早的 `en-water/` 单文件实现,结构不同——
> 确认 `git branch --show-current` 后再动手。

## 这个项目是什么

数字图像处理(DIP)大作业。目标:**设计一种鲁棒图像水印,使图片经 AI 修图/再生成后仍保持高水印留存率**
(留存率 = 解码端比特准确率)。`test` 分支把两个真实项目融合落地为 `SWE/` 工程:

- **guetzli**(Google 的 DCT 感知 JPEG 编码器)→ 自实现 JPEG 九步编解码器 + DCT 域水印;
- **VINE**(ICLR 2025 潜空间扩散水印)→ 深度潜空间水印 + "攻击层入环"训练思路。

蓝本是仓库根目录 `鲁棒图像水印_实施方案.md`,把任务拆成六大模块(JPEG 编解码 / 经典+JPEG 域水印 /
深度潜空间水印 / 攻击套件 / 评估可视化 / 纠错码)。根目录其余中文 md 是配套材料:
`实现流程.md`、`优劣势.md`、`VINE项目核心实现与算法原理分析.md`、`数字图像处理_DIP_课程完整知识点总结.md`。

**算法取舍铁律**(贯穿全工程):以实施方案为目标蓝本;**算法对不上时以 guetzli / VINE 两个真实项目为准**;
缺失的实验自行补齐。

**当前进度**(按实施方案第八节 P1–P4 里程碑):**P1–P4 主链路已打通**。
- **P1 频带诊断**:`swe/eval/diagnose.py` + `scripts/09` + `results/diagnose/freq_survival.png`。
- **P2 主创新 `ImprovedDwtDct`**(`swe/watermark/improved_dwtdct.py`):多低频带(默认 #1–7)QIM + 纹理自适应
  + Watson JND,合成步长 `Δ_eff=δ0·texture_gain·jnd_weight·luminance_mask`,三开关可独立消融;
  自适应步长只用嵌入后稳定统计量(块均值=DC、非载体高频能量)→ 盲提取干净准确率=1.0(`tests/test_improved_dwtdct.py`)。
- **P3 对比阶梯 + 消融四档**:`scripts/07` adapters = SpreadSpec(下界)/ DwtDct-default / Imp+multiband /
  Imp+texture / Imp+JND(主角)/ DCT-QIM-JPEG / Deep-Latent(上界,`--include-deep`)→ `results/attack_suite.csv`。
- **P4 图表**:`scripts/08` 出鲁棒表 / 强度曲线 / 攻防权衡 / **消融曲线**(`swe/eval/plots.py::plot_ablation`,
  档名见 `ABLATION_TIERS`);分析见 `docs/EXPERIMENTS.md` §6 与 `docs/EVALUATION.md` §2.2。

- **真扩散攻击(本地 GPU)**:`scripts/07 --ai-img2img` 用本地 SD1.5 img2img 跑 held-out 再生成
  (`swe/attacks/ai.py::diffusion_img2img`,默认模型 `stable-diffusion-v1-5/stable-diffusion-v1-5`,可用
  环境变量 `SD_IMG2IMG_MODEL` 覆盖)→ `results/attack_suite_ai.csv` + `sweep_diffusion_img2img.png`。

**诚实口径(已含真扩散结论)**:消融逐档抬升(0.65→0.76→0.82,JND 档鲁棒性持平、换的是可见性)是针对
**压缩/模糊/CPU 代理**的;**真扩散 img2img 下所有 QIM 类(默认+改进三档+DCT-QIM-JPEG)在 strength=0.1 即集体
跌到 ~0.5(死),只有 Deep-Latent 幸存 ~0.84**(真 held-out,tiny 模型没见过真扩散)。即改进版增益对低通退化成立、
对真生成式重绘通杀——这正是"为何仍需深度水印"的硬证据。绝对数值仍 PoC(tiny-VAE/3 图),论文级需冻结 SD-VAE
充分训练 + ~100 张 DIV2K/COCO。详见 `docs/EXPERIMENTS.md §6/§6.1`、`docs/EVALUATION.md §2.2/§4`。

> 本机依赖装在 `SWE/.venv`(系统 Python 为 PEP 668 externally-managed);跑脚本用 `.venv/bin/python …`。
> torch 为 **CUDA 版**(`cu124`,RTX 4060 可用);SD1.5 权重已缓存到 `~/.cache/huggingface`(~5GB)。

## 常用命令

所有命令在 `SWE/` 目录下运行。**本机只有 `python3`(无 `python` 别名),且依赖尚未安装**——先装依赖。

```bash
cd SWE
pip install -r requirements.txt        # 经典全链路(纯 CPU:numpy/scipy/skimage/Pillow/matplotlib)
pip install -r requirements-deep.txt   # 深度水印 + 真 AI 攻击(可选,见下"分层依赖")

python3 -m pytest tests/ -q            # 单元测试(经典链路;需先装 requirements.txt)
python3 -m pytest tests/test_codec.py -q                 # 跑单个文件
python3 -m pytest tests/test_codec.py::test_xxx -q       # 跑单个用例

# 9 个 demo 脚本(各自末尾"自动判定":结论由实测算出、非预设)
python3 scripts/01_test_jpeg_codec.py            # 自实现 JPEG vs PIL 校准
python3 scripts/02_demo_classic_watermarks.py    # 8 种经典水印按三攻击均值自动分档
python3 scripts/03_demo_dct_qim_jpeg.py          # DCT-QIM 嵌 JPEG 流水线 + ECC
python3 scripts/04_demo_fragile_tamper.py        # 脆弱/半脆弱篡改定位
python3 scripts/05_train_deep_watermark.py --smoke   # 深度水印 CPU 冒烟训练(~3 分钟,需 torch)
python3 scripts/06_demo_deep_watermark.py        # 深度 vs DWT-SVD 经典基线 同攻击对照
python3 scripts/07_run_attack_suite.py --include-deep # 方法×攻击×强度全套 → results/attack_suite.csv
python3 scripts/08_make_report_figures.py        # 由 CSV 生成三张核心图表
python3 scripts/09_diagnose_frequency_survival.py    # 频带存活诊断
```

`scripts/05/06`、`scripts/07 --include-deep` 必需 `torch`;其余脚本(`01–04`、不带 `--include-deep` 的 `07`、`08`)
只靠 `requirements.txt` 即可。脚本统一用 `_bootstrap.py` 把根目录加进 `sys.path`(故可 `import swe` / `import config`),
新脚本第一行应 `import _bootstrap`。

## 架构大图

数据流:`原图 + 个人信息m(经 ECC 编码)` →【经典:JPEG 编码时 DCT 域 QIM 嵌入】或【深度:VAE 潜空间嵌入】
或【脆弱:块级校验】→ 攻击套件 → 提取器得 m' / 篡改热图 → 评估(PSNR/SSIM/LPIPS + 比特准确率/BER + IoU/F1)。

`swe/` 包按六模块组织,理解时抓这几个**跨文件的关键约定**:

1. **统一水印接口 `swe/watermark/base.BaseWatermark`**:所有经典/JPEG 域方法实现
   `embed(channel, bits) -> 含水印通道` 与 `extract(channel, n_bits) -> bits`,只在**单通道浮点数组(0..255)**上操作。
   变换域方法**共用 QIM**(量化索引调制),差异仅在选哪个载体系数;鲁棒性主要靠各方法构造函数的 `delta`(步长越大越鲁棒、PSNR 越低,默认值集中在 `config.DELTA`)。新增经典方法照此签名即可接入。

2. **"在 JPEG 压缩过程中嵌水印"的精确落点**:`swe/codec/jpeg.JPEGCodec.encode(image, coeff_hook=...)`
   在**第 4 步 DCT 之后、第 5 步量化之前**对亮度 Y 的未量化系数回调 `coeff_hook`。
   `swe/watermark/dct_qim_jpeg.py` 就是用它在 Y 中频系数上做 QIM(Δ≈3× 量化步长 → 天然抗同质量再压缩),
   这是**核心基线**。它有两个入口要分清:`embed/extract`(评估用)与 `embed_in_jpeg`(真走流水线);
   对应两种 PSNR 口径——水印扰动本身 ≈43dB,`embed_in_jpeg` 真实输出 ≈30dB(含 JPEG Q50 损失)。

3. **深度潜空间水印 `swe/watermark/deep/`**:`build_latent_watermark(cfg, device)` 组装
   VAE 主干(`vae_backbone`,`tiny` 自带小 VAE 免下载 / `sd` 冻结 SD-VAE 需 diffusers)+ 秘密编码器/解码器
   (`secret_encoder`/`secret_decoder`,解码器 `cnn` 轻量 / `convnext` 走 VINE CustomConvNeXt)+ **攻击层
   `attack_layer`(VINE 灵魂:模糊/噪声/重采样/VAE 往返入环,把模糊当 AI 编辑的可微代理)**。
   `tiny` VAE 仅供 CPU 冒烟跑通,论文级结果需 `--vae sd` + GPU + 真数据集。

4. **评估驱动靠两个适配器** `swe/eval/runner.py`:`ChannelAdapter` 把经典水印嵌到 RGB 图的**绿色通道**
   (亮度主成分,刻意不走 YCbCr 以免取整破坏 LSB);`ImageAdapter` 把深度水印作用于整张 RGB。
   `run_experiments` 用它们统一跑 方法×攻击×强度 并记录均值±标准差。

5. **攻击套件 `swe/attacks/`**:`classic.py`(噪声/模糊/中值/JPEG 再压缩/裁剪/缩放/旋转,各带强度网格)
   与 `ai.py`(`vae_roundtrip`/`diffusion_img2img` 需 diffusers;`regeneration_surrogate` 是纯 CPU 代理)。
   **实际生效的扫描网格在 `CLASSIC_ATTACKS` / `AI_ATTACKS` 注册表里**,`config.SWEEP` 仅作参考。

6. **纠错码 `swe/ecc/codec.py`**:纯 Python 实现 RS(GF(2^8))/汉明(7,4)/重复码,无第三方依赖
   (有 `reedsolo` 则优先)。

## 关键约定与注意事项

- **分层可选依赖,缺失自动回退**:只跑 CPU 冒烟仅需 `torch`;SD-VAE 缺失→tiny VAE,真扩散攻击缺失→
  `regen_surrogate` 代理,LPIPS 缺失→MSE。改深度/攻击代码时保持这种"缺库不报错、给提示并降级"的风格。
- **结论一律实测自动判定,不预设**:demo 脚本末尾的"弱/强""明显/不明显"由当次实测算出。
  当前已知诚实结论:**冒烟阶段深度 vs 经典在纯 CPU 代理攻击下留存率相当(~0.8),并未拉开差距**——
  VINE"抗 AI 编辑"的优势要真扩散(GPU)或充分训练的冻结 SD-VAE 才显现。改文案/报告时勿夸大。
- **`config.py` 的双份默认**:`swe/` 包内各算法类自带与 `config.py` 一致的默认参数,直接 `import swe` 无需依赖
  `config`;改默认值时**两处保持同步**。
- **生成物不入库**:`results/figures/`、`results/**/*.csv|json`、`checkpoints/`、`*.pth`、数据集(`data/div2k/`、
  `DIV2K_train_HR/`)均被 gitignore;`results/` 下只保留少量演示用 PNG 与 `.gitkeep`。

## 外部参考代码

- `references/guetzli/`(<1MB,入库备查):Google guetzli C++ 源码,JPEG/DCT 的算法出处。
- `references/VINE-main/`(被 gitignore,本地约 74MB):VINE 官方实现,深度水印思路出处。
- `VINE/`:`master` 分支带来的嵌套独立 git 仓库,在 `test` 分支上为**未跟踪**残留目录,通常不动它。

模块→代码→算法出处的完整映射见 `SWE/docs/ARCHITECTURE.md`;复现步骤与预期数值见 `SWE/docs/EXPERIMENTS.md`;
工程总览见 `SWE/README.md`。
