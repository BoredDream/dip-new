# SWE 复现指南与预期结果

所有命令在 `SWE/` 目录下运行。经典链路纯 CPU 即可;深度部分 CPU 可冒烟、GPU 出正式结果。

---

## 0. 环境自检

```bash
pip install -r requirements.txt          # 经典链路
python -m pytest tests/ -q               # 期望:60 passed(含 test_improved_dwtdct 10 项)
```

---

## 1. JPEG 编解码器校准(脚本 01)

```bash
python scripts/01_test_jpeg_codec.py
```
**预期**:灰度各质量因子下,本项目 PSNR 与 PIL 的差 `ΔPSNR≈0`(Q≤90);bpp 随 Q 单调下降。
例:Q50 → PSNR≈32.8 dB,bpp≈0.62;Q90 → PSNR≈42.0 dB。

## 2. 经典水印对比(脚本 02)

```bash
python scripts/02_demo_classic_watermarks.py
```
**预期**:8 种方法干净准确率均 1.00(文本完整还原)。攻击后鲁棒性由 `02` 末尾**自动判定**
按"三攻击(JPEG50/噪声σ10/模糊1.5)均值"实测分档(<0.65=弱基线 / ≥0.85=鲁棒),**不预设结论**——
实测倾向:LSB/DCT/DFT 等接近随机(弱基线)、扩频最鲁棒,但**具体分档以脚本实际打印为准**。结果图写入 `results/classic/`。

## 3. DCT-QIM 嵌入 JPEG + ECC(脚本 03)

```bash
python scripts/03_demo_dct_qim_jpeg.py
```
**预期**(注意两种 PSNR 口径):水印扰动本身 PSNR≈43 dB(`embed()` 含水印图 vs 原图、不含压缩);
`embed_in_jpeg()` 真实输出 PSNR≈30 dB(含 JPEG Q50 压缩损失)、bpp≈0.74;**同质量再压缩后准确率 1.00**;
ECC 部分:强攻击(JPEG30+噪声)下,无 ECC 常无法完整恢复 ID,RS 纠错后可完整恢复。

## 4. 篡改定位(脚本 04)

```bash
python scripts/04_demo_fragile_tamper.py
```
**预期**:脆弱水印 clean 误报 0、对篡改 IoU 高;半脆弱 benign JPEG80 误报≈0% 且能定位重绘。
四联热图写入 `results/tamper/`。

## 5. 深度潜空间水印(脚本 05/06)

```bash
python scripts/05_train_deep_watermark.py --smoke      # CPU ~3 分钟,存 checkpoints/deep_wm.pth
python scripts/06_demo_deep_watermark.py               # 加载并测鲁棒性
```
**预期(tiny VAE,128px,32bit)**:训练 bit_acc 由 ~0.5 升到 1.0;推理对 JPEG/模糊/
再生成代理攻击准确率显著高于经典基线(冒烟设置下常达 ~1.0)。结果图写入 `results/deep/`。

正式训练(GPU + diffusers):
```bash
python scripts/05_train_deep_watermark.py --data-dir /path/to/DIV2K \
       --vae sd --decoder convnext --image-size 256 --secret-bits 100 \
       --steps 20000 --batch-size 8 --device cuda
```

## 6. 对比阶梯 + 消融四档 + 核心图(脚本 07/08)

`07` 按实施方案 §3.2/§六 的**对比阶梯**组织(下界<改进前<改进版<上界)并内置改进版**消融四档**:

| 适配器键 | 角色 | 配置 |
|---|---|---|
| `SpreadSpec` | FF 式扩频(**下界**) | 整图 DCT 加性,α=6 |
| `DwtDct-default` | 经典 DwtDct(**改进前**=消融档0) | 中频 #18,单带 |
| `Imp+multiband` | 消融档1 | 低频 #1–7 多频带重复 |
| `Imp+texture` | 消融档2 | +纹理自适应 |
| `Imp+JND` | **主角**(=消融档3) | +Watson JND |
| `DCT-QIM-JPEG` | JPEG 域核心基线 | Q50,repeat=4 |
| `Deep-Latent` | 深度潜空间(**上界**) | 冻结 SD-VAE + CNN,GPU 在 DIV2K 训练,`--include-deep` |

```bash
python scripts/05_train_deep_watermark.py --smoke      # 先出深度 ckpt(上界基线)
python scripts/07_run_attack_suite.py --include-deep --n-images 10
python scripts/08_make_report_figures.py
```
**产出**:`results/attack_suite.csv` 与 `results/figures/` 下:
- `robustness_table.png`:方法×攻击 比特准确率热表;
- `sweep_<attack>.png`:各攻击的强度衰减曲线(标 0.5 随机基线);
- `tradeoff.png`:攻防权衡(横轴画面改变量、纵轴 BER,标"水印死但图还像"的危险区);
- `ablation.png`:**消融曲线**(默认→+多频带→+纹理→+JND 的逐档增益)。

**实测结果(10 张样图冒烟,以 CSV/图为准;数值随抽样波动)**:
- **消融逐档抬升**(全攻击均值):`DwtDct-default 0.65 → +multiband 0.76 → +texture 0.82 → +JND 0.81`。
  即"多低频带 + 纹理自适应"带来主要鲁棒性增益;**JND 档鲁棒性基本持平**——符合预期:**JND 主要换的是可见性**
  (PSNR≈49 dB),而非鲁棒性。
- **再生成战场**(`regen_surrogate`,strength=0.2):`DwtDct-default≈0.52(死)` → `Imp+JND≈0.82`,改进版显著抬升;
  `DCT-QIM-JPEG≈0.55(死)`——它是 JPEG 域基线、对再生成无招;`Deep-Latent` 全强度稳定 **~0.92–0.95**(上界)。
- **不可见性**:改进版 PSNR≈49 dB / SSIM≈0.997,远高于 `SpreadSpec`(31 dB / 0.86)——后者虽对该代理攻击鲁棒,
  但**以可见性为代价**;在相同画质下改进版完胜默认 DwtDct(见 `tradeoff.png`)。

**诚实口径(勿夸大)**:`regen_surrogate` 是**纯 CPU 代理(模糊+重采样+JPEG+噪声)**,且深度水印的攻击层
入环训练与该代理高度同源(近似 train/test 同分布),故 `Deep-Latent` 在该代理下的"全程稳定"含乐观成分。
真扩散 img2img 的结果见下。

### 6.1 真扩散 img2img 攻击(本地 SD1.5 / GPU,held-out)

```bash
python scripts/07_run_attack_suite.py --include-deep --ai-img2img \
       --data-dir data/eval20 --n-images 20 --out results/attack_suite_ai.csv
# 出图(单独命名,不覆盖 §6 的图):sweep_diffusion_img2img.png / robustness_table_ai.png
```
用本地 Stable Diffusion 1.5 img2img(strength∈{0,0.1,0.2,0.3},25 步,fp16,RTX 4060)做**真·再生成攻击**。
**真正的 held-out**:深度模型用冻结 SD-VAE + CNN 解码器 + GPU 在 **DIV2K(100 张,与 22 张测试图不重叠)**上训练,
攻击层只见过 模糊/噪声/缩放/VAE,**从未见过真扩散**。实测(20 张 held-out 图):

| 方法 (diffusion_img2img acc) | s=0.1 | s=0.2 | s=0.3 |
|---|---|---|---|
| **Deep-Latent(上界)** | **0.96** | **0.95** | **0.92** |
| SpreadSpec(下界,31dB 可见) | 0.68 | 0.59 | 0.57 |
| Imp+JND / Imp+texture / Imp+multiband | ≈0.50 | ≈0.50 | ≈0.50 |
| DwtDct-default | 0.50 | 0.51 | 0.51 |
| DCT-QIM-JPEG | 0.53 | 0.56 | 0.52 |

**关键结论(诚实且重要)**:
1. **真扩散下,所有 QIM 类水印(默认 + 改进三档 + DCT-QIM-JPEG)在 strength=0.1 就集体跌到 ~0.5(死)**
   ——印证 Zhao et al.(NeurIPS 2024)"像素/变换域隐形水印可被再生成证明性移除"。
2. **改进版相对默认的优势,在真扩散下消失**:真 img2img 连低频内容都重写,把低频带嵌入也一起洗掉。
   即**改进版的增益是针对压缩/模糊/低通型退化(含 CPU 代理)的,不是针对真生成式重绘的**——这正是"为何还需要
   深度水印"的硬证据。
3. **只有深度水印幸存**(真扩散 0.92–0.96,且是正经训练的真 held-out,无泄漏),清晰兑现"深度抗再生成"。
   深度模型 clean_acc=0.978(泛化好),但**画质代价明显:PSNR≈23dB / SSIM≈0.72**,远不如经典(49–66dB)
   ——鲁棒性换可见性。

> SD 权重首次下载约 5GB(已缓存到 `~/.cache/huggingface`,再跑命中本地);
> 模型 ID 可用环境变量 `SD_IMG2IMG_MODEL` 覆盖。

---

## 7. 评估纪律(实施方案 4.5)

- 在多张测试图上报均值±标准差;测试集模型未见过;
- 不可见性(PSNR/SSIM/LPIPS)攻击前算,鲁棒性(比特准确率/BER)攻击后算,不混;
- 比特准确率 0.5 = 随机 = 水印死,图中以虚线标出。

## 8. 已知限制

- `tiny` VAE 联合训练,画质受限(冒烟用);论文级需冻结 SD-VAE + 长训练。
- 真扩散攻击(`diffusion_img2img`/`vae_roundtrip`)需 `diffusers`;否则用 `regen_surrogate` 代理。
- 几何攻击(裁剪/旋转)会破坏分块对齐,经典方法跌幅大属预期(可加同步/模板做进阶项)。
