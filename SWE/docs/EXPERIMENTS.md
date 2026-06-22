# SWE 复现指南与预期结果

所有命令在 `SWE/` 目录下运行。经典链路纯 CPU 即可;深度部分 CPU 可冒烟、GPU 出正式结果。

---

## 0. 环境自检

```bash
pip install -r requirements.txt          # 经典链路
python -m pytest tests/ -q               # 期望:39 passed
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
**预期**:8 种方法干净准确率均 1.00;JPEG50/噪声/模糊后,LSB/DCT/DFT 明显跌向 0.5,
SVD/DWT-SVD/扩频较稳。结果图写入 `results/classic/`。

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

## 6. 全套实验 + 三张核心图(脚本 07/08)

```bash
python scripts/07_run_attack_suite.py --include-deep --n-images 4
python scripts/08_make_report_figures.py
```
**产出**:`results/attack_suite.csv` 与 `results/figures/` 下:
- `robustness_table.png`：方法×攻击 比特准确率热表;
- `sweep_<attack>.png`：各攻击的强度衰减曲线(标 0.5 随机基线);
- `tradeoff.png`：攻防权衡(横轴画面改变量、纵轴 BER,标"水印死但图还像"的危险区)。

**预期结论**:经典 DCT-QIM 抗 JPEG 强、但在 `regen_surrogate` 强度上升时跌向 0.5;
深度潜空间水印在再生成代理攻击下留存率更高 —— 即"经典抗压缩、深度抗再生成"的互补结论。

---

## 7. 评估纪律(实施方案 4.5)

- 在多张测试图上报均值±标准差;测试集模型未见过;
- 不可见性(PSNR/SSIM/LPIPS)攻击前算,鲁棒性(比特准确率/BER)攻击后算,不混;
- 比特准确率 0.5 = 随机 = 水印死,图中以虚线标出。

## 8. 已知限制

- `tiny` VAE 联合训练,画质受限(冒烟用);论文级需冻结 SD-VAE + 长训练。
- 真扩散攻击(`diffusion_img2img`/`vae_roundtrip`)需 `diffusers`;否则用 `regen_surrogate` 代理。
- 几何攻击(裁剪/旋转)会破坏分块对齐,经典方法跌幅大属预期(可加同步/模板做进阶项)。
