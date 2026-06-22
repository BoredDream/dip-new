# VINE 项目核心实现流程与算法原理分析

> 对象：`dip/VINE-main/`（ICLR 2025 论文 *Robust Watermarking Using Generative Priors Against Image Editing* 官方实现）
> 目的：拆解它"让水印扛住 AI 修图"的具体做法。在本项目（SWE）里，**VINE/RoSteALS 这条深度潜空间路线被定位为"上界基线"**——
> 回答答辩必问"既然要抗再生成、为何不直接上深度"（答：深度更强但黑盒+要 GPU，我们用信号处理改进 DwtDct 去逼近它）；
> 同时它的两个核心思路（**攻击层入环、把水印嵌到"再生成幸存的低频/语义子空间"**）直接指导我们的主线改进（见 §9）。
> 本文所有结论均来自源码精读，关键处给出 `文件:行号` 以便核对。

---

## 0. 一句话概括

VINE 把水印嵌入做成一个 **"编码器–攻击层–解码器" 端到端联合训练** 的深度模型，并用两个关键手段实现"抗 AI 编辑"：

1. **用一个预训练扩散模型（SD‑Turbo）当"生成先验"做嵌入器** —— 让水印以"扩散模型自己能生成的纹理"形式藏进图里，既隐蔽又稳。
2. **训练时用"模糊等失真"当作 AI 编辑的代理攻击（surrogate attack）** —— 因为论文发现"扩散编辑"在频域上和"模糊"很像，于是用便宜可微的模糊去逼近昂贵不可微的扩散重绘，逼解码器学会抗编辑。

最终产出两个模型：**VINE‑B**（Base，只过预训练）和 **VINE‑R**（Robust，再过"真实编辑入环"的微调）。

---

## 1. 整体架构：三件套

| 组件 | 类 / 文件 | 作用 | 是否参与推理 |
|---|---|---|---|
| **密码适配器** ConditionAdaptor | `stega_encoder_decoder.py:97` | 把 100bit 密码变成和图同尺寸的"条件残差" | ✅ 编码端 |
| **嵌入器** VINE_Turbo | `vine_turbo.py:135` | SD‑Turbo（VAE+UNet）一步生成出含水印图 | ✅ 编码端 |
| **攻击层** TransformNet | `training_src/transformations.py:17` | 训练时模拟各种失真/编辑（含模糊代理攻击） | ❌ 仅训练 |
| **判别器** Discriminator | `training_src/wm_modules.py:26` | GAN 对抗，提升画质真实感 | ❌ 仅训练 |
| **解码器** CustomConvNeXt | `stega_encoder_decoder.py:175` | 从（被攻击后的）图里回读 100bit | ✅ 解码端 |

数据流（训练时）：

```
secret(100bit) ─┐
                ├─► VINE_Turbo 编码器 ─► 含水印图 ──► TransformNet 攻击 ──► CustomConvNeXt 解码器 ─► 预测 100bit
cover image ────┘         │(嵌入)            (代理AI编辑/失真)              (回读)         │
                          └──────────────── 画质损失 (L2/LPIPS/GAN) ───────────────────────┘ 比特损失(BCE)
```

推理时去掉中间的"攻击层"和判别器：编码器把密码写进图 → 图在真实世界被 AI 编辑 → 解码器回读 → 比对比特准确率。

---

## 2. 核心创新一：用扩散模型当"嵌入器"（VINE_Turbo）

传统水印（你 `en-water` 里那 8 种）是在 DCT/DWT/SVD 系数上**手工**改几个数。VINE 则把"加水印"变成"用扩散模型重画一遍这张图，但把密码混进去"。

### 2.1 前向流程 `vine_turbo.py:214`

```python
def forward(self, x, secret, timesteps):
    x_sec  = self.sec_encoder(secret, x)                 # ① 密码+图 → 条件图(3通道)
    x_enc  = self.vae_enc(x_sec, "a2b")                  # ② VAE编码到潜空间 z (4×32×32)
    pred   = self.unet(x_enc, timesteps, ∅text).sample   # ③ 一步UNet去噪预测
    x_out  = sched.step(pred, t, x_enc).prev_sample      # ④ 调度器走一步
    return self.vae_dec(x_out, "a2b")                    # ⑤ VAE解码回图(含水印)
```

逐步拆解：

**① ConditionAdaptor `stega_encoder_decoder.py:97`** —— 把密码"画"成一张图，再和原图拼接：
```
100bit ─Dense─► 64×64 ─Dense─► 3×64×64 ─Upsample×4─► 3×256×256
                                                      └─concat 原图─► [6,256,256] ─conv─► 残差条件图[3,256,256]
```

**②③④ SD‑Turbo 一步生成** —— 这是"生成先验"的核心：
- 用的是 **`stabilityai/sd-turbo`**（注意：README 摘要写的是 SDXL‑Turbo，但发布代码实际加载的是 sd‑turbo，`vine_turbo.py:47,78,85,115`）。
- **只走一步**：`timesteps` 固定为 `num_train_timesteps-1`（=999，最大噪声步，`vine_turbo.py:155`），调度器用 `make_1step_sched`（`model.py:7`）设成 1 步。这就是 "Turbo" 一步出图，不用迭代 50 步，所以编码很快。
- 文本条件用**空字符串的 CLIP 嵌入**（`vine_turbo.py:143`），即"无提示词"，纯靠图像条件。

**⑤ 带跳连的 VAE 解码器 `model.py:30` + `vine_turbo.py:84`** —— 这是保证"画质不崩"的关键改造：
- 标准 VAE 解码会丢细节。VINE 给解码器加了 **4 个 skip_conv 跳连**（`vine_turbo.py:92-99`），把编码器各层特征直接搭桥到解码器对应层（`my_vae_decoder_fwd` 里 `sample = sample + skip_in`，`model.py:40-43`）。
- 跳连卷积权重**初始化成 1e‑5**（近乎 0），训练中再慢慢长大 —— 保证一开始输出≈原图，水印是"微扰"而非"重画"。
- 这套 VAE/UNet 改造借鉴自 **img2img‑turbo（CycleGAN‑Turbo）**，VINE 把它从"图像翻译"改用到"水印嵌入"。

> 直觉：水印不再是"叠加的噪声图案"，而是"由扩散模型生成先验托管的、看起来像自然纹理的扰动"。AI 编辑器再去重绘时，这种扰动更容易被保留下来（因为它本就处在扩散模型的"自然图像流形"上）。

### 2.2 训练哪些参数
UNet 和 VAE 主干都可训（`train.py:55,61`；推理版 `initialize_*_no_lora` 全量微调，另有 LoRA 版 `initialize_unet`/`initialize_vae` 作备选）。可训参数收集见 `VINE_Turbo.get_traininable_params`（`vine_turbo.py:175`）。

---

## 3. 核心创新二：模糊作为"AI 编辑的代理攻击"（TransformNet）

这是论文标题"From Benchmarking to Advances"里 advances 的灵魂，也是**最值得你借鉴**的一点。

### 3.1 为什么要代理攻击
理想训练：含水印图 → **真的过一遍扩散编辑** → 解码。但扩散编辑①慢（几十步）②**不可微**（没法回传梯度去教编码器）。论文的洞察：**扩散编辑对图像的破坏，频域特征上类似"模糊"**。于是用"模糊+一堆常规失真"这种**便宜、可微**的操作当替身，在训练时反复折磨水印。

### 3.2 TransformNet 失真池 `transformations.py:54`
每步以一定概率，从下面这些失真里挑一种/几种作用在含水印图上（全部可微或近似可微）：

| 失真 | 代码位置 | 说明 |
|---|---|---|
| **运动/高斯模糊**（代理攻击核心） | `:128-134` | `random_blur_kernel` 生成随机模糊核做卷积，σ∈[1,3] |
| 高斯噪声 | `:136-141` | |
| 对比度/亮度/色调/饱和度 | `:143-165` | YUV 域扰动 |
| JPEG 压缩（可微近似） | `:167-202` | 自实现可微 JPEG + MBRS 噪声 |
| Cropout / Dropout / Resize | `:80-97` | 局部篡改、缩放 |
| **ImageNet‑C 腐蚀** | `:62-73, 208` | 18 种通用腐蚀，按训练进度逐级加难 |
| Instagram 滤镜 (pilgram) | `:115-118, 229` | 26 种 IG 风格滤镜 |

**课程表式渐进（curriculum）**：失真强度随 `global_step` 增大（`ramp_fn`、`level` 计算，如 `:62-68, 81-89`）—— 先易后难，避免一上来太狠学不动。

### 3.3 真实编辑入环：只在微调阶段（VINE‑R）
`finetune.py:298-328` 里，**80% 的步**直接把含水印图喂给**真实的 InstructPix2Pix 扩散编辑管线**（`pipe_pix2pix`，25 步，`:318`），20% 的步才用 TransformNet 代理失真（`:301`）。即：
- **预训练（train.py）**：纯代理失真，梯度端到端回传 → 编码器+解码器一起学（产出 **VINE‑B**）。
- **微调（finetune.py）**：把真扩散编辑塞进回路，让解码器适应真实编辑痕迹（产出 **VINE‑R**，更鲁棒）。编辑数据集用 InstructPix2Pix 的 clip‑filtered 编辑对（`editing_dataset.py:19`）。

---

## 4. 解码器：CustomConvNeXt `stega_encoder_decoder.py:175`

极简：一个 **ConvNeXt‑Base 主干** + `Linear(1000→100)` + `Sigmoid`，直接把整张（可能被编辑过的）图回归成 100 个 [0,1] 概率，四舍五入即得比特。
- 训练目标：`BCELoss(decoded, secret)`（`train.py:298`）。
- 注意这里**没有用 STN/对齐网络**（`model.py:72` 里那个老式 StegaStamp `Decoder` 和 `ConditionAdaptor_orig` 都是历史遗留/备选，推理不用）。

---

## 5. 损失函数与训练策略 `train.py:269-366`

总损失（非纯比特阶段）：
```
loss = λ_l2 · image_loss(YUV加权L2)      # 不可感知性，按 y/u/v 不同权重 :320-327
     + λ_lpips · LPIPS(orig, encoded)     # 感知相似 :297
     + λ_secret · BCE(decoded, secret)    # 水印可读 :298
     + λ_G · G_loss                       # 对抗真实感 :329-330
```
配套技巧：
- **YUV 加权 + 边缘 falloff**（`:300-327`）：让扰动更多藏在不敏感的色度通道、避开图像边缘。
- **GAN（WGAN 风格）**：判别器分真图/含水印图，权重裁剪到 [−0.01,0.01]（`train.py:354-366`）。
- **两段式优化**：早期 `no_im_loss` 只优化比特损失（`optimizer_sec`，先保证水印读得出），之后再加画质损失（`optimizer_gen`）。`:332-352`。
- **fixed_input 预热**：先在**单张固定图**上把比特准确率练到 >0.9，再切换到整个数据集（`:262-267, 368`）。
- 启动命令见 `README.md` Training 段（`accelerate launch ... train.py` / `finetune.py`）。

---

## 6. 推理流程（这才是你跑 demo 会用到的）

### 6.1 编码 `watermark_encoding.py`
```
原图(任意分辨率) ─► 居中裁方 ─► resize到256 ─► VINE_Turbo(256, secret) ─► 含水印256图
                                                          │
            残差 = 含水印256 − 原256 ─► 把残差上采样回原分辨率 ─► 原图 + 残差 ─► 含水印图(原分辨率)
```
**关键技巧（`:60-71`）：模型只在 256×256 工作，但通过"在 256 算残差、把残差放大回原分辨率再叠加到原图"，实现任意分辨率水印**，省显存又保高清。密码：12 字符 UTF‑8 → 96bit + 4 个 0 = 100bit（`:44-58`）。

### 6.2 解码 `watermark_decoding.py`
```
(被编辑的)含水印图 ─► resize到256 ─► CustomConvNeXt ─► 100概率 ─► round ─► 与groundtruth逐位比对 ─► Bit Accuracy
```
就是 `:37-47`，输出比特准确率。**注意：VINE 是"零比特/已知消息"检测式评测**——解码端需要知道原始消息来算准确率（这点和你 `en-water` 的 `bit_accuracy` 思路一致）。

### 6.3 评测用的"编辑攻击" `editing_pipes.py`
W‑Bench 用四类编辑折磨水印：
- **Regeneration**：DDIM Inversion 反演再重建（`ddim_inversion`，`:58`）/ 随机再生。
- **Global Editing**：UltraEdit（`:10`）、InstructPix2Pix（`:27`）、MagicBrush。
- **Local Editing**：UltraEdit、ControlNet‑Inpainting。
- **Image‑to‑Video**：Stable Video Diffusion。
完整跑法见 `README.md` 的 W‑Bench 段（编码全集 → 编辑 → 解码统计 TPR@FPR / AUROC）。

---

## 7. 消息编码格式

| 场景 | 容量 | 编码 | 代码 |
|---|---|---|---|
| 推理 encode/decode | 100bit | 12 字符 UTF‑8 + 补 4 个 0 | `watermark_encoding.py:44` |
| 训练 validation | 100bit | "Hello"(56bit) + **BCH 纠错码** + 补零 | `train.py:440-453` |

BCH(`BCH_POLYNOMIAL=137, BCH_BITS=5`) 提供纠错冗余 —— 这也是个可借鉴点：**给水印加纠错码**能显著抬高编辑后的"消息级"成功率。

---

## 8. 关键文件索引（速查）

```
vine/src/
├─ vine_turbo.py            编码器主体 VINE_Turbo + VAE改造 + 一步调度
│   ├─ :135 VINE_Turbo      ├─ :214 forward(一步生成)  ├─ :84 initialize_vae(skip跳连)
├─ stega_encoder_decoder.py
│   ├─ :97  ConditionAdaptor(密码→条件图)  ├─ :175 CustomConvNeXt(解码器)
├─ model.py                 :7 一步调度器  :30 带skip的VAE解码forward
├─ train.py                 预训练(代理失真)：损失:269  优化:332  GAN:354
├─ finetune.py              微调(真实编辑入环)：:298-328
├─ watermark_encoding.py    推理编码 + 分辨率缩放技巧 :60
├─ watermark_decoding.py    推理解码 + 比特准确率 :37
├─ editing_pipes.py         评测用编辑攻击(UltraEdit/IP2P/DDIM反演)
└─ training_src/
    ├─ transformations.py   ★攻击层TransformNet(模糊代理攻击) :17,:128
    ├─ wm_modules.py        判别器 Discriminator :26
    └─ editing_dataset.py   微调用编辑数据集 + get_secret_acc :46
```

---

## 9. 对本项目（SWE）主线的启示——VINE 思路如何落到"改进 DwtDct"上

VINE 太重（要 conda、`torch==2.0.1`、下 HuggingFace checkpoint、基本要 GPU），**主线不复刻它**（它只作上界基线对照），
但它的思路恰好是我们"诊断驱动改进经典 DwtDct"主线的灵感来源——逐条对应到已落地的 P1/P2：

1. **【最该抄】模糊/再生成代理攻击 → 已落地为 P1 诊断的攻击源**：VINE 用"模糊"当 AI 编辑的可微代理。我们把它做成
   `swe/attacks/ai.py::regeneration_surrogate`（纯 CPU），并用它驱动 **P1 频带存活诊断**（`scripts/09`），
   客观找出"再生成幸存的频带"。这正是改进 DwtDct"往哪嵌"的依据。
2. **往低频/鲁棒子带嵌 → 已落地为改进①（多低频带 QIM）**：VINE 把水印托管在 VAE 潜空间的低频/语义结构上。
   对应到信号处理域,就是 P1 诊断指向的 **LL 低频幸存带**——改进版 DwtDct 据此把嵌入位置从默认中频 #18 改到低频带。
   （张力:低频更稳但更可见 → 见第 4 点 + 改进③ Watson JND。）
3. **加纠错码（BCH/重复码）→ 已落地为模块六 ECC**：像 VINE 的 BCH 那样给比特流加冗余（项目内自实现 Reed–Solomon），
   编辑后部分比特翻转也能纠回，提升"消息级"留存率。
4. **同步报告画质 / 显式可见性 → 改进③ Watson JND**：VINE 同时优化 PSNR/SSIM/LPIPS。我们更进一步,
   用 **Watson DCT JND 感知模型**（`swe/watermark/jnd.py`）给低频嵌入一个显式可见性上限,并在对比表里
   用 **LPIPS(内容代价) vs BitAcc(生存率)** 权衡曲线呈现 trade-off。

> 一句话:VINE 走"深度非线性"路线到达"再生成不变子空间";我们走"诊断 + 改进经典信号处理"路线逼近同一目标——
> 这就是 SWE 把 VINE 当**上界基线**、而非主方法的原因。

---

## 10. 需要注意的几个细节 / 坑

- **sd‑turbo ≠ SDXL‑turbo**：摘要说 SDXL‑Turbo，发布代码实际用 `stabilityai/sd-turbo`（`vine_turbo.py:47` 等）。引用论文时按摘要写，复现按代码看。
- **真实编辑不回传梯度**：微调时 25 步扩散编辑无法端到端回传，主要是**让解码器去适应**真实编辑痕迹；编码器的鲁棒性主要靠预训练阶段的**可微代理失真**学到。这正是"代理攻击"存在的根本原因。
- **强依赖环境**：`saicinpainting/`、`diffusers/`（被 vendored 进仓库）是给 W‑Bench 编辑/评测用的外部代码，不是 VINE 算法本体；跑通完整 pipeline 成本高，分析/借鉴思路无需安装。
- **评测是"已知消息"式**：算 bit accuracy 要有 groundtruth 消息，属可追溯水印而非盲提取。
```
