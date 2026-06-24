# ARIA 实验细节完整记录（交叉验证版）

> 生成方式：3 个 subagent 独立抽取（架构/训练、评测/结果、审计）+ 交叉核对。
> 每个数值后标注 **来源文件**（`file:line`）。文件中找不到的标 **未核实**，不臆测。
> 根目录：`/zfsstore/user/dingyr/golf/`。最后核对：2026-06。

---

## 0. 版本 ↔ 配置映射

| 变体 | 顶层 config (`cfg/ae/`) | decoder config (`cfg/ae/decoder/`) |
|---|---|---|
| v1 (F024) | `mandarin_f024.yaml` | `aria_golf_16k.yaml` |
| v2 (F024) | `f024_aria_v2_16k.yaml` | `aria_golf_v2_16k.yaml` |
| v3 (F024) | `f024_aria_v3_16k.yaml` | `aria_golf_v3_16k.yaml` |
| v4 (F024) | `f024_aria_v4_16k.yaml` | `aria_golf_v4_16k.yaml` |
| v5 (F024) | `f024_aria_v5_16k.yaml` | `aria_golf_v5_16k.yaml` |
| v6 (F024) | `f024_aria_v6_16k.yaml` | **复用 v4 decoder** (`train_f024_aria_v6.slurm:19`) |
| v7 (F024) | `f024_aria_v7_16k.yaml` | `aria_golf_v7_16k.yaml` |
| v7b (F024) | `f024_aria_v7b_16k.yaml` | `aria_golf_v7b_16k.yaml` |
| CSMSC v2/v3/v4/v5 | `csmsc_aria_v{2..5}_24k.yaml` | `aria_golf_v{2..5}_24k.yaml` |
| LJSpeech v2/v3/v4/v5 | `ljspeech_aria_v{2..5}_22k.yaml` | `aria_golf_v{2..5}_22k.yaml` |

> ⚠️ `f024_aria_v7{,b}_16k.yaml:2` 注释写"Use with aria_golf_v4_16k"是**过期注释**；slurm（`train_f024_aria_v7b.slurm:46`）实际传的是 v7/v7b decoder。

---

## 1. 架构

### 1.1 编码器（所有数据集共用类与主干）
类 `models.enc.VocoderParameterEncoderInterface`，主干 `models.unet.UNetEncoder`，
`channels:[32,64,128,256]`，`strides:[4,4,4,4]`，`lstm_hidden_size:256`，`num_layers:3`，
`dropout:0.1`，`learn_f0:false`，`learn_voicing:false`（源：`f024_aria_v4_16k.yaml:43-55`）。
**用真值 F0**（pitch/voicing 来自分析，不预测）。

| 数据集 | n_fft | hop | f0_min | f0_max | 来源 |
|---|---|---|---|---|---|
| F024 16 kHz (v1) | 512 | 160 | 100 | 500 | `mandarin_f024.yaml:33-49` |
| F024 16 kHz (v2–v7b) | 512 | 160 | 100 | 500 | `f024_aria_v4_16k.yaml:47-55` |
| CSMSC 24 kHz (v2) | 1024 | 240 | 80 | 600 | `csmsc_aria_v2_24k.yaml:41-52` |
| CSMSC 24 kHz (v3/4/5) | 1024 | 240 | 80 | 500 | `csmsc_aria_v4_24k.yaml:46-57` |
| LJ 22 kHz (v2) | 1024 | 220 | 80 | 600 | `ljspeech_aria_v2_22k.yaml:31-42` |
| LJ 22 kHz (v3/4/5) | 1024 | 220 | 100 | 500 | `ljspeech_aria_v4_22k.yaml:45-56` |

> 帧移：F024 hop160 @16k = **10 ms**（计算值）。

### 1.2 解码器共享主干（所有版本相同，源 `aria_golf_v3_16k.yaml:7-43`）
- **声门源** `models.synth.DownsampledIndexedGlottalFlowTable`：`R_d∈[0.3,2.7]`、
  `points:2048`、`oversampling:4`、`table_type:derivative`（LF 流导数）、`lf_v2:true`、
  `trainable:false`、`align_peak:true`、`normalize_method:constant_power`、`hop_rate:10`、`in_channels:64`。
- **噪声** `models.noise.StandardNormalNoise`；噪声滤波 `LTVZeroPhaseFIRFilter`，`n_mag:128`(16k)/`256`(22k/24k)。
- **房间滤波** `models.filters.LTIAcousticFilter`，`length:128`，`conv_method:fft`。

### 1.3 解析声道滤波器（核心贡献）
闭式极点放置（`models/analytic_filter.py:177-183`）：由中心频率 F、带宽 B（Hz）映射到共轭极点对，
`r = exp(−πB/f_s)`，`θ = 2πF/f_s`，biquad `[1, −2r·cosθ, r²]`。谱倾斜 = 一阶实极点退化 biquad
`[1, −α, 0]`（`:171-175`）。级联在**系数域**经 `biquads2lpc` 合成单一 AR 多项式 `A(z)`，
单次 `sample_wise_lpc` 施加 `1/A(z)`（避免串联高 Q LPC 反传爆炸，`:90-111`）。
操纵 API：`get/set_formant_params`（绝对 Hz，逆 sigmoid）、`scale_formants`（相对、保轮廓）。

### 1.4 版本谱系（16 kHz / F024）

| ver | decoder 类 | use_tilt | n_learned | f1_range | f2_range | b1/b2_range | 特有 | 来源 |
|---|---|:--:|:--:|---|---|---|---|---|
| v1 | `SourceFilterSynth` | true(默认) | 6 | [150,1300] | [600,3200] | 默认(30,330)/(30,430) | — | `aria_golf_16k.yaml`;`analytic_filter.py:116` |
| v2 | `SourceFilterSynth` | **false** | **7** | **[200,1000]** | **[700,3000]** | [50,200]/[50,250] | — | `aria_golf_v2_16k.yaml:4,32-37` |
| v3 | `SourceFilterSynth` | true | 6 | [150,1300] | [600,3200] | [50,200]/[50,250] | — | `aria_golf_v3_16k.yaml:4,32-38` |
| **v4** | **`SourceFilterSynthAP`** | true | 6 | [150,1300] | [600,3200] | [50,200]/[50,250] | **band-aperiodicity 头** `AperiodicityFIRFilter` hf_bias2.0 | `aria_golf_v4_16k.yaml:7,31-47` |
| v5 | `SourceFilterSynth` | true | 6 | [150,1300] | [600,3200] | [50,200]/[50,250] | `subtract_harmonics:true`+噪声滤波 sigmoid | `aria_golf_v5_16k.yaml:9,49` |
| v6 | = v4 decoder | true | 6 | 同 v4 | 同 v4 | 同 v4 | v4 架构 + D4C 非周期监督 | `f024_aria_v6_16k.yaml:71,84` |
| v7 | **`VocalTractCascadeARMA`** | true | 6 | [150,1300] | [600,3200] | [50,200]/[50,250] | `n_zeros:2` 反共振 nasal_fz[700,2500] | `aria_golf_v7_16k.yaml:6,41-43` |
| v7b | `VocalTractCascadeARMA` | true | 6 | [150,1300] | [600,3200] | [50,200]/[50,250] | `n_zeros:3` nasal_fz**[180,3000]** | `aria_golf_v7b_16k.yaml:11,46-48` |

逐版改动：
- **v1→v2**：去 tilt、收紧 BW、收窄 F 范围、n_learned 6→7（保 LPC 阶 18）、加共振峰监督。
- **v2→v3**：恢复 tilt、F 范围回 v1、n_learned 7→6、监督 1.5→2.0+平滑 0.05。
- **v3→v4**：`SourceFilterSynth`→`SourceFilterSynthAP`，加解耦 band-aperiodicity `A(f)∈[0,1]`，
  `src=(1−A)·harm + A·H(noise)`（`models/sf.py:137-138`；`SourceFilterSynthAP` 强制 `subtract_harmonics=False` `:112`）。声道不变。
- **v3→v5**：`subtract_harmonics=true`+噪声滤波 sigmoid 限幅，`src=(1−H)·harm+H·noise`，不加参数。
- **v4→v6**：同 v4 架构，加 WORLD-D4C band-aperiodicity 监督（`aperiodicity_loss_weight:2.0`，`load_aperiodicity:true`）。
- **v3→v7**：声道 all-pole→ARMA（极零），加 `n_zeros` 反共振（可控鼻化），两段式 FIR 接在 `1/A` 之后。`n_zeros=0` 等价 v3。
- **v7→v7b**：nasal_fz 范围下移到能压到 F1 附近 + velar /ŋ/，`n_zeros` 2→3。

### 1.5 ⚠️ v2 vs v4 是**两套不同架构**（审计 C8 确认）

| 字段 | v2 | v4 |
|---|---|---|
| decoder 类 | `SourceFilterSynth` | `SourceFilterSynthAP` |
| band-aperiodicity 头 | **无** | 有（`AperiodicityFIRFilter`） |
| use_tilt（谱倾斜控制） | **false** | true |
| n_learned | 7 | 6 |
| f1/f2_range | [200,1000]/[700,3000] | [150,1300]/[600,3200] |

含义：把 v2→v4 说成"更多数据的质量增强版"是**错的**——它是**架构变更**（多了 tilt 与非周期头、极点数与范围都不同）。
24 kHz CSMSC 同理：v2 `use_tilt=false n_learned=9`，v4 `use_tilt=true n_learned=8`。

### 1.6 LPC / AR 阶数
公式（`analytic_filter.py:103,157`）：`阶 = 2 × (use_tilt + n_formants(=2) + n_learned)`。
- F024 16k：v1/v3/v4/v5/v7 = 2·(1+2+6)=**18**；v2 = 2·(2+7)=**18**。
- CSMSC/LJ 24k/22k：v1/v3/v4/v5 = 2·(1+2+8)=**22**；v2 = 2·(2+9)=**22**。
- v7/v7b ARMA：分子另加 `2·n_zeros` 个 MA 通道（v7 +4，v7b +6），all-pole 阶仍 18。

### 1.7 参数量（来自 Lightning 训练日志，审计 C5 确认）

| 变体 | Trainable | Size(MB) | 来源 `slurm/out/` |
|---|---|---|---|
| v1 F024 16k | **5.5 M** | 21.958 | `aria_golf_3170016.out:14-17` |
| **v4 F024 16k** | **5.6 M** | 22.220 | `f024_av4_3560464.out:19-22` |
| v7 F024 16k | 5.5 M | 21.966 | `f024_av7_3688589.out:19-22` |
| v4 CSMSC 24k | 6.2 M | 24.851 | `csmsc_v4_3584703.out:475-478` |
| v5 CSMSC 24k | 6.1 M | 24.326 | `csmsc_v5_3584704.out:163-166` |
| v4 LJ 22k | 6.2 M | 24.851 | `ljspeech_v4_3584701.out:211-214` |

> v2/v3/v5/v6/v7b 的 F024 精确参数量 **未核实**（日志未抓到 summary）。非训练参数=0（声门表是 buffer）。

---

## 2. 数据集与训练

### 2.1 数据集
| 数据集 | SR | 数据类 | wav_dir | 时长 |
|---|---|---|---|---|
| F024 | 16000 | `ltng.data.SingleSpeaker` | `…/F024` | 各版本均 1 h |
| CSMSC v2/v3 | 24000 | `SingleSpeaker` | `…/csmsc_1h` | 1 h |
| CSMSC v4/v5 | 24000 | `SingleSpeaker` | `…/csmsc_5h` | **5 h** |
| LJSpeech v2/v3 | 22050 | `SingleSpeaker` | `…/ljspeech_1h` | 1 h |
| LJSpeech v4/v5 | 22050 | `SingleSpeaker` | `…/ljspeech_5h` | **5 h** |

> ⚠️ **5h vs 1h 混淆项**：CSMSC/LJ 的 v4/v5 用 5 h，v1–v3 用 1 h（`f024_v1_v6_eval.md:21-31`）。F024 全部 1 h。

### 2.2 训练超参（F024）
公共：`Adam`，**lr 2e-4**，`gradient_clip_val:0.5`，`train_with_true_f0:true`，MSSLoss(`alpha:1.0`,hanning,center)。

| ver | MSS n_ffts | max_steps | batch | formant_w | smooth_w | reg_w | aperiodicity_w |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| v1 | [255,511,1023] | 200000 | 64 | 无 | — | — | — |
| v2 | [255,511,1023] | 80000 | 64 | 1.5 | — | 0.02 | — |
| v3 | [127,255,511,1023] | 40000 | 128 | 2.0 | 0.05 | 0.02 | — |
| v4 | [127,255,511,1023] | 40000 | 128 | 2.0 | 0.05 | 0.02 | — |
| v5 | [127,255,511,1023] | 40000 | 128 | 2.0 | 0.05 | 0.02 | — |
| v6 | [127,255,511,1023] | 40000（doc 说 30k） | 128 | 2.0 | 0.05 | 0.02 | **2.0** |
| v7/v7b | [127,255,511,1023] | 40000 | 128 | 2.0 | 0.05 | 0.02 | — |

来源：各 `f024_aria_v*_16k.yaml`。CSMSC/LJ：v2 batch32/40k步/fw1.5；v3 batch64/20k/fw2.0；v4/v5 batch64/80k、MSS 末位加 2047。

---

## 3. 评测

### 3.1 参数独立性（5×5 泄漏矩阵）
**方法**（`eval/independence_matrix.py:31-89`）：对每个控制做 **±扰动**，signed = (measure(+)−measure(−))/2；
扰动量 F0/F1/F2 = ±10%，**tilt 轴 = glottal `rd` 权重**（绝对 0.30↔0.70），energy = ±3 dB。
各 measure 除以其跨句自然标准差 σ，再每行除对角线 → 相对泄漏（对角=1）。`nanmedian` 聚合，默认 N=**20** 句。

> ⚠️ **关键口径**：矩阵里的 "tilt" 行/列指的是**声门 R_d 权重**（用 alpha-ratio 度量），**不是**一阶谱倾斜极点 α。
> ⚠️ **模型来源（审计 C4）**：三个矩阵都来自 **v1 / sup**，**没有一个是 v4**。

**F024（模型 `aria_golf_16k`=v1，val_loss 2.643，20 句；`independence_3282960.out`）**
σ=F0 31.1/F1 66.2/F2 282.1。均 off-diag **0.299**，max 1.599。

| 控制\度量 | F0 | F1 | F2 | tilt | energy |
|---|---|---|---|---|---|
| F0 | 1.000 | 0.046 | 0.022 | 0.312 | 0.138 |
| F1 | 0.002 | 1.000 | 0.009 | **0.994** | 0.045 |
| F2 | 0.001 | **0.796** | 1.000 | 0.089 | **1.599** |
| tilt | 0.002 | 0.417 | 0.077 | 1.000 | **1.402** |
| energy | 0.000 | 0.019 | 0.005 | 0.006 | 1.000 |

**F024-sup（模型 `aria_golf_sup_16k`，监督极点消融，`paper_evals_3282961.out`）** σ=31.1/83.7/290.4。均 0.251。

| 控制\度量 | F0 | F1 | F2 | tilt | energy |
|---|---|---|---|---|---|
| F0 | 1.000 | 0.052 | 0.028 | 0.360 | 0.140 |
| F1 | 0.000 | 1.000 | 0.010 | 1.057 | 0.014 |
| F2 | 0.010 | **0.240** | 1.000 | 0.018 | 1.354 |
| tilt | 0.000 | 0.317 | 0.065 | 1.000 | 1.316 |
| energy | 0.000 | 0.011 | 0.010 | 0.013 | 1.000 |

**CSMSC（模型 `csmsc_aria_golf`=v1）** σ=F0 19.7/F1 114.8/F2 220.5。均 0.352，max 2.704。

| 控制\度量 | F0 | F1 | F2 | tilt | energy |
|---|---|---|---|---|---|
| F0 | 1.000 | 0.012 | 0.001 | 0.198 | 0.177 |
| F1 | 0.006 | 1.000 | 0.032 | **2.704** | 1.282 |
| F2 | 0.010 | 0.001 | 1.000 | 0.388 | 1.363 |
| tilt | 0.006 | 0.163 | 0.003 | 1.000 | 0.680 |
| energy | 0.002 | 0.001 | 0.001 | 0.004 | 1.000 |

**结论**：**F0/F1/F2 三者互相近独立**（交叉项 ≤0.05，CSMSC 全 ≤0.03）；F024-unsup 的 F2→F1=0.80
是极端 /y/→/u/ 下共振峰跟踪器误配的伪影（sup/CSMSC 无此现象）。**tilt(R_d) 与 energy 强纠缠**（≥1.0），
不计入"独立"集合。

**逐模型标量泄漏（含 v4；`aria_v2_manip_3582341.out`，N=10，随机性大）**
- F024：v1 **0.123**、v2 0.093、v3 0.101、v3+GAN 0.107、**v4 0.093**、v5 **0.082**（最佳）。
- CSMSC：v1 0.291、v2 0.284、v3 0.273。LJ：v1 0.451、v2 0.355、v3 0.391（最差）。

### 3.2 操纵保真度

**(a) |measured−target| 中位数（Hz）** —— `paper_evals_3282961.out:52-55`，脚本 `eval/enhance_preserves_control.py`。

| 参数 | coarse 中位 | enhanced 中位 |
|---|---|---|
| F0 | **0.9** | 1.3 |
| F1 | **14.6** | 17.6 |
| F2 | **20.6** | 28.1 |

> ⚠️ **来源模型（审计 C2）**：该评测 base_run=**`csmsc_aria_golf`（CSMSC v1）** + `csmsc_mel_flow` 增强器，N=15 句/75 测量。
> **不是 F024 v4**。增强反而略降控制精度。
> ⚠️ **σ 口径（审计 C3）**：上面的中位数是 CSMSC，但常被配的 σ(31/84/290) 是 **F024-sup** 的；
> CSMSC 自己的 σ = **19.7/114.8/220.5**。引用"误差占自然方差比例"时勿跨数据集混用。

**(b) R²/斜率（measured-vs-target，`aria_v2_manip_3582341.out`，N=10）**

| F024 | F1 R²/slope | F2 R²/slope |
|---|---|---|
| v1 | 0.777 / 0.56 | 0.926 / 0.94 |
| v2 | 0.807 / 0.65 | 0.982 / 0.99 |
| v3 | 0.913 / 0.90 | 0.985 / 1.01 |
| v3+GAN | 0.900 / 0.90 | 0.984 / 1.00 |
| **v4** | 0.890 / 0.88 | 0.986 / 1.00 |
| v5 | 0.902 / 0.88 | 0.988 / 0.97 |

CSMSC：F1 v1 0.963/v2 0.958/v3 0.975；F2 ≥0.987。LJ：F1 ≥0.990；F2 ≥0.988。
**F2 近乎精确（R²≈0.99）**，**F1 在 F024 上欠冲**（R² 0.78→0.91，斜率<1），CSMSC/LJ 上 F1 强（R²≈0.96–0.99）。

### 3.3 自然度 UTMOS（n=50，含 std；`runs/_mos_utmos.json`，审计 C1 全部确认）

| 数据集 | reference | v2 | **v4** | v5 | golf(v1) |
|---|---|---|---|---|---|
| F024 16k | 2.696±0.678 | 2.400±0.593 | **2.526±0.634** | 2.527±0.635 | 2.422±0.632 |
| CSMSC 24k | 3.901±0.370 | 2.986±0.393 | **3.351±0.442** | 2.823±0.334 | 3.160±0.423 |
| LJ 22k | 4.370±0.088 | 3.427±0.313 | **4.002±0.228** | 3.530±0.275 | 3.653±0.284 |

（F024 另有 v3_fast 2.428、v6 2.473、gan3_v3 2.505。）UTMOS 是**神经代理，非人类听测**；宽带训练，16k F024 偏低，
应读**同一嗓音内 resynth−reference 差**。

**DNSMOS_ovrl / SQUIM_PESQ / SQUIM_STOI（`runs/_mos_dnsmos.json`，无 std）**：DNSMOS 各系统近乎持平（无判别力）；
例 CSMSC v4 3.327/3.002/0.971 vs ref 3.363/3.877/0.989。SQUIM 为非侵入估计，16k 上不可靠。

### 3.4 客观指标 MCD / LSD / MSS-STFT / SNR（`paper/html/aria_v2_*.html`，6-19 最新；审计 C6 确认 v4）

| F024 | MCD↓ | LSD↓ | MSS↓ | SNR↑ | | CSMSC | MCD | | LJ | MCD |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 3.87 | 7.32 | 3.52 | −2.74 | | v1 | 3.66 | | v1 | 4.08 |
| v3 | 3.91 | 7.39 | 3.50 | −2.56 | | v3 | 3.80 | | v3 | 4.27 |
| **v4** | **3.72** | 7.29 | 3.38 | −2.63 | | **v4** | **3.52** | | **v4** | **3.98** |
| v5 | 3.95 | 7.44 | 3.45 | −2.50 | | v5 | 3.83 | | v5 | 4.33 |
| v6 | 3.81 | 7.35 | 3.46 | −2.55 | | | | | | |

> MCD ≈3.5–4.0 dB（版本间差 ~0.2 < ~0.3 dB JND，勿过度排序）。**SNR 全为负（≈−2.5~−3 dB）**——
> 分析重合成相位/激励失配所致，**不建议作为卖点**。v4 在三数据集 MCD 均最佳。

### 3.5 高频周期性 / band-aperiodicity（`docs/f024_v1_v6_eval.md` §3.3）
参考基线：3–7k=0.509，5–7k=0.316。Δ>0 = 合成过于周期/谐波。

| 模型 | 3–7k Δ | 5–7k Δ |
|---|---|---|
| v3 | +0.044 | +0.046 |
| v3+GAN | +0.034 | +0.031 |
| v4 | +0.044 | +0.049 |
| v5 | +0.050 | +0.050 |
| **v6** | **+0.028** | +0.033 |

> ⚠️ **仅文档来源、无 slurm 日志佐证**（已全量搜 `slurm/out/` 无对应值）。脚本 `eval/measure_highband_periodicity.py` 似交互式运行。
> 跨数据集（`highband_harmonic_asymmetry.md`）：**F024 合成 HF 过量**，**LJ/CSMSC 合成 HF 不足**——方向相反；
> v4/v5/v6 只能压 F024 那约 ⅓ 的过量，无法给 LJ/CSMSC 补 HF。

### 3.6 鼻音 / 反共振（v7/v7b ARMA）
**鼻音 gap（`nasal_gap_3696432.out`，60 音节/组；审计 C7 确认）**

| 模型 | oral MCD/UTMOS | nasal_coda ΔMCD/ΔUTMOS | nasal_onset ΔMCD/ΔUTMOS |
|---|---|---|---|
| v4 | 3.745 / 2.792 | **+0.041 / +0.196** | +0.015 / +0.121 |
| v7 | 3.877 / 2.703 | +0.028 / +0.266 | +0.025 / +0.127 |

> **关键负结果**：ΔMCD 仅 +0.01~0.04 dB（≪JND），**ΔUTMOS 为正**（鼻音句 UTMOS *更高*）。
> 即"all-pole 做不了鼻音"的缺陷在**聚合指标上不可见**。鼻化的价值是**可控性**，不是重建。

**反共振探针（`nasal_af_3716996.out`）**
- v7：零点基本**休眠**（medBz≈460–500=浅缺口；nasal_onset fracBz<150=0.021=oral 的 3.5×，方向对但弱）。
- **v7b**（范围下移+n_zeros3，40k 步，val_loss 3.475）：第一个可测的 ARMA 改进——
  fracFz<700 由 0.000→**0.276**，fracBz<150（nasal_coda）0.007→0.055（↑8×），低频/反共振带 band-LSD 最优。
- demo 鼻化连续体 LTAS：F1 区峰 −2.8→−21.1 dB（−18 dB），F2+ 不变（解耦）。

> ⚠️ **未做**：定向鼻腔 murmur 带（~800–2500 Hz / 近 F1）的反共振缺陷量化；~250 Hz 鼻极点 P0（A1–P0 主线索）不可表达。

---

## 4. 跨源不一致与口径（交叉验证发现，**写论文/汇报必须遵守**）

1. **论文 §2 写的是 v4，但证据混了模型**：独立性 5×5 矩阵=**v1/sup**，保真度中位数=**CSMSC v1**，
   且常配的 σ(31/84/290)=**F024-sup**（CSMSC 自己 σ=19.7/114.8/220.5）。要么统一到 v4 重跑，要么文中点明各图各表的模型。
2. **UTMOS**：`runs/_mos_utmos.json`(n=50,带 std) 与 `docs/f024_v1_v6_eval.md` §3.2 早期 pass **数值不同**（doc 偏低）。以 JSON 为准。
3. **MCD/LSD**：doc §3.1 与 `paper/html`(6-19) 略有出入（如 F024 v1 3.90 vs 3.87）；MSS-STFT/SNR **只有 html 有**。以 html 为准。
4. **高频 Δ（§3.5）**：仅文档、无日志——可信度低，需重跑落盘。
5. **"tilt" 命名**：独立性矩阵的 tilt 轴 = 声门 R_d，不是一阶谱倾斜极点。
6. **v2→v4 是架构变更**（§1.5），不是单纯"更多数据"。

---

## 5. 评测规模 N 与总体 caveats（`docs/f024_v1_v6_eval.md`）
- 独立性矩阵 N=20；R²/斜率与标量泄漏 N=10（随机性大）；增强保真 N=15/75 测量；UTMOS/DNSMOS N=50；鼻音 60 音节/组。
- doc 客观/MOS 表 N=10–12。
- **明确声明**：(a) "preliminary，非最终结论"；(b) **无人类听测**（MUSHRA/AB），所有"主观"数=神经代理；
  (c) 在**训练集句子**上评测（重建快照，非泛化）；(d) UTMOSv2 未安装未跑。

---

## 附录：各 eval 脚本 → 输出 → checkpoint
| 评测 | 脚本 | 输出 | 关键 ckpt |
|---|---|---|---|
| 独立性 | `eval/independence_matrix.py` | `paper/data/independence_*.csv`、`slurm/out/independence_3282960.out`、`paper_evals_3282961.out` | F024 v1 `epoch=1804…2.643`；CSMSC v1 `epoch=279…3.109`；sup `…2.635` |
| 增强保真 | `eval/enhance_preserves_control.py` | `slurm/out/paper_evals_3282961.out`、`paper/fig/enhance_control.png` | base `csmsc_aria_golf`；flow `…0.0231` |
| R²/泄漏 | `eval/aria_v2_manip.py` | `slurm/out/aria_v2_manip_3582341.out`、`paper/html/aria_v2_*.html` | F024 v4 `epoch=3599…3.425` 等 |
| 客观+音频 | `eval/aria_v2_compare.py` | `paper/html/aria_v2_*.html`、`paper/audio/aria_v2_*/` | 同上（N=50 重合成） |
| UTMOS/DNSMOS | `eval/mos_scores.py` | `runs/_mos_utmos.json`、`runs/_mos_dnsmos.json` | 评 `paper/audio/*` |
| 高频周期 | `eval/measure_highband_periodicity.py` | 仅 `docs/f024_v1_v6_eval.md` §3.3 | — |
| 鼻音 gap | `eval/nasal_gap.py` | `slurm/out/nasal_gap_3696432.out` | v4 `…3.425`；v7 `…3.505` |
| 反共振探针 | `eval/nasal_antiformant_probe.py` | `slurm/out/nasal_af_3716996.out` | v7b `…3.475` |
