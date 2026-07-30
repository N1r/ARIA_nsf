# ARIA — 学术定位分析 (Academic Positioning)

> ARIA = **A**nalytic-**R**esidual **I**nterpolatable **A**rticulation
> 一个 DDSP 框架下、半解析半学习的全极点声道滤波器,为低资源、语音学可控的语音合成提供
> 对 F0 / F1 / F2 / tilt 的**结构性保证**的操控。

---

## 一句话定位

> 我们证明:**时域全极点级联(极点=共振峰)的 formant 操控是架构内禀、数学保证的**——
> 控制忠实度 r>0.97,**无需 formant 监督、仅 ~40% 训练**即成立;而频域谐波幅度参数化
> 的可控性是"涌现的",需监督才能从 formant-residual 冲突中救回。这是一个清晰、可证明的
> structural-vs-emergent controllability 二分。

---

## 核心贡献 (Claims)

### 1. 结构性可控 (Structural Controllability) — 主贡献
- VocalTractCascade(时域 all-pole)的 F1/F2 = 真实极点。`set F2 → 极点频率移动 → 共振峰
  必然移动`,这是**全极点滤波器的数学性质**,与训练程度/数据/是否监督**无关**。
- 实测:F1/F2/tilt 操控 r>0.97,元音连续统 measured≈target(绝对值也准),**全部无监督**。
- 对照:频域 A_k(AnalyticHarmonicsOscillator)无监督 r≈0.57 —— F1/F2 摆到边界、80 维
  residual 包办谱形(over-parameterization / non-identifiability)。
- → **结论框架**:可控性可以是"结构内禀的"(structural)或"涌现需诱导的"(emergent)。
  这个二分,据我们所知,是新的 framing。

### 2. 半解析半学习的声道滤波器
- 前段 F1/F2 + tilt 解析(可解释、可控),高阶共振峰 learned biquads(重建质量)。
- **系数域级联**成单个 AR 多项式(单次 `sample_wise_lpc`)→ 稳定、与基线同速。
- 加 analytic 结构**几乎零重建代价**:aria_golf val 2.658 ≈ baseline GOLF 2.621。

### 3. 低资源语音学操控
- 30 分钟单说话人语料。演示 F0/F1/F2/tilt 操控 + **元音到元音连续统**(/a/→/i/ 等)
  —— 语音学经典刺激 —— 用现代神经 DDSP 合成,且操控忠实。

### 4. 控制验证度量 (Control-Verification Metric)
- 用 `target vs 实测谱峰的相关系数 r` 量化操控忠实度。让"操控是否忠实"可测量,
  而非只靠听。tilt 用谱质心验证(亮度)。

---

## 新颖性 / 相对前作的定位

| 前作 | 提供 | ARIA 的增量 |
|---|---|---|
| DDSP (Engel 2020) | 可微谐波+噪声合成 | 显式、可验证的 formant 控制 |
| GOLF (Yu & Fazekas) | glottal-flow LPC 源-滤波 | 解析 formant 控制 + 结构可控性发现 |
| 可控 TTS (监督解纠缠) | 需大数据/监督 | 控制"免费"来自 all-pole 结构,低资源 |
| Klatt / 共振峰合成 | 可解释但非神经/音质有限 | 解析可解释性 + 神经音质,二者兼得 |

**关键差异化**:别人靠监督/解纠缠/大数据换可控性;ARIA 指出**时域级联的可控性是架构白送的**。

---

## 适合的会议 / 期刊 (Venue Analysis)

### 首选
- **Interspeech**(最对口):语音合成 + 可控性 + 语音学应用,同时覆盖工程与语音科学两个
  听众群。元音连续统直接打动语音感知社区。把"结构可控性"作为 headline。
- **ICASSP**:信号处理 framing(DDSP、LPC 级联、稳定性)。方法/架构角度强。
- **SSW (Speech Synthesis Workshop)**:专门的可控/参数化合成场子,方法+demo 理想。

### 次选
- **WASPAA / EUSIPCO**:音频信号处理 workshop。
- **ISMIR**:仅当 reframe 到歌声/音乐(GOLF 的主场),但 ARIA 是语音,不太对口。

### 语音科学角度(若主打"语音学工具")
- **JASA-EL / Journal of Phonetics**:若贡献定位为"用于感知实验(元音连续统、formant
  操控)的可控合成器"。可配一个小的音位边界感知实验。

### 推荐
> **Interspeech 为主**(方法 + 语音学应用 + demo),"结构可控性"作头条发现;
> **ICASSP 为备选**(若写得更偏信号处理)。

---

## 论文 framing / 标题候选

- "Structurally Controllable Formants: An Analytic–Learned All-Pole Vocal Tract for
  Low-Resource Speech Manipulation"
- "Control for Free: Time-Domain All-Pole Cascades Give Faithful Formant Manipulation
  without Supervision"
- 头条实验:F1/F2/tilt + 元音连续统 r>0.97,**无监督、30min 数据**,对照频域基线需监督。

---

## 直接竞品对比:HiFi-Glot (arXiv 2409.14823, KTH/Aalto)

HiFi-Glot 是最接近的工作:source-filter + differentiable resonant filters + 面向语音学
+ 对比 Praat。但关键技术选择不同,ARIA 在"低资源/解析可控/易训练"维度胜出。

| 维度 | HiFi-Glot | ARIA |
|---|---|---|
| 数据 | VCTK 多说话人, 1M iterations | 30min 单说话人, 40k steps |
| 滤波器 | **频域** all-pole P=31 (reflection coef→Levinson→FFT 域) | **时域** cascade order 18 (极点=共振峰) |
| formant→滤波器 | **网络学** (需 spectral-envelope LSD 监督 anchor) | **解析公式** formant_to_cd (无需学/监督) |
| 可控来源 | 端到端学 + envelope 监督 | 结构内禀, 无监督 r>0.97 |
| GAN | **必需**, 完整 HiFi-GAN 端到端 | **可选**, warm-start finetune (可解耦) |
| 模型 | ~20M params (13.9M+6.54M) | 5.5M |
| 采样率 | 22kHz | 16kHz (可升) |
| formant 精度 | F1<50Hz, F2<150Hz error (scale 0.7-1.3) | r>0.97 (F1/F2/tilt + 元音连续统) |

**三个核心差异化:**
1. **解析 vs 学习的 formant 控制**(最强):HiFi-Glot 的 formant→滤波器是网络学的,需
   envelope 监督;ARIA 的 F1/F2→共振峰是解析公式(极点=共振峰),改 F2 数学保证移动,
   无监督。可控对 HiFi-Glot 是"训练成果",对 ARIA 是"架构白送"。
2. **低资源 + 易训练**:ARIA 用 ~1/25 训练量(30min vs VCTK, 40k vs 1M, 5.5M vs 20M)
   达到可比可控性,模块化(可先无 GAN)。
3. **解耦可控与音质**:ARIA 先无 GAN/无监督验证可控(r>0.97),再 GAN 提音质 → 能 ablate
   证明"可控不靠 GAN/监督"。HiFi-Glot 端到端,无法解耦此论证。

**诚实劣势 + 应对:**
- 绝对音质:HiFi-Glot 22kHz+VCTK+HiFi-GAN 可能更高。应对:卖点是"小数据可控+音质性价比",
  两步走(GAN→CSMSC)冲音质;不争绝对 SOTA。
- 多说话人:HiFi-Glot speaker-independent,ARIA 单说话人。应对:语音学家要的就是单说话人
  精细可控(感知实验刺激),单说话人小数据是场景优势而非劣势。

**定位口径**(互补 + 推进,非推翻):
> HiFi-Glot 证明神经 formant 合成能高保真;ARIA 证明它能在**低资源、无监督、易训练**下做到
> ——因为时域 all-pole 级联让 formant 可控性解析白送,不需大数据/监督/端到端 GAN 去学。

## 还能加强论文的点 (To strengthen)

1. **Formant 监督消融**(已编码就绪):证明监督只是把绝对 F1/F2 校准到 Praat、**不损重建**
   ,但**对控制本身不必需**(这正是 key point)。
2. **MOS / 听测**:对操控刺激做听测(UTMOS 作 proxy;已知 F024 ground-truth UTMOS≈2.15,
   并已说明 16kHz 域偏差)。
3. **连续统感知实验**:配一个小的音位 ID 实验(phoneme boundary),证明刺激可用于语音学。
4. **客观重建表**(MCD/MSS-STFT,已有基础设施):aria_golf vs baseline 逐指标对比。
5. **结构可控性的"证明"**:把 all-pole 极点↔共振峰的对应写成一个 proposition,
   再用 r>0.97 实测佐证——理论+实验双支撑,是论文的骨架。

---

## 一句话给审稿人 (the pitch)

> 现有可控语音合成把可控性当作需要监督/解纠缠/大数据去"学"的东西。我们指出:把声道建成
> 时域全极点级联后,formant 可控性是架构数学**白送**的——无监督、低资源、可验证(r>0.97),
> 还能做语音学元音连续统。而频域谐波参数化做不到(需监督救场)。这给"可控合成"提供了一个
> 结构性的、可证明的新视角。
