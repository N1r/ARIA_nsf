# GOLF Codebase Guide

Repository: `/zfsstore/user/dingyr/golf`
Paper: Differentiable Time-Varying Linear Prediction (Interspeech 2024)

---

## 目录结构总览

```
golf/
├── autoencode.py          ← 入口 A：Analysis-by-Synthesis 训练
├── main.py                ← 入口 B：DDSP Vocoder 训练（mel → 合成）
├── cfg/
│   ├── ae/
│   │   ├── vctk.yaml      ← 训练基础配置（encoder、数据、优化器）
│   │   └── decoder/       ← 各 decoder 模型配置（可互换）
│   │       ├── ddsp.yaml
│   │       ├── golf.yaml
│   │       ├── golf-precise.yaml
│   │       ├── golf-v1.yaml
│   │       ├── nhv.yaml
│   │       ├── mlsa.yaml
│   │       ├── mlsa-taylor.yaml
│   │       └── world.yaml
├── models/                ← 核心模型组件
│   ├── synth.py           ← 激励源（振荡器）
│   ├── filters.py         ← 声道滤波器
│   ├── hpn.py             ← HarmonicPlusNoiseSynth
│   ├── sf.py              ← SourceFilterSynth
│   ├── noise.py           ← 噪声生成器
│   ├── enc.py             ← 编码器
│   ├── unet.py            ← UNet / Transformer 骨干网络
│   ├── mel.py             ← Mel2Control
│   ├── lpc.py             ← LPC 合成
│   └── utils.py           ← rc2lpc, biquads2lpc 等工具函数
├── ltng/                  ← PyTorch Lightning 模块
│   ├── ae.py              ← VoiceAutoEncoder（入口 A 使用）
│   ├── vocoder.py         ← DDSPVocoder（入口 B 使用）
│   ├── data.py            ← VCTK, LJSpeech, M4Singer 数据集
│   └── cli.py             ← CLI 工具
├── loss/
│   └── spec.py            ← MSSLoss（多尺度谱损失）
└── scripts/
    ├── resample_dir.py    ← 批量重采样
    └── wav2f0.py          ← F0 提取（DIO/PENN/SWIPE）
```

---

## 两个训练入口

### 入口 A：`autoencode.py`（主要入口）

**Analysis-by-Synthesis**：从真实语音提取参数（F0、voicing），用这些参数重合成，与原始语音计算损失。

```bash
python autoencode.py fit \
    --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/{MODEL}.yaml \
    --trainer.logger false
```

配置加载逻辑：
```
vctk.yaml         → 固定部分：encoder、数据、优化器、训练参数
decoder/{}.yaml   → 可替换部分：decoder 架构
```

两个 yaml 会被 Lightning CLI 合并，`decoder` 字段由第二个 yaml 覆盖。

### 入口 B：`main.py`（DDSPVocoder）

**Mel → 合成**：mel 谱 → 参数预测 → 合成。
使用较少，适合 TTS 场景。

---

## 所有 Decoder 模型

### 模型 1：DDSP（原版 DDSP H+N）

```yaml
# cfg/ae/decoder/ddsp.yaml
decoder: HarmonicPlusNoiseSynth
  harm_oscillator: AdditiveSynthesizer      # 155 次谐波叠加，A_k 自由预测
  noise_generator: StandardNormalNoise
  harm_filter: PassThrough                  # 谐波不额外滤波
  noise_filter: LTVZeroPhaseFIRFilter       # FIR 滤波噪声
  end_filter: LTIAcousticFilter             # 固定声学室响
```

**对应论文模型**：DDSP
**特点**：谐波幅度 A_k 由网络自由预测（无物理约束），FIR 做谱整形
**命令**：
```bash
python autoencode.py fit --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/ddsp.yaml --trainer.logger false
```

---

### 模型 2：NHV（Neural Homomorphic Vocoder）

```yaml
# cfg/ae/decoder/nhv.yaml
decoder: HarmonicPlusNoiseSynth
  harm_oscillator: AdditivePulseTrain       # 脉冲串叠加谐波
  noise_generator: StandardNormalNoise
  harm_filter: LTVCepFilter                 # 倒谱滤波（min-phase）
  noise_filter: LTVZeroPhaseFIRFilter
  end_filter: LTIAcousticFilter
```

**对应论文模型**：NHV
**特点**：谐波通过倒谱滤波器，类似传统同态声码器
**命令**：
```bash
python autoencode.py fit --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/nhv.yaml --trainer.logger false
```

---

### 模型 3：GOLF-ff（Frame-wise filter，论文主模型）

```yaml
# cfg/ae/decoder/golf.yaml
decoder: SourceFilterSynth
  harm_oscillator: DownsampledIndexedGlottalFlowTable
    # LF 声门流模型，R_d ∈ [0.3, 2.7]，2048 个波形点
    hop_rate: 10        # 帧率 = sr / 240 = 100 Hz
    lf_v2: true
    table_type: derivative
  noise_generator: StandardNormalNoise
  noise_filter: LTVZeroPhaseFIRFilter       # 噪声谱整形
  end_filter: LTVMinimumPhaseFilter         # ← Frame-wise LPC（OLA）
    window_length: 960
    lpc_order: 22
    lpc_parameterisation: rc2lpc            # 反射系数参数化
  room_filter: LTIAcousticFilter
```

**对应论文模型**：GOLF-ff（Feed-Forward）
**特点**：
- 声门源：LF 模型波形表（物理声源）
- 声道：22 阶 LPC，frame-wise OLA
- 反射系数参数化（tanh 保证稳定）
**命令**：
```bash
python autoencode.py fit --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/golf.yaml --trainer.logger false
```

---

### 模型 4：GOLF-ss（Sample-wise filter，精确版）

```yaml
# cfg/ae/decoder/golf-precise.yaml
decoder: SourceFilterSynth
  harm_oscillator: DownsampledIndexedGlottalFlowTable   # 同 golf.yaml
  noise_filter: LTVZeroPhaseFIRFilter
  end_filter: LTVMinimumPhaseFilterPrecise              # ← Sample-wise LPC
    lpc_order: 22
    lpc_parameterisation: rc2lpc
  room_filter: LTIAcousticFilter
```

**对应论文模型**：GOLF-ss（Sample-wise / Precise）
**特点**：
- 与 golf.yaml 唯一区别：`LTVMinimumPhaseFilterPrecise` 代替 `LTVMinimumPhaseFilter`
- 系数插值到采样率后用 torchlpc 单次递推，无 OLA
- 音质略优于 golf-ff
**命令**：
```bash
python autoencode.py fit --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/golf-precise.yaml --trainer.logger false
```

---

### 模型 5：∇WORLD（可微分 WORLD）

```yaml
# cfg/ae/decoder/world.yaml
decoder: SourceFilterSynth
  harm_oscillator: AdditivePulseTrain
  noise_filter: LTVZeroPhaseFIRFilter
  end_filter: DiffWorldSPFilter             # 可微分 WORLD 谱包络滤波
```

**对应论文模型**：∇WORLD
**命令**：
```bash
python autoencode.py fit --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/world.yaml --trainer.logger false
```

---

### 模型 6 & 7：MLSA / MLSA-Taylor

```yaml
# cfg/ae/decoder/mlsa.yaml
end_filter: LTVMLSAFilter
  alpha: 0.46          # 频率轴翘曲（Mel）
  cep_order: 99
  phase: min
```

**对应论文模型**：MLSA（Mel-Log Spectral Approximation）
Mel 倒谱滤波，频率轴做 Mel 翘曲。

---

## 模型对比一览

| Config | 论文名称 | 激励源 | 声道滤波器 | Frame/Sample |
|---|---|---|---|---|
| `ddsp.yaml` | DDSP | 谐波叠加（A_k 自由）| FIR（zero-phase）| Frame |
| `nhv.yaml` | NHV | 脉冲串谐波 | 倒谱 FIR | Frame |
| `golf.yaml` | GOLF-ff | LF 声门流 | 22阶 LPC + OLA | **Frame** |
| `golf-precise.yaml` | GOLF-ss | LF 声门流 | 22阶 LPC torchlpc | **Sample** |
| `golf-v1.yaml` | GOLF-v1 | LF 声门流 | LPC（旧版）| Frame |
| `world.yaml` | ∇WORLD | 脉冲串 | WORLD 谱包络 | Frame |
| `mlsa.yaml` | MLSA | 脉冲串 | MLSA Mel倒谱 | Frame |

> **SawSing** 不在这个 repo 中。
> `SawToothOscillator` 类存在于 `models/synth.py`，但无对应 yaml，
> 需要手动配置或参考 SawSing 原始仓库。

---

## 核心组件详解

### 激励源（`models/synth.py`）

| 类名 | 说明 | 用于 |
|---|---|---|
| `AdditiveSynthesizer` | A_k 自由预测的谐波叠加 | DDSP |
| `AdditivePulseTrain` | 脉冲串 + 谐波叠加 | NHV, ∇WORLD, MLSA |
| `DownsampledIndexedGlottalFlowTable` | LF 模型声门流波形表 | GOLF |
| `SawToothOscillator` | 锯齿波（SawSing 激励）| 未配置 |
| `HarmonicOscillator` | 通用谐波振荡器基类 | 继承用 |

### 声道滤波器（`models/filters.py`）

| 类名 | 类型 | 特点 |
|---|---|---|
| `LTVMinimumPhaseFilter` | LPC all-pole | Frame-wise + OLA |
| `LTVMinimumPhaseFilterPrecise` | LPC all-pole | **Sample-wise**（torchlpc）|
| `LTVZeroPhaseFIRFilter` | FIR | 零相位，frame-wise |
| `LTVCepFilter` | 倒谱 FIR | min-phase |
| `LTVMLSAFilter` | MLSA | Mel 翘曲 |
| `DiffWorldSPFilter` | WORLD 谱包络 | 可微分 |
| `LTIAcousticFilter` | 固定 FIR | 声学室响（固定）|

### LPC 参数化方式（`lpc_parameterisation`）

| 值 | 方法 | 稳定性 |
|---|---|---|
| `rc2lpc` | 反射系数 → AR（`utils.rc2lpc`）| ✓ tanh 保证 |
| `lsp2lpc` | 线谱对 → AR | ✓ softmax 保证 |
| `coef` | 直接预测 biquad 系数 | 需手动约束 |
| `real` | 实数极点参数化 | 需手动约束 |

---

## 编码器（`models/enc.py` + `models/unet.py`）

```
音频 → STFT → UNetEncoder → LSTM → 输出各参数
                                   ├─ log_f0 (1维)
                                   ├─ voicing (1维)
                                   └─ ctrl (N维，输入给 decoder)
```

UNetEncoder 参数（vctk.yaml 中）：
```yaml
n_fft: 1024
hop_length: 240        # 帧移 = 10ms at 24kHz
channels: [32,64,128,256]
strides: [4,4,4,4]     # 时间下采样
lstm_hidden_size: 256
num_layers: 3
```

---

## 常用训练命令

```bash
# 基础训练（GOLF-ss，推荐）
python autoencode.py fit \
    --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/golf-precise.yaml \
    --trainer.logger false

# 修改数据目录
python autoencode.py fit \
    --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/golf-precise.yaml \
    --data.wav_dir /your/data/path \
    --trainer.logger false

# 修改采样率（如 16kHz）
python autoencode.py fit \
    --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/golf-precise.yaml \
    --model.sample_rate 16000 \
    --trainer.logger false

# 从 checkpoint 继续训练
python autoencode.py fit \
    --config cfg/ae/vctk.yaml \
    --model cfg/ae/decoder/golf-precise.yaml \
    --trainer.logger false \
    --ckpt_path path/to/checkpoint.ckpt

# 评估
python autoencode.py test \
    --config cfg/ae/vctk.yaml \
    --ckpt_path path/to/checkpoint.ckpt \
    --data.duration 2 --data.overlap 0 \
    --seed_everything false \
    --data.wav_dir data/vctk \
    --data.batch_size 32 \
    --trainer.logger false

# 生成音频（predict）
python autoencode.py predict \
    --config cfg/ae/vctk.yaml \
    --ckpt_path path/to/checkpoint.ckpt \
    --trainer.logger false \
    --seed_everything false \
    --data.wav_dir data/vctk \
    --trainer.callbacks+=ltng.cli.MyPredictionWriter \
    --trainer.callbacks.output_dir output/
```

---

## 添加自定义 Decoder 的方法

1. 在 `models/` 下实现新的 Filter 或 Oscillator 类
2. 新建 `cfg/ae/decoder/my_model.yaml`
3. 按上述格式组合 `harm_oscillator` / `noise_filter` / `end_filter`
4. 训练命令中替换 `--model cfg/ae/decoder/my_model.yaml`

例如，将 GOLF-ss 的 end_filter 替换为自定义混合滤波器：

```yaml
# cfg/ae/decoder/aria.yaml
decoder:
  class_path: models.sf.SourceFilterSynth
  init_args:
    harm_oscillator:
      class_path: models.synth.DownsampledIndexedGlottalFlowTable
      init_args:
        hop_rate: 10
        lf_v2: true
        # ... 同 golf.yaml
    noise_filter:
      class_path: models.filters.LTVZeroPhaseFIRFilter
      init_args:
        window: hanning
        n_mag: 256
    end_filter:
      class_path: models.filters.HybridFormantFilter   # ← 自定义
      init_args:
        sr: 16000
        n_analytic: 2       # F1, F2
        n_learned: 5
        use_tilt: true
```

---

## 关键工具函数（`models/utils.py`）

```python
rc2lpc(k)            # 反射系数 → AR 系数（Levinson 递推，可微分）
biquads2lpc(biquads) # biquad 二阶系数 → AR 系数（多项式乘法）
complex2biquads(z)   # 复数极点 → biquad 系数
params2biquads(p)    # 参数化极点 → biquad 系数
get_logits2biquads() # 工厂函数，返回参数化方法
linear_upsample(x)   # 帧级 → 采样级线性插值（reduce_hop_length 使用）
```
