# Hybrid Analytic-Learned Vocal Tract Filter Design

Phonetically-interpretable DDSP for speech manipulation (small dataset).
Control targets: F0, F1, F2, F3, spectral tilt.

---

## System Overview

```
Input speech
     │
     ▼
  [Analysis]
  F0 (CREPE/Praat)
  F1, F2, F3, BW (Praat/LPC)
  Tilt α
     │
     ▼
  [Parameter Manipulation]   ← 操控入口：直接改 F0/F1/F2/tilt
     │
     ├─── Excitation model (H+N)
     │         F0 → harmonic + noise
     │
     └─── Vocal tract filter
               ↓
           两种方案（见下文）
     │
     ▼
  Output speech
```

Harmonic component: `x_h = Σ_k A_k · sin(2π·k·F0·t + φ_k)`
Noise component: `x_n = filtered_noise`
Output: `y = x_h + x_n`

---

## Method 1 — Time Domain: Analytic + Learned LPC Cascade

### 思路

声道滤波器为全极点（all-pole）级联：

```
excit → [α tilt] → [F1] → [F2] → [learned×N] → output
         1阶实极点  解析    解析    模型预测
```

前三节参数解析给定，后 N 节由神经网络预测。
使用 `torchlpc.sample_wise_lpc`，sample-wise 时变，无 OLA artifact。

### 各节参数化

**Tilt 节（1 阶实极点）**

```
H_tilt(z) = 1 / (1 - α·z⁻¹),   α ∈ (-0.99, 0.99)

α → +1 : 低通，暗/厚重
α →  0 : 平坦
α → -1 : 高通，亮/单薄
```

**Formant 节（2 阶复极点对）**

由 (Fk, BWk) 解析计算二阶系数：

```
r_k     = exp(-π · BWk / sr)
θ_k     = 2π · Fk / sr
c_k     = -2 · r_k · cos(θ_k)      # AR 系数 a1
d_k     = r_k²                      # AR 系数 a2

H_k(z)  = 1 / (1 + c_k·z⁻¹ + d_k·z⁻²)
```

**Learned 节（N 个 2 阶，频率约束在 F2 以上）**

```python
f_learned  = sigmoid(x_f)  * (sr/2*0.9 - f2) + f2   # F2 以上
bw_learned = sigmoid(x_bw) * 1950 + 50               # 50–2000 Hz
r          = exp(-π · bw_learned / sr)                # 保证 r < 1
```

### 16kHz 推荐节数

```
Tilt   1 阶实极点   :  1 极
F1     biquad       :  2 极
F2     biquad       :  2 极
F3     biquad       :  2 极  (可选，鼻音等)
Learned × 5         : 10 极
─────────────────────────────
合计                : 17 极  ≈ sr/1000 + 1 经验值
```

F3 默认关闭（鼻音不考虑时去掉该节）。

### 实现代码

```python
import torch
import torch.nn as nn
import math
from torchlpc import sample_wise_lpc


def formant_to_cd(f, bw, sr):
    """(B,T) -> c, d : 二阶 AR 系数"""
    r     = torch.exp(-math.pi * bw / sr)
    theta = 2 * math.pi * f / sr
    c     = -2 * r * torch.cos(theta)
    d     = r ** 2
    return c, d


class LearnedPoles(nn.Module):
    """输入特征 → N 对 learned all-pole biquad 系数"""

    def __init__(self, input_dim, n_sections=5, sr=16000):
        super().__init__()
        self.sr = sr
        self.n  = n_sections
        self.proj = nn.Linear(input_dim, n_sections * 2)

    def forward(self, h, f2_hz):
        """
        h     : (B, T, D)
        f2_hz : (B, T)   — 当前帧 F2，用于频率下限约束
        returns c, d : each (B, T, n_sections)
        """
        x   = self.proj(h)
        xf  = x[..., :self.n]
        xbw = x[..., self.n:]

        nyq = self.sr / 2 * 0.9
        f   = torch.sigmoid(xf)  * (nyq - f2_hz.unsqueeze(-1)) + f2_hz.unsqueeze(-1)
        bw  = torch.sigmoid(xbw) * 1950 + 50

        r     = torch.exp(-math.pi * bw / self.sr)
        theta = 2 * math.pi * f / self.sr
        c     = -2 * r * torch.cos(theta)
        d     = r ** 2
        return c, d


class VocalTractLPC(nn.Module):
    """Method 1: cascade all-pole LPC filter"""

    def __init__(self, sr=16000, use_f3=False, n_learned=5):
        super().__init__()
        self.sr       = sr
        self.use_f3   = use_f3
        self.n_learned = n_learned
        # 总阶数 = 1(tilt) + 2 + 2 + (2 if f3) + 2*n_learned
        self.order = 1 + 2 + 2 + (2 if use_f3 else 0) + 2 * n_learned

    def _apply_section(self, x, c, d):
        """
        x : (B, T)
        c, d : (B, T)  — 二阶系数
        """
        B = x.shape[0]
        A = -torch.stack([c, d], dim=-1).unsqueeze(-1)   # (B, T, 2, 1)
        zi = x.new_zeros(B, 2)
        return sample_wise_lpc(x, A, zi)

    def _apply_first_order(self, x, alpha):
        """一阶实极点 tilt 节"""
        B = x.shape[0]
        A = -alpha.unsqueeze(-1).unsqueeze(-1)            # (B, T, 1, 1)
        zi = x.new_zeros(B, 1)
        return sample_wise_lpc(x, A, zi)

    def forward(self, excit, alpha, f1, bw1, f2, bw2,
                c_learned, d_learned, f3=None, bw3=None):
        """
        excit      : (B, T)
        alpha      : (B, T)        tilt，tanh 约束在 (-0.99, 0.99)
        f1/bw1     : (B, T)        F1 Hz / 带宽 Hz
        f2/bw2     : (B, T)        F2 Hz / 带宽 Hz
        f3/bw3     : (B, T) | None F3 (可选)
        c_learned,
        d_learned  : (B, T, n_learned)
        """
        x = excit

        # 1. Tilt
        x = self._apply_first_order(x, torch.tanh(alpha) * 0.99)

        # 2. F1
        c1, d1 = formant_to_cd(f1, bw1, self.sr)
        x = self._apply_section(x, c1, d1)

        # 3. F2
        c2, d2 = formant_to_cd(f2, bw2, self.sr)
        x = self._apply_section(x, c2, d2)

        # 4. F3 (optional)
        if self.use_f3 and f3 is not None:
            c3, d3 = formant_to_cd(f3, bw3, self.sr)
            x = self._apply_section(x, c3, d3)

        # 5. Learned sections
        for i in range(self.n_learned):
            x = self._apply_section(x, c_learned[..., i], d_learned[..., i])

        return x
```

### A_k 配置（配合 Method 1）

Method 1 中，谐波幅度 A_k 只编码 **声门源 tilt**，formant 全部交给 LPC：

```python
def tilt_amplitudes(f0, alpha_src, sr, n_harmonics):
    """声门源谱斜率 → 谐波幅度，与 LPC 的 tilt 节分离"""
    k      = torch.arange(1, n_harmonics+1, device=f0.device).float()
    omega  = 2 * math.pi * k * f0.unsqueeze(-1) / sr
    a      = alpha_src.unsqueeze(-1)
    denom  = torch.sqrt((1 - a*torch.cos(omega))**2 + (a*torch.sin(omega))**2)
    return 1.0 / denom    # (B, T, K)
```

---

## Method 2 — Frequency Domain: Formant-Parameterized A_k

### 思路

不使用 LPC 时域滤波，直接在谐波幅度 A_k 中编码所有谱包络信息：

```
A_k = exp( log|H_F1(k·F0)| + log|H_F2(k·F0)| + log|H_F3(k·F0)|
         + log|G_tilt(k·F0)| + R_k )
```

对数域加法分解，每个 formant 独立贡献一个谱峰，R_k 为学习残差。

### 可视化

```
log|A_k|
   │
   │    [F1峰]      [F2峰]   [F3峰]
   │   /     \    /      \ /     \
   │  /       \  /        X       \
   │_/    R_k  \/     R_k  \  R_k  \___  (学习的背景谱形)
   └──────────────────────────────────→ k·F0
```

### 实现代码

```python
def formant_log_mag(k_freq, f, bw, sr):
    """
    单个 formant 在谐波频率处的对数幅度
    k_freq : (B, T, K)  — 谐波频率 Hz
    f, bw  : (B, T, 1)
    returns: (B, T, K)
    """
    r     = torch.exp(-math.pi * bw / sr)
    theta = 2 * math.pi * f / sr
    omega = 2 * math.pi * k_freq / sr
    diff  = omega - theta
    log_denom = 0.5 * torch.log(
        (1 - r * torch.cos(diff))**2 +
        (r * torch.sin(diff))**2 + 1e-8
    )
    return -log_denom   # (B, T, K)


def tilt_log_mag(k_freq, alpha, sr):
    """声门源 tilt 的对数幅度"""
    omega = 2 * math.pi * k_freq / sr
    a     = alpha.unsqueeze(-1)
    log_denom = 0.5 * torch.log(
        (1 - a * torch.cos(omega))**2 +
        (a * torch.sin(omega))**2 + 1e-8
    )
    return -log_denom   # (B, T, K)


class HarmonicAmplitudes(nn.Module):
    """Method 2: formant-parameterized A_k"""

    def __init__(self, input_dim, n_harmonics=80, sr=16000):
        super().__init__()
        self.sr          = sr
        self.n_harmonics = n_harmonics
        # 学习残差谱：网络输出 K 维对数残差
        self.residual_net = nn.Linear(input_dim, n_harmonics)

    def forward(self, h, f0, formants, bandwidths, alpha):
        """
        h          : (B, T, D)         特征
        f0         : (B, T)            基频 Hz
        formants   : (B, T, n_f)       F1/F2/F3 Hz
        bandwidths : (B, T, n_f)       对应带宽 Hz
        alpha      : (B, T)            tilt 参数
        returns    : A_k (B, T, K)
        """
        k      = torch.arange(1, self.n_harmonics+1,
                               device=f0.device).float()
        k_freq = k * f0.unsqueeze(-1)      # (B, T, K)

        # 各 formant 独立叠加（对数域）
        log_A = torch.zeros(*f0.shape, self.n_harmonics, device=f0.device)
        for i in range(formants.shape[-1]):
            log_A += formant_log_mag(
                k_freq,
                formants[..., i:i+1],
                bandwidths[..., i:i+1],
                self.sr
            )

        # Tilt
        log_A += tilt_log_mag(k_freq, alpha, self.sr)

        # 学习残差（捕捉 F4/F5、个人音色、细粒度谐波变化）
        log_A += self.residual_net(h)      # (B, T, K)

        return torch.exp(log_A)            # (B, T, K)
```

---

## 方案对比

| 维度 | Method 1 (LPC Cascade) | Method 2 (A_k Parameterized) |
|---|---|---|
| **实现域** | 时域递推 | 频域并行 |
| **F0 低音区** | 连续频率响应，精度不受影响 | 谐波稀疏，F1 峰可能只有 1 个谐波 |
| **F0 高音区** | 正常 | 正常，效果更好 |
| **计算** | 串行递推（N 次 torchlpc）| 完全并行 |
| **噪声部分** | N 部分需单独建模 | N 部分天然分离 |
| **代码复杂度** | 中（需 LPC 级联）| 低（纯矩阵运算）|
| **适合场景** | F0 范围宽、需要低音准确 | F0 较高、需要快速实验 |

### 选择建议

- **Method 1**：数据 F0 范围宽（含男声低音），需要精确 formant 控制
- **Method 2**：快速原型验证，F0 主要在 150Hz 以上，或想避免递推实现

---

## 共用参数说明

| 参数 | 符号 | 范围 | 控制效果 |
|---|---|---|---|
| 基频 | F0 | 60–500 Hz | 音高 |
| 第一共振峰 | F1, BW1 | F1: 200–900 Hz | 开口度（/a/ vs /i/）|
| 第二共振峰 | F2, BW2 | F2: 800–2500 Hz | 舌位前后 |
| 第三共振峰 | F3, BW3 | F3: 1800–3500 Hz | 音色，鼻音 |
| 谱斜率 | α | (-0.99, 0.99) | 明暗/气声 |

---

## 训练目标（Analysis-by-Synthesis）

```
Input speech
     │
     ▼
  [Analysis]  →  F0, F1, F2, F3, α  (Praat / differentiable LPC)
     │                  │
     │         [Parameter Manipulation]  ← 训练时直接用提取值
     │                  │
     ▼                  ▼
  [Synthesis]  →  ŷ
     │
  L = L_spec(y, ŷ) + L_mel(y, ŷ) + λ · L_reg(poles)

L_reg: 对 learned 极点半径的正则，防止极点贴近单位圆
       L_reg = mean(relu(r_learned - 0.95))
```

---

## ⚠️ 可辨识性问题：formant 与 residual 的冲突

> 这是「半 analytic 半 learned」方案的核心隐患，直接决定 F1/F2 能否真正操控。

### 问题：过参数化导致 F1/F2 失去物理意义

两种方法都把谱包络分解为「解析 formant + 学习残差」：

- **Method 2**：`log A_k = log|H_F1| + log|H_F2| + log|G_tilt| + R_k`
- **Method 1**：解析 F1/F2 极点 + learned 极点（conj 参数化，频率自由）

其中 **R_k（80 维逐谐波）/ learned 极点** 有足够容量表达**任意谱形**，
甚至可在 F1/F2 的位置自己加峰。给定一个目标谱，存在**无穷多组**
`(F1, F2, tilt, residual)` 都能拟合 → **不可辨识 (non-identifiable)**。

**仅用重建 loss 训练时**，网络可能：

- 把 F1/F2 学成任意值（甚至退化、乱跑）
- 用 residual / learned 极点补偿，使总和 = 真实谱
- 重建 loss 照样很低

**后果**：操控 F1/F2 时 residual 不跟随 → 合成谱共振峰**不移动** → **操控失效**。
即「重建好」≠「可操控」。

### 解法：formant 监督锚定（已有现成数据）

`extract_formants.py` 已将全部 1517 文件的 Praat F1/F2/B1/B2 存入 `.feat.npz`，
直接用作监督信号：

```
L = L_spec (MSS)
  + λ_f · ( ‖F1_pred − F1_praat‖ + ‖F2_pred − F2_praat‖ )   # 锚定共振峰频率
  + λ_b · ( ‖B1_pred − B1_praat‖ + ‖B2_pred − B2_praat‖ )   # 可选，锚定带宽
  + λ_r · ‖residual‖²                                          # 压小残差
```

- **formant loss**：强制 F1/F2 对应真实共振峰 → 恢复物理意义、保证可控
- **residual 正则**：让网络优先用 formant 解释谱形，residual 只补微扰（F3+、个人音色）
- 两者配合 → **F1/F2 可控，residual 不抢戏**

### 实现要点

1. `SingleSpeakerDataset.__getitem__` 额外返回对齐到帧的 `(F1, F2, B1, B2, voiced_mask)`
   （从 `.feat.npz` 读，按 hop 对齐）
2. `training_step` 从 encoder ctrl 解出 F1/F2（用 decoder 的 `get_formant_params`），加监督项
3. **仅在 voiced 帧**（F0>0 且 Praat 给出有效 formant）施加 formant loss
4. 频率建议归一化到 kHz 量级或取 log 再算 L1；初值 `λ_f ≈ 0.1`、`λ_r ≈ 1e-3`，需调
5. Method 1 / Method 2 通用（两者 `get_formant_params` 接口一致）

### 验证方式

训练后用 `eval/continuum.py` 扫 F1/F2，看合成谱共振峰是否随参数**正确移动**。
对比「有 / 无 formant 监督」两版，量化冲突的缓解程度。

> 当前正在训练的 aria_ddsp / aria_golf 只有重建 loss，**只验证了"能重建"，
> 尚未保证"F1/F2 可操控"**。需先实测操控失效程度，再决定是否加此监督重训。
