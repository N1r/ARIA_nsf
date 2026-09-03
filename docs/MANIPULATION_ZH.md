# 参数操控指南：用训练好的模型生成实验刺激

假设你已经训练好一个模型，希望对同一批测试句子系统施加某项模型控制——例如偏移 F0 条件、缩放 F1 轨迹或改变声门源 `R_d`——再生成用于声学测量或知觉实验的一组音频。这就是 `manipulate` 命令的用途。它的特点是“可审计”：每次运行都会记录所用 checkpoint（含哈希值）、每个条件的完整参数，以及输出中是否出现硬限幅采样点，便于复核和报告。

一个最小的例子——把随机源分支的数字增益降低 6 dB：

```bash
uv run aris manipulate EXP CHECKPOINT OUTPUT \
  --variant 'less_noise:noise_gain_db=-6'
```

`EXP` 是包含 `experiment.json` 的实验目录，`CHECKPOINT` 通常是 `EXP/runs/checkpoints/last.ckpt`，`OUTPUT` 是一个还不存在的目录（后面会解释为什么必须是新目录）。安装与训练流程见仓库 [README](../README.md)。

## 第一步：看看你的模型能改什么

不同的模型架构暴露的"控制柄"不一样，所以动手之前先问一下：

```bash
uv run aris controls EXP
```

它会列出你这个实验的模型支持的所有参数，以及各自的取值范围和默认值。加 `--json` 可以得到机器可读的版本，适合写进分析脚本。

三种模型的能力可以概括为一个递进关系：

| 模型 | 可用控制 |
|---|---|
| `ddsp` | 音高、输出增益、噪声增益 |
| `golf` | 以上三项 + 声门 `R_d` |
| `aria-golf` | 以上四项 + 显式 F1、F2、谱倾斜 |

`aria-golf` 是 ARIS 解码器在代码里的名字。如果你的研究问题涉及共振峰，就需要用它训练——只有它的解析声道级联真正暴露了逐帧的 F1/F2 轨迹。

检查其实有两层：`controls` 命令根据实验声明的模型类型做静态检查；真正运行时还会载入 checkpoint，确认 decoder 确实有对应的执行路径。参数名写错、数值越界、模型不支持，都会直接报错，不会静默忽略——这是有意设计的，避免你以为改了什么其实没改。

## 每个参数是什么意思

所有范围都是闭区间。要提醒一句：这些是软件接受的范围，不是验证过的"语音学安全范围"——正式实验建议从小变化开始，先做声学测量和试听。

| 参数 | 范围（默认） | 支持模型 | 含义 |
|---|---|---|---|
| `pitch_semitones` | `-36..36`（`0`） | 全部 | 音高，以半音计 |
| `output_gain_db` | `-24..12` dB（`0`） | 全部 | 保存 WAV 前施加的数字增益 |
| `noise_gain_db` | `-24..24` dB（`0`） | 全部 | 噪声源分支的增益 |
| `glottal_rd_scale` | `0.5..2.0`（`1.0`） | `golf`、`aria-golf` | 声门 `R_d` 的乘性缩放 |
| `f1_scale` | `0.7..1.3`（`1.0`） | 仅 `aria-golf` | F1 轨迹的乘性缩放 |
| `f2_scale` | `0.7..1.3`（`1.0`） | 仅 `aria-golf` | F2 轨迹的乘性缩放 |
| `f1_hz` | `150..1300`（`500`） | 仅 `aria-golf` | F1 的绝对目标频率（Hz） |
| `f2_hz` | `600..3200`（`1500`） | 仅 `aria-golf` | F2 的绝对目标频率（Hz） |
| `tilt_alpha_delta` | `-0.25..0.25`（`0`） | 仅 `aria-golf` | 谱倾斜系数的加性偏移 |

逐个说说值得注意的地方：

**`pitch_semitones`** 把有声帧的 F0 条件乘以 `2^(半音/12)`；无声帧保持为 0，不会被"提"出声来。它不改变时长。输出中实测的 F0 通常接近但不保证严格等于目标比例，正式实验请复测。

**`output_gain_db`** 只是在保存 WAV 前给数字波形乘一个增益。它既不是说话人的发声强度，也不等于经耳机或扬声器呈现时的声压级或感知响度。正增益容易触发削波（见后文）。

**`noise_gain_db`** 改变随机噪声分支的强度，听感上影响气声/噪声感。但它不等于直接设定 HNR 或气声度这些校准过的量；改完之后应该在输出音频上测 HNR、CPP 等指标。

**`glottal_rd_scale`** 缩放模型预测的 LF/GOLF 声门 `R_d` 参数（受模型内部表范围约束），改变声门脉冲形状和声源谱包络。方向和效应量以输出上的测量为准——比如谱倾斜、H1–H2——这个值本身不是 EGG 或声门几何的测量。

**`f1_scale` / `f2_scale`** 按帧缩放 ARIS 的显式共振峰轨迹，保持原有轮廓的相对变化，而不是设一个固定的 Hz 目标。轨迹会被夹在 decoder 配置的频率范围内，所以接近边界时比例可能不再严格成立。F1、F2 一起动的时候，注意可能同时改变元音类别和说话人线索。

**`f1_hz` / `f2_hz`** 把共振峰设为一个绝对 Hz 目标，完全覆盖该帧原有的轨迹，适合搭建跨条目、跨说话人一致的 Hz 连续统。`f1_scale`/`f2_scale` 则保留每帧原有轮廓的相对形状、只做整体比例平移，更适合"保留自然语调、只整体移一档"的操控。同一个共振峰不能同时指定绝对值和比例——`f1_hz` 和 `f1_scale` 一起用会报错。

**`tilt_alpha_delta`** 给一阶谱倾斜滤波器的 `alpha` 系数加偏移。`alpha` 不是 dB/octave——相同的增量在不同基线和频段上产生的声学变化不一定相同，所以效应还是要测。

### 一个常见误区：普通 GOLF 的 LPC 不是共振峰控制

普通 `golf` 的声道滤波器预测一组 LPC/全极点系数，谱包络里当然会出现共振峰，但它没有名为 F1、F2 的独立控制柄——改某个 LPC 系数不能被解释为"只改了 F1"。这就是为什么 `f1_scale`、`f2_scale`、`tilt_alpha_delta` 只对 `aria-golf` 开放。如果你的研究需要显式共振峰操控，请用 ARIS-GOLF 训练，并在输出上用独立的共振峰跟踪方法复测。

## 设计一组条件

真实实验很少只有一个条件。每个 `--variant` 定义一个命名条件，格式是：

```text
条件名:参数=值,参数=值
```

条件名会直接成为输出子目录名，所以要求是安全的 ASCII 名字：以字母或数字开头，只用字母、数字、`_`、`-`，且一次运行内不能重复。建议取能自我说明的名字，比如 `f1_up_10`、`breathy_mild`，几个月后回看数据时你会感谢自己。

`--variant` 可以重复任意多次，一个条件里也可以组合多个参数：

```bash
uv run aris manipulate EXP CHECKPOINT OUTPUT \
  --variant 'less_noise:noise_gain_db=-6' \
  --variant 'raised_clean:pitch_semitones=3,noise_gain_db=-6,output_gain_db=-3' \
  --variant 'source_shift:glottal_rd_scale=1.2,noise_gain_db=-3'
```

做连续统（continuum）刺激时，就是一档一个条件。比如一个五档的 F1 连续统：

```bash
uv run aris manipulate EXP CHECKPOINT OUTPUT \
  --variant 'f1_085:f1_scale=0.85' \
  --variant 'f1_092:f1_scale=0.92' \
  --variant 'f1_100:f1_scale=1.00' \
  --variant 'f1_108:f1_scale=1.08' \
  --variant 'f1_115:f1_scale=1.15'
```

`--semitones` 是音高专用的简写：`--semitones -4 4` 会自动生成名为 `pitch_minus_4st` 和 `pitch_plus_4st` 的条件。想让音高和其他参数组成联合条件时，改用 `--variant` 把 `pitch_semitones` 写进去。注意联合条件回答的是"这组操作的总效应"——想估计单个因素的贡献，还是要分别生成只改一个参数的条件，外加一个不改任何参数的 baseline。

正式跑之前，可以先用 `--dry-run` 检查一遍拼写和取值：

```bash
uv run aris manipulate EXP CHECKPOINT OUTPUT \
  --variant 'less_noise:noise_gain_db=-6' \
  --dry-run
```

它会验证条件名、范围和模型声明并打印将要执行的命令，但不载入 checkpoint，所以最终的 decoder 能力检查还是发生在真正运行时。

如果你已经用 `synthesize` 生成过 baseline 重建，可以顺便让它生成一个试听网页，把原音、重建和所有条件并排放好：

```bash
uv run aris manipulate EXP CHECKPOINT OUTPUT/manipulations \
  --semitones -4 4 \
  --variant 'less_noise:noise_gain_db=-6' \
  --baseline OUTPUT/reconstruction \
  --report OUTPUT/manipulation.html
```

不指定 `--report` 时网页默认写到输出目录下的 `comparison.html`。这个页面适合质检和给合作者预览材料；它没有随机化、盲法或响度校准，不能直接当正式知觉实验用。

顺带一提：`synthesize` 命令本身也接受 `--semitones` 和可重复的 `--control 参数=值`，适合快速试听一个组合。但批量出条件请用 `manipulate`——它多做的那些记录（下一节）正是科研需要的。

## 输出长什么样，元数据里有什么

一次运行的输出结构类似：

```text
OUTPUT/
├── manipulation.json
├── less_noise/
│   ├── RECORD_ID.wav
│   └── _render.json
└── raised_clean/
    ├── RECORD_ID.wav
    └── _render.json
```

**输出目录必须是新的。** 如果 `OUTPUT` 已存在，命令会直接报错。这是为了保证一个目录里的所有音频来自同一次运行、同一个 checkpoint——你不能往旧目录里"追加"条件，改了参数就换一个新的、名字能区分的输出路径。渲染过程中系统还会反复校验 checkpoint 的哈希，如果训练进程恰好在覆盖它，运行会中止而不是混入不同模型状态的输出。因此正式实验建议复制一份明确命名的 checkpoint，别直接用还在被训练更新的 `last.ckpt`。

顶层的 `manipulation.json` 是条件级记录：模型类型、实验路径、数据集 fingerprint、checkpoint 路径和 SHA-256，以及每个条件的名称、完整控制值、F0 scale 和子目录。每个条件目录里的 `_render.json` 是渲染级记录：传给 decoder 的控制值、运行时检测到的能力、写出的文件数，以及每个文件的削波统计。

有一个细节：音高是通过数据侧的 F0 缩放实现的，不走 decoder 的运行时控制通道。所以核对完整的音高条件要看 `manipulation.json` 里的 `controls`、`semitones` 和 `f0_scale`，不要只看 `_render.json`。

### 关于削波（clipping）

保存 16-bit 音频前，系统先应用 `output_gain_db`，统计绝对值超过 1 的样本数，再把波形硬限制到 `[-1, 1]`。在 `_render.json` 里：

- `peak_after_gain_unclipped > 1`：增益后的理论峰值已越界；
- `clipped_samples > 0`：落盘的音频确实被硬截断了。

削波不只是"更响"——它会引入失真和额外的频谱变化，污染你本想单独操控的那个参数。正式比较中建议要求所有条件 `clipped_samples == 0`；出现削波就降低 `output_gain_db` 或重新设计条件，然后完整重渲染。不要事后只对出问题的 WAV 单独归一化，那会让各条件的处理不一致。另外注意 prepared dataset 里也有一个 `clipped_fraction`，那个描述的是输入数据质量，和这里的输出削波不是一回事。

归档结果时，把 `experiment.json`、`manipulation.json`、各 `_render.json`、checkpoint 哈希和分析脚本一起存下来，而不是只留 WAV——将来审稿人（或你自己）想复现时会用到。

关于确定性：同一台机器、同一块 GPU、同一版驱动上，用同一个 checkpoint 和同样的参数重跑 `manipulate`，输出是逐比特相同的（推理过程不含随机采样）；但这不是跨硬件的保证——换 GPU 型号或驱动版本，浮点结果可能有细微差异，如果这对你的分析有影响，把所用硬件记下来。

## 科学使用建议

- 先生成不改任何参数的 baseline 重建，和原音、各条件并排听：模型的重建误差和你施加的控制效应是两回事，得分开看。
- 软件允许的范围不是推荐效应量。预试可以从小处起步——音高 ±2 到 ±4 半音、`R_d` 比例 0.8..1.2、F1/F2 比例 0.9..1.1——再根据可懂度和独立测量调整。
- 一次只改一个参数最容易解释；多参数条件用来检验明确的组合假设，并保留对应的单参数条件作参照。
- 改了就测：`noise_gain_db` 之后测 HNR/CPP，`glottal_rd_scale` 之后测谱倾斜和 H1–H2，F1/F2 之后用独立的共振峰跟踪加人工抽查。同时记得这些声学量也不是声门生理的真值。
- 如果响度不是你的研究变量，用一个事先定义、对所有条件一致的响度方案，并保证不削波。
- 成对比较务必用同一个 checkpoint——`manipulation.json` 里的 SHA-256 就是为此存在的。
- 论文里报告：模型类型、参数取值、条件名、数据 split、F0 提取方法、checkpoint SHA-256、削波检查结果和排除标准。

最后一点措辞建议：最稳妥的说法是"在该 checkpoint 下施加了某个模型控制，输出呈现了某种经测量的声学变化"，而不是直接声称"独立操控了声带、声道或某个感知属性"。前者是你确实做到并且能证明的事。

## 功能边界

两点边界说清楚，别以为工具能做但其实做不到：

- **没有时长/语速控制。** ARIS 不改变音频时长；元音时长、VOT 这类需要操控时长的实验设计，要在 ARIS 之前或之后用别的工具处理。
- **所有控制作用于整条渲染音频，不能只改句子里的一段。** 想只给一个词升调、其余部分不动，`manipulate` 做不到局部操控。要做到目标单元级别的刺激，提前在切分数据（`uv run aris split`/`uv run aris prepare`）阶段把目标单元切成独立的一条，训练和合成都以这条为单位。
