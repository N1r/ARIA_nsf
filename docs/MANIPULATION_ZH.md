# 面向语音学家的参数 Manipulation 指南

PhonLab-DDSP 的 manipulation 是“给同一 checkpoint 和同一测试材料施加命名控制条件，再生成可审计音频”。它适合构造试听材料、检查模型可控性和形成后续实验假设；它本身不证明某个控制等同于独立的生理或构音操作。

## Capability-driven：先问模型能做什么

控制采用两层检查：

1. 根据实验 `experiment.json` 中的模型类型，检查参数名、数值范围和模型支持矩阵；
2. 真正载入 checkpoint 后，再检查 decoder 是否具有相应的振荡器、噪声滤波器或显式 formant 接口。

未知参数、越界值、模型不支持的参数，或 checkpoint 缺少真实执行路径时都会报错，不会静默忽略。先查看实验声明的能力：

```bash
.venv/bin/phonlab controls EXP
.venv/bin/phonlab controls EXP --json
```

这里的 `EXP` 是包含 `experiment.json` 的实验目录。第一条适合人读，第二条适合脚本记录。`CHECKPOINT` 通常是
`EXP/runs/checkpoints/last.ckpt`；用于正式实验时建议保存明确命名的 checkpoint，
不要让训练进程继续覆盖它。最终能否执行仍由载入 checkpoint 后的运行时检查确认。

## 参数、范围与解释

所有范围都是闭区间。它们是软件接受范围，不是已经验证的语音学“安全效应范围”；正式实验应从更小变化开始做声学测量和试听预试。

| 参数 | 范围；默认值 | 支持模型 | 操作与科学解释 |
|---|---|---|---|
| `pitch_semitones` | `-36..36`；`0` | `golf`、`ddsp`、`aria-golf` | 仅把有声帧的 F0 条件乘以 `2^(st/12)`；F0 为 0 的无声帧仍为 0。它不改变时长，也不保证输出中实测 F0 完全按同一比例变化。 |
| `output_gain_db` | `-24..12` dB；`0` | 三种模型 | 在保存前对最终波形乘增益。它是播放/输出电平控制，不是声门能量或说话强度的独立生理变量。正增益可能导致 clipping。 |
| `noise_gain_db` | `-24..24` dB；`0` | 三种模型 | 改变随机源分支的增益。它通常影响噪声感、周期/非周期成分关系，但不等于直接设定经过校准的 HNR、气声度或湍流强度；应在输出音频上重新测量。 |
| `glottal_rd_scale` | `0.5..2.0`；`1.0` | `golf`、`aria-golf` | 对模型预测的 LF/GOLF 声门 `R_d` 表位置做乘性偏移，并受可用表范围约束。它会改变脉冲形状和声源谱包络，但方向与效应量应在输出上测量；这里的值不是 EGG 或声门几何的直接测量。 |
| `f1_scale` | `0.7..1.3`；`1.0` | 仅 `aria-golf` | 按帧乘 ARIA 显式解析 F1 轨迹，并夹在该 decoder 配置的 F1 频率范围内；保持原轮廓的相对变化，不是一个固定 Hz 目标。 |
| `f2_scale` | `0.7..1.3`；`1.0` | 仅 `aria-golf` | 与 F1 相同，但作用于显式解析 F2。F1/F2 同时变化可能改变元音类别和说话人线索。 |
| `tilt_alpha_delta` | `-0.25..0.25`；`0` | 仅 `aria-golf` | 对 ARIA 一阶谱倾斜滤波器的 `alpha` 加偏移。`alpha` 不是 dB/octave；相同增量在不同基线和频率处不一定产生相同声学变化。 |

模型支持矩阵可以概括为：

| 模型 | 可用控制 |
|---|---|
| `ddsp` | pitch、输出增益、噪声增益 |
| `golf` | DDSP 的三项 + 声门 `R_d` |
| `aria-golf` | GOLF 的四项 + 显式 F1、F2、谱倾斜 |

### 关于普通 GOLF 的重要限制

**不要把普通 GOLF 的 LPC 当成显式 formant 控制。** 普通 `golf` 的声道末端滤波器预测一组 LPC/全极点系数；其中可能出现共振峰，但它没有名为 F1/F2 的独立解析控制柄。直接改某个 LPC 系数不能被标注为“只改变 F1”或“只改变 F2”。

只有 `aria-golf` 的解析声道级联明确暴露 F1/F2 轨迹和 `scale_formants` 执行路径，因此 `f1_scale`、`f2_scale` 与 `tilt_alpha_delta` 只对它开放。如果研究问题要求显式 formant 操控，应选择 ARIA-GOLF，并在输出上用独立声学方法复测 formant。

自动化测试已经验证真实 `VocalTractCascade` 的 F1、F2 和 tilt 轨迹按请求
变化。ARIA-GOLF 的 Slurm Job `4600588` 在 F024 上训练 400 steps；最佳
checkpoint 随后由 Job `4600593` 渲染 F1、F2 和 tilt 各一组成对条件，全部
4/4 held-out WAV 发生变化且 clipping 为 0。`make verify-aria` 会重新核对
checkpoint 哈希、dataset fingerprint、运行时 capability、decoder 调用、
成对音频差异和削波。这个工程闭环仍不能替代独立 formant 估计或感知实验；
正式研究需要另外复测实际声学变化的方向和幅度。

## CLI：生成一个或多个命名条件

最小示例：

```bash
.venv/bin/phonlab manipulate EXP CHECKPOINT OUTPUT \
  --variant 'less_noise:noise_gain_db=-6'
```

`OUTPUT` 必须尚不存在，避免把不同运行混进同一目录。条件写法为：

```text
条件名:参数=值,参数=值
```

条件名同时成为输出子目录名：必须以 ASCII 字母或数字开头，只使用字母、数字、`_`、`-`，并且在一次运行中唯一。可以重复 `--variant`，也可以在一个条件内组合多个参数：

```bash
.venv/bin/phonlab manipulate EXP CHECKPOINT OUTPUT \
  --semitones -4 4 \
  --variant 'less_noise:noise_gain_db=-6' \
  --variant 'raised_clean:pitch_semitones=3,noise_gain_db=-6,output_gain_db=-3' \
  --variant 'source_shift:glottal_rd_scale=1.2,noise_gain_db=-3'
```

对 ARIA-GOLF：

```bash
.venv/bin/phonlab manipulate EXP CHECKPOINT OUTPUT \
  --variant 'vowel_shift:f1_scale=1.05,f2_scale=0.95' \
  --variant 'vowel_tilt:f1_scale=1.05,f2_scale=0.95,tilt_alpha_delta=0.05'
```

`--semitones -4 4` 是兼容旧流程的音高 sweep，会产生 `pitch_minus_4st` 和 `pitch_plus_4st`。把 pitch 写进 `--variant` 则可与其他参数组成一个联合条件。联合条件回答的是“这组操作的总效应”；如果要估计单个因素，仍需分别生成只改一个参数的条件和未改 baseline。

先检查参数和底层命令而不运行 GPU 推理：

```bash
.venv/bin/phonlab manipulate EXP CHECKPOINT OUTPUT \
  --variant 'less_noise:noise_gain_db=-6' \
  --dry-run
```

dry-run 能检查命名、范围和实验模型声明，但不会载入 checkpoint，因此不能替代真正运行时的 decoder capability 检查。

已有 baseline 重建时，可以同时生成试听页面：

```bash
.venv/bin/phonlab manipulate EXP CHECKPOINT OUTPUT/manipulations \
  --semitones -4 4 \
  --variant 'less_noise:noise_gain_db=-6' \
  --baseline OUTPUT/reconstruction \
  --report OUTPUT/manipulation.html
```

试听页并未提供随机化、盲法、响度校准或被试记录，只能用作质检和材料预览，不能直接替代正式知觉实验。

单次重建也可重复使用 `--control` 组合非音高参数：

```bash
.venv/bin/phonlab synthesize EXP CHECKPOINT OUTPUT \
  --semitones 3 \
  --control noise_gain_db=-6 \
  --control output_gain_db=-3
```

批量条件优先使用 `manipulate`，因为它会额外生成条件级 provenance。

## Slurm：用 `init-postprocess` 生成 GPU 作业包

checkpoint 推理应在 GPU 计算节点执行。登录节点只生成作业包：

```bash
.venv/bin/phonlab init-postprocess EXP CHECKPOINT OUTPUT \
  --semitones -4 4 \
  --variant 'less_noise:noise_gain_db=-6' \
  --variant 'source_shift:glottal_rd_scale=1.2'
```

`init-postprocess` 若不指定 `--semitones`，默认仍包括 `-4` 和 `+4` 半音。它只创建作业目录，不会自动提交。命令会打印作业包路径；确认路径和其中的 `job.json`、`train.slurm` 后再提交：

```bash
.venv/bin/phonlab submit-job PRINTED_JOB_BUNDLE_PATH --confirm
```

只生成非 F0 条件时，显式加入 `--no-pitch`，并至少提供一个 `--variant`。
提交命令返回数字 `JOB_ID` 后可查询状态和受限长度日志：

```bash
.venv/bin/phonlab job-status JOB_ID --json
.venv/bin/phonlab job-log PRINTED_JOB_BUNDLE_PATH JOB_ID --tail-lines 200
```

还可用 `--partition`、`--gres`、`--time`、`--cpus`、`--memory` 和 `--exclude` 设置集群资源。生成的作业按顺序执行：

1. baseline checkpoint reconstruction；
2. 原音与 reconstruction 对照页；
3. 所有 pitch/命名 manipulation 条件及试听页；
4. 训练 metrics 报告。

输出目录和作业包都要求是新目录。同一输出路径不能用来“追加条件”；改变参数后应使用新的、可辨识的输出路径。
成功后主要页面位于 `OUTPUT/reconstruction.html`、`OUTPUT/manipulation.html`
和 `OUTPUT/metrics.html`，条件级记录位于
`OUTPUT/manipulations/manipulation.json`。

## GUI：不用拼接命令

启动本地工作台：

```bash
.venv/bin/phonlab gui
```

在“7 · 推理与 Manipulation”卡片中：

1. 输入实验目录并点击“查看实验模型声明的参数”；
2. 填写 checkpoint、新输出目录和半音条件；
3. 在“其他参数条件”中每行写一个 `名称:参数=值,...`；
4. 生成 GPU 后处理作业；
5. 到“6 · Slurm 作业中心”选择返回的作业包，显式确认后提交。

GUI 和 CLI 使用同一套参数解析、范围验证和作业生成函数。GUI 只绑定 `127.0.0.1`；远程集群上应通过 SSH 端口转发访问，不要把具有本地文件权限的工作台公开到网络。

## 输出、元数据与 clipping

一次 manipulation 的核心结构类似：

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

`manipulation.json` 是条件级记录，包含：

- 模型、实验路径和数据集 fingerprint；
- checkpoint 路径与 SHA-256；
- 每个条件的名称、完整控制值、F0 scale、标签和子目录；
- 无声帧保持为 0 的策略。

每个条件目录中的 `_render.json` 是渲染级记录，包含：

- 传入 waveform/decoder callback 的非 F0 控制；
- 载入模型检测到的 `runtime_capabilities`；
- decoder control hook 的调用次数和写出的文件数；
- 总样本数、`clipped_samples` 与 `clipped_fraction`；
- 每个文件的相对路径、增益前峰值、未截断的增益后峰值、样本数和削波数。

pitch 是通过数据侧的 F0 scale 应用的，因此完整 pitch 条件以 `manipulation.json` 中的 `controls`、`semitones` 和 `f0_scale` 为准；不要只看 `_render.json` 的 runtime-control 字段。

保存 PCM16 前，系统先计算 `output_gain_db`，统计绝对值大于 1 的样本，然后把波形硬限制到 `[-1, 1]`。所以：

- `peak_after_gain_unclipped > 1` 表示增益后的理论峰值已越界；
- `clipped_samples > 0` 表示落盘音频已经发生硬截断；
- clipping 不只是“更响”，还会引入失真和频谱变化，破坏参数独立性。

正式比较中最好要求所有条件的 `clipped_samples == 0`。若出现 clipping，应降低 `output_gain_db` 或重新设计条件并完整重渲染；不要事后只对有问题的 WAV 单独归一化。还要注意，prepared dataset 中的 `clipped_fraction` 描述输入数据质量，不等同于 `_render.json` 的推理输出 clipping。

Slurm 作业包的 `job.json` 另外记录数据 fingerprint、checkpoint SHA-256、全部条件和资源参数。归档研究结果时，应同时保存 `experiment.json`、`job.json`、`manipulation.json`、各 `_render.json`、checkpoint 哈希和分析脚本，而不是只保留 WAV。

渲染结束后可用独立验收器检查 schema、每条 WAV 是否实际变化、hook 次数、
clipping、−/+ 对称噪声条件和纯输出增益条件，并保存 JSON：

```bash
.venv/bin/python tools/check_control_manipulation.py \
  OUTPUT/reconstruction \
  OUTPUT/manipulations \
  --output OUTPUT/control-acceptance.json
```

成功时只打印 `PHONLAB_CONTROL_MANIPULATION_OK`。失败时报告逐项问题并返回非零
状态；不要只看试听页就宣称参数已生效。

## 科学使用建议

- 先生成未改 reconstruction，并把它与原音和所有条件并排检查；模型重建误差与控制效应是两个不同因素。
- 软件允许范围不是推荐效应量。预试可从较小变化开始，例如 pitch `±2/±4` st、`R_d` 比例约 `0.8..1.2`、F1/F2 比例约 `0.9..1.1`，再根据研究对象、可懂度和独立测量调整。
- 一次只改一个参数有利于解释主效应；多参数条件用于检验明确的组合假设，并应保留对应的单参数条件。
- `noise_gain_db` 后应测量 HNR/CPP 或任务相关噪声指标；`glottal_rd_scale` 后应测量谱倾斜、H1–H2 等，并说明这些声学量也不是声门生理真值。
- F1/F2 操作后用独立 formant 跟踪和人工抽查确认实际输出；接近模型频率边界时可能发生夹取，比例变化不再严格成立。
- `output_gain_db` 会改变响度并可能造成 clipping。若响度不是研究变量，应在不削波前提下采用事先定义、对所有条件一致的响度校准方案。
- 报告模型类型、参数范围、条件名称、数据 split、F0 提取方法、checkpoint SHA-256、clipping 结果和排除标准。

最稳妥的表述是“在该 checkpoint 下施加了某个模型控制，输出呈现了某种经测量的声学变化”，而不是直接声称“独立操控了声带、声道或感知属性”。
