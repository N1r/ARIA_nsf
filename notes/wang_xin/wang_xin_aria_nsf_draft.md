# ARIA-NSF 给王鑫老师邮件初稿

日期：2026-06-12  
用途：邮件正文 + 可附在邮件后的技术说明草稿。  
当前建议：真正发邮件时，正文保留「邮件正文初稿」部分即可；「技术说明」可以作为附件或网页 demo 的补充说明。

## 邮件主题备选

1. 请教一个面向语音学实验的小数据 neural source-filter / Klatt-style synthesizer 方向
2. 关于小数据、可控语音合成实验刺激生成的一点初步结果，想请您指正
3. 受 NSF 启发的可微 Klatt-style synthesizer 初步尝试，想向您请教

## 邮件正文初稿

王鑫老师，您好！

距离上一次给您写信，转眼已经过去三年了。最初正是受到您在 neural source-filter 方向工作的启发，我才开始持续关注这一领域。我本身是语音学背景，计算机和 DSP 基础并不算强，因此在理解滤波器、声码器和神经源滤波模型时，确实花了不少时间。但这几年里，我一直断断续续地学习和尝试，也希望把这些方法真正用于语音学实验中一些具体而实际的问题。

我之所以一直关注这个方向，是因为在语音学研究中，speech synthesis 往往有几个比较特殊的需求。首先是数据量问题。语音学实验中希望合成或操控的材料，很多时候并不是大语料中容易覆盖的自然语音，而是研究者自己录制的、围绕特定现象设计的实验材料，例如 tone sandhi、category perception、prosodic prominence，或者某些特定发声类型。这样的数据通常只有 30 分钟到 1 小时，甚至更少。

其次是音质问题。实验刺激需要被被试反复听辨，因此我们在设计阶段会非常仔细地检查音频，希望尽可能减少 artifacts。如果合成刺激与自然语音差距过大，可能会影响实验结果的生态效度与可解释性。第三是可控性问题。语音学家通常并不只是希望“生成一段自然语音”，而是希望能够独立、精细地操控某些声学维度。例如，通过 F0 操控声调和语调，通过 F1/F2 的独立变化构建元音连续统，而不是对整体频谱做简单缩放；又如，通过声源或谱倾斜相关参数研究 breathy、creaky、pressed 等发声类型。因此，对语音学实验而言，合成系统不仅要自然，还需要可解释、可操控，并且不同声学维度之间的耦合最好尽可能小。

这也是我近几年一直关注 modern speech synthesis for phonetic / speech-science studies 的原因。最近几年，KTH、Aalto 等团队在这一方向上的工作给了我很多启发。例如 Zofia Malisz 等人关于 speech technology and phonetics interface 的讨论，以及近期的 HiFi-Glot，它尝试用 differentiable resonant / all-pole filters 实现 neural formant synthesis，在提升音质的同时保留 formant-level control。GOLF 这类工作也让我意识到，现代神经语音合成与传统 source-filter / formant synthesis 之间其实存在一个很有潜力的交汇点：用可微 DSP 模块保留可解释结构，再让神经网络学习难以显式建模的 residual 部分。

过去一段时间里，我尝试过几种不同方案，包括基于原版 DDSP 的 harmonic-plus-noise synthesis，在模型中 condition F1/F2；也尝试参考 HiFi-Glot 的配置，在我们自己的小数据上复现；此外，还尝试过基于 NSF 或 GOLF 思路的 source-filter 方法。最近，最后这一类方法终于取得了一些让我比较欣喜的初步结果，所以很想向您分享，也希望能得到您的一些建议。

目前我的模型可以简单理解为一个“可微的 Klatt-style synthesizer”。其中，F0、F1、F2、B1、B2、energy / gain、spectral tilt / alpha，以及声门源相关的 Rd 等参数，被设计为 analytic 或 deterministic 的可控参数；而其余较难显式建模的部分，则交给神经网络从数据中学习。更具体地说，模型先由 F0 和 Rd 生成 LF-style glottal excitation，同时生成一个受控的 noise component；二者合并后再通过 vocal-tract cascade / LPC filter。当前配置中，vocal tract 部分包含 deterministic 的低阶 formant all-pole sections，以及 8 个 learned biquad sections，因此总 LPC order 为

$$
\mathrm{order}=2(3+8)=22.
$$

我希望这种设计能够把传统 formant / source-filter synthesis 的可解释性，与现代神经模型的 residual modeling 能力结合起来。对我而言，比较重要的一点不是完全追求通用 TTS 或 speaker-independent vocoder，而是面向语音学实验室常见的 small-data、single-speaker、strong-control 场景。

我也做了一个简单的 HTML demo，用来展示模型的 copy synthesis 和 phonetic manipulation 效果，包括 F0、F1/F2 以及 Rd 等维度的操控结果：

demo 页面：<https://n1r.github.io/ARIA_nsf/>

初步来看，这个模型在小数据场景下似乎比较适合语音学实验的需求。相较于我目前理解中的 HiFi-Glot，这个方案可能有三点不同的优势。第一，对数据量的需求比较小。本次实验中，我只使用了自己在北京语言大学张劲松老师课题组时录制的一小时普通话单字词材料，约 1400 个词，总有效音频大约 30 分钟。后续测试中，我发现即使用 5 分钟或 10 分钟的数据，模型也可以收敛，而且音质下降并不明显。HiFi-Glot 这类模型在音质、full-band synthesis 和 generalizability 上非常强，但它的目标和训练配置明显更偏向大规模、高容量的 neural vocoder；而我的目标更偏向语音学实验室中常见的小数据、单说话人、强控制场景。

第二，由于关键参数是 analytic / deterministic 的，声学操控误差比较小。例如，F0 本身来自 deterministic source，误差通常只有几 Hz；F1/F2 的控制误差在目前材料上也基本可以控制在 50 Hz 以内。模型还可以通过 Rd 控制声门源，从而实现 pressed、breathy 等发声维度上的连续操控。对语音学实验而言，这种低误差、低耦合的参数控制，可能比纯粹追求通用生成能力更重要。

第三，模型本身比较小，对训练资源的需求也比较低。我在自己的笔记本 RTX 4060 上就可以完成训练；如果移动到服务器 A100 上，训练通常只需要几十分钟。这一点对语音学实验室来说可能也比较重要，因为很多实验室并没有持续使用大规模 GPU 资源的条件。

当然，目前这个工作还处在比较初步的阶段，很多地方仍然需要进一步验证。例如，如何更系统地评估 copy synthesis 的自然度，如何设计 perceptual test 来验证 manipulation 的可靠性，以及如何在保持可控性的同时进一步提升音质，都是我接下来想继续推进的问题。

因此，我很希望能向您请教：从 NSF、source-filter modeling 以及神经声码器的角度来看，您觉得这个方向是否有继续推进的学术价值？目前这样的设计中，是否有明显的技术风险，或者我因为 DSP 和神经声码器知识不够扎实而忽略的问题？如果您方便，也非常希望听到您的批评和建议。

再次感谢您早期工作的启发。对我这样一个语音学背景、但一直希望理解并使用这些方法的人来说，您的工作、文章和 slides 确实给了我很大的帮助和鼓励。

祝好！  
Yiran Ding（丁怡然）  
Leiden University Centre for Linguistics

## 可附技术说明初稿

### 1. 背景定位

这个项目可以暂时定位为：

> A small-data, controllable, differentiable Klatt-style neural source-filter synthesizer for phonetic stimulus generation.

它不是一个通用 TTS/vocoder benchmark，而是面向 speech-science experiments 的 controllable copy synthesis / manipulation tool。核心诉求是：

- 小数据：适合 5-30 分钟、单说话人、实验材料级数据。
- 高可控：F0、F1/F2、bandwidth、gain、tilt / alpha、Rd 等参数尽量显式、可解释。
- 低耦合：改 F1/F2 时尽量不破坏 source；改 Rd 时尽量不引入不可解释的 spectral shift。
- 资源友好：可在消费级 GPU 上训练，方便语音学实验室使用。

### 2. 与 NSF、GOLF、HiFi-Glot 的关系

NSF 提供了最重要的启发：把 waveform generation 写成 source module、filter module 和 conditional module 的组合，并用 spectral loss 直接训练 waveform model。我的模型保留了 source-filter 的分解思想，但把 filter module 从较黑箱的 neural filter，进一步约束为 formant / LPC / biquad cascade。

GOLF 提供了另一个很接近的参照：用 glottal-flow wavetable 作为 harmonic source，用 LPC filter 模拟 vocal tract，并强调 DDSP 结构带来的解释性、效率和较低参数量。我的模型与 GOLF 的共同点是都希望将 human voice production 的物理结构放进合成器；不同点是我的目标不是 singing voice synthesis，而是 speech-science stimulus generation，并且我更强调 F1/F2、Rd 等实验变量的独立操控。

HiFi-Glot 则是更强的 high-fidelity neural formant synthesis 参照。它使用 neural excitation / NSF-HiFiGAN decoder 和 differentiable all-pole filters，在大规模数据和强 neural vocoder 配置下实现高质量、可操控的 formant synthesis。我的模型可以被看作一个更小、更 Klatt-like、更面向单说话人实验刺激的版本：牺牲一部分通用性，换取小数据可训练性和更直接的参数控制。

### 3. 当前模型公式草稿

给定采样率 \(f_s\)，逐样本相位由 deterministic F0 积分得到：

$$
\phi_n = \phi_{n-1} + 2\pi \frac{F_{0,n}}{f_s}.
$$

声门激励由 LF-style source 产生，其中 \(R_d\) 控制声门脉冲形状：

$$
e_n = g_{\mathrm{LF}}(\phi_n; R_{d,n}).
$$

噪声分支由 white noise \(\epsilon_n\) 经过 learned / conditioned noise shaper 得到：

$$
\eta_n = \mathcal{N}(\epsilon_n; m_n).
$$

我的当前结构中，glottal excitation 与 shaped noise 先合并，再进入 vocal-tract cascade：

$$
u_n = e_n + \eta_n,
$$

$$
y_n = H_{\mathrm{VT}}(z; v_n)\,u_n.
$$

当前 vocal-tract cascade 可写成 deterministic sections 与 learned sections 的乘积：

$$
H_{\mathrm{VT}}(z; v_n)
= H_{\mathrm{det}}(z; F_{1,n},F_{2,n},B_{1,n},B_{2,n},\alpha_n,g_n)
\prod_{i=1}^{8} H_i(z; \theta_{i,n}).
$$

其中 \(H_{\mathrm{det}}\) 包含当前配置中的 3 个 deterministic biquad sections，显式暴露 F1/F2 相关 all-pole control；8 个 learned biquads 用于补偿较难手工建模的 residual spectral structure。因此：

$$
\mathrm{LPC\ order}=2(3+8)=22.
$$

训练目标目前可以简化写为 multi-scale spectral reconstruction loss：

$$
\mathcal{L}_{\mathrm{MSSTFT}}
=
\sum_{r\in\mathcal{R}}
\left(
\left\| |S_r(x)|-|S_r(y)| \right\|_1
+
\left\| \log(|S_r(x)|+\epsilon)-\log(|S_r(y)|+\epsilon) \right\|_1
\right),
$$

其中 \(x\) 是目标 waveform，\(y\) 是合成 waveform，\(S_r\) 表示第 \(r\) 个 STFT resolution。若后续加入更严格的操控评估，可以额外报告：

$$
E_{F_i}=\frac{1}{T}\sum_t
\left|
\widehat{F_i}(y_t)-F_i^{\mathrm{target}}(t)
\right|,
\quad i\in\{1,2\},
$$

以及 \(F0\)、\(R_d\)、spectral tilt / alpha 的 tracking error 或 perceptual consistency。

### 4. 目前最值得强调的优势

1. 小数据适应性：目前 30 分钟普通话单字词材料即可得到可用结果，5-10 分钟数据也能收敛。
2. 参数控制误差低：F0 来自 deterministic source，F1/F2 通过显式 resonant / all-pole sections 控制，适合元音连续统和声调/语调实验刺激。
3. Source 与 filter 的结构清楚：Rd 主要影响 source，F1/F2 主要影响 filter，便于做 phonetic manipulation。
4. 训练成本低：RTX 4060 可训练，A100 上通常几十分钟级别。
5. 面向 speech-science use case：目标不是替代大规模 neural TTS，而是给语音学实验提供高自然度、低 artifact、可解释的刺激生成工具。

### 5. 需要主动承认的风险与待验证问题

1. Copy synthesis 的主观自然度需要系统评估，不能只依赖 demo。
2. F1/F2 的低误差需要在更多元音、声调、speaker 和 speaking styles 上验证。
3. Rd 的 perceptual interpretation 需要实验验证，不能只用参数可调来替代听感证据。
4. Learned biquads 虽然提高音质，但也可能重新引入不可解释耦合，需要分析它们对 formant / spectral envelope 的影响。
5. 若未来扩展到多说话人或跨语言，small-data advantage 可能会变弱，需要明确当前工作的边界。

## 参考文献与可引用依据

### 核心参考文献

Yu, C.-Y., & Fazekas, G. (2024). GOLF: A Singing Voice Synthesiser with Glottal Flow Wavetables and LPC Filters. *Transactions of the International Society for Music Information Retrieval, 7*(1), 316-330. https://doi.org/10.5334/tismir.210

Gu, Y., Perez Zarazaga, P., Wang, C., Wu, Z., Malisz, Z., Henter, G. E., & Juvela, L. (2026). HiFi-Glot: High-Fidelity Neural Formant Synthesis with Differentiable Resonant Filters. *arXiv:2409.14823*. https://arxiv.org/abs/2409.14823

Wang, X., Takaki, S., & Yamagishi, J. (2020). Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis. *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28*, 402-415. https://doi.org/10.1109/TASLP.2019.2956145

### BibTeX 草稿

BibTeX 已单独放在 `wang_xin_aria_nsf_refs.bib`，这样 PDF 预览不会因为长作者行溢出。

## 发出前建议

1. 如果目标是得到技术建议，邮件正文最好控制在 1200-1800 中文字之间，把技术公式作为附件。
2. 如果目标是建立后续合作或请教关系，可以在结尾加一句：“如果您愿意，我也可以把更完整的技术说明或音频样例整理后发给您。”
3. 不建议在邮件正文中直接写“优于 HiFi-Glot”。更稳的说法是：“目标不同：HiFi-Glot 面向 high-fidelity / generalizable neural formant synthesis，我的模型面向小数据、单说话人、实验刺激级强控制场景。”
4. 目前 IEEE Xplore 链接 `https://ieeexplore.ieee.org/abstract/document/9747442/` 的元数据我没有从网页工具中可靠抓取出来；如果这是你想额外引用的特定论文，建议发出前再手动核对标题、作者和 DOI。
