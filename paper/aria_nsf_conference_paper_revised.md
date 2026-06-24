---
title: "ARIA-NSF: A Differentiable Klatt-Style Neural Source-Filter Synthesizer for Small-Data Phonetic Stimulus Generation"
author:
  - "Yiran Ding"
  - "Leiden University Centre for Linguistics"
date: "Draft, 12 June 2026"
bibliography: aria_nsf_conference_refs_revised.bib
link-citations: true
geometry: margin=0.85in
fontsize: 10pt
---

# Abstract

Speech-science experiments often require synthetic stimuli that are natural enough for repeated listening, but also explicitly controllable along phonetically meaningful dimensions such as fundamental frequency, formant frequency, spectral tilt, intensity, and voice quality. Modern neural vocoders provide high perceptual quality, but their data requirements and latent acoustic representations can make them difficult to use in small-data phonetic studies. Classical formant synthesis provides interpretability and control, but often lacks the naturalness required for ecological experimental stimuli. This paper presents ARIA-NSF, a differentiable Klatt-style neural source-filter synthesizer designed for small-data, single-speaker phonetic stimulus generation. The model combines an analytic glottal excitation controlled by \(F_0\) and \(R_d\), a learned noise component, and a vocal-tract cascade that combines deterministic formant/all-pole sections with learned biquad residual sections. In the current configuration, the vocal-tract cascade contains three deterministic biquad sections and eight learned biquads, yielding an LPC-equivalent order of \(2(3+8)=22\). Rather than proposing a general-purpose neural vocoder, ARIA-NSF targets a narrower experimental setting: controllable copy synthesis and acoustic manipulation from laboratory-scale speech recordings. We describe the architecture, the pilot Mandarin corpus, and a concrete evaluation plan covering acoustic error, parameter disentanglement, baseline comparison, and perceptual validation.

**Keywords:** neural source-filter model; formant synthesis; differentiable DSP; phonetic stimulus generation; small-data speech synthesis; voice quality

# 1. Introduction

Controlled speech synthesis occupies an unusual position between speech technology and phonetic science. In text-to-speech and neural vocoding, the primary goal is often perceptual naturalness, speaker similarity, or large-scale generalization. In phonetic and psycholinguistic experiments, however, the research goal is frequently different: the stimulus must be natural enough to be listened to repeatedly, but the experimental manipulation must also be interpretable, precise, and preferably low-dimensional. A phonetician may need to shift \(F_0\) to create a tone or intonation continuum, move \(F_1\) and \(F_2\) independently to construct a vowel continuum, or alter source-related parameters to probe breathy, creaky, or pressed voice quality. A system that produces plausible audio but hides these variables inside a high-dimensional latent representation is therefore not always suitable for experimental work.

This requirement is especially challenging in typical speech-science settings. Unlike modern speech-generation pipelines, which may rely on many hours or thousands of hours of training data, experimental materials are often collected for a specific study. A dataset may contain only 5--60 minutes of speech from one speaker, recorded under a controlled design for a particular phenomenon such as tone sandhi, categorical perception, prosodic prominence, vowel normalization, or phonation type. Such datasets are valuable precisely because they are tailored to a hypothesis, but they are usually too small to support large generic neural vocoders without overfitting, artifacts, or loss of controllability.

Classical source-filter theory [@fant1960acoustic] and formant synthesis [@klatt1980software] offer the right conceptual primitives: a source, a vocal-tract filter, and interpretable acoustic controls. Tools such as Praat [@boersma2001praat] remain central in phonetics because they support analysis and manipulation in terms that phoneticians understand. Yet purely classical synthesis can sound unnatural, especially when used to create full experimental stimuli rather than isolated demonstrations. Conversely, modern neural vocoders and neural source-filter models can generate high-quality waveforms [@wang2020nsf], but their control variables are often less directly aligned with experimental manipulations.

This paper presents ARIA-NSF, a small-data differentiable Klatt-style neural source-filter synthesizer. The central design hypothesis is that phonetic stimulus generation benefits from a hybrid allocation of responsibility: parameters that are experimentally meaningful and acoustically well-understood should remain analytic or deterministic, while the residual aspects of timbre and spectral detail can be learned from data. In the current implementation, \(F_0\), \(F_1\), \(F_2\), bandwidths, gain, spectral tilt, and glottal-source parameters are exposed as controllable variables. A small neural model predicts or refines the remaining control streams needed for copy synthesis.

The intended contribution is not a universal TTS system. Instead, ARIA-NSF targets a narrower but common research use case: creating high-quality, interpretable speech stimuli from a small, single-speaker corpus. The paper makes three contributions:

1. It formulates the requirements of neural speech synthesis for phonetic stimulus generation as a small-data, high-control source-filter problem.
2. It introduces a differentiable Klatt-style architecture that combines an analytic glottal source, shaped noise, deterministic formant sections, and learned residual biquads.
3. It defines a pilot Mandarin evaluation protocol for measuring copy-synthesis quality, manipulation reliability, and unwanted acoustic coupling.

# 2. Related Work

## 2.1 Classical Formant Synthesis and Phonetic Manipulation

The source-filter model of speech production provides a compact account of speech as the interaction between a laryngeal source and a vocal-tract filter [@fant1960acoustic]. Klatt's cascade/parallel formant synthesizer [@klatt1980software] made this account operational by exposing parameters such as formant frequencies, bandwidths, voicing amplitude, aspiration, frication, and spectral tilt. For phonetic experiments, this level of control remains attractive because it allows a researcher to manipulate one acoustic dimension while holding others approximately constant.

The limitation is perceptual quality. Even when a classical formant synthesizer accurately implements an acoustic hypothesis, the resulting stimulus may sound synthetic in ways that affect listener behavior. This matters because perceptual experiments are not only tests of acoustic discriminability; they are also embedded in listeners' expectations about natural speech. A stimulus continuum with audible artifacts may therefore confound the intended manipulation.

Praat [@boersma2001praat] and LPC-based copy synthesis remain widely used because they allow practical analysis and manipulation of pitch and formant-related properties. However, inverse filtering and source-filter separation are imperfect. If formant information remains in the residual excitation, formant manipulation can introduce artifacts or unintended acoustic coupling.

## 2.2 Neural Source-Filter and Differentiable DSP Models

Neural source-filter models reintroduced speech-production structure into neural waveform generation. Wang et al. [-@wang2020nsf] proposed an NSF framework in which a source module generates an excitation signal, a neural filter module transforms it into a waveform, and a condition module preprocesses acoustic features. This design preserves explicit \(F_0\)-driven excitation while avoiding autoregressive generation.

Differentiable digital signal processing (DDSP) similarly argues that neural audio models can benefit from embedding known signal-processing operators inside trainable systems [@engel2020ddsp]. Rather than learning the entire waveform generator as an unconstrained network, DDSP systems can learn control parameters for interpretable modules such as oscillators, filters, and reverberation. This is especially relevant for small-data settings, where inductive bias can reduce the amount of data required to learn a usable synthesizer.

GOLF [@yu2024golf] is a particularly close reference point. It uses glottal-flow wavetables as the harmonic source and LPC filters to model the vocal tract, achieving a compact and efficient singing voice synthesizer. GOLF shows that voice-production constraints can be integrated into differentiable synthesis while maintaining competitive quality and efficiency. ARIA-NSF follows the same general philosophy but targets speech-science stimulus generation rather than singing voice synthesis, and gives priority to explicit manipulation of \(F_0\), \(F_1/F_2\), and source-shape parameters.

HiFi-Glot [@gu2026hifiglot] addresses a related problem from the direction of high-fidelity neural formant synthesis. It combines neural excitation with differentiable resonant/all-pole filters to provide explicit formant control at high perceptual quality. Compared with such full-band, large-capacity systems, ARIA-NSF is deliberately smaller and more specialized. It sacrifices some generality in favor of small-data trainability, explicit experimental controls, and a structure that remains close to traditional formant synthesis.

## 2.3 Source-Filter Positioning

The source-filter taxonomy used in uSFGAN [@yoneyama2021usfgan] is useful for locating ARIA-NSF among conventional vocoders, neural source-filter models, and fully neural waveform generators. In that taxonomy, excitation generation and resonance filtering can each be implemented as deterministic/parametric modules, neural modules, or a unified neural waveform generator. Figure 1 adapts this view and places ARIA-NSF as a dual-hybrid source-filter model.

![Positioning of ARIA-NSF within source-filter vocoder architectures. The diagram is adapted from the source-filter taxonomy in uSFGAN [@yoneyama2021usfgan], but adds the ARIA-NSF column to emphasize that both excitation generation and resonance filtering are hybrid rather than purely deterministic or purely neural.](figures/aria_source_filter_positioning_tikz.pdf){width=100%}

ARIA-NSF is therefore better described as structurally hybrid than as a literal 50/50 split between deterministic and neural components. On the excitation side, the periodic glottal source is generated deterministically from \(F_0\) and \(R_d\), while the aperiodic/noise component is learned or conditioned. On the filter side, the experimentally important vocal-tract controls remain deterministic and interpretable, while learned residual biquads supply speaker-specific spectral detail. The model thus splits responsibility in both places: explicit DSP handles the variables that should remain editable, and neural residual capacity handles details that are difficult to specify by hand.

# 3. Design Requirements

ARIA-NSF is designed around five requirements that are common in phonetic stimulus generation but less central in general-purpose neural vocoding.

**R1. Small-data trainability.** The system should be usable with 5--60 minutes of single-speaker recordings, because phonetic corpora are often collected for a narrow experimental question rather than for large-scale speech generation.

**R2. Explicit intervention variables.** Experimental variables such as \(F_0\), \(F_1/F_2\), bandwidth, gain, spectral tilt, and \(R_d\) should remain directly editable. They should not be recoverable only as post-hoc measurements from a generated waveform.

**R3. Low cross-parameter coupling.** Manipulating one dimension should minimally disturb others. For example, an \(F_1\) manipulation should not substantially alter \(F_0\), energy, or source-shape cues.

**R4. Residual expressivity.** Purely analytic formant synthesis may not capture enough speaker-specific spectral detail for natural-sounding stimuli. The architecture therefore needs limited learned capacity, but this capacity must be constrained enough that it does not erase interpretability.

**R5. Laboratory accessibility.** Training should be feasible on consumer or modest institutional GPUs. This requirement is practical rather than theoretical, but it is important for speech-science laboratories without sustained access to large compute clusters.

These requirements motivate a dual-hybrid source-filter design: deterministic components implement the experimental variables, while learned residual components model aspects of both the excitation and the filter that are difficult to specify analytically.

# 4. Model

Figure 2 gives a schematic overview of the current ARIA-NSF pipeline. The architecture operates at two rates. Frame-rate acoustic controls are predicted, extracted, or specified by the user; these controls are then upsampled to the audio sample rate for differentiable waveform synthesis.

![Current ARIA-NSF source-filter diagram. The glottal excitation and shaped noise are summed before passing through the vocal-tract cascade / LPC filter.](figures/aria_golf_f024_formula_flow_tikz.pdf){width=80%}

## 4.1 Sample-Rate Excitation

Given sampling rate \(f_s\), the deterministic phase trajectory is obtained by integrating the frame- or sample-rate \(F_0\) contour:

$$
\phi_n = \phi_{n-1} + 2\pi \frac{F_{0,n}}{f_s}.
$$

The voiced excitation is generated by an LF-style glottal source:

$$
e_n = g_{\mathrm{LF}}(\phi_n; R_{d,n}),
$$

where \(R_d\) controls the glottal pulse shape and is intended to support continuous manipulation of phonation-related dimensions such as pressed or breathy voice. This formulation follows the spirit of LF-family source modeling [@fant1985lf], but is used here as a differentiable component inside a neural source-filter synthesizer rather than as a complete standalone voice model.

A noise branch models turbulent or aperiodic energy. White noise \(\epsilon_n\) is transformed by a learned or conditioned noise shaper:

$$
\eta_n = \mathcal{N}(\epsilon_n; m_n),
$$

where \(m_n\) denotes the sample-rate noise control stream. In contrast to architectures that filter harmonic and noise components through separate output paths, ARIA-NSF currently combines excitation and shaped noise before the vocal-tract filter:

$$
u_n = e_n + \eta_n.
$$

This design choice follows the intended speech-production interpretation: the periodic and aperiodic components form a source-like excitation, which is then shaped by the vocal tract.

## 4.2 Vocal-Tract Cascade

The output waveform is generated by passing the combined excitation through a time-varying vocal-tract cascade:

$$
y_n = H_{\mathrm{VT}}(z; v_n) u_n.
$$

The filter is factorized into deterministic, interpretable sections and learned residual sections:

$$
H_{\mathrm{VT}}(z; v_n)
=
H_{\mathrm{det}}(z; F_{1,n}, F_{2,n}, B_{1,n}, B_{2,n}, \alpha_n, g_n)
\cdot
\prod_{i=1}^{8} H_i(z; \theta_{i,n}).
$$

Each biquad section can be written as an all-pole resonator:

$$
H_i(z; \theta_i)
=
\frac{G_i}{1+a_{i,1}z^{-1}+a_{i,2}z^{-2}}.
$$

In the current configuration, the deterministic component contains three biquad sections and the neural residual component contains eight learned biquads. The resulting LPC-equivalent order is:

$$
\mathrm{order} = 2(3+8)=22.
$$

This configuration is intended to make the first formant-related degrees of freedom explicitly controllable while leaving enough residual capacity to learn speaker-specific spectral detail that would be difficult to hand-code.

## 4.3 Training Objective

The pilot implementation is trained as a copy-synthesis model using a multi-scale spectral reconstruction objective:

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

where \(x\) is the target waveform, \(y\) is the synthesized waveform, and \(S_r\) is an STFT operator at resolution \(r\). This objective is consistent with the spectral-loss tradition in DDSP and neural source-filter work [@engel2020ddsp; @wang2020nsf; @yu2024golf], while keeping the training procedure lightweight.

# 5. Pilot Corpus and Experimental Conditions

The current pilot corpus consists of Mandarin monosyllabic word recordings collected for phonetic research. The recording session contains approximately 1,400 word tokens, with about 30 minutes of effective usable audio after trimming and quality control. This corpus is representative of a common phonetic-laboratory scenario: the material is carefully designed for a linguistic question, but it is much smaller than the datasets normally used to train high-capacity neural vocoders.

The pilot evaluation is organized around three operations:

1. **Copy synthesis:** reconstructing held-out utterances from extracted or predicted acoustic controls.
2. **Pitch manipulation:** modifying \(F_0\) trajectories while preserving other source-filter properties.
3. **Vowel and source manipulation:** independently altering \(F_1/F_2\) and \(R_d\) to create continua relevant to vowel perception and voice-quality perception.

The planned experimental conditions are summarized in Table 1. The 30-minute condition is treated as the primary pilot setting. The 5- and 10-minute conditions test whether the architecture remains useful when the available speech material is closer to the lower bound of many phonetic laboratory datasets.

Table 1. Planned pilot and ablation conditions.

| Condition | Training material | Purpose |
|---|---:|---|
| Full pilot | approximately 30 min | Main copy-synthesis and manipulation condition |
| Reduced data | approximately 10 min | Small-data robustness check |
| Minimal data | approximately 5 min | Lower-bound feasibility check |
| Ablation: no learned biquads | approximately 30 min | Tests contribution of residual spectral modeling |
| Ablation: deterministic formants disabled | approximately 30 min | Tests whether explicit formant control is necessary |
| Ablation: no shaped noise | approximately 30 min | Tests contribution of aperiodic excitation |

Early listening during development suggests that the model can converge on the 30-minute corpus and can still produce usable output under 5--10 minute training conditions. These observations motivate the evaluation but should not be interpreted as final benchmark results.

The model is lightweight enough to train on a laptop-class RTX 4060 GPU. On an A100-class server GPU, training time is typically on the order of tens of minutes for the current single-speaker setting. This resource profile is important for speech-science laboratories, where continuous access to large-scale GPU clusters is often unrealistic.

# 6. Evaluation Protocol

A formal evaluation should measure both synthesis quality and manipulation validity. For phonetic stimulus generation, these two dimensions are equally important. A system that sounds natural but fails to control the intended variable is not useful for controlled experiments; likewise, a precisely controlled but unnatural stimulus may not support ecologically valid listening behavior.

## 6.1 Baselines and Ablations

The evaluation should compare ARIA-NSF against baselines that represent both phonetic practice and neural synthesis practice:

Table 2. Proposed baselines and ablation systems.

| System | Role in evaluation |
|---|---|
| Natural recording | Upper reference for perceptual quality |
| Praat / LPC copy synthesis | Conventional phonetic manipulation baseline |
| DDSP harmonic-plus-noise | Differentiable synthesis baseline without explicit vocal-tract cascade |
| HiFi-Glot-style source-filter model | Strong neural formant-synthesis reference, where feasible |
| ARIA-NSF without learned biquads | Tests whether residual capacity improves quality |
| ARIA-NSF without deterministic formant sections | Tests whether explicit formant control is responsible for manipulation accuracy |
| ARIA-NSF without shaped noise | Tests whether aperiodic excitation improves naturalness |

## 6.2 Acoustic Evaluation

For copy synthesis, standard reconstruction metrics can be reported, including multi-scale STFT loss, mel-cepstral distortion, \(F_0\) error, and spectral-envelope error. For manipulation tasks, the more important metric is target-tracking error. For formants:

$$
E_{F_i}
=
\frac{1}{T}
\sum_t
\left|
\widehat{F_i}(y_t)-F_i^{\mathrm{target}}(t)
\right|,
\quad i\in\{1,2\}.
$$

The same logic can be applied to \(F_0\), intensity, spectral tilt, and any estimated correlate of voice quality. To measure unwanted coupling, one can compute a manipulation cross-effect matrix. For example, when \(F_1\) is manipulated, the analysis should report not only the \(F_1\) error but also the induced changes in \(F_0\), \(F_2\), energy, tilt, and \(R_d\)-related measures.

For reporting, the most important outcome is not a single aggregate score but a separation between quality, target accuracy, and side effects:

| Dimension | Example metric | Interpretation |
|---|---|---|
| Copy quality | MS-STFT, MCD, spectral-envelope error | How well the model reconstructs held-out speech |
| Pitch control | \(F_0\) RMSE after manipulation | Whether deterministic excitation tracks the target |
| Formant control | \(F_1/F_2\) target-tracking error | Whether vowel-space manipulations are accurate |
| Source control | acoustic/perceptual correlate of \(R_d\) | Whether source-shape manipulation is interpretable |
| Coupling | cross-effect matrix | Whether manipulating one parameter disturbs others |
| Efficiency | training time, inference RTF, model size | Whether the system is usable in laboratory settings |

## 6.3 Perceptual Evaluation

For copy synthesis, a MUSHRA-like or CMOS listening test can compare ARIA-NSF with natural recordings, Praat/LPC copy synthesis, and neural baselines such as DDSP harmonic-plus-noise or a HiFi-Glot-style source-filter system. For manipulation, the evaluation should be task-based rather than only naturalness-based. Suitable designs include:

1. identification or goodness-rating tests along an \(F_1/F_2\) vowel continuum;
2. tone or intonation perception tests along an \(F_0\) continuum;
3. voice-quality rating tests along an \(R_d\) continuum;
4. ABX or oddity tests to assess whether listeners perceive the intended dimension rather than artifacts.

The key criterion is whether the synthesized stimuli support the same phonetic contrast that the experiment is designed to test.

# 7. Discussion

ARIA-NSF is motivated by a gap between two mature traditions. Classical formant synthesis gives phoneticians transparent control, but it can sound artificial. Neural vocoders provide high-quality audio, but their control variables are often too entangled for experimental stimulus generation. A differentiable Klatt-style neural source-filter model offers a middle path: keep the experimentally meaningful variables explicit, and learn the residual structure needed for naturalness.

For speech scientists, the main advantage is experimental control. \(F_0\), \(F_1/F_2\), spectral tilt, intensity, and \(R_d\) can be treated as intervention variables rather than as post-hoc measurements of a generated waveform. For machine-learning researchers, the model is an example of structured inductive bias: by placing source-filter assumptions inside the architecture, the system can operate in a data regime where large black-box neural vocoders are poorly matched to the task.

The learned biquad cascade is both useful and potentially risky. It increases expressive power and improves speaker-specific spectral modeling, but it may also reintroduce coupling between parameters that are meant to remain independent. Future work should analyze the learned sections directly, for example by tracking pole trajectories, inspecting formant interactions, and ablating the learned residual cascade.

The current pilot implementation should therefore be read as a system and evaluation proposal rather than as a completed benchmark. Its strongest claim is architectural: phonetic stimulus generation may benefit from keeping experimentally meaningful parameters analytic while assigning only the residual spectral detail to learned modules. The empirical claim will require controlled comparison against Praat/LPC manipulation, DDSP-style harmonic-plus-noise synthesis, and a stronger neural source-filter or HiFi-Glot-inspired model.

# 8. Conclusion

This paper introduced ARIA-NSF, a differentiable Klatt-style neural source-filter synthesizer for small-data phonetic stimulus generation. The model combines deterministic source and formant controls with learned residual spectral modeling, aiming to preserve the interpretability of classical formant synthesis while improving the naturalness of copy synthesis and manipulation. The current architecture uses an LF-style glottal excitation, a shaped noise branch, and a vocal-tract cascade consisting of deterministic all-pole sections plus eight learned biquads. The revised evaluation plan focuses on the criteria that matter most for phonetic experiments: target accuracy, low cross-parameter coupling, perceptual adequacy, and small-data trainability. The next step is to complete the objective and perceptual evaluations needed to turn this system draft into a fully supported conference submission.

# Acknowledgements

The pilot recordings were collected during work with the research group of Jinsong Zhang at Beijing Language and Culture University. The author thanks earlier neural source-filter, DDSP, GOLF, and neural formant synthesis work for motivating this hybrid design.

# Data Availability

The current pilot corpus contains speech recordings collected for phonetic research and is not yet publicly released. A demo page with selected copy-synthesis and manipulation examples is available at <https://n1r.github.io/ARIA_nsf/>.

# Ethics Statement

Future perceptual evaluation will require informed consent from listeners and appropriate handling of speaker recordings. No perceptual listener data are reported in this draft.

# Author Contributions

Yiran Ding designed the model, prepared the pilot corpus, implemented the current system, and wrote the draft.

# Conflict of Interest

The author declares no conflict of interest.

# Funding

No dedicated funding is reported for the current pilot work.

# AI Assistance Disclosure

AI writing assistance was used to organize the draft structure and polish language. The author is responsible for all technical claims, experiments, and final manuscript content.

# References
