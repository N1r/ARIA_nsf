# ARIS

[简体中文](README.md) | **English**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/N1r/ARIS_nsf/blob/main/notebooks/ARIS_Tutorial_and_Workflow.ipynb)
[![Listening demo](https://img.shields.io/badge/demo-listen-blue)](https://n1r.github.io/ARIS_nsf/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

## 0. Introduction

ARIS (Analytic Resonance for Interpretable Synthesis) is a differentiable analysis-by-synthesis tool designed for phonetics and speech science. Built upon a differentiable source-filter vocoder architecture, ARIS enables precise, orthogonal manipulation of fundamental frequency (F0), vocal-tract formants (F1/F2), glottal pulse shape (`R_d`), and spectral tilt while preserving speaker identity and naturalness, facilitating reproducible batch generation of paired stimuli and acoustic continua for perceptual experiments.

- Listening demo: <https://n1r.github.io/ARIS_nsf/>
- Tutorial notebook: [`notebooks/ARIS_Tutorial_and_Workflow.ipynb`](notebooks/ARIS_Tutorial_and_Workflow.ipynb)

Model training requires a CUDA-capable NVIDIA GPU (a free Google Colab T4 instance is sufficient); resynthesis and stimulus manipulation can run directly on CPU.

Depending on your workflow, ARIS provides multiple ways to interact:

1. **Google Colab Cloud Tutorial**: Click the **Open in Colab** badge at the top of this page to run the complete analysis-by-synthesis pipeline (covering audio analysis, lightweight training, and acoustic manipulation) on a cloud GPU without local configuration.
2. **Interactive Listening Demo**: Open the [online listening demo](https://n1r.github.io/ARIS_nsf/) to explore acoustic parameter effects.
3. **Local GUI Workbench (Studio)**: Run `uv sync --locked --all-extras && uv run aris studio` to launch an interactive slider workspace in your browser for parameter exploration and continuum generation.
4. **Command-Line Interface**: Suitable for batch stimulus generation, automated experiment scripts, and large-scale training (Sections 2–5 and Section 7).

### Controls and validation for phonetics research

| Research dimension | ARIS control | Independently remeasure in the output |
|---|---|---|
| Pitch and intonation | `pitch_semitones` | median F0, contour shape, voiced proportion |
| Vowel resonance | `f1_*`, `f2_*` | F1/F2 tracks, vowel space, intelligibility |
| Voice quality | `glottal_rd_scale`, `noise_gain_db`, `tilt_alpha_delta` | H1–H2, CPP, HNR, spectral slope |
| Stimulus waveform gain (not vocal effort) | `output_gain_db` | peak, RMS/LUFS, clipped-sample count |

## 1. Installation & Quickstart

Supported Platforms: Linux (recommended) or Windows (via [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)).

### Recommended Installation (Using uv)

ARIS uses `pyproject.toml`, `uv.lock`, and [uv](https://docs.astral.sh/uv/) for Python and dependency management. After cloning the repository, run these commands from its root:

```bash
# Create the project environment and sync exactly from the lockfile
uv sync --locked --all-extras

# Run diagnostics inside the project environment
uv run aris doctor
```

All commands below use `uv run aris ...` without needing to manually activate the virtual environment. Run
`uv sync --locked --all-extras` again whenever dependencies change. If you intentionally
change dependencies, run `uv lock` first and commit the updated lockfile.

`uv run aris doctor` checks Python, audio dependencies, PyTorch, CUDA, and GPU status. Continue once all required checks pass.

### Google Colab

Open the Colab link above, select a **T4 GPU** runtime, and choose
**Runtime → Run all**. The tutorial uses uv throughout, requiring no manual environment configuration.
The demo setup (batch size 32, 1,500 steps) requires ~2.2 GB VRAM and provides quick audio feedback on a T4 GPU. For higher synthesis quality, refer to the 40,000-step checkpoint provided in the release.

**Quickstart with Pretrained Model:**
The official release provides a pretrained model (Mandarin female speaker, 16 kHz) and test recordings. Download and extract `aris_f024_demo.zip` to run reconstruction and manipulation directly:

```bash
# Download the official demo package (once)
curl -LO https://github.com/N1r/ARIS_nsf/releases/download/v0.1.0/aris_f024_demo.zip
unzip -q aris_f024_demo.zip

# 1. Reconstruct baseline audio:
uv run aris synthesize demo_f024/experiment demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_recon

# 2. Generate acoustic manipulation stimuli:
uv run aris manipulate demo_f024/experiment demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_stimuli \
  --variant 'f1_up:f1_scale=1.2' \
  --variant 'f1_down:f1_scale=0.85' \
  --variant 'pitch_down:pitch_semitones=-4' \
  --variant 'rd_high:glottal_rd_scale=1.6'
```

Upon completion, `out/demo_recon/` contains the reconstructed WAV files, and `out/demo_stimuli/` contains condition-specific audio files along with JSON metadata describing the generation parameters.

## 2. Preparing data

Place single-speaker recordings into a dedicated directory (WAV format, recorded in a quiet, low-reverberation environment):

```text
recordings/
├── session1.wav
├── session2.wav
└── ...
```

Run segmentation, feature extraction, and validation:

```bash
uv run aris split recordings/ segments/ --mode silence               # segment into short utterances at silences
uv run aris prepare segments/audio data/my_voice --extract-formants # extract F0 and Praat F1/F2, split dataset
uv run aris validate data/my_voice                                  # verify dataset integrity
```

Data preparation notes:

1. **Recording Specs**: 20–60 minutes of clean audio recommended. Maintain consistent microphone, distance, gain, and acoustics; avoid clipping and reverberation.
2. **Segmentation & Splits**: Segmenting long recordings into short utterances speeds up training. Plan train/val/test splits carefully to avoid data leakage across sessions.
3. **F0 Extraction**: `--f0-method pyworld` is recommended (WORLD DIO+StoneMask, robust and fast); defaults to autocorrelation if WORLD is unavailable.
4. **Formant Supervision**: Training `aria-golf` requires `--extract-formants`, which runs Praat Burg analysis on a 10 ms time step for F1/F2 supervision. The default 5,500 Hz ceiling fits female speech; use `--formant-ceiling 5000` for male voices if appropriate.
5. **External Pitch Tracks (Sidecar)**: For tone languages or F0-sensitive tasks, extract pitch contours using [RMVPE](https://github.com/Dream-High/RMVPE) or Praat, and load them via `--f0-method sidecar`. `.pv` files must follow a fixed 5 ms hop size (`0.0` for unvoiced). If the frame count does not match audio duration, `prepare` raises an error.
6. **Sample Rate**: Inputs need not share sample rates; `prepare` resamples them automatically. Retain original unresampled recordings as master copies.
7. **Example Corpus**: If no recordings are readily available, run `uv run aris fetch-corpus data/arctic` to download ~30 minutes of CMU ARCTIC for pipeline testing.

## 3. Training

Once data preparation is complete, initialize experiment configuration and start training:

```bash
uv run aris init-experiment data/my_voice experiments/my_voice --model aria-golf
uv run aris train experiments/my_voice --dry-run   # print the training command without executing
uv run aris train experiments/my_voice
```

Notes:

1. **Model Selection**: Choose the model architecture suitable for your study:

   | Model | Available controls | Typical use |
   |---|---|---|
   | `ddsp` | F0, waveform gain, stochastic-source gain | baseline reconstruction and pitch studies |
   | `golf` | above + glottal `R_d` | source and voice-quality studies |
   | `aria-golf` | above + F1/F2, spectral tilt | explicit source–filter joint manipulation |

   `aria-golf` incorporates F1/F2 supervision and temporal smoothness constraints alongside multi-scale spectral loss, ensuring formant parameters have identifiable physical meaning.
2. **Checkpoints**: Model weights are saved under `experiments/my_voice/runs/checkpoints/`.
3. **CUDA & PyTorch**: The default installation includes CUDA-enabled PyTorch. If your GPU driver requires a different build, refer to the [PyTorch selector](https://pytorch.org/get-started/locally/) and install with `uv pip install`.
4. **Cluster Support (Slurm)**: `init-experiment` generates a ready-to-`sbatch` `train.slurm` script. Adjust GPU parameters for your cluster environment, or ignore this file for local runs.
5. **Directory Portability**: Experiment directories support relative relocation. Moving an experiment directory alongside its dataset directory preserves relative paths for training and inference.

## 4. Reconstruction (inference)

After training, reconstruct the test set recordings to evaluate synthesis quality:

```bash
uv run aris synthesize experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/reconstruction
```

The command outputs reconstructed WAV files. Compare originals and reconstructions on the test set, inspecting intelligibility, artifacts, F0/formant deviation, and confirming the absence of hard-clipped samples.

## 5. Generating manipulated stimuli

Modify target acoustic parameters independently on top of reconstructed speech to generate paired stimuli or continua:

```bash
uv run aris manipulate experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/stimuli \
  --variant 'pitch_down:pitch_semitones=-4' \
  --variant 'f1_up:f1_scale=1.2'
```

Each `--variant` defines a named condition (`name:param=value,param=value`), producing a separate set of WAV files and a JSON record of configuration parameters.

Supported parameters and ranges:

| Parameter | Range | Meaning | DDSP | GOLF | ARIS-GOLF |
|---|---|---|:---:|:---:|:---:|
| `pitch_semitones` | `-36..36` | pitch shift (semitones) | ✓ | ✓ | ✓ |
| `output_gain_db` | `-24..12` | post-synthesis waveform gain (dB) | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` | digital gain on the stochastic-source branch (dB) | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0` | scaling of the model's glottal-source $R_d$ | — | ✓ | ✓ |
| `f1_scale` / `f2_scale` | `0.7..1.3` | F1 / F2 relative scaling (preserves contour) | — | — | ✓ |
| `f1_hz` / `f2_hz` | `150..1300` / `600..3200` | F1 / F2 absolute target frequency (Hz) | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | spectral tilt offset | — | — | ✓ |

### Parameter Notes and Quality Control

- **Continuum Construction**: `f1_hz` / `f2_hz` fix formants to absolute frequencies across items; `f1_scale` / `f2_scale` shift contours proportionally. The two cannot be combined on the same formant.
- **Provenance Tracking**: Each output directory includes `manipulation.json`, recording control parameters, dataset fingerprints, and checkpoint SHA-256 hashes.
- **Clipping Verification**: Check `clipped_samples` in `_render.json`; if clipping occurs, reduce `output_gain_db`.
- **Audio Demos & Details**: Interactive audio samples are available at the [online demo](https://n1r.github.io/ARIS_nsf/); parameter details are covered in the [Manipulation guide](docs/MANIPULATION_ZH.md).

## 6. Browser workbench (Studio)

ARIS provides an interactive browser-based workbench for parameter exploration and listening comparison:

```bash
# Launch locally (opens http://127.0.0.1:8765/):
uv run aris studio

# Launch with a public shareable URL (ideal for Google Colab and remote servers):
uv run aris studio --share
```

Features:
- **Parameter Sliders & Continuum Builder**: Dynamically generated controls based on model capabilities, with one-click continuum stimulus generation.
- **A/B Listening & Visual Comparison**: Quick A/B switching between baseline and variants, with time-aligned waveform and spectrogram displays.
- **Boundary & Clipping Warnings**: Visual red alerts for hard clipping or parameters approaching model boundaries.
- **Consistent Output Specifications**: Rendered audio and metadata match the output structure of command-line `manipulate`.

## 7. Command reference

```text
uv run aris doctor             check audio, training dependencies, CUDA, and GPU hardware
uv run aris fetch-corpus       download the CMU ARCTIC example corpus
uv run aris split              segment continuous recordings
uv run aris prepare            resample, extract F0, split the dataset
uv run aris validate           check dataset integrity
uv run aris init-experiment    create a training experiment directory
uv run aris train              start training
uv run aris controls           list manipulation parameters supported by a model
uv run aris synthesize         reconstruct recordings from a checkpoint
uv run aris manipulate         generate manipulated stimuli
uv run aris studio             launch the browser workbench (supports --share for public URL)
```

## 8. Citation

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

The ARIS vocoder implementation derives from GOLF:

- C.-Y. Yu and G. Fazekas, "Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis," *Interspeech 2024*. DOI: `10.21437/Interspeech.2024-1187`
- C.-Y. Yu and G. Fazekas, "Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables," *ISMIR 2023*. DOI: `10.5281/zenodo.10265377`

The earlier methodological foundations are differentiable DSP and neural
source-filter models:

- J. Engel, L. Hantrakul, C. Gu, and A. Roberts, "DDSP: Differentiable Digital Signal Processing," *ICLR 2020*. arXiv: `2001.04643`
- X. Wang, S. Takaki, and J. Yamagishi, "Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis," *IEEE/ACM TASLP*, 2020. arXiv: `1904.12088`

Code is released under the MIT license; see [LICENSE](LICENSE).

## 9. Contact

If you run into problems or have suggestions, feel free to open an
[issue](https://github.com/N1r/ARIS_nsf/issues), or email
<dingyr@hum.leidenuniv.nl> (Leiden University).
