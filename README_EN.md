# ARIS

[简体中文](README.md) | **English**

[![Listening demo](https://img.shields.io/badge/demo-listen-blue)](https://n1r.github.io/ARIS_nsf/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

## 0. Introduction

ARIS (Analytic Resonance for Interpretable Synthesis) is a differentiable
analysis-by-synthesis tool made for phonetics researchers: train a DDSP/GOLF vocoder
on a few tens of minutes of single-speaker recordings, then change F0,
energy, noise, glottal source shape (`R_d`), or formants (F1/F2, spectral
tilt) one at a time — with everything else held fixed — to batch-generate
paired experimental stimuli.

- Listening demo: <https://n1r.github.io/ARIS_nsf/>

No dedicated compute is required: training runs on an ordinary gaming GPU
(e.g. an RTX 4060) in a few hours for a few tens of minutes of data;
reconstruction and stimulus generation need no GPU and run on a regular CPU.

There are three ways in, by increasing commitment:

1. **Zero install**: open the [listening demo](https://n1r.github.io/ARIS_nsf/)
   and drag one parameter at a time to hear its effect.
2. **About half an hour**: install the dependencies, download the
   pretrained model, and run reconstruction and manipulation once
   (Section 1).
3. **The full pipeline**: take your own recordings through
   data preparation → training → stimulus generation (Sections 2–5).

## 1. Installing dependencies

A note on platforms first: the commands below are meant for a **Linux or
macOS terminal**. On Windows, we recommend installing
[WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Microsoft's
official Linux subsystem — a single `wsl --install` command sets it up)
and following the steps inside the WSL terminal.

All you need on your machine is [uv](https://docs.astral.sh/uv/) (a
single-file Python environment manager). From the repository root:

```bash
source scripts/project_env.sh      # enter the project environment
./scripts/setup_project_env.sh     # install all dependencies (inside the repo, system untouched)
.venv/bin/aris doctor              # check audio and training dependencies
```

Once installed, all commands are invoked as `.venv/bin/aris`; whenever
you are unsure about usage, every command supports `--help`.

Each `doctor` line starting with `[OK]` means that dependency is ready;
a `[--]` line means it is missing and includes the install command to
fix it. Once the training rows (`torch`, `lightning`) show `[OK]`, you
are ready for the steps below.

**Start with a pretrained model.** We provide a trained model (Mandarin
female speaker, 16 kHz) with a matching example dataset; after
downloading and extracting it, you can run reconstruction and
manipulation directly and see what each step produces:

```bash
curl -LO https://github.com/N1r/ARIS_nsf/releases/download/v0.1.0/aris_f024_demo.tar.gz
tar -xzf aris_f024_demo.tar.gz    # extract in the repository root; creates demo_f024/

.venv/bin/aris synthesize demo_f024/experiment \
  demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_recon
.venv/bin/aris manipulate demo_f024/experiment \
  demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_stimuli \
  --variant 'f1_up:f1_scale=1.2'
```

What success looks like: reconstructed WAV files appear under
`out/demo_recon/`, and `out/demo_stimuli/` contains one set of WAVs per
condition plus JSON metadata recording how they were generated. Listen
to the `f1_up` variant against the reconstruction — the raised first
formant should be clearly audible.

## 2. Preparing data

First, put your recordings in one folder (WAV, single speaker, quiet room):

```text
recordings/
├── session1.wav
├── session2.wav
└── ...
```

Then segment, preprocess, and check:

```bash
.venv/bin/aris split recordings/ segments/ --mode silence        # cut into short utterances at silences
.venv/bin/aris prepare segments/audio data/my_voice              # resample, extract F0, split the dataset
.venv/bin/aris validate data/my_voice                            # confirm the dataset is complete and usable
```

A few practical notes on data:

1. 20–60 minutes of audio in total is recommended; cutting long recordings
   into utterances of a few seconds speeds up training.
2. For F0 extraction, `--f0-method pyworld` (WORLD's DIO+StoneMask, stable
   and fast) is recommended; the default `auto` falls back to
   autocorrelation when WORLD is not installed.
3. For tonal languages such as Mandarin, or other F0-sensitive work,
   extract a more reliable pitch track first with
   [RMVPE](https://github.com/Dream-High/RMVPE) or Praat, save it as a
   `.pv` file next to each WAV, and load it via `--f0-method sidecar`.
   `.pv` is plain text: one float per line, one line per frame, at a
   fixed 5 ms hop; `0.0` means unvoiced, a positive number is that
   frame's F0 in Hz. Praat's default pitch time step is not 5 ms, and it
   marks unvoiced frames `--undefined--`, not `0` — so a Praat export
   needs resampling to the 5 ms grid and an `--undefined--` → `0.0`
   substitution before `prepare` will accept it. A frame count that does
   not match the audio duration now raises a clear error instead of
   silently misaligning.
4. Sample rates need not be unified beforehand; `prepare` resamples to the
   model sample rate automatically.
5. No recordings at hand? `.venv/bin/aris fetch-corpus data/arctic`
   downloads ~30 minutes of the public CMU ARCTIC corpus for a full trial
   run; once downloaded, continue with
   `.venv/bin/aris prepare data/arctic/selected data/arctic_prepared`.

## 3. Training

With your data ready, create an experiment directory and start training:

```bash
.venv/bin/aris init-experiment data/my_voice experiments/my_voice --model aria-golf
.venv/bin/aris train experiments/my_voice --dry-run   # print the training command without running it
.venv/bin/aris train experiments/my_voice
```

A few notes:

1. `--model` is one of `ddsp`, `golf`, `aria-golf`; choose `aria-golf` if
   you need formant (F1/F2) and spectral-tilt control (`aria-golf` is the
   code name of the ARIS decoder).
2. Checkpoints are saved under `experiments/my_voice/runs/checkpoints/`.
3. Training requires a CUDA-enabled PyTorch. The default Linux install
   already ships with CUDA support; if it does not match your GPU or
   driver, install the matching build for your machine — tell an AI
   assistant (ChatGPT, Claude, …) your GPU model and operating system
   and ask for the install command, or use the selector at
   [pytorch.org](https://pytorch.org).
4. On a Slurm cluster, `init-experiment` has already generated a
   ready-to-`sbatch` `train.slurm`; adjust cluster parameters (partition,
   GPU type, etc.) via `init-experiment` options, and fill in the CUDA
   module for your cluster before submitting. Ignore it if you have no
   cluster.
5. Experiment directories created from now on are portable: move a fresh
   `init-experiment` output together with its dataset directory (keeping
   their relative position), and `train`/`synthesize`/`manipulate` still
   find the dataset from any working directory. Older experiment
   directories (like the demo above) keep resolving against their
   original path.

## 4. Reconstruction (inference)

Once training is done, first reconstruct the held-out test recordings with
the checkpoint to see how the model sounds:

```bash
.venv/bin/aris synthesize experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/reconstruction
```

The output is reconstructed WAV files; listen against the originals, and if
they sound close, you are ready for the next step.

## 5. Generating manipulated stimuli

This step is what ARIS is really for: change one parameter at a time on top
of the reconstruction, keeping everything else fixed:

```bash
.venv/bin/aris manipulate experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/stimuli \
  --variant 'pitch_down:pitch_semitones=-4' \
  --variant 'f1_up:f1_scale=1.2'
```

Each `--variant` is a named condition (`name:param=value,param=value`) and
produces one set of WAVs plus JSON metadata recording how they were
generated. Available parameters and ranges:

| Parameter | Range | Meaning | DDSP | GOLF | ARIS-GOLF |
|---|---|---|:---:|:---:|:---:|
| `pitch_semitones` | `-36..36` | pitch (semitones) | ✓ | ✓ | ✓ |
| `output_gain_db` | `-24..12` | overall energy (dB) | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` | noise component (dB) | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0` | glottal shape `R_d` (breathy–pressed) | — | ✓ | ✓ |
| `f1_scale` / `f2_scale` | `0.7..1.3` | first/second formant (ratio) | — | — | ✓ |
| `f1_hz` / `f2_hz` | `150..1300` / `600..3200` | first/second formant (absolute Hz) | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | spectral tilt | — | — | ✓ |

`f1_hz` / `f2_hz` set a formant to an absolute Hz target, useful for
building an evenly-spaced Hz continuum across different stimulus items;
`f1_scale` / `f2_scale` instead shift each frame's natural contour
proportionally, keeping its shape. The two cannot be combined on the
same formant.

Hear what each parameter does on the [demo page](https://n1r.github.io/ARIS_nsf/);
parameter semantics and condition design are covered in the
[Manipulation guide (Chinese)](docs/MANIPULATION_ZH.md).

## 6. Browser workbench (Studio)

If you would rather not assemble `--variant` strings on the command line,
you can design conditions and compare results in the browser:

```bash
source scripts/project_env.sh && uv sync --extra studio   # install the studio extra once
.venv/bin/aris studio        # opens http://127.0.0.1:8765/
```

The page generates parameter sliders for your model, supports named
conditions and a continuum builder (e.g. F1 from 400 to 600 Hz in five
steps, one click for the whole stimulus set), one-click rendering,
original/variant A/B listening, and time-aligned waveform and
spectrogram comparison; clipping or formants hitting the model's range
edge are flagged in red. Renders land in `studio_output/` with exactly
the same output and metadata format as the command-line `manipulate`.

## 7. Command reference

```text
aris doctor             check audio and training dependencies
aris fetch-corpus       download the CMU ARCTIC example corpus
aris split              segment continuous recordings
aris prepare            resample, extract F0, split the dataset
aris validate           check dataset integrity
aris init-experiment    create a training experiment directory
aris train              start training
aris controls           list manipulation parameters supported by a model
aris synthesize         reconstruct recordings from a checkpoint
aris manipulate         generate manipulated stimuli
aris studio             launch the browser workbench
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
