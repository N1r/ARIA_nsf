# ARIS

[简体中文](README.md) | **English**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/N1r/ARIS_nsf/blob/main/notebooks/ARIS_Tutorial_and_Workflow.ipynb)
[![Listening demo](https://img.shields.io/badge/demo-listen-blue)](https://n1r.github.io/ARIS_nsf/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

## 0. Introduction

ARIS (Analytic Resonance for Interpretable Synthesis) is a differentiable
analysis-by-synthesis tool for phonetics research. Train a DDSP/GOLF vocoder on
single-speaker recordings, then apply recorded controls to F0, post-synthesis waveform
gain, stochastic-source gain, glottal-source shape (`R_d`), or vocal-tract resonances (F1/F2 and spectral
tilt) on the same reconstruction to generate paired stimuli or continua.

- Listening demo: <https://n1r.github.io/ARIS_nsf/>
- Tutorial notebook: [`notebooks/ARIS_Tutorial_and_Workflow.ipynb`](notebooks/ARIS_Tutorial_and_Workflow.ipynb)

Training requires a CUDA-capable NVIDIA GPU (a Google Colab T4 is enough for a trial run); resynthesis and stimulus manipulation can also run on CPU.

Depending on your workflow, ARIS provides multiple ways to interact:

1. **Google Colab Cloud Tutorial**: Click the **Open in Colab** badge at the top of this page to run the full workflow (from audio analysis and training to stimulus manipulation) on a cloud GPU without local configuration.
2. **Interactive Listening Demo**: Open the [online listening demo](https://n1r.github.io/ARIS_nsf/) to explore acoustic parameter effects.
3. **Local GUI Workbench (Studio)**: Run `uv sync --locked --all-extras && uv run aris studio` to launch an interactive slider workspace in your browser for parameter exploration and continuum generation.
4. **Command-Line Interface & Python API**: Suitable for batch stimulus generation, automated experiment scripts, and large-scale training (Sections 2–5 and Section 7).

### Controls and validation for phonetics research

| Research dimension | ARIS control | Independently remeasure in the output |
|---|---|---|
| Pitch and intonation | `pitch_semitones` | median F0, contour shape, voiced proportion |
| Vowel resonance | `f1_*`, `f2_*` | F1/F2 tracks, vowel space, intelligibility |
| Voice quality | `glottal_rd_scale`, `noise_gain_db`, `tilt_alpha_delta` | H1–H2, CPP, HNR, spectral slope |
| Stimulus waveform gain (not vocal effort) | `output_gain_db` | peak, RMS/LUFS, clipped-sample count |

These parameters are **model controls**, not physiological measurements or perceptual
labels. For example, `output_gain_db` is not vocal effort, and `glottal_rd_scale` is
not an EGG measurement. For confirmatory work, preregister the target acoustic
measures, remeasure the actual effect in the generated WAV files, and report
reconstruction error. ARIS currently provides no duration, speech-rate, or
within-utterance interval control.

### Recommended research workflow

1. Define the hypothesis, target acoustic measures, and exclusion criteria.
2. Record one speaker with a consistent signal chain and preserve the source files.
3. Prepare data, freeze the split, train, and evaluate reconstruction on held-out data.
4. Start with small single-parameter changes; retain single-factor controls for combinations.
5. Generate baseline and variants, then blind-listen and independently remeasure them.
6. Archive the checkpoint, configuration, dataset fingerprint, control metadata, and scripts.

## 1. Installation & Quickstart

Platforms: Linux, macOS, or Windows (via [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)).

### Recommended Installation (Using uv)

ARIS uses `pyproject.toml`, `uv.lock`, and [uv](https://docs.astral.sh/uv/) for Python and dependency management. After cloning the repository, run these commands from its root:

```bash
# Create the project environment and sync exactly from the lockfile
uv sync --locked --all-extras

# Run diagnostics inside the project environment
uv run aris doctor
```

Every command below uses `uv run aris ...`; activating `.venv` is unnecessary. Run
`uv sync --locked --all-extras` again whenever dependencies change. If you intentionally
change dependencies, run `uv lock` first and commit the updated lockfile.

`aris doctor` checks Python, audio dependencies, PyTorch, CUDA, and GPU status. Continue once all required checks pass.

### Google Colab

Open the Colab link above, select a **T4 GPU** runtime, and choose
**Runtime → Run all**. The tutorial uses uv throughout: uv installs Python 3.11, creates
the project environment from `uv.lock`, and executes ARIS through `uv run aris ...`.
Plots and audio players remain in the Colab kernel, so changes to Colab's system Python
do not affect ARIS and no runtime restart is needed. The tutorial trains for 1,000 steps,
then uses that checkpoint for reconstruction and manipulation. This is enough to hear
an initial result, but it is not evidence of convergence or research-ready quality.
The default batch size is 8 to
fit a Colab T4; the release's 40,000-step checkpoint provides a stable quality reference.

**Try the Pretrained Model Immediately:**
The official release provides a trained model (Mandarin female speaker, 16 kHz) and
matching test recordings. Download and extract `aris_f024_demo.zip`, then run:

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
uv run aris split recordings/ segments/ --mode silence        # cut into short utterances at silences
uv run aris prepare segments/audio data/my_voice              # resample, extract F0, split the dataset
uv run aris validate data/my_voice                            # confirm the dataset is complete and usable
```

A few practical notes on data and study design:

1. 20–60 minutes is recommended. Keep microphone, distance, gain, room, and task
   consistent; avoid waveform overload, reverberation, background sound, and automatic gain.
   Confirm consent and the permitted uses of the recordings.
2. Cutting long recordings into utterances of a few seconds speeds up training. If
   material spans sessions, word lists, or speaking styles, plan the split before a
   confirmatory study so near-duplicate items do not leak across train and test.
3. For F0 extraction, `--f0-method pyworld` (WORLD's DIO+StoneMask, stable
   and fast) is recommended; the default `auto` falls back to
   autocorrelation when WORLD is not installed.
4. For tonal languages such as Mandarin, or other F0-sensitive work,
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
5. Sample rates need not be unified beforehand; `prepare` resamples to the
   model sample rate automatically. Preserve the unresampled, unnormalized source
   recordings rather than overwriting the research archive.
6. No recordings at hand? `uv run aris fetch-corpus data/arctic`
   downloads ~30 minutes of the public CMU ARCTIC corpus for a full trial
   run; once downloaded, continue with
   `uv run aris prepare data/arctic/selected data/arctic_prepared`.

## 3. Training

With your data ready, create an experiment directory and start training:

```bash
uv run aris init-experiment data/my_voice experiments/my_voice --model aria-golf
uv run aris train experiments/my_voice --dry-run   # print the training command without running it
uv run aris train experiments/my_voice
```

A few notes:

1. Choose the simplest model that answers the research question:

   | Model | Available controls | Typical use |
   |---|---|---|
   | `ddsp` | F0, waveform gain, stochastic-source gain | baseline reconstruction and pitch studies |
   | `golf` | above + glottal `R_d` | source and voice-quality studies |
   | `aria-golf` | above + F1/F2, spectral tilt | explicit source–filter manipulation |

   `aria-golf` is the code name of the ARIS decoder. The LPC coefficients in regular
   `golf` cannot be interpreted as independent F1/F2 controls.
2. Checkpoints are saved under `experiments/my_voice/runs/checkpoints/`.
3. Training requires a CUDA-enabled PyTorch. The default Linux install
   already ships with CUDA support; if it does not match your GPU or
   driver, use the [PyTorch installation selector](https://pytorch.org/get-started/locally/),
   then install the matching build with `uv add` or `uv pip install`.
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
uv run aris synthesize experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/reconstruction
```

The output is reconstructed WAV files. Do not proceed on a general impression of
similarity alone: compare original and reconstruction on the held-out test set and
record at least intelligibility, artifacts, F0 and formant error, duration, waveform amplitude,
and clipping. Reconstruction error and the experimental control effect are separate
sources of variation and should be reported separately.

## 5. Generating manipulated stimuli

This step is what ARIS is really for: change one parameter at a time on top
of the reconstruction, keeping everything else fixed:

```bash
uv run aris manipulate experiments/my_voice \
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
| `output_gain_db` | `-24..12` | post-synthesis waveform gain (dB) | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` | digital gain on the stochastic-source branch (dB) | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0` | scaling of the model's glottal-source `R_d` | — | ✓ | ✓ |
| `f1_scale` / `f2_scale` | `0.7..1.3` | first/second formant (ratio) | — | — | ✓ |
| `f1_hz` / `f2_hz` | `150..1300` / `600..3200` | first/second formant (absolute Hz) | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | spectral tilt | — | — | ✓ |

`f1_hz` / `f2_hz` set a formant to an absolute Hz target, useful for
building an evenly-spaced Hz continuum across different stimulus items;
`f1_scale` / `f2_scale` instead shift each frame's natural contour
proportionally, keeping its shape. The two cannot be combined on the
same formant.

### Experimental design and quality control

- The accepted software range is not a validated phonetic range. Pilot small steps,
  then select effect sizes from acoustic remeasurement and intelligibility checks.
- Single-parameter conditions are easiest to interpret. For combined controls, retain
  the baseline and corresponding single-factor conditions.
- Use one frozen checkpoint for every formal condition. Verify its SHA-256, the dataset
  fingerprint, and complete control values in `manipulation.json`.
- Inspect `_render.json` in every condition directory. `clipped_samples` is the number
  of samples hard-limited after reaching digital full scale; formal materials should
  normally have a value of `0`. Do not normalize only selected conditions afterward.
- The comparison report and Studio support quality control, but provide no randomization,
  blinding, or playback-level calibration and are not perception-experiment platforms.

Hear what each parameter does on the [demo page](https://n1r.github.io/ARIS_nsf/);
parameter semantics and condition design are covered in the
[Manipulation guide (Chinese)](docs/MANIPULATION_ZH.md).

## 6. Browser workbench (Studio)

If you would rather not assemble `--variant` strings on the command line,
you can design conditions and compare results in the browser:

```bash
# Launch locally (opens http://127.0.0.1:8765/):
uv run aris studio

# Launch with a public shareable URL (ideal for Google Colab and remote servers):
uv run aris studio --share
```

The page generates parameter sliders for your model, supports named
conditions and a continuum builder (e.g. F1 from 400 to 600 Hz in five
steps, one click for the whole stimulus set), one-click rendering,
original/variant A/B listening, and time-aligned waveform and
spectrogram comparison; clipping or formants hitting the model's range
edge are flagged in red. Renders land in `studio_output/` with exactly
the same output and metadata format as the command-line `manipulate`.

## 7. Python API (Jupyter / Colab / Scripts)

ARIS also provides a Python API. Save the following as `workflow.py` and run it with
`uv run python workflow.py`; launch Jupyter with
`uv run --with jupyter jupyter lab`.

```python
import aris

# 1. Segment audio & prepare dataset manifest
aris.split("recordings/", "segments/", mode="silence")
manifest = aris.prepare("segments/audio", "data/my_voice", sample_rate=16000)

# 2. Validate manifest integrity
errors = aris.validate("data/my_voice")
assert not errors

# 3. Create experiment and run training
exp_dir = aris.init_experiment("data/my_voice", "experiments/my_voice", model="aria-golf")
aris.train(exp_dir)

# 4. Reconstruct speech (inference)
aris.synthesize(
    exp_dir,
    "experiments/my_voice/runs/checkpoints/last.ckpt",
    "out/recon",
)

# 5. Generate acoustic manipulation stimuli
aris.manipulate(
    exp_dir,
    "experiments/my_voice/runs/checkpoints/last.ckpt",
    "out/stimuli",
    variants=[
        "f1_up:f1_scale=1.2",
        "pitch_down:pitch_semitones=-4",
        "rd_high:glottal_rd_scale=1.6",
    ],
)

# 6. Launch interactive Gradio Web Studio inside Jupyter / Colab
aris.launch_studio(workspace=".", share=True)
```

## 8. Command reference

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

## 9. Citation

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

The ARIS vocoder implementation derives from GOLF:

- C.-Y. Yu and G. Fazekas, "Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis," *Interspeech 2024*. DOI: `10.21437/Interspeech.2024-1187`
- C.-Y. Yu and G. Fazekas, "Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables," *ISMIR 2023*. DOI: `10.5281/zenodo.10265377`

The earlier methodological foundations are differentiable DSP and neural
source-filter models:

- J. Engel, L. Hantrakul, C. Gu, and A. Roberts, "DDSP: Differentiable Digital Signal Processing," *ICLR 2020*. arXiv: `2001.04643`
- X. Wang, S. Takaki, and J. Yamagishi, "Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis," *IEEE/ACM TASLP*, 2020. arXiv: `1904.12088`

Code is released under the MIT license; see [LICENSE](LICENSE).

## 10. Contact

If you run into problems or have suggestions, feel free to open an
[issue](https://github.com/N1r/ARIS_nsf/issues), or email
<dingyr@hum.leidenuniv.nl> (Leiden University).
