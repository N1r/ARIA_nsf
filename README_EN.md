# ARIS

[简体中文](README.md) | **English**

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

## 1. Installing dependencies

All you need on your machine is [uv](https://docs.astral.sh/uv/) (a
single-file Python environment manager). From the repository root:

```bash
source scripts/project_env.sh      # enter the project environment
./scripts/setup_project_env.sh     # install all dependencies (inside the repo, system untouched)
.venv/bin/aris doctor              # check audio and training dependencies
```

Once installed, all commands are invoked as `.venv/bin/aris`; whenever
you are unsure about usage, every command supports `--help`.

**Want to hear it first?** We provide a trained model (Mandarin female
speaker, 16 kHz) with a matching example dataset, so you can try
reconstruction and manipulation without training anything:

```bash
curl -LO https://github.com/N1r/ARIS_nsf/releases/download/v0.1.0/aris_f024_demo.tar.gz
tar -xzf aris_f024_demo.tar.gz    # extract in the repository root; creates demo_f024/

.venv/bin/aris synthesize demo_f024/experiment \
  demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_recon
.venv/bin/aris manipulate demo_f024/experiment \
  demo_f024/experiment/runs/checkpoints/last.ckpt out/demo_stimuli \
  --variant 'f1_up:f1_scale=1.2'
```

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
3. On a Slurm cluster, `init-experiment` has already generated a
   ready-to-`sbatch` `train.slurm`; adjust cluster parameters (partition,
   GPU type, etc.) via `init-experiment` options, and fill in the CUDA
   module for your cluster before submitting. Ignore it if you have no
   cluster.

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
| `f1_scale` / `f2_scale` | `0.7..1.3` | first/second formant | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | spectral tilt | — | — | ✓ |

Hear what each parameter does on the [demo page](https://n1r.github.io/ARIS_nsf/);
parameter semantics and condition design are covered in the
[Manipulation guide (Chinese)](docs/MANIPULATION_ZH.md).

## 6. Command reference

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
```

## 7. Citation

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff).

The ARIS vocoder implementation derives from GOLF:

- C.-Y. Yu and G. Fazekas, "Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis," *Interspeech 2024*. DOI: `10.21437/Interspeech.2024-1187`
- C.-Y. Yu and G. Fazekas, "Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables," *ISMIR 2023*. DOI: `10.5281/zenodo.10265377`

The earlier methodological foundations are differentiable DSP and neural
source-filter models:

- J. Engel, L. Hantrakul, C. Gu, and A. Roberts, "DDSP: Differentiable Digital Signal Processing," *ICLR 2020*. arXiv: `2001.04643`
- X. Wang, S. Takaki, and J. Yamagishi, "Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis," *IEEE/ACM TASLP*, 2020. arXiv: `1904.12088`

Code is released under the MIT license; see [LICENSE](LICENSE).

## 8. Contact

If you run into problems or have suggestions, feel free to open an
[issue](https://github.com/N1r/ARIS_nsf/issues), or email
<dingyr@hum.leidenuniv.nl> (Leiden University).
