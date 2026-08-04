# ARIS

[简体中文](README.md) | **English**

## 0. Introduction

ARIS (Analytic Resonance for Interpretable Synthesis) is a differentiable
analysis-by-synthesis tool for phonetic research: train a DDSP/GOLF vocoder
on a few tens of minutes of single-speaker recordings, then change F0,
energy, noise, glottal source shape (`R_d`), or formants (F1/F2, spectral
tilt) one at a time — with everything else held fixed — to batch-generate
paired experimental stimuli.

- Listening demo: <https://n1r.github.io/ARIS_nsf/>
- Paper: SLT 2026 (citation at the end)

Training runs on an ordinary gaming GPU (e.g. an RTX 4060) in a few hours
for a few tens of minutes of data; reconstruction and stimulus generation
need no GPU and run on a regular CPU.

## 1. Installing dependencies

The only prerequisite is [uv](https://docs.astral.sh/uv/) (a single-file
Python environment manager). From the repository root:

```bash
source scripts/project_env.sh      # enter the project environment
./scripts/setup_project_env.sh     # install all dependencies (inside the repo, system untouched)
.venv/bin/aris doctor              # self-check: audio, F0, GPU environment
```

All commands are then invoked as `.venv/bin/aris`; every command supports
`--help`.

## 2. Preparing data

Put your recordings in one folder (WAV, single speaker, quiet room):

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

1. 20–60 minutes of audio in total is recommended; cutting long recordings
   into utterances of a few seconds speeds up training.
2. F0 is extracted with autocorrelation by default — nothing extra to
   install; `.pv` pitch tracks already extracted with Praat or similar can
   be reused via `--f0-method sidecar`.
3. Sample rates need not be unified beforehand; `prepare` resamples to the
   model sample rate automatically.
4. No recordings at hand? `.venv/bin/aris fetch-corpus data/arctic`
   downloads ~30 minutes of the public CMU ARCTIC corpus for a full trial
   run.

## 3. Training

Create an experiment directory, then start training:

```bash
.venv/bin/aris init-experiment data/my_voice experiments/my_voice --model aria-golf
.venv/bin/aris train experiments/my_voice --dry-run   # check the configuration only
.venv/bin/aris train experiments/my_voice
```

1. `--model` is one of `ddsp`, `golf`, `aria-golf`; choose `aria-golf` if
   you need formant (F1/F2) and spectral-tilt control.
2. Checkpoints are saved under `experiments/my_voice/runs/checkpoints/`.
3. On a Slurm cluster, `init-experiment` has already generated a
   ready-to-`sbatch` `train.slurm`; ignore it if you have no cluster.

## 4. Reconstruction (inference)

Reconstruct the held-out test recordings with the trained checkpoint to
check model quality:

```bash
.venv/bin/aris synthesize experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/reconstruction
```

The output is reconstructed WAV files; listen against the originals, and if
they are close, move on.

## 5. Generating manipulated stimuli

Change one parameter at a time on top of the reconstruction, keeping
everything else fixed:

```bash
.venv/bin/aris manipulate experiments/my_voice \
  experiments/my_voice/runs/checkpoints/last.ckpt out/stimuli \
  --variant 'pitch_down:pitch_semitones=-4' \
  --variant 'f1_up:f1_scale=1.2'
```

Each `--variant` is a named condition (`name:param=value,param=value`) and
produces one set of WAVs. Available parameters and ranges:

| Parameter | Range | Meaning | DDSP | GOLF | ARIS-GOLF |
|---|---|---|:---:|:---:|:---:|
| `pitch_semitones` | `-36..36` | pitch (semitones) | ✓ | ✓ | ✓ |
| `output_gain_db` | `-24..12` | overall energy (dB) | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` | noise component (dB) | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0` | glottal shape `R_d` (breathy–pressed) | — | ✓ | ✓ |
| `f1_scale` / `f2_scale` | `0.7..1.3` | first/second formant | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | spectral tilt | — | — | ✓ |

1. `.venv/bin/aris controls experiments/my_voice` lists the parameters your
   trained model actually supports.
2. Each output directory carries JSON metadata — checkpoint hash, all
   control values, and clipping statistics — for reporting exactly how the
   stimuli were generated.
3. Condition design and more examples: [Manipulation guide (Chinese)](docs/MANIPULATION_ZH.md).

## 6. Command reference

```text
aris doctor             check audio, F0, and GPU training environment
aris fetch-corpus       download the CMU ARCTIC example corpus
aris split              segment continuous recordings
aris prepare            resample, extract F0, split the dataset
aris validate           check dataset integrity
aris init-experiment    create a training experiment directory
aris train              start or resume training
aris controls           list manipulation parameters supported by a model
aris synthesize         reconstruct recordings from a checkpoint
aris manipulate         generate manipulated stimuli
```

## 7. Citation

The ARIS (SLT 2026) entry will be added once the paper is online;
machine-readable metadata is in [CITATION.cff](CITATION.cff). This tool
builds on the GOLF vocoder:

- C.-Y. Yu and G. Fazekas, "Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis," *Interspeech 2024*. DOI: `10.21437/Interspeech.2024-1187`
- C.-Y. Yu and G. Fazekas, "Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables," *ISMIR 2023*. DOI: `10.5281/zenodo.10265377`

Code is released under the MIT license; see [LICENSE](LICENSE).
