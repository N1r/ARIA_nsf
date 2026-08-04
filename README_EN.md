# ARIS

[简体中文](README.md) | **English**

Official toolkit for ARIS (Analytic Resonance for Interpretable Synthesis):
differentiable DDSP/GOLF analysis-by-synthesis and controlled stimulus
generation for phonetic research. Given a few tens of minutes of single-speaker
recordings, train a differentiable vocoder, then manipulate F0, energy, noise,
glottal source shape (`R_d`), or formants (F1/F2, spectral tilt) one at a time
while everything else stays fixed — producing paired experimental stimuli with
full provenance from raw-file hashes to output WAVs.

- Listening demo: <https://n1r.github.io/ARIS_nsf/>
- Paper: SLT 2026 (citation below and in `CITATION.cff`)

## Installation

The only prerequisite is [`uv`](https://docs.astral.sh/uv/) (a Python
environment manager, a single executable). The three lines below enter the
project environment, install all dependencies, and check that everything is
ready. Python and all dependencies stay inside the repository directory:

```bash
source scripts/project_env.sh
./scripts/setup_project_env.sh
.venv/bin/aris doctor
```

## Quick start

```bash
# 1. Data preparation: segment, resample, extract F0, quality-check
.venv/bin/aris split recordings/ segments/ --mode silence
.venv/bin/aris prepare segments/audio data/my_voice --f0-method autocorr
.venv/bin/aris validate data/my_voice

# 2. Create and launch an experiment (--dry-run checks the config first)
.venv/bin/aris init-experiment data/my_voice experiments/my_voice_golf --model golf
.venv/bin/aris train experiments/my_voice_golf --dry-run

# 3. After training: reconstruct and render manipulation conditions
.venv/bin/aris synthesize experiments/my_voice_golf CKPT out/reconstruction
.venv/bin/aris manipulate experiments/my_voice_golf CKPT out/manipulations \
  --variant 'pitch_down:pitch_semitones=-4'
```

`CKPT` is the model file produced by training (e.g.
`runs/checkpoints/last.ckpt`). Inference and manipulation run on an ordinary
CPU; training is best done on a GPU. If you have a Slurm cluster,
`init-experiment` generates ready-to-submit job scripts, but Slurm is optional.

**Data preparation tip**: the default F0 backend `autocorr` needs no extra
installation; pitch tracks already extracted with Praat or other tools can be
reused via `--f0-method sidecar`.

## Controllable parameters

| Parameter | Range | DDSP | GOLF | ARIS-GOLF |
|---|---|:---:|:---:|:---:|
| `pitch_semitones` | `-36..36` | ✓ | ✓ | ✓ |
| `output_gain_db` | `-24..12` dB | ✓ | ✓ | ✓ |
| `noise_gain_db` | `-24..24` dB | ✓ | ✓ | ✓ |
| `glottal_rd_scale` | `0.5..2.0` | — | ✓ | ✓ |
| `f1_scale` / `f2_scale` | `0.7..1.3` | — | — | ✓ |
| `tilt_alpha_delta` | `-0.25..0.25` | — | — | ✓ |

Usage and condition design: [Manipulation guide (Chinese)](docs/MANIPULATION_ZH.md).

## CLI

```text
aris doctor             check audio, F0, and GPU-training environment
aris fetch-corpus       download and verify the CMU ARCTIC example corpus
aris split              segment continuous recordings
aris prepare            resample, downmix, extract F0, split, and hash data
aris validate           validate manifests, paths, and file integrity
aris init-experiment    create config, provenance, and training launchers
aris train              verify dataset fingerprint, then launch training
aris controls           list controls declared by an experiment model
aris synthesize         reconstruct held-out audio from a checkpoint
aris manipulate         render named source/filter control conditions
```

Every command has `--help`.

## Repository layout

```text
src/aris/          single Python package
├── cli.py …       CLI and data preparation (audio/corpus/segment/manifest/doctor)
├── controls/      manipulation control parameters
├── presets/       built-in model configurations (ddsp / golf / aria_golf)
├── models/        differentiable synthesis models (GOLF / DDSP / ARIS)
├── training/      Lightning training modules
├── losses/        spectral losses
└── engine.py      training/inference engine entry point (python -m aris.engine)
configs/           example decoder configurations
scripts/           environment setup scripts
tests/  docs/
```

## Design conventions

- No silent per-file loudness/peak normalisation, so research variables are never erased implicitly.
- F0 bounds, method, sample rate, and split seed enter provenance; dataset paths are relative, so datasets can be moved between machines.
- Raw-file SHA-256 and dataset fingerprints guard against silent data drift.
- Manipulation records checkpoint SHA-256, all control values, and clipping statistics.

## Development

```bash
make test    # full test suite
make lint    # ruff
```

Other Makefile targets: `install`, `test-lightweight`, `format`, `doctor`, `verify`.

## Citation

The ARIS (SLT 2026) citation entry will be added once the paper is online;
machine-readable metadata is in [CITATION.cff](CITATION.cff). Underlying GOLF
methods:

- Chin-Yun Yu and György Fazekas, “Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis,” Interspeech 2024, DOI: `10.21437/Interspeech.2024-1187`.
- Chin-Yun Yu and György Fazekas, “Singing Voice Synthesis Using Differentiable LPC and Glottal-Flow-Inspired Wavetables,” ISMIR 2023, DOI: `10.5281/zenodo.10265377`.

## License

MIT; see [LICENSE](LICENSE).
