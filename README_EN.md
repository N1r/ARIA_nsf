# PhonLab-DDSP

[简体中文](README.md) | **English**

PhonLab-DDSP is a reproducible DDSP/GOLF analysis-and-synthesis toolkit for
phoneticians. It turns the frozen SLT 2026 research code into an operational
workflow for **recording import or download, segmentation, F0 and acoustic
parameter extraction, quality control, Slurm training, loss visualisation,
checkpoint inference, and auditable multi-parameter manipulation**.

> Status: research preview `0.1.0`. Frozen import paths and checkpoint
> interfaces remain compatible. The data workflow, GOLF, and ARIA-GOLF have
> real GPU end-to-end tests. ARIA F1/F2/tilt controls have been verified through
> checkpoint loading, Slurm inference, unclipped WAV output, paired-condition
> differences, and WebUI export. Synthesised parameters must not be interpreted
> as physiological or articulatory measurements without independent validation.

## Where to start

| Goal | Entry point |
|---|---|
| Use a graphical workflow | `phonlab webui` and the [WebUI guide (Chinese)](docs/WEBUI_ZH.md) |
| Prepare your recordings | [Quick start (Chinese)](docs/QUICKSTART_ZH.md) |
| Reproduce the public 30-minute pipeline | [CMU ARCTIC pipeline (Chinese)](docs/CMU_ARCTIC_PIPELINE_ZH.md) |
| Understand hashes and reproducibility | [Data and reproducibility](docs/DATA_AND_REPRODUCIBILITY.md) |
| Manipulate F0, source, noise, or formants | [Manipulation guide (Chinese)](docs/MANIPULATION_ZH.md) |
| Understand the directory structure | [Repository map (Chinese)](docs/REPOSITORY_MAP_ZH.md) |
| Understand stable and frozen code | [Architecture](docs/ARCHITECTURE.md) |

## Five-minute start

The login node must provide `uv`, either through the system or a module. If it
is not on `PATH`, pass `UV_BIN=/absolute/path/to/uv` to the setup script. Python,
dependencies, build caches, and model downloads remain inside this repository.

```bash
source scripts/project_env.sh
./scripts/setup_project_env.sh
.venv/bin/phonlab doctor

.venv/bin/phonlab split recordings/ segments/ --mode silence
.venv/bin/phonlab prepare segments/audio data/my_voice --f0-method autocorr
.venv/bin/phonlab validate data/my_voice
.venv/bin/phonlab inspect data/my_voice

.venv/bin/phonlab init-experiment \
  data/my_voice experiments/my_voice_golf --model golf
.venv/bin/phonlab train experiments/my_voice_golf --dry-run
```

The portable default F0 backend is `autocorr`, so `pyworld` is optional.
Existing Praat or other `.pv` tracks can be reused with `--f0-method sidecar`.

## Local WebUI

```bash
.venv/bin/phonlab webui
```

The self-contained WebUI requires neither Streamlit nor a frontend build. It
provides model-aware sliders, named manipulation conditions, Slurm submission
and status tracking, baseline/variant listening, individual WAV downloads,
workspace-local exports, and provenance-bearing ZIP archives. `gui` is a
compatibility alias.

For a remote cluster, keep the server on loopback and use an SSH tunnel:

```bash
# Cluster login node
.venv/bin/phonlab webui --host 127.0.0.1 --port 8765 --no-browser

# Local computer
ssh -N -L 8765:127.0.0.1:8765 USER@CLUSTER
```

Open `http://127.0.0.1:8765/` locally. Do not bind the workbench to `0.0.0.0`;
it is a single-user local research tool, not a public multi-user service.

`scripts/project_env.sh` keeps uv, Python, Torch, Numba, Matplotlib, and model
caches below `.venv/`, `.cache/`, and `artifacts/`. GPU training and checkpoint
inference must be submitted through Slurm, not run on a login node.

## Reproducible acceptance scenarios

### F024 wiring smoke test

```bash
source scripts/project_env.sh
./scripts/setup_project_env.sh
sbatch slurm/f024_e2e_smoke.slurm
```

This job requests one GPU, performs a one-step wiring test, writes a checkpoint,
reconstructs held-out records, and creates a listening report. One step verifies
integration; it does not demonstrate convergence or model quality.

### Public 30-minute corpus pipeline

The public-data scenario uses the official CMU ARCTIC `slt` speaker:

```bash
source scripts/project_env.sh
.venv/bin/phonlab fetch-corpus artifacts/cmu_arctic_slt_demo/corpus
.venv/bin/phonlab split \
  artifacts/cmu_arctic_slt_demo/corpus/continuous.wav \
  artifacts/cmu_arctic_slt_demo/segments
# Continue with docs/CMU_ARCTIC_PIPELINE_ZH.md
```

The recorded acceptance run trained for 400 steps on an NVIDIA L4 and produced
63 held-out reconstructions plus 63 files at each of -4 and +4 semitones.
`train_loss` fell from 297.72 to 5.52, final `val_loss` was 6.036, and no NaN or
Inf was reported. The predicate returned `PHONLAB_COMPLETE_PIPELINE_OK`. These
values are engineering reproducibility evidence, not proof of full convergence
or phonetic validity.

### GOLF and ARIA-GOLF manipulation

The GOLF acceptance run generated a baseline plus eight F0, output-level,
noise, `R_d`, and combined conditions: 567 WAV files. Every non-default
condition changed all 63 held-out files, with zero clipping.

ARIA-GOLF was separately trained for 400 F024 steps. The checkpoint with the
best validation loss (`5.05988`) rendered isolated F1 `0.9/1.1`, F2 `0.9/1.1`,
and tilt `-0.1/+0.1` pairs. Every pair differed for all four held-out files and
all outputs were unclipped.

```bash
make verify-webui
make verify-aria
```

The ARIA predicate checks checkpoint SHA-256, dataset fingerprint, runtime
capabilities, decoder-hook calls, complete WAV sets, pairwise differences, and
clipping. It does not replace independent formant tracking or a controlled
perception experiment.

## CLI

```text
phonlab doctor             inspect audio, F0, and GPU-training dependencies
phonlab fetch-corpus       download and verify the CMU ARCTIC example corpus
phonlab split              segment continuous audio
phonlab prepare            resample, downmix, extract F0, split, and hash data
phonlab parameters         export per-record acoustic parameters
phonlab validate           validate manifests, paths, and file integrity
phonlab inspect            create an offline HTML quality-control report
phonlab init-experiment    create config, provenance, Shell, and Slurm launchers
phonlab train              verify provenance and launch Lightning training
phonlab metrics            visualise losses, learning rate, and numerical issues
phonlab controls           list controls declared by an experiment model
phonlab synthesize         reconstruct held-out audio from a checkpoint
phonlab manipulate         render named source/filter control conditions
phonlab init-postprocess   create a reconstruction/manipulation GPU job bundle
phonlab compare            create an original/reconstruction listening page
phonlab webui / gui        launch the local browser workbench
```

Every command provides `--help`.

## Repository layout

```text
src/phonlab_ddsp/  stable shared library, controls, CLI, and WebUI
models/            imported GOLF/DDSP/ARIA synthesis models
ltng/              imported Lightning training code
loss/              spectral losses
cfg/               paper-era and ARIA model configurations
tests/             workflow, acceptance, and compatibility tests
docs/              user, architecture, and reproducibility documentation
provenance/        provenance for the 2026-07-08 frozen snapshot
slurm/             real GPU acceptance jobs
tools/             mechanical acceptance and repository-audit programs
```

New application code belongs in `src/phonlab_ddsp/`. Top-level `models/`,
`ltng/`, `loss/`, and several entry modules retain historical paths for
checkpoint compatibility.

Some frozen research configurations contain site-specific `/zfsstore/...`
paths preserved as provenance. They are not portable presets and are not used
by the stable `phonlab` workflow.

## Research defaults

- Per-file peak or loudness normalisation is not silently applied.
- F0 bounds, method, sample rate, and split seed enter provenance.
- Dataset paths are relative and source files are protected by SHA-256.
- HTML reports expose clipping, missing F0, and record-level audio.
- Training logs use local CSV without requiring an online tracking service.
- Manipulation records controls, dataset/checkpoint identity, hook calls, and
  clipping.

Manipulations are controlled synthesis conditions, not automatically
physiological, articulatory, or causal measurements.

## Development

```bash
source scripts/project_env.sh
./scripts/setup_project_env.sh
make test
make lint
python tools/repo_audit.py --strict
python tools/engine_checksums.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing the frozen engine.

## Publishing on GitHub

Follow the [GitHub release checklist (Chinese)](docs/GITHUB_RELEASE_ZH.md).
Replace placeholder maintainer information in `pyproject.toml` and
`CITATION.cff`, add the final repository URL, confirm licensing and attribution,
run all verification gates, and inspect the staged file list. Training data,
checkpoints, experiments, `.venv/`, and caches are ignored and must not be
force-added. See [SECURITY.md](SECURITY.md) for the WebUI threat model.

## Source and citation

The research engine comes from the sibling frozen snapshot
`golf_frozen_slt2026_20260708`. Freeze state, patches, and environment records
are under `provenance/`; large runs and checkpoints are not copied into Git.

Underlying GOLF methods:

- Chin-Yun Yu and György Fazekas, “Differentiable Time-Varying Linear
  Prediction in the Context of End-to-End Analysis-by-Synthesis,” Interspeech
  2024, DOI: `10.21437/Interspeech.2024-1187`.
- Chin-Yun Yu and György Fazekas, “Singing Voice Synthesis Using
  Differentiable LPC and Glottal-Flow-Inspired Wavetables,” ISMIR 2023, DOI:
  `10.5281/zenodo.10265377`.

Add final maintainers, repository URL, and release DOI before publication.
Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License

MIT; see [LICENSE](LICENSE). Confirm attribution before publishing derived
research code.
