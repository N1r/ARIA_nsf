# Architecture

```text
official corpus / the researcher's recordings
       │ fetch-corpus (optional)       │
       └───────────────┬───────────────┘
                       │ split
                       ▼
              WAV segments + segments.csv
                       │ prepare
                       ▼
 portable dataset + fingerprint + F0 tracks
          │ parameters             │ inspect
          ▼                        ▼
  parameters.csv            HTML audit/listening report
          └──────────────┬─────────┘
                         │ init-experiment
                         ▼
           config + hashes + Slurm launcher
                         │ confirmed Slurm submission
                         ▼
        GOLF/DDSP engine → metrics + checkpoints
                         │ synthesize / manipulate
                         ▼
       reconstruction + controlled variants + reports
```

## Package boundary

- `src/phonlab_ddsp/` is the stable, documented workflow library.
  - `corpus.py`: verified download, safe extraction, deterministic duration subset.
  - `segment.py`, `audio.py`, `manifest.py`: splitting, F0/acoustic extraction,
    portable datasets and validation.
  - `parameters.py`, `report.py`, `metrics.py`: CSV/HTML research outputs.
  - `experiment.py`, `lightning.py`: provenance-bound configuration and manifest
    adapters for the frozen engine.
  - `jobs.py`, `launchers.py`: argv-only Slurm control and post-processing bundles.
  - `manipulation.py`: checkpoint-bound voiced-F0 control and listening reports.
  - `cli.py`, `gui.py`: thin command-line and loopback-only web adapters over the
    same library functions.
- `models/`, `ltng/`, `loss/`, and selected top-level Python entry points retain
  their historical import names for frozen configs and checkpoints. New workflow
  features should not be added there.
- `provenance/` pins the imported engine state and checksums.
- `slurm/` contains real cluster acceptance jobs; `tools/` contains mechanical
  acceptance and repository checks.
- `tests/` separates dependency-light workflow tests from Torch/model compatibility
  tests.

The core package avoids importing Torch for corpus acquisition, splitting,
preparation, validation, parameter export, reports, metrics parsing and Slurm
inspection. A phonetician can prepare and audit data without a GPU, then submit
only training and checkpoint inference to a compute node.

## Data and execution boundaries

`manifest.csv` plus `dataset.json` is the portable dataset contract. Relative
artifact paths permit moving a complete dataset; the dataset fingerprint detects
silent changes. `experiment.json` binds that fingerprint to config and decoder
hashes. Manipulation metadata additionally binds the checkpoint SHA-256 and exact
F0 scale.

The GUI never interpolates shell commands. CPU actions call Python APIs directly;
Slurm actions pass argument arrays to `sbatch`, `squeue`, `sacct` and `scancel`.
Submission and cancellation require explicit confirmation, and logs are bounded
and confined to the selected job directory.

Mutable environments, downloads, caches and experimental outputs stay in
`.venv/`, `.cache/` and `artifacts/`. They are not package source and are excluded
from repository commits. The wheel retains the legacy `models`, `ltng` and `loss`
names because serialized checkpoints use them; the old generic `datasets`
namespace stays source-only to avoid a collision with the unrelated Hugging Face
package.
