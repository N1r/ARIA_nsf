# Data contract and reproducibility

## Dataset contract

`manifest.csv` is the stable interface between preparation, visualisation, and
training. Paths are relative to the dataset root, so the directory can be moved.
Each row records the SHA-256 of the original input, the deterministic split, audio
statistics, and the F0 backend. `dataset.json` records preparation parameters and a
dataset fingerprint computed from source hashes, identifiers, and splits.
`phonlab parameters` exports the researcher-facing fixed-schema `parameters.csv`;
it is derived from, not a replacement for, the manifest contract.

Preparation uses a staging directory and renames it only after every record and
manifest have been written. Existing output directories are rejected. This prevents
a failed conversion from looking like a complete dataset.

## Artifact policy

Git contains source, small configs, tests, and documentation. It does not contain
raw recordings, checkpoints, run logs, or generated reports. Store those in an
artifact service or an explicitly backed-up filesystem. A publication archive
should include:

1. repository commit;
2. `dataset.json` and `manifest.csv` (subject to participant privacy policy);
3. experiment directory;
4. best and last checkpoints with SHA-256;
5. evaluation tables and the command used to generate them;
6. dependency lock or container digest and GPU/CUDA information.
7. raw Lightning metrics CSV plus the generated loss dashboard;
8. `training_job.json`/scheduler evidence and manipulation metadata, including
   checkpoint SHA-256 and exact control values.

## Frozen baseline

The executable source was copied from
`../golf_frozen_slt2026_20260708`, frozen on 2026-07-08. Its freeze marker,
working-tree state, patch, and exact environment are retained in `provenance/`.
Large runs, checkpoints, paper PDFs, and web-demo binaries were intentionally not
duplicated. They remain in the immutable frozen snapshot.

The imported `models/`, `ltng/`, `loss/`, `datasets/`, and `cfg/` directories are
the compatibility engine. New user workflows live under `src/phonlab_ddsp/`.
Future model refactors should add parity tests before moving engine modules.

Run `make verify-engine` to compare imported engine files with
`provenance/ENGINE_FILES.sha256`. An intentional engine edit will fail this check;
after review, update the manifest with
`python tools/engine_checksums.py --write`.
