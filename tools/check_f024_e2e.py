#!/usr/bin/env python3
"""Mechanical success predicate for the real F024 smoke workflow."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


def missing_units(root: Path) -> tuple[int, list[str]]:
    missing = []
    dataset = root / "dataset"
    experiment = root / "experiment"
    reconstructions = root / "reconstructions"

    manifest = dataset / "manifest.csv"
    metadata = dataset / "dataset.json"
    report = dataset / "report.html"
    for path in (manifest, metadata, report):
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(f"missing or empty: {path}")

    experiment_json = experiment / "experiment.json"
    config = experiment / "config.yaml"
    decoder = experiment / "decoder.yaml"
    for path in (experiment_json, config, decoder):
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(f"missing or empty: {path}")

    checkpoints = list((experiment / "runs" / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        missing.append("no checkpoint")

    wavs = sorted(reconstructions.glob("*.wav"))
    if not wavs:
        missing.append("no reconstructed WAV")
    else:
        for path in wavs:
            try:
                with wave.open(str(path), "rb") as stream:
                    if stream.getnframes() <= 0 or stream.getframerate() != 16000:
                        missing.append(f"invalid reconstructed WAV: {path}")
            except (wave.Error, EOFError):
                missing.append(f"unreadable reconstructed WAV: {path}")

    comparison = root / "comparison.html"
    if not comparison.is_file() or comparison.stat().st_size < 500:
        missing.append("missing or too-small comparison.html")

    if experiment_json.is_file():
        try:
            payload = json.loads(experiment_json.read_text())
            if not payload.get("dataset_fingerprint"):
                missing.append("experiment missing dataset fingerprint")
        except (OSError, json.JSONDecodeError):
            missing.append("invalid experiment.json")
    return len(missing), missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()
    count, details = missing_units(args.root.resolve())
    if args.count:
        print(count)
        return 0
    if details:
        print("\n".join(details), file=sys.stderr)
        return 1
    print("F024_E2E_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
