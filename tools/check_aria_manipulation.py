#!/usr/bin/env python3
"""Strict end-to-end acceptance for ARIA-GOLF formant and tilt controls.

This checker builds on the general control-artifact validator and adds the
conditions needed to close the ARIA-GOLF integration gap: a real checkpoint,
an ``aria-golf`` experiment, isolated up/down F1, F2, and tilt conditions, and
pairwise-distinct rendered audio.  It validates artifacts only; it never loads
the checkpoint with PyTorch and is therefore safe to run on a CPU/login node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import soundfile as sf

try:
    from tools.check_control_manipulation import check_control_manipulation
except ModuleNotFoundError:  # Direct execution adds tools/, not the repository, to sys.path.
    from check_control_manipulation import check_control_manipulation

SUCCESS_TOKEN = "PHONLAB_ARIA_MANIPULATION_OK"
REQUIRED_CENTERS = {
    "f1_scale": 1.0,
    "f2_scale": 1.0,
    "tilt_alpha_delta": 0.0,
}


def check_aria_manipulation(baseline: Path, manipulations: Path) -> dict[str, Any]:
    """Validate one real ARIA-GOLF checkpoint/control render set."""

    baseline = Path(baseline).expanduser().resolve()
    manipulations = Path(manipulations).expanduser().resolve()
    base = check_control_manipulation(baseline, manipulations)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "tool": "check_aria_manipulation",
        "success": False,
        "success_token": None,
        "inputs": {"baseline": str(baseline), "manipulations": str(manipulations)},
        "general_control_acceptance": {
            "success": bool(base.get("success")),
            "success_token": base.get("success_token"),
            "issues": base.get("issues", []),
        },
        "checkpoint": {},
        "experiment": {},
        "control_pairs": [],
        "issues": [],
    }
    issues: list[dict[str, Any]] = report["issues"]
    if not base.get("success"):
        _issue(
            issues,
            "general_control_acceptance",
            "General control artifact acceptance failed; inspect embedded issues",
        )

    metadata = _load_json(manipulations / "manipulation.json", issues, "metadata")
    if metadata is None:
        return _finish(report)
    if metadata.get("model") != "aria-golf":
        _issue(issues, "metadata.model", "model must be exactly 'aria-golf'")

    _check_checkpoint(metadata, report, issues)
    _check_experiment(metadata, report, issues)
    pairs = _collect_control_pairs(metadata.get("outputs"), issues)
    for control, directions in pairs.items():
        pair_report = {
            "control": control,
            "down": directions.get("down", {}).get("name"),
            "up": directions.get("up", {}).get("name"),
            "down_value": directions.get("down", {}).get("value"),
            "up_value": directions.get("up", {}).get("value"),
            "files_compared": 0,
            "files_different": 0,
            "max_abs_difference": None,
        }
        down = directions.get("down")
        up = directions.get("up")
        if down is not None and up is not None:
            down_root = _safe_pair_directory(manipulations, down.get("directory"))
            up_root = _safe_pair_directory(manipulations, up.get("directory"))
            if down_root is None or up_root is None:
                _issue(
                    issues,
                    "control.pair_directory",
                    f"{control} up/down directories must be direct, non-symlink children",
                )
            else:
                comparison = _compare_pair(down_root, up_root, control, issues)
                pair_report.update(comparison)
        report["control_pairs"].append(pair_report)

    return _finish(report)


def _check_checkpoint(
    metadata: Mapping[str, Any], report: dict[str, Any], issues: list[dict[str, Any]]
) -> None:
    raw_path = metadata.get("checkpoint")
    expected = metadata.get("checkpoint_sha256")
    path = Path(raw_path).expanduser().resolve() if isinstance(raw_path, str) else None
    actual = None
    if path is None or not path.is_file() or path.is_symlink():
        _issue(issues, "checkpoint.missing", "Recorded checkpoint is not a regular file")
    else:
        actual = _sha256(path)
        if actual != expected:
            _issue(issues, "checkpoint.sha256", "Recorded checkpoint SHA-256 has changed")
    report["checkpoint"] = {
        "path": str(path) if path is not None else raw_path,
        "recorded_sha256": expected,
        "actual_sha256": actual,
        "matches": actual is not None and actual == expected,
    }


def _check_experiment(
    metadata: Mapping[str, Any], report: dict[str, Any], issues: list[dict[str, Any]]
) -> None:
    raw_path = metadata.get("experiment")
    root = Path(raw_path).expanduser().resolve() if isinstance(raw_path, str) else None
    experiment = (
        _load_json(root / "experiment.json", issues, "experiment") if root is not None else None
    )
    if root is None:
        _issue(issues, "experiment.path", "Recorded experiment path is missing")
    if experiment is not None:
        if experiment.get("model") != "aria-golf":
            _issue(issues, "experiment.model", "Experiment model must be exactly 'aria-golf'")
        if experiment.get("dataset_fingerprint") != metadata.get("dataset_fingerprint"):
            _issue(
                issues,
                "experiment.dataset_fingerprint",
                "Experiment and manipulation dataset fingerprints differ",
            )
    report["experiment"] = {
        "path": str(root) if root is not None else raw_path,
        "model": experiment.get("model") if experiment else None,
        "dataset_fingerprint": experiment.get("dataset_fingerprint") if experiment else None,
        "matches": bool(
            experiment
            and experiment.get("model") == "aria-golf"
            and experiment.get("dataset_fingerprint") == metadata.get("dataset_fingerprint")
        ),
    }


def _collect_control_pairs(
    raw_outputs: Any, issues: list[dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {control: {} for control in REQUIRED_CENTERS}
    if not isinstance(raw_outputs, list):
        _issue(issues, "metadata.outputs", "outputs must be a list")
        return pairs
    for output in raw_outputs:
        if not isinstance(output, dict) or not isinstance(output.get("controls"), dict):
            continue
        controls = output["controls"]
        relevant = set(controls) & set(REQUIRED_CENTERS)
        if not relevant:
            continue
        if len(controls) != 1 or len(relevant) != 1:
            _issue(
                issues,
                "control.isolation",
                "ARIA acceptance conditions must change exactly one of F1, F2, or tilt",
                condition=output.get("name"),
            )
            continue
        control = next(iter(relevant))
        value = controls[control]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            continue
        center = REQUIRED_CENTERS[control]
        direction = "down" if value < center else "up" if value > center else "default"
        if direction == "default":
            _issue(
                issues,
                "control.default",
                f"{control} acceptance condition must be non-default",
                condition=output.get("name"),
            )
            continue
        if direction in pairs[control]:
            _issue(
                issues,
                "control.duplicate_direction",
                f"Multiple {direction} conditions were supplied for {control}",
                condition=output.get("name"),
            )
            continue
        pairs[control][direction] = {
            "name": output.get("name"),
            "directory": output.get("directory"),
            "value": float(value),
        }
    for control, directions in pairs.items():
        for direction in ("down", "up"):
            if direction not in directions:
                _issue(
                    issues,
                    "control.pair_missing",
                    f"Missing isolated {direction} condition for {control}",
                )
    return pairs


def _compare_pair(
    down_root: Path,
    up_root: Path,
    control: str,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    down_files = {path.name: path for path in down_root.glob("*.wav") if path.is_file()}
    up_files = {path.name: path for path in up_root.glob("*.wav") if path.is_file()}
    common = sorted(set(down_files) & set(up_files))
    if not common or set(down_files) != set(up_files):
        _issue(
            issues,
            "control.pair_files",
            f"{control} up/down conditions must contain the same non-empty WAV set",
        )
    different = 0
    maximum = 0.0
    for name in common:
        down, down_sr = sf.read(str(down_files[name]), dtype="float32", always_2d=False)
        up, up_sr = sf.read(str(up_files[name]), dtype="float32", always_2d=False)
        if down_sr != up_sr or np.shape(down) != np.shape(up):
            _issue(
                issues,
                "control.pair_shape",
                f"{control} pair differs in sample rate or shape for {name}",
            )
            continue
        difference = np.asarray(up, dtype=np.float64) - np.asarray(down, dtype=np.float64)
        peak = float(np.max(np.abs(difference))) if difference.size else 0.0
        maximum = max(maximum, peak)
        if np.any(difference != 0.0):
            different += 1
    if common and different != len(common):
        _issue(
            issues,
            "control.pair_unchanged",
            f"{control} up/down conditions differ for only {different}/{len(common)} WAVs",
        )
    return {
        "files_compared": len(common),
        "files_different": different,
        "max_abs_difference": maximum,
    }


def _safe_pair_directory(root: Path, raw_directory: Any) -> Optional[Path]:
    if not isinstance(raw_directory, str):
        return None
    relative = Path(raw_directory)
    if len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        return None
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_dir():
        return None
    try:
        if candidate.resolve().parent != root.resolve():
            return None
    except OSError:
        return None
    return candidate


def _load_json(path: Path, issues: list[dict[str, Any]], label: str) -> Optional[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        _issue(issues, f"{label}.missing", f"Missing regular JSON file: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _issue(issues, f"{label}.read", f"Cannot read {path}: {error}")
        return None
    if not isinstance(value, dict):
        _issue(issues, f"{label}.type", f"JSON root must be an object: {path}")
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _issue(issues: list[dict[str, Any]], code: str, message: str, *, condition: Any = None) -> None:
    item = {"code": code, "message": message}
    if condition is not None:
        item["condition"] = str(condition)
    issues.append(item)


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    report["success"] = not report["issues"]
    report["success_token"] = SUCCESS_TOKEN if report["success"] else None
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("manipulations", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = check_aria_manipulation(args.baseline, args.manipulations)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if report["success"]:
        print(SUCCESS_TOKEN)
        return 0
    if not args.output:
        print(payload, end="")
    else:
        for issue in report["issues"]:
            print(f"{issue['code']}: {issue['message']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
