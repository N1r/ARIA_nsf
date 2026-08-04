import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from tools import check_aria_manipulation as checker


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.astype(np.float32), 16000, subtype="FLOAT")


def _fixture(root: Path):
    baseline = root / "reconstruction"
    manipulations = root / "manipulations"
    experiment = root / "experiment"
    checkpoint = experiment / "runs" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"real checkpoint fixture")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fingerprint = "b" * 64
    _json(
        experiment / "experiment.json",
        {"model": "aria-golf", "dataset_fingerprint": fingerprint},
    )
    signals = {}
    for index, frames in enumerate((1600, 1920)):
        time = np.arange(frames, dtype=np.float64) / 16000
        signal = 0.12 * np.sin(2 * math.pi * (180 + index * 20) * time)
        signal += 0.04 * np.sin(2 * math.pi * (700 + index * 30) * time)
        signals[f"item-{index}.wav"] = signal.astype(np.float32)
        _wav(baseline / f"item-{index}.wav", signal)

    variants = [
        ("f1_down", "f1_scale", 0.9, -0.006),
        ("f1_up", "f1_scale", 1.1, 0.006),
        ("f2_down", "f2_scale", 0.9, -0.009),
        ("f2_up", "f2_scale", 1.1, 0.009),
        ("tilt_down", "tilt_alpha_delta", -0.1, -0.012),
        ("tilt_up", "tilt_alpha_delta", 0.1, 0.012),
    ]
    outputs = []
    capabilities = [
        "f1_scale",
        "f2_scale",
        "glottal_rd_scale",
        "noise_gain_db",
        "output_gain_db",
        "tilt_alpha_delta",
    ]
    for name, control, value, offset in variants:
        rows = []
        total = 0
        for filename, signal in signals.items():
            rendered = signal + offset
            _wav(manipulations / name / filename, rendered)
            total += rendered.size
            rows.append(
                {
                    "path": filename,
                    "peak_before_gain": float(np.max(np.abs(signal))),
                    "peak_after_gain_unclipped": float(np.max(np.abs(rendered))),
                    "clipped_samples": 0,
                    "samples": int(rendered.size),
                }
            )
        controls = {control: value}
        render = {
            "schema_version": "1.0",
            "controls": controls,
            "runtime_capabilities": capabilities,
            "decoder_control_calls": len(signals),
            "files_written": len(signals),
            "clipped_samples": 0,
            "samples": total,
            "clipped_fraction": 0.0,
            "files": rows,
        }
        _json(manipulations / name / "_render.json", render)
        outputs.append(
            {
                "name": name,
                "directory": name,
                "controls": controls,
                "f0_scale": 1.0,
                "label": f"{control}={value}",
                "render_audit": {
                    "metadata": f"{name}/_render.json",
                    "runtime_capabilities": capabilities,
                    "decoder_control_calls": len(signals),
                    "files_written": len(signals),
                    "clipped_fraction": 0.0,
                },
            }
        )
    _json(
        manipulations / "manipulation.json",
        {
            "schema_version": "2.0",
            "created_at": "2026-08-03T12:00:00+00:00",
            "operation": "ddsp-multi-parameter-control",
            "unvoiced_policy": "preserve-zero",
            "model": "aria-golf",
            "experiment": str(experiment),
            "dataset_fingerprint": fingerprint,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "outputs": outputs,
        },
    )
    return baseline, manipulations, checkpoint


def _codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_accepts_real_checkpoint_and_isolated_aria_pairs():
    with tempfile.TemporaryDirectory() as temporary:
        baseline, manipulations, _ = _fixture(Path(temporary))
        report = checker.check_aria_manipulation(baseline, manipulations)
        assert report["success"], report["issues"]
        assert report["success_token"] == checker.SUCCESS_TOKEN
        assert report["checkpoint"]["matches"]
        assert report["experiment"]["matches"]
        assert {pair["control"] for pair in report["control_pairs"]} == {
            "f1_scale",
            "f2_scale",
            "tilt_alpha_delta",
        }
        assert all(
            pair["files_compared"] == pair["files_different"] == 2
            for pair in report["control_pairs"]
        )


def test_rejects_missing_pair_changed_checkpoint_and_wrong_model():
    with tempfile.TemporaryDirectory() as temporary:
        baseline, manipulations, checkpoint = _fixture(Path(temporary))
        metadata_path = manipulations / "manipulation.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["model"] = "golf"
        metadata["outputs"] = [item for item in metadata["outputs"] if item["name"] != "f1_down"]
        _json(metadata_path, metadata)
        checkpoint.write_bytes(b"changed")

        report = checker.check_aria_manipulation(baseline, manipulations)
        assert not report["success"]
        assert "metadata.model" in _codes(report)
        assert "checkpoint.sha256" in _codes(report)
        assert "control.pair_missing" in _codes(report)


def test_cli_writes_machine_report():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        baseline, manipulations, _ = _fixture(root)
        output = root / "aria-acceptance.json"
        assert checker.main([str(baseline), str(manipulations), "--output", str(output)]) == 0
        assert json.loads(output.read_text())["success_token"] == checker.SUCCESS_TOKEN


def test_rejects_unsafe_pair_directory_without_reading_outside_artifact():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        baseline, manipulations, _ = _fixture(root)
        metadata_path = manipulations / "manipulation.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["outputs"][0]["directory"] = "../outside"
        _json(metadata_path, metadata)

        report = checker.check_aria_manipulation(baseline, manipulations)
        assert not report["success"]
        assert "control.pair_directory" in _codes(report)
