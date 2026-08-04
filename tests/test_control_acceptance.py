import copy
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import soundfile as sf

from tools import check_control_manipulation as checker


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), sample_rate, subtype="FLOAT")


def _fixture(root: Path):
    baseline = root / "reconstruction"
    manipulations = root / "manipulations"
    sample_rate = 16000
    factor = 10.0 ** (6.0 / 20.0)
    signals = {}
    harmonic_parts = {}
    noise_parts = {}
    for index, frames in enumerate((1600, 1920, 2240)):
        time = np.arange(frames, dtype=np.float64) / sample_rate
        harmonic = 0.18 * np.sin(2 * math.pi * (170 + index * 23) * time)
        noise = 0.025 * np.sin(2 * math.pi * (911 + index * 17) * time + 0.3)
        name = f"item-{index}.wav"
        signals[name] = (harmonic + noise).astype(np.float32)
        harmonic_parts[name] = harmonic.astype(np.float32)
        noise_parts[name] = noise.astype(np.float32)
        _write_wav(baseline / name, signals[name], sample_rate)

    variants = [
        ("quieter", {"output_gain_db": -6.0}),
        ("less_noise", {"noise_gain_db": -6.0}),
        ("more_noise", {"noise_gain_db": 6.0}),
        ("rd_higher", {"glottal_rd_scale": 1.25}),
    ]
    outputs = []
    capabilities = ["glottal_rd_scale", "noise_gain_db", "output_gain_db"]
    for name, controls in variants:
        condition = manipulations / name
        rows = []
        total_samples = 0
        for filename, baseline_samples in signals.items():
            if name == "quieter":
                samples = baseline_samples * (10.0 ** (-6.0 / 20.0))
            elif name == "less_noise":
                samples = harmonic_parts[filename] + noise_parts[filename] / factor
            elif name == "more_noise":
                samples = harmonic_parts[filename] + noise_parts[filename] * factor
            else:
                samples = baseline_samples + 0.0025
            _write_wav(condition / filename, samples, sample_rate)
            sample_count = int(samples.size)
            total_samples += sample_count
            rows.append(
                {
                    "path": filename,
                    "peak_before_gain": float(np.max(np.abs(baseline_samples))),
                    "peak_after_gain_unclipped": float(np.max(np.abs(samples))),
                    "clipped_samples": 0,
                    "samples": sample_count,
                }
            )
        render = {
            "schema_version": "1.0",
            "controls": controls,
            "runtime_capabilities": capabilities,
            "decoder_control_calls": 0 if name == "quieter" else len(signals),
            "files_written": len(signals),
            "clipped_samples": 0,
            "samples": total_samples,
            "clipped_fraction": 0.0,
            "files": rows,
        }
        _write_json(condition / "_render.json", render)
        outputs.append(
            {
                "name": name,
                "directory": name,
                "controls": controls,
                "f0_scale": 1.0,
                "label": ", ".join(f"{key}={value:g}" for key, value in controls.items()),
                "render_audit": {
                    "metadata": f"{name}/_render.json",
                    "runtime_capabilities": capabilities,
                    "decoder_control_calls": render["decoder_control_calls"],
                    "files_written": len(signals),
                    "clipped_fraction": 0.0,
                },
            }
        )
    metadata = {
        "schema_version": "2.0",
        "created_at": "2026-07-30T12:00:00+00:00",
        "operation": "ddsp-multi-parameter-control",
        "unvoiced_policy": "preserve-zero",
        "model": "golf",
        "experiment": str(root / "experiment"),
        "dataset_fingerprint": "b" * 64,
        "checkpoint": str(root / "last.ckpt"),
        "checkpoint_sha256": "a" * 64,
        "outputs": outputs,
    }
    _write_json(manipulations / "manipulation.json", metadata)
    return baseline, manipulations


def _issue_codes(report):
    return {item["code"] for item in report["issues"]}


class ControlAcceptanceTest(unittest.TestCase):
    def test_accepts_complete_multi_parameter_render_and_measures_acoustics(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline, manipulations = _fixture(Path(temporary))
            report = checker.check_control_manipulation(baseline, manipulations)

            self.assertTrue(report["success"], report["issues"])
            self.assertEqual(report["success_token"], checker.SUCCESS_TOKEN)
            self.assertEqual(report["baseline"]["files"], 3)
            self.assertEqual(len(report["conditions"]), 4)
            self.assertTrue(
                all(condition["changed_samples"] > 0 for condition in report["conditions"])
            )

            gain = next(
                item["output_gain"] for item in report["conditions"] if item["name"] == "quieter"
            )
            self.assertTrue(gain["within_tolerance"])
            self.assertAlmostEqual(
                gain["measured_median_rms_ratio"],
                10.0 ** (-6.0 / 20.0),
                places=6,
            )

            self.assertEqual(len(report["noise_symmetric_pairs"]), 1)
            pair = report["noise_symmetric_pairs"][0]
            self.assertTrue(pair["within_tolerance"])
            self.assertAlmostEqual(
                pair["measured_difference_energy_ratio"],
                10.0 ** (6.0 / 10.0),
                places=4,
            )

    def test_cli_writes_json_and_prints_unique_success_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, manipulations = _fixture(root)
            output = root / "acceptance.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = checker.main([str(baseline), str(manipulations), "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue().strip(), checker.SUCCESS_TOKEN)
            self.assertTrue(json.loads(output.read_text())["success"])

    def test_reports_each_independent_artifact_failure(self):
        mutations = {
            "schema": (
                lambda metadata, _root: metadata.__setitem__("schema_version", "1.0"),
                "metadata.schema_version",
            ),
            "checkpoint_hash": (
                lambda metadata, _root: metadata.__setitem__("checkpoint_sha256", "BAD"),
                "metadata.checkpoint_sha256",
            ),
            "unsafe_render_path": (
                lambda metadata, _root: metadata["outputs"][0]["render_audit"].__setitem__(
                    "metadata", "../_render.json"
                ),
                "render.metadata_path",
            ),
            "render_controls": (
                lambda _metadata, root: _mutate_render(
                    root, "quieter", "controls", {"output_gain_db": -5.0}
                ),
                "render.controls",
            ),
            "files_written": (
                lambda _metadata, root: _mutate_render(root, "quieter", "files_written", 2),
                "render.files_written",
            ),
            "clipping": (
                _add_clipping,
                "render.clipping_nonzero",
            ),
            "missing_peak_audit": (
                _remove_peak_audit,
                "render.file_peak",
            ),
            "sample_rate": (
                _change_sample_rate,
                "audio.sample_rate",
            ),
            "shape": (
                _change_shape,
                "audio.shape",
            ),
            "unchanged": (
                _make_condition_unchanged,
                "audio.unchanged",
            ),
            "partly_unchanged": (
                _make_one_file_unchanged,
                "audio.partly_unchanged",
            ),
            "decoder_control_not_applied": (
                _zero_decoder_calls,
                "render.decoder_control_not_applied",
            ),
        }
        for label, (mutation, expected_code) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                baseline, manipulations = _fixture(root)
                metadata_path = manipulations / "manipulation.json"
                metadata = json.loads(metadata_path.read_text())
                mutation(metadata, root)
                _write_json(metadata_path, metadata)

                report = checker.check_control_manipulation(baseline, manipulations)
                self.assertFalse(report["success"])
                self.assertIn(expected_code, _issue_codes(report), report["issues"])
                self.assertIsNone(report["success_token"])

    def test_failure_cli_emits_valid_json_without_output_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, manipulations = _fixture(root)
            metadata_path = manipulations / "manipulation.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["checkpoint_sha256"] = "not-a-hash"
            _write_json(metadata_path, metadata)

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = checker.main([str(baseline), str(manipulations)])
            report = json.loads(stdout.getvalue())
            self.assertEqual(status, 1)
            self.assertFalse(report["success"])
            self.assertEqual(stderr.getvalue(), "")


def _mutate_render(root: Path, condition: str, key: str, value) -> None:
    path = root / "manipulations" / condition / "_render.json"
    render = json.loads(path.read_text())
    render[key] = value
    _write_json(path, render)


def _add_clipping(_metadata, root: Path) -> None:
    path = root / "manipulations" / "quieter" / "_render.json"
    render = json.loads(path.read_text())
    render["files"][0]["clipped_samples"] = 1
    render["clipped_samples"] = 1
    render["clipped_fraction"] = 1 / render["samples"]
    _write_json(path, render)
    return None


def _remove_peak_audit(_metadata, root: Path) -> None:
    path = root / "manipulations" / "quieter" / "_render.json"
    render = json.loads(path.read_text())
    render["files"][0].pop("peak_before_gain")
    _write_json(path, render)


def _change_sample_rate(_metadata, root: Path) -> None:
    source = root / "manipulations" / "rd_higher" / "item-0.wav"
    samples, _ = sf.read(str(source), always_2d=False)
    _write_wav(source, samples, sample_rate=22050)
    return None


def _change_shape(_metadata, root: Path) -> None:
    source = root / "manipulations" / "rd_higher" / "item-0.wav"
    samples, sample_rate = sf.read(str(source), always_2d=False)
    _write_wav(source, samples[:-5], sample_rate=sample_rate)
    return None


def _make_condition_unchanged(_metadata, root: Path) -> None:
    baseline = root / "reconstruction"
    condition = root / "manipulations" / "rd_higher"
    render_path = condition / "_render.json"
    render = json.loads(render_path.read_text())
    for row in render["files"]:
        samples, sample_rate = sf.read(str(baseline / row["path"]), always_2d=False)
        _write_wav(condition / row["path"], samples, sample_rate)
    _write_json(render_path, copy.deepcopy(render))
    return None


def _make_one_file_unchanged(_metadata, root: Path) -> None:
    baseline = root / "reconstruction" / "item-0.wav"
    condition = root / "manipulations" / "rd_higher" / "item-0.wav"
    samples, sample_rate = sf.read(str(baseline), always_2d=False)
    _write_wav(condition, samples, sample_rate)
    return None


def _zero_decoder_calls(metadata, root: Path) -> None:
    _mutate_render(root, "rd_higher", "decoder_control_calls", 0)
    for output in metadata["outputs"]:
        if output["name"] == "rd_higher":
            output["render_audit"]["decoder_control_calls"] = 0
            break


if __name__ == "__main__":
    unittest.main()
