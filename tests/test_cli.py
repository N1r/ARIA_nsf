import csv
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

import aris.cli as cli
import aris.experiment as experiment_module
from aris.experiment import create_experiment
from aris.manifest import prepare_dataset


def _tone(path: Path, frequency: float, seconds: float = 0.30, sample_rate: int = 8000):
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    audio = (0.25 * np.sin(2 * math.pi * frequency * time) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(audio.tobytes())


def _make_experiment(tmp_path: Path) -> Path:
    source = tmp_path / "raw"
    for index in range(30):
        _tone(source / f"speaker-{index % 3}" / f"item-{index}.wav", 100 + index * 3)
    dataset = tmp_path / "prepared"
    prepare_dataset(source, dataset, sample_rate=16000, f0_method="autocorr", min_duration=0.2)
    return create_experiment(dataset, tmp_path / "experiment", max_steps=2)


def _make_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "raw"
    for index in range(4):
        _tone(source / f"item-{index}.wav", 140 + index)
    dataset = tmp_path / "dataset"
    prepare_dataset(
        source, dataset, f0_method="autocorr", min_duration=0.2, validation_ratio=0, test_ratio=0
    )
    return dataset


def _corrupt_manifest_split(dataset: Path) -> None:
    csv_path = dataset / "manifest.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["split"] = "bogus"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_engine_failure_during_train_prints_clean_error(tmp_path, monkeypatch, capsys):
    experiment = _make_experiment(tmp_path)

    def _fail(command, cwd=None, check=None):
        raise subprocess.CalledProcessError(returncode=17, cmd=command)

    monkeypatch.setattr(experiment_module.subprocess, "run", _fail)

    exit_code = cli.main(["train", str(experiment)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.strip() == (
        "error: aris train failed: the underlying engine exited with status 17 "
        "(see output above for details)"
    )
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("metadata", [{}, {"model": 123}])
def test_controls_rejects_malformed_model_field(tmp_path, capsys, metadata):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "experiment.json").write_text(json.dumps(metadata))

    exit_code = cli.main(["controls", str(experiment)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_cli_validate_reports_ok_for_a_valid_dataset(tmp_path, capsys):
    dataset = _make_dataset(tmp_path)

    exit_code = cli.main(["validate", str(dataset)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("OK:")


def test_cli_prepare_reports_progress(tmp_path, capsys):
    source = tmp_path / "raw"
    for index in range(4):
        _tone(source / f"item-{index}.wav", 140 + index)

    exit_code = cli.main(
        [
            "prepare",
            str(source),
            str(tmp_path / "dataset"),
            "--f0-method",
            "autocorr",
            "--min-duration",
            "0.2",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "[prepare] 1/4 ( 25.0%) | audio and F0" in output
    assert "[prepare] 4/4 (100.0%) | audio and F0" in output


def test_cli_validate_json_reports_ok_for_a_valid_dataset(tmp_path, capsys):
    dataset = _make_dataset(tmp_path)

    exit_code = cli.main(["validate", str(dataset), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["errors"] == []


def test_cli_validate_reports_errors_for_a_corrupted_dataset(tmp_path, capsys):
    dataset = _make_dataset(tmp_path)
    _corrupt_manifest_split(dataset)

    exit_code = cli.main(["validate", str(dataset)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out.startswith("INVALID:")
    assert "invalid split" in captured.out


def test_cli_validate_json_reports_errors_for_a_corrupted_dataset(tmp_path, capsys):
    dataset = _make_dataset(tmp_path)
    _corrupt_manifest_split(dataset)

    exit_code = cli.main(["validate", str(dataset), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert any("invalid split" in error for error in payload["errors"])


def test_cli_controls_json_lists_declared_controls(tmp_path, capsys):
    experiment = _make_experiment(tmp_path)

    exit_code = cli.main(["controls", str(experiment), "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["model"] == "golf"
    names = {item["name"] for item in payload["controls"]}
    assert {"pitch_semitones", "output_gain_db", "noise_gain_db", "glottal_rd_scale"} <= names


def test_cli_synthesize_dry_run_combines_multiple_controls(tmp_path, capsys):
    experiment = _make_experiment(tmp_path)
    checkpoint = experiment / "fake.ckpt"
    checkpoint.touch()
    output = tmp_path / "synth-out"

    exit_code = cli.main(
        [
            "synthesize",
            str(experiment),
            str(checkpoint),
            str(output),
            "--control",
            "noise_gain_db=3",
            "--control",
            "output_gain_db=-6",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '{"noise_gain_db":3.0,"output_gain_db":-6.0}' in captured.out


def test_cli_manipulate_dry_run_with_baseline_prints_command_and_skips_report(
    tmp_path, monkeypatch, capsys
):
    # --dry-run takes precedence over --baseline in cli.py's dispatch (the
    # branches are mutually exclusive), so the report path must not run here.
    experiment = _make_experiment(tmp_path)
    checkpoint = experiment / "fake.ckpt"
    checkpoint.touch()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    output = tmp_path / "output"
    called = []
    monkeypatch.setattr(cli, "build_manipulation_report", lambda *a, **k: called.append(a))

    exit_code = cli.main(
        [
            "manipulate",
            str(experiment),
            str(checkpoint),
            str(output),
            "--variant",
            "boosted:noise_gain_db=3",
            "--baseline",
            str(baseline),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert called == []
    assert "aris.engine predict" in captured.out


def test_cli_manipulate_dispatches_to_report_path_when_baseline_given_without_dry_run(
    tmp_path, monkeypatch, capsys
):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.touch()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    output = tmp_path / "output"
    monkeypatch.setattr(cli, "manipulate_controls", lambda *a, **k: ["predict"])
    recorded = {}

    def fake_report(experiment_arg, baseline_arg, output_arg, destination):
        recorded["args"] = (experiment_arg, baseline_arg, output_arg, destination)
        return str(destination)

    monkeypatch.setattr(cli, "build_manipulation_report", fake_report)

    exit_code = cli.main(
        [
            "manipulate",
            str(experiment),
            str(checkpoint),
            str(output),
            "--variant",
            "boosted:noise_gain_db=3",
            "--baseline",
            str(baseline),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert recorded["args"] == (experiment, baseline, output, output / "comparison.html")
    assert str(output / "comparison.html") in captured.out
