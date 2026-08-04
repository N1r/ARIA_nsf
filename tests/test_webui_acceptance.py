import json
import math
import struct
import wave
from pathlib import Path

from tools import check_webui as checker


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_wav(path: Path, *, gain: float, frames: int = 800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        int(max(-1.0, min(1.0, gain * math.sin(2.0 * math.pi * index / 40.0))) * 32767)
        for index in range(frames)
    ]
    payload = struct.pack("<" + "h" * len(samples), *samples)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(payload)


def _render(path: Path, controls, *, frames: int = 800) -> None:
    _write_json(
        path,
        {
            "schema_version": "1.0",
            "controls": controls,
            "runtime_capabilities": ["output_gain_db"],
            "decoder_control_calls": 0 if not controls else 1,
            "files_written": 1,
            "clipped_samples": 0,
            "samples": frames,
            "clipped_fraction": 0.0,
            "files": [
                {
                    "path": "fixture.wav",
                    "clipped_samples": 0,
                    "samples": frames,
                    "peak_before_gain": 0.2,
                    "peak_after_gain_unclipped": 0.2,
                }
            ],
        },
    )


def _result_fixture(workspace: Path) -> Path:
    result = workspace / "artifacts" / "tiny-control-result"
    baseline = result / "reconstruction"
    variant = result / "manipulations" / "quieter"
    _write_wav(baseline / "fixture.wav", gain=0.2)
    _write_wav(variant / "fixture.wav", gain=0.1)
    _render(baseline / "_render.json", {})
    _render(variant / "_render.json", {"output_gain_db": -6.0})
    _write_json(
        result / "manipulations" / "manipulation.json",
        {
            "schema_version": "2.0",
            "created_at": "2026-08-03T12:00:00+00:00",
            "operation": "ddsp-multi-parameter-control",
            "unvoiced_policy": "preserve-zero",
            "model": "golf",
            "dataset_fingerprint": "b" * 64,
            "checkpoint_sha256": "a" * 64,
            "outputs": [
                {
                    "name": "quieter",
                    "directory": "quieter",
                    "label": "output_gain_db=-6",
                    "controls": {"output_gain_db": -6.0},
                    "render_audit": {
                        "metadata": "quieter/_render.json",
                    },
                }
            ],
        },
    )
    (result / "manipulation.html").write_text(
        "<!doctype html><title>Tiny manipulation</title>\n",
        encoding="utf-8",
    )
    return result


def test_checks_catalog_range_download_zip_and_safe_temp_export(tmp_path):
    result = _result_fixture(tmp_path)

    report = checker.check_webui(tmp_path, result, check_export=True)

    assert report["success"], report["issues"]
    assert report["success_token"] == checker.SUCCESS_TOKEN
    checks = {item["name"]: item for item in report["checks"]}
    assert set(checks) >= {
        "server",
        "homepage",
        "catalog",
        "wav_range",
        "wav_download",
        "results_zip",
        "results_export",
    }
    assert all(item["ok"] for item in checks.values())
    assert checks["wav_range"]["status"] == 206
    assert checks["results_zip"]["integrity"] == "ok"
    assert not Path(checks["results_zip"]["archive"]).exists()
    assert checks["results_export"]["cleaned"]
    assert not Path(checks["results_export"]["destination"]).exists()


def test_cli_writes_json_report_and_prints_marker(tmp_path, capsys):
    result = _result_fixture(tmp_path)
    output = tmp_path / ".cache" / "webui-acceptance.json"

    status = checker.main(
        [
            "--workspace",
            str(tmp_path),
            "--result",
            str(result.relative_to(tmp_path)),
            "--check-export",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert capsys.readouterr().out.strip() == checker.SUCCESS_TOKEN
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["success"]
    assert report["success_token"] == checker.SUCCESS_TOKEN


def test_rejects_result_outside_selected_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    report = checker.check_webui(workspace, outside)

    assert not report["success"]
    assert report["success_token"] is None
    assert [issue["code"] for issue in report["issues"]] == ["inputs.invalid"]
    assert not (workspace / ".cache").exists()
