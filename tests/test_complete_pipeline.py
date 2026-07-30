import csv
import hashlib
import json
import struct
from pathlib import Path

from tools import check_complete_pipeline as checker


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_file(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _write_report(path: Path) -> None:
    _write_file(path, b"<html>" + b"x" * 600 + b"</html>")


def _write_wav(path: Path, frames: int = 1, sample_rate: int = 16000) -> None:
    """Write a valid sparse PCM WAV; long fixtures consume almost no disk blocks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data_size = frames * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    with path.open("wb") as stream:
        stream.write(header)
        stream.seek(44 + data_size - 1)
        stream.write(b"\0")


def _write_sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.seek(size - 1)
        stream.write(b"\0")


def _fingerprint(rows) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["id"]):
        digest.update(row["id"].encode())
        digest.update(row["source_sha256"].encode())
        digest.update(row["split"].encode())
    return digest.hexdigest()


def _complete_fixture(root: Path) -> Path:
    corpus = root / "corpus"
    archive = corpus / "cmu_us_slt_arctic-0.95-release.tar.bz2"
    selected = corpus / "selected" / "arctic_a0001.wav"
    continuous = corpus / "continuous.wav"
    license_path = corpus / "extracted" / "cmu_us_slt_arctic" / "COPYING"
    order_path = corpus / "extracted" / "cmu_us_slt_arctic" / "etc" / "txt.done.data"
    _write_sparse(archive, checker.CMU_ARCTIC_SIZE)
    speech_frames = 1801 * 16000
    _write_wav(selected, speech_frames)
    _write_wav(continuous, speech_frames)
    _write_file(license_path, b"license")
    _write_file(order_path, b"( arctic_a0001 test )")
    fixed = checker.CMU_ARCTIC_SHA256
    _write_json(
        corpus / "corpus.json",
        {
            "schema_version": "1.0",
            "complete": True,
            "corpus": {
                "name": "CMU ARCTIC",
                "release": "0.95",
                "speaker": "slt",
            },
            "source": {
                "url": "http://festvox.org/cmu_arctic/cmu_arctic/packed/"
                "cmu_us_slt_arctic-0.95-release.tar.bz2",
                "archive": archive.relative_to(corpus).as_posix(),
                "size_bytes": checker.CMU_ARCTIC_SIZE,
                "sha256": fixed,
            },
            "license": {
                "file": license_path.relative_to(corpus).as_posix(),
                "size_bytes": license_path.stat().st_size,
                "sha256": fixed,
            },
            "provenance": {
                "order_file": order_path.relative_to(corpus).as_posix(),
                "order_file_size_bytes": order_path.stat().st_size,
                "order_file_sha256": fixed,
            },
            "selection": {
                "target_duration_s": 1800.0,
                "max_duration_s": 3600.0,
                "selected_duration_s": 1801.0,
                "reached_target": True,
                "utterance_count": 1,
                "utterances": [
                    {
                        "id": "arctic_a0001",
                        "selected_path": selected.relative_to(corpus).as_posix(),
                        "frames": speech_frames,
                        "sample_rate": 16000,
                        "channels": 1,
                        "sample_width_bytes": 2,
                        "size_bytes": selected.stat().st_size,
                        "sha256": fixed,
                    }
                ],
            },
            "outputs": {
                "selected_dir": "selected",
                "continuous_wav": "continuous.wav",
                "continuous_size_bytes": continuous.stat().st_size,
                "continuous_sha256": fixed,
            },
        },
    )

    segment_rows = []
    dataset_rows = []
    source_sha = "a" * 64
    for index in range(500):
        item_id = "item-{0:04d}".format(index)
        segment_audio = root / "segments" / "audio" / (item_id + ".wav")
        dataset_audio = root / "dataset" / "audio" / (item_id + ".wav")
        f0 = root / "dataset" / "f0" / (item_id + ".f0.txt")
        _write_wav(segment_audio)
        _write_wav(dataset_audio)
        _write_file(f0, b"100.0\n")
        segment_rows.append(
            {
                "id": item_id,
                "segment_path": "audio/" + item_id + ".wav",
                "duration_s": str(1 / 16000),
                "sample_rate": "16000",
                "samples": "1",
            }
        )
        split = "test" if index % 50 == 0 else "validation" if index % 50 == 1 else "train"
        dataset_rows.append(
            {
                "id": item_id,
                "split": split,
                "audio_path": "audio/" + item_id + ".wav",
                "f0_path": "f0/" + item_id + ".f0.txt",
                "source_sha256": source_sha,
                "sample_rate": "16000",
                "samples": "1",
            }
        )

    for path, rows in (
        (root / "segments" / "segments.csv", segment_rows),
        (root / "dataset" / "manifest.csv", dataset_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fingerprint = _fingerprint(dataset_rows)
    _write_json(
        root / "dataset" / "dataset.json",
        {
            "schema_version": "1.0",
            "sample_rate": 16000,
            "dataset_fingerprint": fingerprint,
        },
    )
    _write_report(root / "dataset" / "report.html")

    experiment = root / "experiment"
    config = _write_file(experiment / "config.yaml", b"trainer:\n  max_steps: 200\n")
    decoder = _write_file(experiment / "decoder.yaml", b"decoder: golf\n")
    _write_json(
        experiment / "experiment.json",
        {
            "schema_version": "1.0",
            "dataset_fingerprint": fingerprint,
            "model": "golf",
            "decoder_config": "decoder.yaml",
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "decoder_sha256": hashlib.sha256(decoder.read_bytes()).hexdigest(),
        },
    )
    metrics = experiment / "runs" / "metrics" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        "step,train_loss,val_loss,lr-Adam\n0,2.0,,0.0002\n100,1.0,0.9,0.0001\n",
        encoding="utf-8",
    )
    checkpoint = _write_file(experiment / "runs" / "checkpoints" / "last.ckpt", b"checkpoint")
    _write_report(root / "metrics.html")
    _write_json(
        root / "training_job.json",
        {
            "job_id": "12345",
            "node": "gpu001",
            "gpu": "NVIDIA L4",
            "state": "COMPLETED",
            "stage": "pipeline-complete",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    )

    test_ids = [row["id"] for row in dataset_rows if row["split"] == "test"]
    for item_id in test_ids:
        _write_wav(root / "reconstruction" / (item_id + ".wav"))
    _write_report(root / "comparison.html")

    outputs = []
    for shift, name in ((-4.0, "pitch_minus_4st"), (4.0, "pitch_plus_4st")):
        outputs.append({"semitones": shift, "f0_scale": 1.0, "directory": name})
        for item_id in test_ids:
            _write_wav(root / "manipulations" / name / (item_id + ".wav"))
    _write_json(
        root / "manipulations" / "manipulation.json",
        {
            "outputs": outputs,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    )
    _write_report(root / "manipulation.html")
    return root


def _patch_corpus_hashes(monkeypatch) -> None:
    real_sha256 = checker.sha256_file

    def fixture_sha256(path):
        if "corpus" in Path(path).parts:
            return checker.CMU_ARCTIC_SHA256
        return real_sha256(path)

    monkeypatch.setattr(checker, "sha256_file", fixture_sha256)


def test_complete_fixture_passes_and_prints_exact_token(tmp_path, monkeypatch, capsys):
    root = _complete_fixture(tmp_path / "demo")
    _patch_corpus_hashes(monkeypatch)

    assert checker.collect_issues(root) == []
    assert checker.main([str(root)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "PHONLAB_COMPLETE_PIPELINE_OK\n"
    assert captured.err == ""


def test_failures_are_itemized_and_return_nonzero(tmp_path, monkeypatch, capsys):
    root = _complete_fixture(tmp_path / "demo")
    _patch_corpus_hashes(monkeypatch)

    corpus_path = root / "corpus" / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    corpus["source"]["sha256"] = "0" * 64
    _write_json(corpus_path, corpus)
    (root / "dataset" / "f0" / "item-0000.f0.txt").unlink()
    job_path = root / "training_job.json"
    job = json.loads(job_path.read_text())
    job["state"] = "FAILED"
    job["stage"] = "training"
    _write_json(job_path, job)
    (root / "manipulations" / "pitch_plus_4st" / "item-0000.wav").unlink()

    assert checker.main([str(root)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fixed CMU ARCTIC SLT release digest" in captured.err
    assert "dataset row[0] F0" in captured.err
    assert "neither COMPLETED state nor pipeline-complete marker" in captured.err
    assert "manipulation +4 st item-0000" in captured.err
    assert len(captured.err.strip().splitlines()) >= 4


def test_count_prints_only_issue_count(tmp_path, monkeypatch, capsys):
    root = _complete_fixture(tmp_path / "demo")
    _patch_corpus_hashes(monkeypatch)
    (root / "metrics.html").unlink()
    (root / "comparison.html").unlink()
    expected = len(checker.collect_issues(root))

    assert checker.main([str(root), "--count"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "{0}\n".format(expected)
    assert captured.err == ""


def test_latest_metrics_version_is_the_one_accepted(tmp_path, monkeypatch):
    root = _complete_fixture(tmp_path / "demo")
    _patch_corpus_hashes(monkeypatch)
    latest = root / "experiment" / "runs" / "metrics" / "version_2" / "metrics.csv"
    latest.parent.mkdir(parents=True)
    latest.write_text("step,train_loss\n200,1.0\n", encoding="utf-8")

    issues = checker.collect_issues(root)
    assert "training metrics: val_loss series is missing" in issues
    assert "training metrics: learning-rate series is missing" in issues
