import csv
import dataclasses
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from aris.experiment import create_experiment, synthesize, train
from aris.manifest import DatasetManifest, prepare_dataset, summarize, validate_manifest
from aris.report import build_report, build_synthesis_report


def _tone(path: Path, frequency: float, seconds: float = 0.30, sample_rate: int = 8000):
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    audio = (0.25 * np.sin(2 * math.pi * frequency * time) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(audio.tobytes())


class WorkflowTest(unittest.TestCase):
    def test_prepare_validate_report_and_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            source = tmp_path / "raw"
            for index in range(30):
                _tone(source / f"speaker-{index % 3}" / f"item-{index}.wav", 100 + index * 3)

            dataset = tmp_path / "prepared"
            first = prepare_dataset(
                source,
                dataset,
                sample_rate=16000,
                f0_method="autocorr",
                min_duration=0.2,
            )
            self.assertEqual(len(first.records), 30)
            self.assertFalse(validate_manifest(first))
            self.assertEqual({record.sample_rate for record in first.records}, {16000})
            self.assertTrue(
                all((dataset / record.audio_path).is_file() for record in first.records)
            )
            self.assertEqual(first.fingerprint, DatasetManifest.load(dataset).fingerprint)
            self.assertEqual(
                {record.split for record in first.records}, {"train", "validation", "test"}
            )

            report = build_report(first, dataset / "report.html")
            self.assertIn("Record audit", report.read_text())

            experiment = create_experiment(dataset, tmp_path / "experiment", max_steps=2)
            metadata = json.loads((experiment / "experiment.json").read_text())
            self.assertEqual(metadata["dataset_fingerprint"], first.fingerprint)
            self.assertEqual(metadata["slurm"]["gres"], "gpu:1")
            command = train(experiment, dry_run=True)
            self.assertEqual(command[1:4], ["-m", "aris.engine", "fit"])
            self.assertTrue((experiment / "train.slurm").is_file())
            self.assertIn(".venv/bin/python", (experiment / "train.sh").read_text())
            slurm_script = (experiment / "train.slurm").read_text()
            self.assertIn("# module load <your CUDA module>", slurm_script)
            self.assertIn("srun --ntasks=1", slurm_script)
            cli_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "aris.cli",
                    "train",
                    str(experiment),
                    "--dry-run",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("-m aris.engine fit", cli_result.stdout)

            checkpoint = experiment / "fake.ckpt"
            checkpoint.touch()
            synthesis = tmp_path / "synthesis"
            synth_command = synthesize(experiment, checkpoint, synthesis, dry_run=True)
            self.assertIn("predict", synth_command)
            synthesis.mkdir()
            for record in first.records:
                if record.split == "test":
                    shutil.copy(dataset / record.audio_path, synthesis / f"{record.id}.wav")
            comparison = build_synthesis_report(first, synthesis, tmp_path / "comparison.html")
            self.assertIn("Reconstruction", comparison.read_text())

    def test_experiment_survives_relocation_and_different_cwd(self):
        # Regression test: experiment.json stores the dataset path relative to
        # the experiment directory, and readers must resolve it against that
        # directory (not the process cwd) so a moved experiment+dataset pair
        # keeps working from anywhere.
        with (
            tempfile.TemporaryDirectory() as original,
            tempfile.TemporaryDirectory() as relocated,
            tempfile.TemporaryDirectory() as elsewhere,
        ):
            original_path = Path(original)
            source = original_path / "raw"
            for index in range(30):
                _tone(source / f"speaker-{index % 3}" / f"item-{index}.wav", 100 + index * 3)

            dataset = original_path / "prepared"
            prepare_dataset(
                source, dataset, sample_rate=16000, f0_method="autocorr", min_duration=0.2
            )
            experiment = create_experiment(dataset, original_path / "experiment", max_steps=2)
            metadata = json.loads((experiment / "experiment.json").read_text())
            self.assertFalse(Path(metadata["dataset"]).is_absolute())
            self.assertEqual((experiment / metadata["dataset"]).resolve(), dataset.resolve())

            # Move the experiment+dataset pair together, preserving their
            # relative layout, and prove it resolves from an unrelated cwd.
            relocated_root = Path(relocated) / "moved"
            shutil.copytree(original_path, relocated_root)
            relocated_experiment = relocated_root / "experiment"

            train_result = subprocess.run(
                [sys.executable, "-m", "aris.cli", "train", str(relocated_experiment), "--dry-run"],
                check=True,
                text=True,
                capture_output=True,
                cwd=elsewhere,
            )
            self.assertIn("-m aris.engine fit", train_result.stdout)

            checkpoint = relocated_experiment / "fake.ckpt"
            checkpoint.touch()
            synth_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "aris.cli",
                    "synthesize",
                    str(relocated_experiment),
                    str(checkpoint),
                    str(Path(elsewhere) / "synth-out"),
                    "--dry-run",
                ],
                check=True,
                text=True,
                capture_output=True,
                cwd=elsewhere,
            )
            self.assertIn("predict", synth_result.stdout)

    def test_cli_doctor_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "aris.cli", "doctor", "--json"],
            check=True,
            text=True,
            capture_output=True,
        )
        checks = json.loads(result.stdout)
        self.assertTrue(any(item["name"] == "Python" and item["ok"] for item in checks))

    def test_prepare_reuses_f0_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            for index in range(3):
                path = source / f"item-{index}.wav"
                _tone(path, 120 + index * 10)
                # 60 frames * 5 ms hop = 0.30 s, matching _tone's default duration.
                np.savetxt(path.with_suffix(".pv"), np.full(60, 120 + index, dtype=np.float64))
            manifest = prepare_dataset(
                source,
                root / "prepared",
                f0_method="sidecar",
                min_duration=0.2,
            )
            self.assertEqual({record.f0_backend for record in manifest.records}, {"sidecar-pv"})

    def test_inference_dataset_rejects_sample_rate_drift(self):
        try:
            from aris.lightning import ManifestInferenceDataset
        except ImportError:
            self.skipTest("training extras are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            _tone(source / "item.wav", 150)
            manifest = prepare_dataset(
                source, root / "dataset", f0_method="autocorr", min_duration=0.2
            )
            dataset = ManifestInferenceDataset(manifest.root, split="train")
            audio_path = manifest.root / dataset.records[0].audio_path
            with wave.open(str(audio_path), "rb") as stream:
                params = stream.getparams()
                frames = stream.readframes(params.nframes)
            # Edit the audio file's rate on disk without touching the manifest,
            # simulating drift the manifest no longer reflects.
            with wave.open(str(audio_path), "wb") as stream:
                stream.setnchannels(params.nchannels)
                stream.setsampwidth(params.sampwidth)
                stream.setframerate(params.framerate * 2)
                stream.writeframes(frames)
            with self.assertRaisesRegex(RuntimeError, "Sample-rate drift"):
                dataset[0]

    def test_fingerprint_changes_with_sample_rate(self):
        from aris.manifest import DatasetRecord

        fields = dict(
            id="a",
            split="train",
            audio_path="audio/a.wav",
            f0_path="f0/a.f0.txt",
            source_path="a.wav",
            source_sha256="deadbeef",
            samples=100,
            duration_s=1.0,
            peak=0.5,
            rms_dbfs=-10.0,
            clipped_fraction=0.0,
            dc_offset=0.0,
            f0_backend="autocorr",
            median_f0_hz=120.0,
            voiced_fraction=0.9,
        )
        at_16k = DatasetManifest(
            root=Path("."), records=[DatasetRecord(sample_rate=16000, **fields)], metadata={}
        )
        at_8k = DatasetManifest(
            root=Path("."), records=[DatasetRecord(sample_rate=8000, **fields)], metadata={}
        )
        self.assertNotEqual(at_16k.fingerprint, at_8k.fingerprint)

    def test_fingerprint_detects_sample_rate_edit_in_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            for index in range(6):
                _tone(source / f"item-{index}.wav", 140 + index)
            dataset = root / "dataset"
            prepare_dataset(source, dataset, f0_method="autocorr", min_duration=0.2)
            experiment = create_experiment(dataset, root / "experiment", max_steps=2)

            # A dataset directory edited to a different rate after the experiment
            # was created must not silently pass fingerprint verification.
            csv_path = dataset / "manifest.csv"
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row["sample_rate"] = "8000"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(RuntimeError, "Dataset fingerprint changed"):
                train(experiment, dry_run=True)

    def test_prepare_rejects_mismatched_f0_sidecar_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            good = source / "good.wav"
            _tone(good, 130, seconds=0.5)
            np.savetxt(good.with_suffix(".pv"), np.full(100, 130.0))  # 100 * 5 ms = 0.5 s
            bad = source / "bad.wav"
            _tone(bad, 140, seconds=0.5)
            np.savetxt(bad.with_suffix(".pv"), np.full(5, 140.0))  # 5 * 5 ms = 0.025 s

            manifest = prepare_dataset(
                source,
                root / "dataset",
                f0_method="sidecar",
                min_duration=0.2,
                validation_ratio=0,
                test_ratio=0,
            )
            self.assertEqual([record.id.split("-")[0] for record in manifest.records], ["good"])
            skipped_invalid = manifest.metadata["skipped"]["invalid"]
            self.assertEqual(len(skipped_invalid), 1)
            self.assertIn("bad.wav", skipped_invalid[0]["file"])
            self.assertIn("frame", skipped_invalid[0]["error"].lower())

    def test_prepare_skips_bad_files_and_counts_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            _tone(source / "good.wav", 150, seconds=0.5)
            _tone(source / "short.wav", 150, seconds=0.05)
            corrupt = source / "corrupt.wav"
            corrupt.parent.mkdir(parents=True, exist_ok=True)
            corrupt.write_bytes(b"not a real wav file")

            manifest = prepare_dataset(
                source,
                root / "dataset",
                f0_method="autocorr",
                min_duration=0.2,
                validation_ratio=0,
                test_ratio=0,
            )
            self.assertEqual(len(manifest.records), 1)
            self.assertEqual(manifest.records[0].id.split("-")[0], "good")
            skipped = manifest.metadata["skipped"]
            self.assertEqual(skipped["too_short"], ["short.wav"])
            self.assertEqual(len(skipped["invalid"]), 1)
            self.assertIn("corrupt.wav", skipped["invalid"][0]["file"])

    def test_summarize_reports_skip_counts_when_given(self):
        self.assertNotIn("skipped", summarize([]))
        result = summarize(
            [], skipped={"invalid": [{"file": "a.wav", "error": "boom"}], "too_short": ["b.wav"]}
        )
        self.assertEqual(result["skipped"], {"invalid": 1, "too_short": 1})

    def test_train_and_synthesize_reject_corrupted_dataset_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            for index in range(6):
                _tone(source / f"item-{index}.wav", 140 + index)
            dataset = root / "dataset"
            prepare_dataset(source, dataset, f0_method="autocorr", min_duration=0.2)
            experiment = create_experiment(dataset, root / "experiment", max_steps=2)
            checkpoint = experiment / "fake.ckpt"
            checkpoint.touch()

            # Corrupt the recorded fingerprint directly, as if the dataset had
            # been silently swapped out from under this experiment.
            meta_path = experiment / "experiment.json"
            metadata = json.loads(meta_path.read_text())
            metadata["dataset_fingerprint"] = "corrupted-" + metadata["dataset_fingerprint"]
            meta_path.write_text(json.dumps(metadata))

            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                train(experiment, dry_run=True)
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                synthesize(experiment, checkpoint, root / "synth-out", dry_run=True)

    def test_validate_manifest_reports_each_error_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            for index in range(4):
                _tone(source / f"item-{index}.wav", 140 + index)
            dataset = root / "dataset"
            manifest = prepare_dataset(
                source,
                dataset,
                f0_method="autocorr",
                min_duration=0.2,
                validation_ratio=0,
                test_ratio=0,
            )
            self.assertFalse(validate_manifest(manifest))
            first, second, *rest = manifest.records

            def mutated(records):
                return DatasetManifest(
                    root=manifest.root, records=records, metadata=manifest.metadata
                )

            with self.subTest("duplicate id"):
                duplicate = dataclasses.replace(second, id=first.id)
                errors = validate_manifest(mutated([first, duplicate, *rest]))
                self.assertTrue(any("duplicate id" in error for error in errors))

            with self.subTest("invalid split"):
                bad_split = dataclasses.replace(first, split="bogus")
                errors = validate_manifest(mutated([bad_split, second, *rest]))
                self.assertTrue(any("invalid split" in error for error in errors))

            with self.subTest("audio_path escapes dataset root"):
                escaping = dataclasses.replace(first, audio_path="../outside.wav")
                errors = validate_manifest(mutated([escaping, second, *rest]))
                self.assertTrue(any("escapes dataset root" in error for error in errors))

            with self.subTest("missing referenced file"):
                missing = dataclasses.replace(first, audio_path="audio/does-not-exist.wav")
                errors = validate_manifest(mutated([missing, second, *rest]))
                self.assertTrue(any("missing audio_path" in error for error in errors))

            with self.subTest("non-positive duration"):
                zero_duration = dataclasses.replace(first, duration_s=0.0)
                errors = validate_manifest(mutated([zero_duration, second, *rest]))
                self.assertTrue(any("non-positive duration" in error for error in errors))

    def test_prepare_sidecar_missing_file_is_skipped_not_raised(self):
        # prepare_dataset's per-file try/except (see test_prepare_skips_bad_
        # files_and_counts_them) also covers sidecar lookup failures: a missing
        # .pv alongside one good file lands in skipped_invalid rather than
        # aborting the whole preparation.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            good = source / "good.wav"
            _tone(good, 130, seconds=0.5)
            np.savetxt(good.with_suffix(".pv"), np.full(100, 130.0))
            missing = source / "missing.wav"
            _tone(missing, 140, seconds=0.5)  # deliberately no .pv sidecar

            manifest = prepare_dataset(
                source,
                root / "dataset",
                f0_method="sidecar",
                min_duration=0.2,
                validation_ratio=0,
                test_ratio=0,
            )
            self.assertEqual([record.id.split("-")[0] for record in manifest.records], ["good"])
            skipped_invalid = manifest.metadata["skipped"]["invalid"]
            self.assertEqual(len(skipped_invalid), 1)
            self.assertIn("missing.wav", skipped_invalid[0]["file"])
            self.assertIn("sidecar", skipped_invalid[0]["error"].lower())

    def test_prepare_sidecar_invalid_values_are_skipped_not_raised(self):
        for label, bad_value in (("negative", -5.0), ("nan", float("nan"))):
            with self.subTest(label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "raw"
                good = source / "good.wav"
                _tone(good, 130, seconds=0.5)
                np.savetxt(good.with_suffix(".pv"), np.full(100, 130.0))
                bad = source / "bad.wav"
                _tone(bad, 140, seconds=0.5)
                values = np.full(100, 140.0)
                values[10] = bad_value
                np.savetxt(bad.with_suffix(".pv"), values)

                manifest = prepare_dataset(
                    source,
                    root / "dataset",
                    f0_method="sidecar",
                    min_duration=0.2,
                    validation_ratio=0,
                    test_ratio=0,
                )
                self.assertEqual([record.id.split("-")[0] for record in manifest.records], ["good"])
                skipped_invalid = manifest.metadata["skipped"]["invalid"]
                self.assertEqual(len(skipped_invalid), 1)
                self.assertIn("bad.wav", skipped_invalid[0]["file"])
                self.assertIn("Invalid F0 values", skipped_invalid[0]["error"])

    def test_prepare_sidecar_failure_raises_value_error_when_no_files_survive(self):
        # When every file fails (here: the only file has no sidecar at all),
        # prepare_dataset raises its own aggregate ValueError rather than
        # propagating the per-file FileNotFoundError/ValueError verbatim.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            _tone(source / "only.wav", 140, seconds=0.5)

            with self.assertRaisesRegex(ValueError, "No usable audio files"):
                prepare_dataset(
                    source,
                    root / "dataset",
                    f0_method="sidecar",
                    min_duration=0.2,
                    validation_ratio=0,
                    test_ratio=0,
                )


if __name__ == "__main__":
    unittest.main()
