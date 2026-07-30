import csv
import tempfile
import unittest
from pathlib import Path

from phonlab_ddsp.manifest import DatasetManifest, DatasetRecord
from phonlab_ddsp.parameters import (
    PARAMETER_SCHEMA,
    export_parameters,
    parameter_summary,
)


def _record(
    root: Path,
    item_id: str,
    *,
    split: str = "train",
    duration_s: float = 30.0,
    median_f0_hz: float = 120.0,
    voiced_fraction: float = 0.5,
    rms_dbfs: float = -24.0,
) -> DatasetRecord:
    audio_path = f"audio/{item_id}.wav"
    f0_path = f"f0/{item_id}.f0.txt"
    (root / audio_path).parent.mkdir(parents=True, exist_ok=True)
    (root / audio_path).write_bytes(b"audio")
    (root / f0_path).parent.mkdir(parents=True, exist_ok=True)
    (root / f0_path).write_text("0\n120\n", encoding="utf-8")
    return DatasetRecord(
        id=item_id,
        split=split,
        audio_path=audio_path,
        f0_path=f0_path,
        source_path=f"source/{item_id}.wav",
        source_sha256=item_id.ljust(64, "0")[:64],
        sample_rate=16000,
        samples=int(duration_s * 16000),
        duration_s=duration_s,
        peak=0.5,
        rms_dbfs=rms_dbfs,
        clipped_fraction=0.0,
        dc_offset=0.001,
        f0_backend="autocorr",
        median_f0_hz=median_f0_hz,
        voiced_fraction=voiced_fraction,
    )


class ParameterExportTest(unittest.TestCase):
    def test_export_is_schema_stable_sorted_and_portable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            records = [
                _record(dataset, "z-item", split="test", median_f0_hz=180.0),
                _record(dataset, "a-item", split="train", median_f0_hz=100.0),
            ]
            manifest = DatasetManifest(root=dataset, records=records, metadata={})

            first = export_parameters(manifest, root / "parameters.csv")
            second = export_parameters(manifest, root / "parameters-copy.csv")

            self.assertEqual(first, (root / "parameters.csv").resolve())
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with first.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames or ()), PARAMETER_SCHEMA)
            self.assertEqual([row["id"] for row in rows], ["a-item", "z-item"])
            self.assertEqual(rows[0]["audio_path"], "audio/a-item.wav")
            self.assertEqual(rows[0]["f0_path"], "f0/a-item.f0.txt")
            self.assertEqual(rows[0]["sample_rate"], "16000")

    def test_export_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            manifest = DatasetManifest(
                root=dataset,
                records=[_record(dataset, "item")],
                metadata={},
            )
            output = root / "parameters.csv"
            output.write_text("keep me\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_parameters(manifest, output)

            self.assertEqual(output.read_text(encoding="utf-8"), "keep me\n")

    def test_export_refuses_to_follow_an_existing_broken_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            manifest = DatasetManifest(
                root=dataset,
                records=[_record(dataset, "item")],
                metadata={},
            )
            output = root / "parameters.csv"
            output.symlink_to(root / "missing-target.csv")

            with self.assertRaises(FileExistsError):
                export_parameters(manifest, output)

            self.assertTrue(output.is_symlink())
            self.assertFalse((root / "missing-target.csv").exists())

    def test_export_accepts_dataset_and_manifest_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            manifest = DatasetManifest(
                root=dataset,
                records=[_record(dataset, "item")],
                metadata={},
            )
            manifest.save()

            by_directory = export_parameters(dataset, root / "directory.csv")
            by_manifest = export_parameters(dataset / "manifest.csv", root / "manifest.csv")

            self.assertEqual(by_directory.read_bytes(), by_manifest.read_bytes())

    def test_invalid_manifest_is_rejected_without_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            record = _record(dataset, "item")
            (dataset / record.f0_path).unlink()
            manifest = DatasetManifest(root=dataset, records=[record], metadata={})
            output = root / "parameters.csv"

            with self.assertRaisesRegex(ValueError, "missing f0_path"):
                export_parameters(manifest, output)

            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".parameters.csv.*.tmp")))


class ParameterSummaryTest(unittest.TestCase):
    def test_summary_reports_minutes_quantiles_and_basic_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            records = [
                _record(
                    dataset,
                    "a",
                    duration_s=60,
                    median_f0_hz=100,
                    voiced_fraction=0.25,
                    rms_dbfs=-30,
                ),
                _record(
                    dataset,
                    "b",
                    split="validation",
                    duration_s=120,
                    median_f0_hz=200,
                    voiced_fraction=0.75,
                    rms_dbfs=-20,
                ),
                _record(
                    dataset,
                    "c",
                    split="test",
                    duration_s=180,
                    median_f0_hz=300,
                    voiced_fraction=1.0,
                    rms_dbfs=-10,
                ),
            ]
            summary = parameter_summary(DatasetManifest(root=dataset, records=records, metadata={}))

            self.assertEqual(summary["files"], 3)
            self.assertEqual(summary["duration_minutes"], 6.0)
            self.assertEqual(summary["splits"], {"train": 1, "validation": 1, "test": 1})
            self.assertEqual(summary["sample_rates"], [16000])
            f0 = summary["median_f0_hz"]
            self.assertEqual(f0["count"], 3)
            self.assertEqual(f0["min"], 100)
            self.assertEqual(f0["q25"], 150)
            self.assertEqual(f0["median"], 200)
            self.assertEqual(f0["q75"], 250)
            self.assertEqual(f0["max"], 300)
            self.assertEqual(f0["mean"], 200)
            self.assertEqual(summary["voiced_fraction"]["mean"], 2 / 3)
            self.assertEqual(summary["rms_dbfs"]["median"], -20)

    def test_summary_handles_no_voiced_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            records = [
                _record(
                    dataset,
                    "a",
                    median_f0_hz=0,
                    voiced_fraction=0,
                ),
                _record(
                    dataset,
                    "b",
                    median_f0_hz=0,
                    voiced_fraction=0,
                ),
            ]

            summary = parameter_summary(DatasetManifest(root=dataset, records=records, metadata={}))

            self.assertEqual(
                summary["median_f0_hz"],
                {
                    "count": 0,
                    "min": None,
                    "q25": None,
                    "median": None,
                    "q75": None,
                    "max": None,
                    "mean": None,
                },
            )
            self.assertEqual(summary["voiced_fraction"]["mean"], 0)

    def test_inconsistent_voicing_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset"
            record = _record(
                dataset,
                "bad",
                median_f0_hz=120,
                voiced_fraction=0,
            )

            with self.assertRaisesRegex(ValueError, "unvoiced record"):
                parameter_summary(DatasetManifest(root=dataset, records=[record], metadata={}))


if __name__ == "__main__":
    unittest.main()
