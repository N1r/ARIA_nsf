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
from aris.manifest import DatasetManifest, prepare_dataset, validate_manifest
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
            self.assertEqual(metadata["slurm"]["gres"], "gpu:l4:1")
            command = train(experiment, dry_run=True)
            self.assertEqual(command[1:4], ["-m", "aris.engine", "fit"])
            self.assertTrue((experiment / "train.slurm").is_file())
            self.assertIn(".venv/bin/python", (experiment / "train.sh").read_text())
            slurm_script = (experiment / "train.slurm").read_text()
            self.assertIn("module load ALICE/default CUDA/12.4.0", slurm_script)
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
                np.savetxt(path.with_suffix(".pv"), np.array([0, 120 + index, 121 + index]))
            manifest = prepare_dataset(
                source,
                root / "prepared",
                f0_method="sidecar",
                min_duration=0.2,
            )
            self.assertEqual({record.f0_backend for record in manifest.records}, {"sidecar-pv"})


if __name__ == "__main__":
    unittest.main()
