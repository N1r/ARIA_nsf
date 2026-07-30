import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from phonlab_ddsp.experiment import create_experiment, synthesize
from phonlab_ddsp.manifest import prepare_dataset
from phonlab_ddsp.manipulation import (
    build_manipulation_report,
    manipulate_pitch,
    pitch_directory_name,
    semitones_to_scale,
)

from .test_workflow import _tone


class ManipulationTest(unittest.TestCase):
    def test_semitones_and_names(self):
        self.assertAlmostEqual(semitones_to_scale(12), 2.0)
        self.assertAlmostEqual(semitones_to_scale(-12), 0.5)
        self.assertEqual(pitch_directory_name(4), "pitch_plus_4st")
        self.assertEqual(pitch_directory_name(-3.5), "pitch_minus_3p5st")
        with self.assertRaises(ValueError):
            semitones_to_scale(math.inf)

    def test_dry_run_and_listening_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            for index in range(10):
                _tone(source / f"item-{index}.wav", 120 + index)
            manifest = prepare_dataset(
                source,
                root / "dataset",
                f0_method="autocorr",
                validation_ratio=0.2,
                test_ratio=0.2,
                min_duration=0.2,
            )
            experiment = create_experiment(manifest.root, root / "experiment")
            checkpoint = root / "checkpoint.ckpt"
            checkpoint.write_bytes(b"test checkpoint")
            commands = manipulate_pitch(
                experiment,
                checkpoint,
                root / "manipulations",
                [-4, 4],
                dry_run=True,
            )
            self.assertEqual(len(commands), 2)
            self.assertIn("--data.init_args.predict_f0_scale", commands[0])
            self.assertAlmostEqual(float(commands[0][-1]), semitones_to_scale(-4))

            baseline = root / "baseline"
            manipulations = root / "manipulations"
            baseline.mkdir()
            manipulations.mkdir()
            variants = []
            for shift in (-4, 4):
                name = pitch_directory_name(shift)
                variants.append(
                    {
                        "semitones": shift,
                        "f0_scale": semitones_to_scale(shift),
                        "directory": name,
                    }
                )
                (manipulations / name).mkdir()
            (manipulations / "manipulation.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": "0" * 64,
                        "operation": "voiced-f0-scale",
                        "outputs": variants,
                    }
                )
            )
            for record in manifest.records:
                if record.split != "test":
                    continue
                shutil.copy(manifest.root / record.audio_path, baseline / f"{record.id}.wav")
                for variant in variants:
                    shutil.copy(
                        manifest.root / record.audio_path,
                        manipulations / variant["directory"] / f"{record.id}.wav",
                    )
            report = build_manipulation_report(
                experiment,
                baseline,
                manipulations,
                root / "manipulation-report.html",
            )
            text = report.read_text()
            self.assertIn("voiced F0 scaling", text)
            self.assertIn("+4 st", text)

    def test_synthesize_rejects_invalid_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "f0_scale"):
                synthesize(root, root / "missing.ckpt", root / "output", f0_scale=0)

    def test_inference_scales_voiced_f0_and_preserves_zeros(self):
        try:
            from phonlab_ddsp.lightning import ManifestInferenceDataset
        except ImportError:
            self.skipTest("training extras are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            for index in range(4):
                _tone(source / f"item-{index}.wav", 150 + index)
            manifest = prepare_dataset(
                source,
                root / "dataset",
                f0_method="autocorr",
                validation_ratio=0.25,
                test_ratio=0.25,
                min_duration=0.2,
            )
            baseline = ManifestInferenceDataset(manifest.root, f0_scale=1.0)
            shifted = ManifestInferenceDataset(manifest.root, f0_scale=2.0)
            _, base_f0, _ = baseline[0]
            _, shifted_f0, _ = shifted[0]
            self.assertTrue((shifted_f0[base_f0 == 0] == 0).all())
            self.assertTrue((shifted_f0[base_f0 > 0] == 2 * base_f0[base_f0 > 0]).all())


if __name__ == "__main__":
    unittest.main()
