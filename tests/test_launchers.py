import json
import tempfile
import unittest
from pathlib import Path

from phonlab_ddsp.cli import main
from phonlab_ddsp.controls import parse_variant
from phonlab_ddsp.launchers import create_postprocess_job


class PostprocessLauncherTest(unittest.TestCase):
    def test_creates_submit_ready_job_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            experiment = root / "experiment"
            experiment.mkdir()
            (experiment / "experiment.json").write_text(
                json.dumps(
                    {
                        "dataset": str(dataset),
                        "dataset_fingerprint": "abc123",
                        "model": "golf",
                    }
                )
            )
            checkpoint = experiment / "last.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "results"

            bundle = create_postprocess_job(
                experiment,
                checkpoint,
                output,
                [-4, 4],
                control_variants=[
                    parse_variant("less_noise:noise_gain_db=-6"),
                    parse_variant("source_shift:glottal_rd_scale=1.2"),
                ],
                partition="gpu-short",
                gres="gpu:l4:1",
                exclude="node857",
            )

            script = (bundle / "train.slurm").read_text()
            metadata = json.loads((bundle / "job.json").read_text())
            self.assertIn("#SBATCH --partition=gpu-short", script)
            self.assertIn("#SBATCH --gres=gpu:l4:1", script)
            self.assertIn("#SBATCH --exclude=node857", script)
            self.assertIn("phonlab manipulate", script)
            self.assertIn("--semitones -4.0 4.0", script)
            self.assertIn("--variant less_noise:noise_gain_db=-6", script)
            self.assertIn("--variant source_shift:glottal_rd_scale=1.2", script)
            self.assertEqual(script.count("verify_checkpoint"), 4)
            self.assertEqual(metadata["semitone_shifts"], [-4.0, 4.0])
            self.assertEqual(
                metadata["control_variants"],
                [
                    {"name": "less_noise", "controls": {"noise_gain_db": -6.0}},
                    {
                        "name": "source_shift",
                        "controls": {"glottal_rd_scale": 1.2},
                    },
                ],
            )
            self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
            self.assertIn(metadata["checkpoint_sha256"], script)
            self.assertFalse(output.exists())

    def test_rejects_missing_checkpoint_and_duplicate_shift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            (experiment / "experiment.json").write_text("{}")
            with self.assertRaises(FileNotFoundError):
                create_postprocess_job(
                    experiment,
                    root / "missing.ckpt",
                    root / "output",
                    [-4, 4],
                )

    def test_cli_can_create_control_only_job_without_default_pitch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            experiment = root / "experiment"
            experiment.mkdir()
            (experiment / "experiment.json").write_text(
                json.dumps(
                    {
                        "dataset": str(dataset),
                        "dataset_fingerprint": "fixture",
                        "model": "golf",
                    }
                )
            )
            checkpoint = root / "last.ckpt"
            checkpoint.write_bytes(b"checkpoint")

            self.assertEqual(
                main(
                    [
                        "init-postprocess",
                        str(experiment),
                        str(checkpoint),
                        str(root / "output"),
                        "--no-pitch",
                        "--variant",
                        "quiet:output_gain_db=-6",
                    ]
                ),
                0,
            )

            bundles = list((experiment / "jobs").iterdir())
            self.assertEqual(len(bundles), 1)
            metadata = json.loads((bundles[0] / "job.json").read_text())
            self.assertEqual(metadata["semitone_shifts"], [])
            self.assertEqual(
                metadata["control_variants"],
                [{"name": "quiet", "controls": {"output_gain_db": -6.0}}],
            )


if __name__ == "__main__":
    unittest.main()
