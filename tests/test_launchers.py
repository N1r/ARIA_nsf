import json
import tempfile
import unittest
from pathlib import Path

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
            self.assertEqual(metadata["semitone_shifts"], [-4.0, 4.0])
            self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
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


if __name__ == "__main__":
    unittest.main()
