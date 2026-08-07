import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aris import __version__ as ARIS_VERSION
from aris.controls import parse_variant
from aris.experiment import create_experiment, synthesize
from aris.manifest import prepare_dataset
from aris.manipulation import (
    build_manipulation_report,
    manipulate_controls,
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

    def test_pitch_directory_name_delegates_to_shared_formatter(self):
        # manipulation.py must not reimplement the magnitude-formatting logic
        # that controls/specs.py's _pitch_variant_name already owns.
        with patch(
            "aris.manipulation._pitch_variant_name", return_value="sentinel"
        ) as mock_formatter:
            self.assertEqual(pitch_directory_name(4.0), "sentinel")
        mock_formatter.assert_called_once_with(4.0)

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

            control_commands = manipulate_controls(
                experiment,
                checkpoint,
                root / "control-manipulations",
                [
                    parse_variant("less_noise:noise_gain_db=-6"),
                    parse_variant("source_shift:glottal_rd_scale=1.2,output_gain_db=-3"),
                ],
                dry_run=True,
            )
            self.assertEqual(len(control_commands), 2)
            self.assertIn(
                '{"noise_gain_db":-6.0}',
                control_commands[0],
            )
            self.assertIn(
                '{"glottal_rd_scale":1.2,"output_gain_db":-3.0}',
                control_commands[1],
            )
            with self.assertRaisesRegex(ValueError, "not supported"):
                manipulate_controls(
                    experiment,
                    checkpoint,
                    root / "invalid-controls",
                    [parse_variant("formant:f1_scale=1.1")],
                    dry_run=True,
                )

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

    def test_synthesize_rejects_conflicting_pitch_semitones_and_f0_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            (experiment / "experiment.json").write_text(
                json.dumps({"model": "golf", "dataset_fingerprint": "fixture"})
            )
            # pitch_semitones=12 implies a 2.0x f0_scale; 1.5 does not match it
            # and is not the neutral default, so the two controls disagree.
            with self.assertRaisesRegex(ValueError, "Conflicting pitch controls"):
                synthesize(
                    experiment,
                    root / "missing.ckpt",
                    root / "output",
                    f0_scale=1.5,
                    controls={"pitch_semitones": 12},
                )

    def test_synthesize_allows_matching_pitch_semitones_and_f0_scale(self):
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

            command = synthesize(
                experiment,
                checkpoint,
                root / "output",
                dry_run=True,
                f0_scale=semitones_to_scale(12),
                controls={"pitch_semitones": 12},
            )
            self.assertAlmostEqual(float(command[-1]), semitones_to_scale(12))

    def test_manipulation_rejects_checkpoint_drift_between_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            (experiment / "experiment.json").write_text(
                json.dumps(
                    {
                        "model": "golf",
                        "dataset_fingerprint": "fixture",
                    }
                )
            )
            checkpoint = root / "last.ckpt"
            checkpoint.write_bytes(b"initial")

            def fake_synthesize(_experiment, _checkpoint, output, **_kwargs):
                output.mkdir()
                (output / "_render.json").write_text(
                    json.dumps(
                        {
                            "runtime_capabilities": ["noise_gain_db"],
                            "decoder_control_calls": 1,
                            "files_written": 1,
                            "clipped_fraction": 0.0,
                        }
                    )
                )
                checkpoint.write_bytes(b"changed")
                return ["predict"]

            with patch(
                "aris.manipulation.synthesize",
                side_effect=fake_synthesize,
            ):
                with self.assertRaisesRegex(RuntimeError, "Checkpoint changed"):
                    manipulate_controls(
                        experiment,
                        checkpoint,
                        root / "output",
                        [parse_variant("noise:noise_gain_db=3")],
                    )

            self.assertFalse((root / "output").exists())

    def test_manipulate_controls_writes_aris_version_and_formant_tracking_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            (experiment / "experiment.json").write_text(
                json.dumps({"model": "golf", "dataset_fingerprint": "fixture"})
            )
            checkpoint = root / "last.ckpt"
            checkpoint.write_bytes(b"stable")

            def fake_synthesize(_experiment, _checkpoint, output, **_kwargs):
                output.mkdir()
                (output / "_render.json").write_text(
                    json.dumps(
                        {
                            "runtime_capabilities": ["noise_gain_db"],
                            "decoder_control_calls": 1,
                            "files_written": 1,
                            "clipped_fraction": 0.0,
                        }
                    )
                )
                return ["predict"]

            with patch("aris.manipulation.synthesize", side_effect=fake_synthesize):
                manipulate_controls(
                    experiment,
                    checkpoint,
                    root / "output",
                    [parse_variant("noise:noise_gain_db=3")],
                )

            metadata = json.loads((root / "output" / "manipulation.json").read_text())
            self.assertEqual(metadata["aris_version"], ARIS_VERSION)
            # The fake _render.json above has no formant_tracking key; the
            # render_audit must default to {} rather than raising KeyError.
            self.assertEqual(metadata["outputs"][0]["render_audit"]["formant_tracking"], {})

    def test_inference_scales_voiced_f0_and_preserves_zeros(self):
        try:
            from aris.lightning import ManifestInferenceDataset
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

    def test_inference_tail_matches_last_frame_voicing(self):
        try:
            from aris.lightning import ManifestInferenceDataset
        except ImportError:
            self.skipTest("training extras are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            path = source / "item.wav"
            _tone(path, 150)  # 0.30 s at 8 kHz, resampled to 0.30 s at 16 kHz
            np.savetxt(path.with_suffix(".pv"), np.full(60, 150.0))  # 60 * 5 ms = 0.30 s
            manifest = prepare_dataset(
                source,
                root / "dataset",
                f0_method="sidecar",
                min_duration=0.2,
                validation_ratio=0,
                test_ratio=0,
            )
            dataset = ManifestInferenceDataset(manifest.root, split="train")
            _, f0_track, _ = dataset[0]
            # The last known frame (index 59, at sample 4720) is voiced, so the
            # trailing fractional-hop samples beyond it must stay voiced too.
            self.assertTrue((f0_track[-10:] > 0).all())

    def test_inference_tail_stays_unvoiced_when_last_frame_is(self):
        try:
            from aris.lightning import ManifestInferenceDataset
        except ImportError:
            self.skipTest("training extras are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            path = source / "item.wav"
            _tone(path, 150)
            np.savetxt(path.with_suffix(".pv"), np.array([150.0] * 59 + [0.0]))
            manifest = prepare_dataset(
                source,
                root / "dataset",
                f0_method="sidecar",
                min_duration=0.2,
                validation_ratio=0,
                test_ratio=0,
            )
            dataset = ManifestInferenceDataset(manifest.root, split="train")
            _, f0_track, _ = dataset[0]
            self.assertTrue((f0_track[-10:] == 0).all())

    def test_segment_dataset_tail_matches_last_frame_voicing(self):
        try:
            from aris.lightning import ManifestSegmentDataset
        except ImportError:
            self.skipTest("training extras are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw"
            path = source / "item.wav"
            _tone(path, 150)
            np.savetxt(path.with_suffix(".pv"), np.full(60, 150.0))
            manifest = prepare_dataset(
                source,
                root / "dataset",
                f0_method="sidecar",
                min_duration=0.2,
                validation_ratio=0,
                test_ratio=0,
            )
            dataset = ManifestSegmentDataset(
                manifest.root, split="train", duration=1.0, overlap=0.0
            )
            _, f0_track = dataset[0]
            # Segment padding extends past the 0.30 s recording; the padded tail
            # is still governed by the last real frame's voicing, index 4720+.
            self.assertTrue((f0_track[5000:5020] > 0).all())


if __name__ == "__main__":
    unittest.main()
