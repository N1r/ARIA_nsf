import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from aris.segment import split_audio


def _write(path: Path, audio: np.ndarray, sample_rate: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def _tone(seconds: float, frequency: float = 180, sample_rate: int = 8000):
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    return (0.3 * np.sin(2 * math.pi * frequency * time)).astype(np.float32)


class AudioSplitTest(unittest.TestCase):
    def test_fixed_split_writes_auditable_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "continuous.wav"
            _write(source, _tone(4.6))
            result = split_audio(
                source,
                root / "segments",
                mode="fixed",
                segment_seconds=2,
                min_duration_seconds=0.25,
            )

            self.assertEqual(len(result.records), 3)
            self.assertAlmostEqual(result.records[-1].duration_s, 0.6, places=2)
            self.assertTrue(
                all((result.root / item.segment_path).is_file() for item in result.records)
            )
            rows = list(csv.DictReader((result.root / "segments.csv").open()))
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(rows[0]["source_sha256"]), 64)
            metadata = json.loads((result.root / "split.json").read_text())
            self.assertEqual(metadata["mode"], "fixed")

    def test_silence_split_finds_two_utterances(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "two-utterances.wav"
            audio = np.concatenate([_tone(0.4), np.zeros(4000), _tone(0.4, 220)])
            _write(source, audio)
            result = split_audio(
                source,
                root / "segments",
                mode="silence",
                silence_threshold_db=-35,
                min_silence_seconds=0.3,
                padding_seconds=0.03,
                min_duration_seconds=0.2,
            )

            self.assertEqual(len(result.records), 2)
            self.assertLess(result.records[0].end_s, result.records[1].start_s)
            self.assertTrue(all(0.4 <= item.duration_s <= 0.5 for item in result.records))

    def test_split_keeps_pv_sidecars_aligned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            _write(source, _tone(1.0))
            np.savetxt(source.with_suffix(".pv"), np.arange(200, dtype=np.float32))
            result = split_audio(
                source,
                root / "segments",
                mode="fixed",
                segment_seconds=0.5,
                split_f0_sidecars=True,
            )

            self.assertEqual(len(result.records), 2)
            for record in result.records:
                sidecar = result.root / record.f0_path
                self.assertTrue(sidecar.is_file())
                self.assertEqual(len(np.loadtxt(sidecar, ndmin=1)), 100)
                self.assertEqual(len(record.source_f0_sha256), 64)

    def test_split_cli_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write(root / "source.wav", _tone(1.3))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "aris.cli",
                    "split",
                    str(root / "source.wav"),
                    str(root / "output"),
                    "--mode",
                    "fixed",
                    "--segment-seconds",
                    "0.5",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(completed.stdout)["segments"], 3)


if __name__ == "__main__":
    unittest.main()
