"""Lightning data adapter for portable ARIS manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    from lightning.pytorch import LightningDataModule
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:
    raise ImportError("Training support is not installed. Run: uv sync --extra train") from error

from .manifest import DatasetManifest


class ManifestSegmentDataset(Dataset):
    """Fixed-length segments with sample-aligned F0 and optional formant targets."""

    def __init__(self, manifest_path, split, duration=1.5, overlap=1.0, load_formants=False):
        self.manifest = DatasetManifest.load(Path(manifest_path))
        self.records = [record for record in self.manifest.records if record.split == split]
        if not self.records and split == "validation":
            raise ValueError("Manifest has no validation records; increase --validation-ratio")
        self.segment_frames = int(duration * self.manifest.metadata["sample_rate"])
        self.hop_frames = int((duration - overlap) * self.manifest.metadata["sample_rate"])
        self.formant_hop_frames = int(0.010 * self.manifest.metadata["sample_rate"])
        self.load_formants = load_formants
        if self.hop_frames <= 0:
            raise ValueError("overlap must be smaller than duration")
        if load_formants:
            missing = [record.id for record in self.records if not record.formant_path]
            if missing:
                raise ValueError(
                    "aria-golf requires prepared F1/F2 targets; rerun `aris prepare` "
                    f"with --extract-formants (missing for {len(missing)} {split} records)"
                )
        self.items = []
        for record_index, record in enumerate(self.records):
            segment_count = max(
                1, 1 + max(0, record.samples - self.segment_frames) // self.hop_frames
            )
            self.items.extend(
                (record_index, index * self.hop_frames) for index in range(segment_count)
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record_index, offset = self.items[index]
        record = self.records[record_index]
        audio, sample_rate = sf.read(
            str(self.manifest.root / record.audio_path), dtype="float32", always_2d=True
        )
        audio = audio.mean(axis=1)
        if sample_rate != record.sample_rate:
            raise RuntimeError(f"Sample-rate drift in {record.audio_path}")
        segment = audio[offset : offset + self.segment_frames]
        if len(segment) < self.segment_frames:
            segment = np.pad(segment, (0, self.segment_frames - len(segment)))
        f0 = np.loadtxt(self.manifest.root / record.f0_path, dtype=np.float32, ndmin=1)
        # Upsample the 5 ms F0 track to sample rate, but keep frames adjacent
        # to unvoiced regions at zero instead of interpolating across them.
        hop = record.sample_rate * 0.005
        positions = np.arange(len(f0)) * hop
        target = np.arange(offset, offset + self.segment_frames)
        unvoiced = (
            np.interp(target, positions, (f0 <= 0).astype(float), right=float(f0[-1] <= 0)) > 0
        )
        interpolated = np.where(unvoiced, 0, np.interp(target, positions, f0))
        segment = segment.astype(np.float32)
        interpolated = interpolated.astype(np.float32)
        if not self.load_formants:
            return segment, interpolated

        with np.load(self.manifest.root / record.formant_path) as features:
            feature_times = features["time_s"].astype(np.float64)
            f1_track = features["f1"].astype(np.float32)
            f2_track = features["f2"].astype(np.float32)
        frame_count = self.segment_frames // self.formant_hop_frames
        target_times = (
            offset / record.sample_rate + np.arange(frame_count, dtype=np.float64) * 0.010
        )
        f1 = _nearest_feature_track(feature_times, f1_track, target_times, 0.010)
        f2 = _nearest_feature_track(feature_times, f2_track, target_times, 0.010)
        frame_f0 = interpolated[:: self.formant_hop_frames][:frame_count]
        voiced_gate = ((frame_f0 > 0) & (f1 > 0) & (f2 > 0)).astype(np.float32)
        return segment, interpolated, f1, f2, voiced_gate


def _nearest_feature_track(times, values, target_times, hop_seconds):
    """Sample a frame track without interpolating across undefined (zero) values."""
    if not len(times):
        return np.zeros(len(target_times), dtype=np.float32)
    right = np.searchsorted(times, target_times, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    choose_left = np.abs(target_times - times[left]) <= np.abs(times[right] - target_times)
    indices = np.where(choose_left, left, right)
    valid = np.abs(times[indices] - target_times) <= hop_seconds * 0.51
    return np.where(valid, values[indices], 0).astype(np.float32)


class ManifestInferenceDataset(Dataset):
    """Complete test recordings in the format expected by the legacy writer."""

    def __init__(self, manifest_path, split="test", f0_scale=1.0):
        self.manifest = DatasetManifest.load(Path(manifest_path))
        if split in (None, "all"):
            self.records = list(self.manifest.records)
        else:
            self.records = [record for record in self.manifest.records if record.split == split]
        self.f0_scale = float(f0_scale)
        if not np.isfinite(self.f0_scale) or self.f0_scale <= 0:
            raise ValueError("f0_scale must be finite and positive")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        audio, sample_rate = sf.read(
            str(self.manifest.root / record.audio_path), dtype="float32", always_2d=True
        )
        audio = audio.mean(axis=1)
        if sample_rate != record.sample_rate:
            raise RuntimeError(f"Sample-rate drift in {record.audio_path}")
        f0 = np.loadtxt(self.manifest.root / record.f0_path, dtype=np.float32, ndmin=1)
        positions = np.arange(len(f0)) * sample_rate * 0.005
        target = np.arange(len(audio))
        unvoiced = (
            np.interp(target, positions, (f0 <= 0).astype(float), right=float(f0[-1] <= 0)) > 0
        )
        interpolated = np.where(unvoiced, 0, np.interp(target, positions, f0) * self.f0_scale)
        return (
            audio.astype(np.float32),
            interpolated.astype(np.float32),
            f"{record.id}.wav",
        )


class ManifestDataModule(LightningDataModule):
    """LightningDataModule serving train/val/test/predict splits from one manifest."""

    def __init__(
        self,
        manifest_path: str,
        batch_size: int = 32,
        duration: float = 1.5,
        overlap: float = 1.0,
        num_workers: int = 4,
        predict_f0_scale: float = 1.0,
        predict_split: str = "test",
        load_formants: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        def factory(split):
            return ManifestSegmentDataset(
                self.hparams.manifest_path,
                split,
                self.hparams.duration,
                self.hparams.overlap,
                self.hparams.load_formants,
            )

        if stage in {None, "fit"}:
            self.train_dataset = factory("train")
            self.valid_dataset = factory("validation")
        if stage in {None, "validate"}:
            self.valid_dataset = factory("validation")
        if stage == "test":
            self.test_dataset = factory("test")
        if stage == "predict":
            self.predict_dataset = ManifestInferenceDataset(
                self.hparams.manifest_path,
                split=self.hparams.predict_split,
                f0_scale=self.hparams.predict_f0_scale,
            )

    def _loader(self, dataset, shuffle=False, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self.valid_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1, num_workers=1, shuffle=False)
