"""Lightning data adapter for portable PhonLab-DDSP manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    from lightning.pytorch import LightningDataModule
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:
    raise ImportError(
        "Training support is not installed. Run: pip install 'phonlab-ddsp[train]'"
    ) from error

from .manifest import DatasetManifest


class ManifestSegmentDataset(Dataset):
    def __init__(self, manifest_path, split, duration=1.5, overlap=1.0):
        self.manifest = DatasetManifest.load(Path(manifest_path))
        self.records = [record for record in self.manifest.records if record.split == split]
        if not self.records and split == "validation":
            raise ValueError("Manifest has no validation records; increase --validation-ratio")
        self.segment_frames = int(duration * self.manifest.metadata["sample_rate"])
        self.hop_frames = int((duration - overlap) * self.manifest.metadata["sample_rate"])
        if self.hop_frames <= 0:
            raise ValueError("overlap must be smaller than duration")
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
        hop = record.sample_rate * 0.005
        positions = np.arange(len(f0)) * hop
        target = np.arange(offset, offset + self.segment_frames)
        unvoiced = np.interp(target, positions, (f0 <= 0).astype(float), right=1) > 0
        interpolated = np.where(unvoiced, 0, np.interp(target, positions, f0))
        return segment.astype(np.float32), interpolated.astype(np.float32)


class ManifestInferenceDataset(Dataset):
    """Complete test recordings in the format expected by the legacy writer."""

    def __init__(self, manifest_path, split="test", f0_scale=1.0):
        self.manifest = DatasetManifest.load(Path(manifest_path))
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
        f0 = np.loadtxt(self.manifest.root / record.f0_path, dtype=np.float32, ndmin=1)
        positions = np.arange(len(f0)) * sample_rate * 0.005
        target = np.arange(len(audio))
        unvoiced = np.interp(target, positions, (f0 <= 0).astype(float), right=1) > 0
        interpolated = np.where(unvoiced, 0, np.interp(target, positions, f0) * self.f0_scale)
        return (
            audio.astype(np.float32),
            interpolated.astype(np.float32),
            f"{record.id}.wav",
        )


class ManifestDataModule(LightningDataModule):
    def __init__(
        self,
        manifest_path: str,
        batch_size: int = 32,
        duration: float = 1.5,
        overlap: float = 1.0,
        num_workers: int = 4,
        predict_f0_scale: float = 1.0,
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
