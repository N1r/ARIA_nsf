from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
import pathlib
import numpy as np
from tqdm import tqdm
import soundfile as sf
from functools import partial

from datasets.mir1k import MIR1KDataset
from datasets.mpop600 import MPop600Dataset


class MPop600InferenceDataset(Dataset):
    def __init__(self, wav_dir: str, split: str = "train"):
        super().__init__()
        wav_dir = pathlib.Path(wav_dir)
        test_files = []
        valid_files = []
        train_files = []
        for f in wav_dir.glob("*.wav"):
            singer, postfix = f.name.split("_")
            if postfix in MPop600Dataset.test_file_postfix:
                test_files.append(f)
            elif postfix in MPop600Dataset.valid_file_postfix:
                valid_files.append(f)
            else:
                train_files.append(f)

        if split == "train":
            self.files = train_files
        elif split == "valid":
            self.files = valid_files
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")

        self.wav_dir = wav_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename: pathlib.Path = self.files[index]
        y, sr = sf.read(filename)
        f0 = np.loadtxt(filename.with_suffix(".pv"))

        tp = np.arange(len(f0)) * sr // 200
        t = np.arange(y.shape[0])
        interp_f0 = np.interp(t, tp, f0)

        f0 = np.where(f0 < 80, 0, f0)
        # base file name
        rel_path = filename.relative_to(self.wav_dir)

        return y.astype(np.float32), interp_f0.astype(np.float32), str(rel_path)


class LJSpeechDataset(MPop600Dataset):
    test_file_postfix = set(f"LJ001-{i:04d}.wav" for i in range(1, 21))
    valid_file_postfix = set(f"LJ001-{i:04d}.wav" for i in range(21, 101))

    def __init__(
        self,
        wav_dir: str,
        split: str = "train",
        duration: float = 2.0,
        overlap: float = 1.0,
    ):
        wav_dir = pathlib.Path(wav_dir)
        test_files = []
        valid_files = []
        train_files = []
        for f in wav_dir.glob("*.wav"):
            postfix = f.name
            if postfix in self.test_file_postfix:
                test_files.append(f)
            elif postfix in self.valid_file_postfix:
                valid_files.append(f)
            else:
                train_files.append(f)

        if split == "train":
            self.files = train_files
        elif split == "valid":
            self.files = valid_files
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")

        self.sample_rate = None

        file_lengths = []
        self.samples = []
        self.f0s = []

        print("Gathering files ...")
        for filename in tqdm(self.files):
            x, sr = sf.read(filename)
            if self.sample_rate is None:
                self.sample_rate = sr
                self.segment_num_frames = int(duration * self.sample_rate)
                self.hop_num_frames = int((duration - overlap) * self.sample_rate)
            else:
                assert sr == self.sample_rate
            f0 = np.loadtxt(filename.with_suffix(".pv"))
            # interpolate f0 to frame level
            f0 = np.interp(
                np.arange(0, len(x)),
                np.arange(0, len(f0)) * self.sample_rate * 0.005,
                f0,
            )
            f0[f0 < 80] = 0

            self.f0s.append(f0)
            self.samples.append(x)
            file_lengths.append(
                max(0, x.shape[0] - self.segment_num_frames) // self.hop_num_frames + 1
            )

        self.file_lengths = np.array(file_lengths)
        self.boundaries = np.cumsum(np.array([0] + file_lengths))


class M4SingerDataset(Dataset):
    test_folder_prefixes = set(["Alto-1", "Soprano-1", "Tenor-1", "Bass-1"])
    valid_folder_prefixes = set(["Alto-2", "Alto-3", "Tenor-2", "Tenor-3"])
    file_suffix = ".wav"

    def __init__(
        self,
        wav_dir: str,
        split: str = "train",
        duration: float = 2.0,
        overlap: float = 1.0,
        f0_suffix: str = ".pv",
    ):
        super().__init__()
        wav_dir = pathlib.Path(wav_dir)
        test_files = []
        valid_files = []
        train_files = []
        for f in wav_dir.glob("**/*" + self.file_suffix):
            parent_prefix = f.parent.name.split("#")[0]
            if parent_prefix in self.test_folder_prefixes:
                test_files.append(f)
            elif parent_prefix in self.valid_folder_prefixes:
                valid_files.append(f)
            else:
                train_files.append(f)

        if split == "train":
            self.files = train_files
        elif split == "valid":
            self.files = valid_files
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")

        self.sample_rate = None

        file_lengths = []
        self.samples = []
        self.f0s = []

        print("Gathering files ...")
        for filename in tqdm(self.files):
            x, sr = sf.read(filename, dtype="float32")
            if self.sample_rate is None:
                self.sample_rate = sr
                self.segment_num_frames = int(duration * self.sample_rate)
                self.hop_num_frames = int((duration - overlap) * self.sample_rate)
                self.f0_hop_num_frames = 0.005 * self.sample_rate
            else:
                assert sr == self.sample_rate
            f0 = np.loadtxt(filename.with_suffix(f0_suffix))

            self.f0s.append(f0)
            self.samples.append(x)
            file_lengths.append(
                max(0, x.shape[0] - self.segment_num_frames) // self.hop_num_frames + 1
            )

        self.file_lengths = np.array(file_lengths)
        self.boundaries = np.cumsum(np.array([0] + file_lengths))

    def __len__(self):
        return self.boundaries[-1]

    def __getitem__(self, index):
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        x = self.samples[bin_pos]
        f0 = self.f0s[bin_pos]
        f0 = np.where(f0 < 60, 0, f0)
        offset = (index - self.boundaries[bin_pos]) * self.hop_num_frames

        x = x[offset : offset + self.segment_num_frames]
        tp = np.arange(len(f0)) * self.f0_hop_num_frames
        t = np.arange(offset, offset + self.segment_num_frames)
        mask = np.interp(t, tp, (f0 == 0).astype(float), right=1) > 0
        interp_f0 = np.where(mask, 0, np.interp(t, tp, f0))

        if x.shape[0] < self.segment_num_frames:
            x = np.pad(x, (0, self.segment_num_frames - x.shape[0]), "constant")
        else:
            x = x[: self.segment_num_frames]
        return x.astype(np.float32), interp_f0.astype(np.float32)


class VCTKDataset(M4SingerDataset):
    test_folder_prefixes = set(
        [
            "p360",
            "p361",
            "p362",
            "p363",
            "p364",
            "p374",
            "p376",
            "s5",
        ]
    )

    valid_folder_prefixes = set(
        [
            "p225",
            "p226",
            "p227",
            "p228",
            "p229",
            "p230",
            "p231",
            "p232",
            "p233",
            "p234",
            "p236",
            "p237",
            "p238",
            "p239",
            "p240",
            "p241",
        ]
    )

    file_suffix = "mic1.wav"


class VCTKInferenceDataset(Dataset):
    def __init__(self, wav_dir: str, split: str = "train", f0_suffix: str = ".pv"):
        super().__init__()
        self.wav_dir = pathlib.Path(wav_dir)
        test_files = []
        valid_files = []
        train_files = []
        for f in self.wav_dir.glob("**/*" + VCTKDataset.file_suffix):
            parent_prefix = f.parent.name.split("#")[0]
            if parent_prefix in VCTKDataset.test_folder_prefixes:
                test_files.append(f)
            elif parent_prefix in VCTKDataset.valid_folder_prefixes:
                valid_files.append(f)
            else:
                train_files.append(f)

        if split == "train":
            self.files = train_files
        elif split == "valid":
            self.files = valid_files
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")

        self.f0_suffix = f0_suffix

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename: pathlib.Path = self.files[index]
        y, sr = sf.read(filename)
        f0 = np.loadtxt(filename.with_suffix(self.f0_suffix))
        f0 = np.where(f0 < 60, 0, f0)
        tp = np.arange(len(f0)) * sr // 200
        t = np.arange(y.shape[0])
        mask = np.interp(t, tp, (f0 == 0).astype(float), right=1) > 0
        interp_f0 = np.where(mask, 0, np.interp(t, tp, f0))

        # base file name
        rel_path = filename.relative_to(self.wav_dir)

        return y.astype(np.float32), interp_f0.astype(np.float32), str(rel_path)


class MIR1K(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_dir: str,
        segment: int,
        overlap: int = 0,
        upsample_f0: bool = False,
        in_hertz: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = MIR1KDataset(
                data_dir=self.hparams.data_dir,
                segment=self.hparams.segment,
                overlap=self.hparams.overlap,
                upsample_f0=self.hparams.upsample_f0,
                in_hertz=self.hparams.in_hertz,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )


class MPop600(LightningDataModule):
    def __init__(
        self, batch_size: int, wav_dir: str, duration: float = 2, overlap: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = MPop600Dataset(
                wav_dir=self.hparams.wav_dir,
                split="train",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

        if stage == "validate" or stage == "fit":
            self.valid_dataset = MPop600Dataset(
                wav_dir=self.hparams.wav_dir,
                split="valid",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

        if stage == "test":
            self.test_dataset = MPop600Dataset(
                wav_dir=self.hparams.wav_dir,
                split="test",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

        if stage == "predict":
            self.predict_dataset = MPop600InferenceDataset(
                wav_dir=self.hparams.wav_dir,
                split="test",
            )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            drop_last=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )


class LJSpeech(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        wav_dir: str,
        duration: float = 2,
        overlap: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = LJSpeechDataset(
                wav_dir=self.hparams.wav_dir,
                split="train",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

        if stage == "validate" or stage == "fit":
            self.valid_dataset = LJSpeechDataset(
                wav_dir=self.hparams.wav_dir,
                split="valid",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

        if stage == "test":
            self.test_dataset = LJSpeechDataset(
                wav_dir=self.hparams.wav_dir,
                split="test",
                duration=self.hparams.duration,
                overlap=self.hparams.overlap,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )


class M4Singer(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        wav_dir: str,
        duration: float = 2,
        overlap: float = 0.5,
        f0_suffix: str = ".pv",
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        factory = partial(
            M4SingerDataset,
            wav_dir=self.hparams.wav_dir,
            duration=self.hparams.duration,
            overlap=self.hparams.overlap,
            f0_suffix=self.hparams.f0_suffix,
        )

        if stage == "fit":
            self.train_dataset = factory(split="train")

        if stage == "validate" or stage == "fit":
            self.valid_dataset = factory(split="valid")

        if stage == "test":
            self.test_dataset = factory(split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )


class VCTK(M4Singer):
    def setup(self, stage=None):
        factory = partial(
            VCTKDataset,
            wav_dir=self.hparams.wav_dir,
            duration=self.hparams.duration,
            overlap=self.hparams.overlap,
            f0_suffix=self.hparams.f0_suffix,
        )

        if stage == "fit":
            self.train_dataset = factory(split="train")

        if stage == "validate" or stage == "fit":
            self.valid_dataset = factory(split="valid")

        if stage == "test":
            self.test_dataset = factory(split="test")

        if stage == "predict":
            self.predict_dataset = VCTKInferenceDataset(
                wav_dir=self.hparams.wav_dir,
                split="test",
                f0_suffix=self.hparams.f0_suffix,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            drop_last=False,
        )


# ──────────────────────────────────────────────────────────────────
# Single-speaker flat directory dataset (e.g. Mandarin tone corpus)
# ──────────────────────────────────────────────────────────────────

class SingleSpeakerDataset(Dataset):
    """Flat directory of wav files for single-speaker experiments."""

    def __init__(
        self,
        wav_dir: str,
        split: str = "train",
        duration: float = 1.5,
        overlap: float = 1.0,
        f0_suffix: str = ".pv",
        val_ratio: float = 0.05,
        seed: int = 42,
        augment=None,
        load_formants: bool = False,
        load_aperiodicity: bool = False,
    ):
        super().__init__()
        import random
        wav_dir = pathlib.Path(wav_dir)
        all_files = sorted(wav_dir.glob("*.wav"))

        rng = random.Random(seed)
        files_shuffled = list(all_files)
        rng.shuffle(files_shuffled)

        n_val = max(1, int(len(files_shuffled) * val_ratio))
        val_files   = files_shuffled[:n_val]
        train_files = files_shuffled[n_val:]

        self.files = train_files if split == "train" else val_files
        self.augment = augment
        self.load_formants = load_formants
        self.load_aperiodicity = load_aperiodicity

        self.sample_rate = None
        self.samples, self.f0s, self.feats, self.aps = [], [], [], []
        file_lengths = []

        print(f"[SingleSpeakerDataset] Loading {len(self.files)} files ({split}) "
              f"{'(+formants)' if load_formants else ''}...")
        for filename in tqdm(self.files):
            x, sr = sf.read(str(filename), dtype="float32")
            if self.sample_rate is None:
                self.sample_rate = sr
                self.segment_frames = int(duration * sr)
                self.hop_frames     = int((duration - overlap) * sr)
                self.f0_hop_frames  = 0.005 * sr
                self.feat_hop       = int(sr * 0.01)   # .feat.npz is at 10ms
            f0 = np.loadtxt(str(filename.with_suffix(f0_suffix)))
            self.samples.append(x)
            self.f0s.append(f0)
            if load_formants:
                d = np.load(str(filename.with_suffix(".feat.npz")))
                self.feats.append({"f1": d["f1"].astype(np.float32),
                                   "f2": d["f2"].astype(np.float32)})
            if load_aperiodicity:
                # band aperiodicity A(f) target, [n_frames, n_mag], 10ms (WORLD D4C)
                self.aps.append(np.load(str(filename.with_suffix(".ap.npy"))
                                        ).astype(np.float32))
            file_lengths.append(
                max(0, len(x) - self.segment_frames) // self.hop_frames + 1
            )

        self.boundaries = np.cumsum([0] + file_lengths)

    def __len__(self):
        return int(self.boundaries[-1])

    def __getitem__(self, index):
        bin_pos = int(np.digitize(index, self.boundaries[1:], right=False))
        x  = self.samples[bin_pos]
        f0 = self.f0s[bin_pos]
        offset = int((index - self.boundaries[bin_pos]) * self.hop_frames)

        tp = np.arange(len(f0)) * self.f0_hop_frames
        t  = np.arange(offset, offset + self.segment_frames)
        mask      = np.interp(t, tp, (f0 == 0).astype(float), right=1) > 0
        interp_f0 = np.where(mask, 0.0, np.interp(t, tp, f0))

        seg = x[offset:offset + self.segment_frames]
        if len(seg) < self.segment_frames:
            seg = np.pad(seg, (0, self.segment_frames - len(seg)))

        seg = seg.astype(np.float32)
        interp_f0 = interp_f0.astype(np.float32)

        if self.augment is not None:
            seg, interp_f0 = self.augment(seg, interp_f0)

        if self.load_formants:
            feat = self.feats[bin_pos]
            foff = offset // self.feat_hop
            nfr  = self.segment_frames // self.feat_hop
            def _sl(arr):
                s = arr[foff:foff + nfr]
                if len(s) < nfr:
                    s = np.pad(s, (0, nfr - len(s)))
                return s[:nfr].astype(np.float32)
            f1 = _sl(feat["f1"]); f2 = _sl(feat["f2"])
            vmask = ((f1 > 0) & (f2 > 0)).astype(np.float32)
            if self.load_aperiodicity:
                ap = self.aps[bin_pos]                       # [n_frames, n_mag]
                s = ap[foff:foff + nfr]
                if len(s) < nfr:
                    s = np.pad(s, ((0, nfr - len(s)), (0, 0)))
                ap_seg = s[:nfr].astype(np.float32)
                return seg, interp_f0, f1, f2, vmask, ap_seg
            return seg, interp_f0, f1, f2, vmask

        return seg, interp_f0


class SingleSpeaker(LightningDataModule):
    """LightningDataModule for single-speaker flat wav directory."""

    def __init__(
        self,
        batch_size: int,
        wav_dir: str,
        duration: float = 1.5,
        overlap: float = 1.0,
        f0_suffix: str = ".pv",
        val_ratio: float = 0.05,
        sample_rate: int = 16000,
        augment: bool = True,
        f0_scale_range: list = None,
        tilt_alpha_range: list = None,
        load_formants: bool = False,
        load_aperiodicity: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _make_augmenter(self):
        from models.augment import SpeechAugment
        f0_range   = tuple(self.hparams.f0_scale_range   or [0.7, 1.3])
        tilt_range = tuple(self.hparams.tilt_alpha_range or [-0.12, 0.12])
        return SpeechAugment(f0_scale_range=f0_range, tilt_alpha_range=tilt_range)

    def setup(self, stage=None):
        from functools import partial
        factory = partial(
            SingleSpeakerDataset,
            wav_dir=self.hparams.wav_dir,
            duration=self.hparams.duration,
            overlap=self.hparams.overlap,
            f0_suffix=self.hparams.f0_suffix,
            val_ratio=self.hparams.val_ratio,
            load_formants=self.hparams.load_formants,
            load_aperiodicity=self.hparams.load_aperiodicity,
        )
        if stage in ("fit", None):
            aug = self._make_augmenter() if self.hparams.augment else None
            self.train_dataset = factory(split="train", augment=aug)
            self.valid_dataset = factory(split="valid", augment=None)
        if stage == "validate":
            self.valid_dataset = factory(split="valid", augment=None)
        if stage == "predict":
            self.predict_dataset = SingleSpeakerInferenceDataset(
                wav_dir=self.hparams.wav_dir,
                f0_suffix=self.hparams.f0_suffix,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=4, shuffle=True, drop_last=True,
                          persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size,
                          num_workers=4, shuffle=False, drop_last=False,
                          persistent_workers=True, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1,
                          num_workers=2, shuffle=False, drop_last=False)


class SingleSpeakerInferenceDataset(Dataset):
    """Returns full-length files (no segmentation) with rel_path for saving."""

    def __init__(self, wav_dir: str, f0_suffix: str = ".pv"):
        super().__init__()
        self.wav_dir = pathlib.Path(wav_dir)
        self.files = sorted(self.wav_dir.glob("*.wav"))
        self.f0_suffix = f0_suffix

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        f = self.files[index]
        x, sr = sf.read(str(f), dtype="float32")
        f0 = np.loadtxt(str(f.with_suffix(self.f0_suffix)))

        f0_hop = int(0.005 * sr)
        tp = np.arange(len(f0)) * f0_hop
        t  = np.arange(len(x))
        f0_interp = np.interp(t, tp, f0).astype(np.float32)

        rel_path = str(f.relative_to(self.wav_dir))
        return x.astype(np.float32), f0_interp, rel_path
