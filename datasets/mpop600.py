import pathlib
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import soundfile as sf


class MPop600Dataset(Dataset):
    """
    MPop600 dataset.
    """

    # sample_rate = 48000
    # the first 3 files, 001 to 003, are test files
    test_file_postfix = set(f"00{i}.wav" for i in range(1, 4))
    # the next 27 files are for validation
    valid_file_postfix = set(f"{i:03d}.wav" for i in range(4, 31))

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
            singer, postfix = f.name.split("_")
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

    def __len__(self):
        return self.boundaries[-1]

    def __getitem__(self, index):
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        x = self.samples[bin_pos]
        f0 = self.f0s[bin_pos]
        offset = (index - self.boundaries[bin_pos]) * self.hop_num_frames

        x = x[offset : offset + self.segment_num_frames]
        f0 = f0[offset : offset + self.segment_num_frames]

        if x.shape[0] < self.segment_num_frames:
            x = np.pad(x, (0, self.segment_num_frames - x.shape[0]), "constant")
            f0 = np.pad(f0, (0, self.segment_num_frames - f0.shape[0]), "constant")
        else:
            x = x[: self.segment_num_frames]
            f0 = f0[: self.segment_num_frames]
        return x.astype(np.float32), f0.astype(np.float32)
