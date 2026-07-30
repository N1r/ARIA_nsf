import os
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from resampy import resample
import soundfile as sf
from pathlib import Path


class FixedSampleRateRandomWAVDataset(Dataset):
    """
    Wave-file-only Dataset.
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int,
        duration: float,
        size: int,
        deterministic: bool = True,
    ):
        self.data_path = os.path.expanduser(data_dir)
        self.sr = sample_rate
        self.size = size
        self.duration = duration
        self.deterministic = deterministic
        self.segment = int(self.sr * self.duration)

        self.waves = []
        self.files = []
        self.origin_sr = []

        file_lengths = []

        print("Gathering training files ...")
        for filename in tqdm(sorted(Path(self.data_path).glob("**/*.wav"))):
            meta = sf.info(filename)
            sr = meta.samplerate
            frames = meta.frames
            file_duration = frames / sr
            resampled_frames = int(file_duration * self.sr)
            file_lengths.append(max(0, resampled_frames - self.segment) + 1)
            self.files.append(filename)
            self.origin_sr.append(sr)

        self.file_lengths = np.array(file_lengths)
        self.boundaries = (
            np.cumsum(np.array([0] + file_lengths)) / self.file_lengths.sum()
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.deterministic:
            uniform_pos = index / self.size
        else:
            uniform_pos = random.uniform(0, 1)
        bin_pos = np.digitize(uniform_pos, self.boundaries[1:], right=False)
        f, length = self.files[bin_pos], self.file_lengths[bin_pos]
        origin_sr = self.origin_sr[bin_pos]
        ratio = self.sr / origin_sr
        offset = int(
            length
            * (uniform_pos - self.boundaries[bin_pos])
            / (self.boundaries[bin_pos + 1] - self.boundaries[bin_pos])
            / ratio
        )
        x, _ = sf.read(
            f,
            start=offset,
            frames=int(self.segment / ratio),
            dtype="float32",
            always_2d=True,
        )

        # resample
        if origin_sr != self.sr:
            x = resample(x, origin_sr, self.sr, axis=0)

        if x.shape[0] < self.segment:
            x = np.pad(x, ((0, self.segment - x.shape[0]), (0, 0)), "constant")
        elif x.shape[0] > self.segment:
            x = x[: self.segment]
        return x.T.astype(np.float32)
