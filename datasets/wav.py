import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchaudio


class WAVDataset(Dataset):
    """
    Wave-file-only Dataset.
    """

    def __init__(self,
                 data_dir: str,
                 segment: int,
                 overlap: int = 0):
        assert segment > overlap and overlap >= 0
        self.segment = segment
        self.data_path = os.path.expanduser(data_dir)
        self.hop_size = segment - overlap

        self.waves = []
        self.sr = None
        self.files = []

        file_chunks = []

        print("Gathering training files ...")
        for f in tqdm(sorted(os.listdir(self.data_path))):
            if f.endswith('.wav'):
                filename = os.path.join(self.data_path, f)
                meta = torchaudio.info(filename)
                self.files.append(filename)
                file_chunks.append(
                    max(0, meta.num_frames - segment) // self.hop_size + 1)

                if not self.sr:
                    self.sr = meta.sample_rate
                else:
                    assert meta.sample_rate == self.sr

        self.size = sum(file_chunks)
        self.boundaries = np.cumsum(np.array([0] + file_chunks))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        f = self.files[bin_pos]
        offset = (index - self.boundaries[bin_pos]) * self.hop_size
        x = torchaudio.load(f, frame_offset=offset,
                            num_frames=self.segment)[0].mean(0)

        # this should not happen but I just want to make sure
        if x.numel() < self.segment:
            x = torch.cat([x, x.new_zeros(self.segment - x.numel())])
        return x
