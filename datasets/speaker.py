from torch.utils.data import Dataset
from .random_wav import RandomWAVDataset
import random
import torch
import torchaudio
import numpy as np


class SpeakerEmbDataset(RandomWAVDataset):
    def __getitem__(self, index):
        if self.deterministic:
            uniform_pos = index / self.size
        else:
            uniform_pos = random.uniform(0, 1)
        bin_pos = np.digitize(uniform_pos, self.boundaries[1:], right=False)
        f, length = self.files[bin_pos], self.file_lengths[bin_pos]
        emb_f = str(f).replace(".wav", "_emb.pt")
        offset = int(length * (uniform_pos - self.boundaries[bin_pos]) / (
            self.boundaries[bin_pos+1] - self.boundaries[bin_pos]))
        x = torchaudio.load(f, frame_offset=offset,
                            num_frames=self.segment)[0].mean(0)
        emb = torch.load(emb_f)
        emb.requires_grad = False

        # this should not happen but I just want to make sure
        if x.numel() < self.segment:
            x = torch.cat([x, x.new_zeros(self.segment - x.numel())])
        return x, emb
