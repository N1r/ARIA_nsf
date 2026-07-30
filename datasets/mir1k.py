import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchaudio


def midi2hz(midi):
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


class MIR1KDataset(Dataset):
    """
    MID-1K Dataset.
    """

    pitch_offset = 640
    pitch_step = 320
    sr = 16000
    wave_folder = "Wavfile"
    pitch_folder = "PitchLabel"

    def __init__(
        self,
        data_dir: str,
        segment: int,
        overlap: int = 0,
        upsample_f0: bool = False,
        in_hertz: bool = True,
    ):
        assert segment > overlap and overlap >= 0
        self.segment = segment
        self.data_path = os.path.expanduser(data_dir)
        self.hop_size = segment - overlap
        self.in_hertz = in_hertz
        self.upsample_f0 = upsample_f0

        self.waves = []
        self.file_names = []
        self.pitches = []

        file_chunks = []
        print("Gathering files ...")
        for f in tqdm(
            sorted(os.listdir(os.path.join(self.data_path, self.pitch_folder)))
        ):
            base_name = f.split(".")[0]
            self.file_names.append(base_name)
            filename = os.path.join(self.data_path, self.pitch_folder, f)
            f0 = np.loadtxt(filename)
            self.pitches.append(f0)

            file_chunks.append(max(0, len(f0) - segment) // self.hop_size + 1)

        self.size = sum(file_chunks)
        self.boundaries = np.cumsum(np.array([0] + file_chunks))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)

        offset = (index - self.boundaries[bin_pos]) * self.hop_size
        f0 = self.pitches[bin_pos][offset : offset + self.segment]
        if self.in_hertz:
            mask = f0 > 0
            f0[mask] = midi2hz(f0[mask])
        f0 = torch.from_numpy(f0).float()
        f0[f0 > (self.sr // 2)] = 0

        basename = self.file_names[bin_pos]
        wave_filename = os.path.join(
            self.data_path, self.wave_folder, basename + ".wav"
        )
        frames = (self.segment - 1) * self.pitch_step + 1
        x = torchaudio.load(
            wave_filename,
            frame_offset=self.pitch_offset + offset * self.pitch_step,
            num_frames=frames,
        )[0]
        if x.shape[1] < frames:
            x = torch.cat([x, x.new_zeros((x.shape[0], frames - x.shape[1]))], dim=1)
        if f0.numel() < self.segment:
            f0 = torch.cat([f0, f0.new_zeros((self.segment - f0.numel()))], dim=0)

        if self.upsample_f0:
            f0 = torch.nn.functional.interpolate(
                f0.unsqueeze(0).unsqueeze(0),
                size=frames,
                mode="linear",
                align_corners=True,
            ).squeeze()

        return x, f0
