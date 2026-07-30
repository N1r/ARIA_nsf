from .random_wav import RandomWAVDataset
from .wav import WAVDataset


class AutoregressiveRandomWAVDataset(RandomWAVDataset):
    """
    Wave-file-only Dataset.
    """

    def __init__(self,
                 data_dir: str,
                 size: int,
                 segment: int,
                 deterministic: bool = True):
        super().__init__(data_dir, size, segment + 1, deterministic)

    def __getitem__(self, index):
        x = super().__getitem__(index)
        return x[:-1], x[1:]


class AutoregressiveWAVDataset(WAVDataset):
    """
    Wave-file-only Dataset.
    """

    def __init__(self,
                 data_dir: str,
                 segment: int,
                 overlap: int = 0):
        super().__init__(data_dir, segment + 1, overlap + 1)

    def __getitem__(self, index):
        x = super().__getitem__(index)
        return x[:-1], x[1:]
