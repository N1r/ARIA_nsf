import wave

import torch

from ltng.cli import MyPredictionWriter
from models.audiotensor import AudioTensor


def test_prediction_writer_emits_interoperable_pcm16(tmp_path):
    output = tmp_path / "predictions"
    writer = MyPredictionWriter(output)

    class Module:
        sample_rate = 16000

    prediction = AudioTensor(torch.linspace(-0.5, 0.5, 320).unsqueeze(0))
    batch = (None, None, ["example.wav"])
    writer.write_on_batch_end(
        trainer=None,
        pl_module=Module(),
        prediction=(prediction, {}),
        batch_indices=None,
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    with wave.open(str(output / "example.wav"), "rb") as stream:
        assert stream.getframerate() == 16000
        assert stream.getsampwidth() == 2
        assert stream.getnframes() == 320
