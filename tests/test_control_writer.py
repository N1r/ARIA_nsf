import json
from types import SimpleNamespace

import pytest
import torch

from models.audiotensor import AudioTensor
from phonlab_ddsp.controls.lightning import ControlledPredictionWriter


def test_controlled_writer_accepts_cli_mapping_and_records_render_audit(tmp_path, monkeypatch):
    saved = {}

    def fake_save(path, waveform, **kwargs):
        saved["path"] = path
        saved["waveform"] = waveform
        saved["kwargs"] = kwargs

    monkeypatch.setattr(
        "phonlab_ddsp.controls.lightning.torchaudio.save",
        fake_save,
    )
    writer = ControlledPredictionWriter(
        str(tmp_path / "render"),
        controls_json={"output_gain_db": -6.0},
    )
    module = SimpleNamespace(decoder=torch.nn.Identity(), sample_rate=16000)
    writer.on_predict_start(None, module)
    prediction = AudioTensor(torch.full((1, 320), 0.5))

    writer.write_on_batch_end(
        trainer=None,
        pl_module=module,
        prediction=(prediction, {}),
        batch_indices=None,
        batch=(None, None, ["sample.wav"]),
        batch_idx=0,
        dataloader_idx=0,
    )
    writer.on_predict_end(None, module)

    expected = torch.full((1, 320), 0.5 * 10 ** (-6.0 / 20.0))
    torch.testing.assert_close(saved["waveform"], expected)
    assert saved["path"] == tmp_path / "render" / "sample.wav"
    assert saved["kwargs"]["sample_rate"] == 16000
    metadata = json.loads((tmp_path / "render" / "_render.json").read_text())
    assert metadata["controls"] == {"output_gain_db": -6.0}
    assert metadata["runtime_capabilities"] == ["output_gain_db"]
    assert metadata["files_written"] == 1
    assert metadata["decoder_control_calls"] == 0
    assert metadata["clipped_samples"] == 0


@pytest.mark.parametrize(
    "controls",
    (
        {"typo_noise_db": 6.0},
        {"noise_gain_db": 24.001},
        {"pitch_semitones": 4.0},
    ),
)
def test_controlled_writer_rejects_unknown_out_of_range_and_data_side_controls(tmp_path, controls):
    with pytest.raises(ValueError):
        ControlledPredictionWriter(
            str(tmp_path / "render"),
            controls_json=controls,
        )
