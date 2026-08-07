import json
from types import SimpleNamespace

import pytest
import torch

from aris import __version__
from aris.controls.lightning import ControlledPredictionWriter
from aris.models.audiotensor import AudioTensor


def test_controlled_writer_accepts_cli_mapping_and_records_render_audit(tmp_path, monkeypatch):
    saved = {}

    def fake_save(path, waveform, **kwargs):
        saved["path"] = path
        saved["waveform"] = waveform
        saved["kwargs"] = kwargs

    monkeypatch.setattr(
        "aris.controls.lightning.torchaudio.save",
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
    assert metadata["aris_version"] == __version__
    assert metadata["formant_tracking"] == {}


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


def test_controlled_writer_warns_on_stderr_when_clipping_exceeds_threshold(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("aris.controls.lightning.torchaudio.save", lambda *a, **k: None)
    writer = ControlledPredictionWriter(
        str(tmp_path / "render"),
        controls_json={"output_gain_db": 12.0},
    )
    module = SimpleNamespace(decoder=torch.nn.Identity(), sample_rate=16000)
    writer.on_predict_start(None, module)
    # 0.5 * 10**(12/20) ~= 1.99: every sample clips at full scale.
    prediction = AudioTensor(torch.full((1, 320), 0.5))

    writer.write_on_batch_end(
        trainer=None,
        pl_module=module,
        prediction=(prediction, {}),
        batch_indices=None,
        batch=(None, None, ["clipped.wav"]),
        batch_idx=0,
        dataloader_idx=0,
    )
    writer.on_predict_end(None, module)

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "clipped.wav" in captured.err


def test_controlled_writer_clamps_and_records_real_clipping(tmp_path, monkeypatch):
    saved = {}

    def fake_save(path, waveform, **kwargs):
        saved["waveform"] = waveform

    monkeypatch.setattr("aris.controls.lightning.torchaudio.save", fake_save)
    writer = ControlledPredictionWriter(
        str(tmp_path / "render"),
        controls_json={"output_gain_db": 12.0},
    )
    module = SimpleNamespace(decoder=torch.nn.Identity(), sample_rate=16000)
    writer.on_predict_start(None, module)
    # 0.99 * 10**(12/20) ~= 3.94: comfortably past full scale on every sample.
    prediction = AudioTensor(torch.full((1, 320), 0.99))

    writer.write_on_batch_end(
        trainer=None,
        pl_module=module,
        prediction=(prediction, {}),
        batch_indices=None,
        batch=(None, None, ["clipped.wav"]),
        batch_idx=0,
        dataloader_idx=0,
    )
    writer.on_predict_end(None, module)

    assert torch.all(saved["waveform"].abs() <= 1.0)
    metadata = json.loads((tmp_path / "render" / "_render.json").read_text())
    assert metadata["clipped_samples"] > 0
    assert metadata["clipped_fraction"] > 0


def test_controlled_writer_does_not_warn_when_clipping_stays_below_threshold(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("aris.controls.lightning.torchaudio.save", lambda *a, **k: None)
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
        batch=(None, None, ["clean.wav"]),
        batch_idx=0,
        dataloader_idx=0,
    )
    writer.on_predict_end(None, module)

    captured = capsys.readouterr()
    assert captured.err == ""
