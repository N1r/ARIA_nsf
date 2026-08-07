"""Tests for the aris studio helpers; the pure helpers need no gradio."""

import json
import sys

import pytest

from aris.studio import (
    auto_variant_name,
    continuum_values,
    discover_checkpoints,
    discover_experiments,
    format_variant_metadata,
    variant_name,
)


class TestVariantNaming:
    def test_absolute_hz_value(self):
        assert variant_name("f1_hz", 550.0) == "f1_hz_550"

    def test_ratio_value_uses_p_for_decimal_point(self):
        assert variant_name("f1_scale", 1.15) == "f1_scale_1p15"

    def test_negative_value_uses_m_for_minus(self):
        assert variant_name("pitch_semitones", -2.5) == "pitch_semitones_m2p5"

    def test_auto_name_sorts_controls(self):
        name = auto_variant_name({"pitch_semitones": 2.0, "f1_scale": 1.1})
        assert name == "f1_scale_1p1_pitch_semitones_2"

    def test_auto_name_rejects_empty_controls(self):
        with pytest.raises(ValueError):
            auto_variant_name({})

    def test_names_are_valid_variant_slugs(self):
        from aris.controls.specs import ControlVariant

        for control, value in [("f1_hz", 550.0), ("f1_scale", 0.85), ("pitch_semitones", -3.25)]:
            ControlVariant(variant_name(control, value), {control: value})


class TestContinuumValues:
    def test_endpoints_and_count(self):
        values = continuum_values("f1_hz", 400.0, 800.0, 5)
        assert values == [400.0, 500.0, 600.0, 700.0, 800.0]

    def test_descending_continuum(self):
        values = continuum_values("f2_scale", 1.2, 0.8, 3)
        assert values == [1.2, 1.0, 0.8]

    def test_step_bounds(self):
        for steps in (1, 21):
            with pytest.raises(ValueError):
                continuum_values("f1_hz", 400.0, 800.0, steps)

    def test_out_of_range_endpoint_rejected(self):
        with pytest.raises(ValueError):
            continuum_values("f1_hz", 100.0, 800.0, 3)

    def test_equal_endpoints_rejected(self):
        with pytest.raises(ValueError):
            continuum_values("f1_hz", 500.0, 500.0, 3)

    def test_unknown_control_rejected(self):
        with pytest.raises(ValueError):
            continuum_values("f9_hz", 400.0, 800.0, 3)

    def test_values_produce_unique_names(self):
        values = continuum_values("f1_scale", 0.8, 1.2, 9)
        names = [variant_name("f1_scale", value) for value in values]
        assert len(set(names)) == len(names)


class TestDiscovery:
    def test_experiments_found_up_to_depth_three(self, tmp_path):
        for relative in ("exp_root", "experiments/a", "demo_f024/experiment", "x/y/z/too_deep"):
            target = tmp_path / relative
            target.mkdir(parents=True)
            (target / "experiment.json").write_text("{}")
        found = discover_experiments(tmp_path)
        names = [str(path.relative_to(tmp_path)) for path in found]
        assert "exp_root" in names
        assert "experiments/a" in names
        assert "demo_f024/experiment" in names
        assert not any("too_deep" in name for name in names)

    def test_hidden_and_output_directories_skipped(self, tmp_path):
        for relative in (".venv/exp", "studio_output/exp"):
            target = tmp_path / relative
            target.mkdir(parents=True)
            (target / "experiment.json").write_text("{}")
        assert discover_experiments(tmp_path) == []

    def test_checkpoints_last_first(self, tmp_path):
        checkpoint_dir = tmp_path / "runs" / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        for name in ("epoch=3.ckpt", "last.ckpt", "epoch=1.ckpt"):
            (checkpoint_dir / name).write_bytes(b"")
        names = [path.name for path in discover_checkpoints(tmp_path)]
        assert names == ["last.ckpt", "epoch=1.ckpt", "epoch=3.ckpt"]

    def test_no_checkpoint_directory(self, tmp_path):
        assert discover_checkpoints(tmp_path) == []


class TestFormatVariantMetadata:
    def entry(self, **audit):
        base = {"clipped_fraction": 0.0, "formant_tracking": {}}
        base.update(audit)
        return {"name": "f1_hz_550", "controls": {"f1_hz": 550.0}, "render_audit": base}

    def test_clean_render_has_no_warnings(self):
        text = format_variant_metadata(self.entry())
        assert "f1_hz_550" in text
        assert "f1_hz=550" in text
        assert "⚠️" not in text

    def test_clipping_above_threshold_warns_in_red(self):
        text = format_variant_metadata(self.entry(clipped_fraction=0.01))
        assert "⚠️" in text
        assert "color:#c00" in text

    def test_formant_tracking_rounded_and_percent(self):
        tracking = {"f1": {"max_abs_hz_discrepancy": 12.34, "clamped_frame_fraction": 0.031}}
        text = format_variant_metadata(self.entry(formant_tracking=tracking))
        assert "12 Hz" in text
        assert "3.1%" in text
        assert "⚠️" not in text

    def test_clamped_fraction_above_threshold_warns(self):
        tracking = {"f2": {"max_abs_hz_discrepancy": 210.0, "clamped_frame_fraction": 0.34}}
        text = format_variant_metadata(self.entry(formant_tracking=tracking))
        assert "210 Hz" in text
        assert "34.0%" in text
        assert "⚠️" in text
        assert "范围边缘" in text


class TestGradioOptional:
    def test_import_works_without_gradio(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gradio", None)
        import aris.studio  # noqa: F401 - import must not require gradio

    def test_build_app_raises_clear_hint_without_gradio(self, monkeypatch, tmp_path):
        from aris.studio import build_app

        monkeypatch.setitem(sys.modules, "gradio", None)
        with pytest.raises(RuntimeError, match=r"aris\[studio\]"):
            build_app(tmp_path)


class TestBuildApp:
    def test_build_app_returns_blocks(self, tmp_path):
        gradio = pytest.importorskip("gradio")
        from aris.studio import build_app

        app = build_app(tmp_path)
        assert isinstance(app, gradio.Blocks)


class TestSpectrogramFigure:
    def test_four_aligned_panels(self, tmp_path):
        pytest.importorskip("matplotlib")
        soundfile = pytest.importorskip("soundfile")
        numpy = pytest.importorskip("numpy")
        from aris.studio import spectrogram_figure

        sample_rate = 16000
        time = numpy.arange(int(0.3 * sample_rate)) / sample_rate
        original = tmp_path / "original.wav"
        variant = tmp_path / "variant.wav"
        soundfile.write(original, 0.5 * numpy.sin(2 * numpy.pi * 220 * time), sample_rate)
        soundfile.write(variant, 0.5 * numpy.sin(2 * numpy.pi * 330 * time), sample_rate)
        figure = spectrogram_figure(original, variant)
        assert len(figure.axes) == 4
        assert figure.axes[-1].get_xlabel() == "Time (s)"


class TestCliRegistration:
    def test_studio_subcommand_parses(self):
        from aris.cli import parser

        args = parser().parse_args(["studio", "--port", "9000", "--no-browser"])
        assert args.command == "studio"
        assert args.port == 9000
        assert args.no_browser is True
        assert str(args.workspace) == "."

    def test_manipulation_metadata_roundtrip(self, tmp_path):
        # format_variant_metadata consumes manipulation.json outputs[] entries.
        entry = {
            "name": "demo",
            "directory": "demo",
            "controls": {"f2_scale": 1.2},
            "render_audit": {
                "clipped_fraction": 0.0,
                "formant_tracking": {
                    "f2": {"max_abs_hz_discrepancy": 3.0, "clamped_frame_fraction": 0.0}
                },
            },
        }
        path = tmp_path / "manipulation.json"
        path.write_text(json.dumps({"outputs": [entry]}))
        loaded = json.loads(path.read_text())["outputs"][0]
        text = format_variant_metadata(loaded)
        assert "demo" in text and "f2_scale=1.2" in text and "3 Hz" in text
