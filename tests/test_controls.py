import math
from types import SimpleNamespace

import pytest
import torch

from aris.controls.runtime import (
    DecoderControlHook,
    runtime_capabilities,
)
from aris.controls.specs import (
    CONTROL_SPECS,
    controls_for_model,
    parse_variant,
    validate_controls,
)
from aris.models.audiotensor import AudioTensor

EXPECTED_CONTROLS = {
    "golf": (
        "pitch_semitones",
        "output_gain_db",
        "noise_gain_db",
        "glottal_rd_scale",
    ),
    "ddsp": (
        "pitch_semitones",
        "output_gain_db",
        "noise_gain_db",
    ),
    "aria-golf": (
        "pitch_semitones",
        "output_gain_db",
        "noise_gain_db",
        "glottal_rd_scale",
        "f1_scale",
        "f2_scale",
        "f1_hz",
        "f2_hz",
        "tilt_alpha_delta",
    ),
}


class _CaptureDecoder(torch.nn.Module):
    def __init__(self, *, noise=False, rd_values=None, end_filter=None):
        super().__init__()
        if noise:
            self.noise_filter = SimpleNamespace(mag_activation="exp")
        if rd_values is not None:
            self.harm_oscillator = SimpleNamespace(
                R_d_values=rd_values,
                rd_control_mode="indexed",
            )
        if end_filter is not None:
            self.end_filter = end_filter
        self.seen_kwargs = None

    def forward(self, **kwargs):
        self.seen_kwargs = kwargs
        return kwargs


class _AriaLikeEndFilter:
    use_tilt = True

    def __init__(self):
        self.scale_call = None

    def scale_formants(self, control, *, f1_scale, f2_scale, alpha_delta):
        self.scale_call = {
            "control": control.clone(),
            "f1_scale": f1_scale,
            "f2_scale": f2_scale,
            "alpha_delta": alpha_delta,
        }
        return control + 0.25


@pytest.mark.parametrize("model", ("golf", "ddsp", "aria-golf"))
def test_control_capability_matrix(model):
    specs = controls_for_model(model)

    assert tuple(spec.name for spec in specs) == EXPECTED_CONTROLS[model]
    assert all(spec.supports(model) for spec in specs)


@pytest.mark.parametrize(
    ("name", "minimum", "maximum"),
    (
        ("pitch_semitones", -36.0, 36.0),
        ("output_gain_db", -24.0, 12.0),
        ("noise_gain_db", -24.0, 24.0),
        ("glottal_rd_scale", 0.5, 2.0),
        ("f1_scale", 0.7, 1.3),
        ("f2_scale", 0.7, 1.3),
        ("f1_hz", 150.0, 1300.0),
        ("f2_hz", 600.0, 3200.0),
        ("tilt_alpha_delta", -0.25, 0.25),
    ),
)
def test_control_ranges_are_inclusive(name, minimum, maximum):
    spec = CONTROL_SPECS[name]

    assert spec.normalize(minimum) == minimum
    assert spec.normalize(maximum) == maximum
    assert minimum <= spec.default <= maximum


def test_parse_variant_normalizes_a_valid_variant():
    variant = parse_variant(" bright_1 : noise_gain_db=6, glottal_rd_scale=1.25, f1_scale=.9 ")

    assert variant.name == "bright_1"
    assert variant.slug == "bright_1"
    assert variant.controls == {
        "noise_gain_db": 6.0,
        "glottal_rd_scale": 1.25,
        "f1_scale": 0.9,
    }


@pytest.mark.parametrize(
    "text",
    (
        "low:pitch_semitones=-36.001",
        "high:f1_scale=1.3001",
        "nonfinite:noise_gain_db=nan",
        "duplicate:noise_gain_db=1,noise_gain_db=2",
        "unknown:not_a_control=1",
        "../escape:pitch_semitones=1",
    ),
)
def test_parse_variant_rejects_invalid_values(text):
    with pytest.raises(ValueError):
        parse_variant(text)


def test_parse_variant_accepts_absolute_formant_controls():
    variant = parse_variant("f1_target:f1_hz=550,f2_hz=1800")

    assert variant.controls == {"f1_hz": 550.0, "f2_hz": 1800.0}


def test_parse_variant_rejects_out_of_range_absolute_formant_control():
    with pytest.raises(ValueError):
        parse_variant("too_low:f1_hz=100")


def test_validate_controls_rejects_unsupported_control_and_model():
    with pytest.raises(ValueError, match="not supported.*'ddsp'"):
        validate_controls("ddsp", {"glottal_rd_scale": 1.0})

    with pytest.raises(ValueError, match="Unknown model"):
        validate_controls("not-a-model", {"pitch_semitones": 0.0})


def test_absolute_formant_controls_are_gated_to_aria_golf():
    with pytest.raises(ValueError, match="not supported.*'golf'"):
        validate_controls("golf", {"f1_hz": 500.0})

    assert validate_controls("aria-golf", {"f1_hz": 550.0, "f2_hz": 1800.0}) == {
        "f1_hz": 550.0,
        "f2_hz": 1800.0,
    }


@pytest.mark.parametrize(
    ("absolute", "ratio"),
    (("f1_hz", "f1_scale"), ("f2_hz", "f2_scale")),
)
def test_validate_controls_rejects_absolute_and_ratio_formant_conflict(absolute, ratio):
    with pytest.raises(ValueError, match="Conflicting formant controls"):
        validate_controls("aria-golf", {absolute: CONTROL_SPECS[absolute].default, ratio: 1.1})


def test_runtime_capabilities_follow_decoder_surfaces():
    end_filter = _AriaLikeEndFilter()
    decoder = _CaptureDecoder(
        noise=True,
        rd_values=torch.tensor([0.3, 0.7, 1.4, 2.7]),
        end_filter=end_filter,
    )

    assert runtime_capabilities(SimpleNamespace(decoder=decoder)) == {
        "output_gain_db",
        "noise_gain_db",
        "glottal_rd_scale",
        "f1_scale",
        "f2_scale",
        "tilt_alpha_delta",
    }


@pytest.mark.parametrize("amplitude_ratio", (0.5, 2.0))
def test_noise_gain_db_adds_exact_log_amplitude(amplitude_ratio):
    decoder = _CaptureDecoder(noise=True)
    hook = DecoderControlHook({"noise_gain_db": 20.0 * math.log10(amplitude_ratio)})
    hook.install(SimpleNamespace(decoder=decoder))
    original = AudioTensor(
        torch.tensor([[[-1.0, -0.25], [0.5, 1.25]]]),
        hop_length=80,
    )
    tail = object()

    result = decoder(noise_filter_params=(original, tail))
    changed, changed_tail = result["noise_filter_params"]
    changed_tensor = changed.as_tensor()
    original_tensor = original.as_tensor()

    assert isinstance(changed, AudioTensor)
    assert changed.hop_length == original.hop_length
    assert changed_tail is tail
    torch.testing.assert_close(
        changed_tensor - original_tensor,
        torch.full_like(original_tensor, math.log(amplitude_ratio)),
    )
    torch.testing.assert_close(
        torch.exp(changed_tensor) / torch.exp(original_tensor),
        torch.full_like(original_tensor, amplitude_ratio),
    )
    assert hook.calls == 1
    hook.verify()
    hook.remove()


@pytest.mark.parametrize(("scale", "direction"), ((0.5, -1), (2.0, 1)))
def test_glottal_rd_scale_clamps_weights_and_moves_in_requested_direction(scale, direction):
    rd_values = torch.tensor([0.3, 0.7, 1.4, 2.7])
    decoder = _CaptureDecoder(rd_values=rd_values)
    hook = DecoderControlHook({"glottal_rd_scale": scale})
    hook.install(SimpleNamespace(decoder=decoder))
    original = AudioTensor(torch.tensor([[0.05, 0.5, 0.95]]), hop_length=64)
    tail = object()

    result = decoder(harm_oscillator_params=(original, tail))
    changed, changed_tail = result["harm_oscillator_params"]
    changed_tensor = changed.as_tensor()
    original_tensor = original.as_tensor()
    expected_delta = math.log(scale) / math.log(float(rd_values[-1] / rd_values[0]))
    expected = torch.clamp(original_tensor + expected_delta, 0.0, 1.0)

    assert isinstance(changed, AudioTensor)
    assert changed.hop_length == original.hop_length
    assert changed_tail is tail
    torch.testing.assert_close(changed_tensor, expected)
    assert torch.all((0.0 <= changed_tensor) & (changed_tensor <= 1.0))
    assert torch.all(direction * (changed_tensor - original_tensor) >= 0.0)
    assert torch.any(direction * (changed_tensor - original_tensor) > 0.0)
    hook.verify()
    hook.remove()


def test_aria_formant_controls_delegate_to_end_filter_and_preserve_audio_tensor():
    end_filter = _AriaLikeEndFilter()
    decoder = _CaptureDecoder(end_filter=end_filter)
    hook = DecoderControlHook(
        {
            "f1_scale": 1.1,
            "f2_scale": 0.85,
            "tilt_alpha_delta": -0.125,
        }
    )
    hook.install(SimpleNamespace(decoder=decoder))
    original = AudioTensor(torch.tensor([[0.1, 0.2, 0.3]]), hop_length=80)
    tail = object()

    result = decoder(end_filter_params=(original, tail))
    changed, changed_tail = result["end_filter_params"]

    assert end_filter.scale_call is not None
    assert not isinstance(end_filter.scale_call["control"], AudioTensor)
    torch.testing.assert_close(
        end_filter.scale_call["control"],
        original.as_tensor(),
    )
    assert end_filter.scale_call["f1_scale"] == 1.1
    assert end_filter.scale_call["f2_scale"] == 0.85
    assert end_filter.scale_call["alpha_delta"] == -0.125
    assert isinstance(changed, AudioTensor)
    assert changed.hop_length == original.hop_length
    torch.testing.assert_close(changed.as_tensor(), original.as_tensor() + 0.25)
    assert changed_tail is tail
    hook.verify()
    hook.remove()


def test_real_aria_filter_changes_interpretable_tracks_by_requested_amount():
    from aris.models.analytic_filter import VocalTractCascade

    end_filter = VocalTractCascade(sr=16000, n_learned=0)
    decoder = _CaptureDecoder(end_filter=end_filter)
    hook = DecoderControlHook(
        {
            "f1_scale": 1.1,
            "f2_scale": 0.9,
            "tilt_alpha_delta": 0.05,
        }
    )
    hook.install(SimpleNamespace(decoder=decoder))
    original = AudioTensor(torch.zeros(1, 3, end_filter.n_ctrl), hop_length=80)

    result = decoder(end_filter_params=(original,))
    changed = result["end_filter_params"][0]
    before = end_filter.get_formant_params(original.as_tensor())
    after = end_filter.get_formant_params(changed.as_tensor())

    torch.testing.assert_close(after["f1"], before["f1"] * 1.1)
    torch.testing.assert_close(after["f2"], before["f2"] * 0.9)
    torch.testing.assert_close(
        after["alpha"],
        before["alpha"] + 0.05,
        atol=1e-6,
        rtol=1e-6,
    )
    hook.verify()
    hook.remove()


def test_absolute_formant_controls_are_not_unlocked_by_scale_only_end_filter():
    end_filter = _AriaLikeEndFilter()
    decoder = _CaptureDecoder(end_filter=end_filter)

    capabilities = runtime_capabilities(SimpleNamespace(decoder=decoder))

    assert "f1_hz" not in capabilities
    assert "f2_hz" not in capabilities


def test_real_aria_filter_absolute_hz_overrides_natural_contour():
    from aris.models.analytic_filter import VocalTractCascade

    end_filter = VocalTractCascade(sr=16000, n_learned=0)
    decoder = _CaptureDecoder(end_filter=end_filter)
    capabilities = runtime_capabilities(SimpleNamespace(decoder=decoder))
    assert {"f1_hz", "f2_hz"} <= capabilities

    hook = DecoderControlHook({"f1_hz": 500.0, "f2_hz": 1800.0})
    hook.install(SimpleNamespace(decoder=decoder))
    original = AudioTensor(torch.zeros(1, 3, end_filter.n_ctrl), hop_length=80)

    result = decoder(end_filter_params=(original,))
    changed = result["end_filter_params"][0]
    after = end_filter.get_formant_params(changed.as_tensor())

    torch.testing.assert_close(after["f1"], torch.full_like(after["f1"], 500.0))
    torch.testing.assert_close(after["f2"], torch.full_like(after["f2"], 1800.0))
    hook.verify()
    hook.remove()


def test_real_aria_filter_combines_absolute_f1_hz_with_relative_f2_scale():
    from aris.models.analytic_filter import VocalTractCascade

    end_filter = VocalTractCascade(sr=16000, n_learned=0)
    decoder = _CaptureDecoder(end_filter=end_filter)
    hook = DecoderControlHook({"f1_hz": 400.0, "f2_scale": 1.1})
    hook.install(SimpleNamespace(decoder=decoder))
    original = AudioTensor(torch.zeros(1, 3, end_filter.n_ctrl), hop_length=80)

    result = decoder(end_filter_params=(original,))
    changed = result["end_filter_params"][0]
    before = end_filter.get_formant_params(original.as_tensor())
    after = end_filter.get_formant_params(changed.as_tensor())

    torch.testing.assert_close(after["f1"], torch.full_like(after["f1"], 400.0))
    torch.testing.assert_close(after["f2"], before["f2"] * 1.1)
    hook.verify()
    hook.remove()


def test_formant_tracking_reports_achieved_vs_requested_after_clamp():
    from aris.models.analytic_filter import VocalTractCascade

    end_filter = VocalTractCascade(sr=16000, n_learned=0, f1_range=(150, 300))
    decoder = _CaptureDecoder(end_filter=end_filter)
    hook = DecoderControlHook({"f1_scale": 1.3})
    hook.install(SimpleNamespace(decoder=decoder))
    # A large positive logit saturates sigmoid() near the top of f1_range, so a
    # 1.3x request cannot fit inside the range and must be clamped every frame.
    ctrl = torch.full((1, 4, end_filter.n_ctrl), 4.0)
    original = AudioTensor(ctrl, hop_length=80)

    decoder(end_filter_params=(original,))
    summary = hook.formant_tracking_summary()

    assert "f1" in summary
    assert summary["f1"]["max_abs_hz_discrepancy"] > 0.0
    assert summary["f1"]["clamped_frame_fraction"] == 1.0
    hook.verify()
    hook.remove()


def test_formant_tracking_is_empty_without_ratio_formant_controls():
    from aris.models.analytic_filter import VocalTractCascade

    end_filter = VocalTractCascade(sr=16000, n_learned=0)
    decoder = _CaptureDecoder(end_filter=end_filter)
    hook = DecoderControlHook({"f1_hz": 500.0})
    hook.install(SimpleNamespace(decoder=decoder))
    original = AudioTensor(torch.zeros(1, 3, end_filter.n_ctrl), hop_length=80)

    decoder(end_filter_params=(original,))

    assert hook.formant_tracking_summary() == {}
    hook.verify()
    hook.remove()


def test_hook_install_rejects_missing_runtime_capability():
    decoder = _CaptureDecoder()
    hook = DecoderControlHook({"noise_gain_db": 3.0})

    with pytest.raises(
        ValueError,
        match=r"cannot apply control\(s\): noise_gain_db",
    ):
        hook.install(SimpleNamespace(decoder=decoder))

    assert hook.handle is None
    assert hook.calls == 0


def test_hook_rejects_unknown_runtime_control_instead_of_silently_ignoring_it():
    with pytest.raises(ValueError, match="Unknown runtime control.*typo_noise_db"):
        DecoderControlHook({"typo_noise_db": 6.0})


def test_weighted_glottal_table_is_not_misreported_as_scalar_rd_control():
    decoder = _CaptureDecoder()
    decoder.harm_oscillator = SimpleNamespace(R_d_values=torch.tensor([0.3, 0.7, 1.4, 2.7]))

    assert "glottal_rd_scale" not in runtime_capabilities(SimpleNamespace(decoder=decoder))
    hook = DecoderControlHook({"glottal_rd_scale": 1.2})
    with pytest.raises(ValueError, match="glottal_rd_scale"):
        hook.install(SimpleNamespace(decoder=decoder))
