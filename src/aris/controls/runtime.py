"""Runtime transformations for auditable DDSP controls.

This module deliberately uses duck typing instead of importing the frozen
research engine.  The stable library can therefore add manipulation features
without changing the checkpoint-compatible ``aris.models`` and ``aris.training`` packages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

_LOG_10_OVER_20 = math.log(10.0) / 20.0

# Discrepancy tolerance between a requested formant ratio and the ratio actually
# achieved after scale_formants clamps to the model's static frequency range.
_FORMANT_CLAMP_TOLERANCE = 0.01

# f1_hz/f2_hz write an absolute target frequency; unlike the ratio controls
# there is no numeric value that means "no-op", so presence alone makes them
# active rather than comparing against a default.
_ABSOLUTE_FORMANT_CONTROLS = frozenset({"f1_hz", "f2_hz"})


def active_runtime_controls(controls: Mapping[str, float]) -> dict[str, float]:
    """Return non-default controls that must be applied inside prediction."""
    defaults = {
        "output_gain_db": 0.0,
        "noise_gain_db": 0.0,
        "glottal_rd_scale": 1.0,
        "f1_scale": 1.0,
        "f2_scale": 1.0,
        "tilt_alpha_delta": 0.0,
    }
    unknown = sorted(set(controls) - set(defaults) - _ABSOLUTE_FORMANT_CONTROLS)
    if unknown:
        raise ValueError("Unknown runtime control(s): " + ", ".join(unknown))
    active = {
        name: float(value)
        for name, value in controls.items()
        if name in defaults and not math.isclose(float(value), defaults[name], abs_tol=1e-12)
    }
    active.update(
        (name, float(value))
        for name, value in controls.items()
        if name in _ABSOLUTE_FORMANT_CONTROLS
    )
    return active


def runtime_capabilities(pl_module: Any) -> set[str]:
    """Inspect a loaded model and report controls that have a real execution path."""
    capabilities = {"output_gain_db"}
    decoder = getattr(pl_module, "decoder", None)
    if decoder is None:
        return capabilities

    noise_filter = getattr(decoder, "noise_filter", None)
    if noise_filter is not None and getattr(noise_filter, "mag_activation", "exp") == "exp":
        capabilities.add("noise_gain_db")

    oscillator = getattr(decoder, "harm_oscillator", None)
    if _has_indexed_rd_control(oscillator):
        capabilities.add("glottal_rd_scale")

    end_filter = getattr(decoder, "end_filter", None)
    if callable(getattr(end_filter, "scale_formants", None)):
        capabilities.update({"f1_scale", "f2_scale"})
        if bool(getattr(end_filter, "use_tilt", False)):
            capabilities.add("tilt_alpha_delta")
    if callable(getattr(end_filter, "set_formant_params", None)):
        capabilities.update({"f1_hz", "f2_hz"})
    return capabilities


class DecoderControlHook:
    """Apply source/filter controls immediately before the decoder forward pass."""

    def __init__(self, controls: Mapping[str, float]):
        self.controls = active_runtime_controls(controls)
        self.handle = None
        self.calls = 0
        self.capabilities: set[str] = set()
        # Per-formant achieved-vs-requested discrepancy, accumulated across all
        # decoder calls; see formant_tracking_summary() for the reported shape.
        self.formant_tracking: dict[str, dict[str, float]] = {}

    @property
    def decoder_controls(self) -> dict[str, float]:
        return {name: value for name, value in self.controls.items() if name != "output_gain_db"}

    def install(self, pl_module: Any) -> None:
        self.capabilities = runtime_capabilities(pl_module)
        unavailable = sorted(set(self.controls) - self.capabilities)
        if unavailable:
            raise ValueError(
                "Loaded checkpoint/model cannot apply control(s): "
                + ", ".join(unavailable)
                + ". Runtime capabilities: "
                + ", ".join(sorted(self.capabilities))
            )
        if not self.decoder_controls:
            return
        decoder = getattr(pl_module, "decoder", None)
        if decoder is None:
            raise ValueError("Loaded model has no decoder control surface")
        self.handle = decoder.register_forward_pre_hook(self._apply, with_kwargs=True)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def verify(self) -> None:
        """Raise if requested decoder controls were never actually applied."""
        if self.decoder_controls and self.calls == 0:
            raise RuntimeError("Decoder controls were requested but the decoder hook never ran")

    def formant_tracking_summary(self) -> dict[str, dict[str, float]]:
        """Achieved-vs-requested formant-scale discrepancy, for metadata only.

        scale_formants clamps its per-frame Hz target to the model's static
        frequency range with no record of it; this reports, per formant, the
        worst absolute Hz miss and the fraction of frames the clamp touched.
        """
        summary: dict[str, dict[str, float]] = {}
        for key, stats in self.formant_tracking.items():
            frames_total = stats["frames_total"]
            summary[key] = {
                "max_abs_hz_discrepancy": stats["max_abs_hz_discrepancy"],
                "clamped_frame_fraction": (
                    stats["clamped_frames"] / frames_total if frames_total else 0.0
                ),
            }
        return summary

    def _apply(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        changed = dict(kwargs)
        if "noise_gain_db" in self.decoder_controls:
            changed["noise_filter_params"] = _scale_noise(
                changed.get("noise_filter_params"),
                self.decoder_controls["noise_gain_db"],
            )
        if "glottal_rd_scale" in self.decoder_controls:
            changed["harm_oscillator_params"] = _scale_glottal_rd(
                changed.get("harm_oscillator_params"),
                self.decoder_controls["glottal_rd_scale"],
                getattr(module, "harm_oscillator", None),
            )
        formant_names = {
            "f1_scale",
            "f2_scale",
            "tilt_alpha_delta",
            "f1_hz",
            "f2_hz",
        } & self.decoder_controls.keys()
        if formant_names:
            end_filter = getattr(module, "end_filter", None)
            end_filter_value = changed.get("end_filter_params")
            changed["end_filter_params"] = _scale_formants(
                end_filter_value,
                end_filter,
                self.decoder_controls,
            )
            self._record_formant_tracking(
                end_filter_value, changed["end_filter_params"], end_filter
            )
        self.calls += 1
        return args, changed

    def _record_formant_tracking(
        self, before_value: Any, after_value: Any, end_filter: Any
    ) -> None:
        ratio_controls = {"f1_scale", "f2_scale"} & self.decoder_controls.keys()
        get_formant_params = getattr(end_filter, "get_formant_params", None)
        if not ratio_controls or not callable(get_formant_params):
            return
        before_ctrl, _ = _first_parameter(before_value, "end_filter_params")
        after_ctrl, _ = _first_parameter(after_value, "end_filter_params")
        before_raw = (
            before_ctrl.as_tensor()
            if callable(getattr(before_ctrl, "as_tensor", None))
            else before_ctrl
        )
        after_raw = (
            after_ctrl.as_tensor()
            if callable(getattr(after_ctrl, "as_tensor", None))
            else after_ctrl
        )
        before_params = get_formant_params(before_raw)
        after_params = get_formant_params(after_raw)
        for index, scale_name in ((1, "f1_scale"), (2, "f2_scale")):
            if scale_name not in ratio_controls:
                continue
            key = f"f{index}"
            if key not in before_params or key not in after_params:
                continue
            requested_scale = self.decoder_controls[scale_name]
            original_hz = before_params[key]
            achieved_hz = after_params[key]
            requested_hz = original_hz * requested_scale
            achieved_ratio = achieved_hz / original_hz
            discrepancy = (achieved_hz - requested_hz).abs()
            clamped = (achieved_ratio - requested_scale).abs() > _FORMANT_CLAMP_TOLERANCE
            running = self.formant_tracking.setdefault(
                key, {"max_abs_hz_discrepancy": 0.0, "clamped_frames": 0, "frames_total": 0}
            )
            running["max_abs_hz_discrepancy"] = max(
                running["max_abs_hz_discrepancy"],
                float(discrepancy.max().detach().cpu()),
            )
            running["clamped_frames"] += int(clamped.sum().detach().cpu())
            running["frames_total"] += int(clamped.numel())


def _first_parameter(value: Any, name: str) -> tuple[Any, tuple[Any, ...]]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"Decoder did not provide {name}")
    return value[0], tuple(value[1:])


def _scale_noise(value: Any, gain_db: float):
    log_magnitude, rest = _first_parameter(value, "noise_filter_params")
    scaled = log_magnitude + float(gain_db) * _LOG_10_OVER_20
    return (scaled, *rest)


def _scale_glottal_rd(value: Any, scale: float, oscillator: Any):
    weight, rest = _first_parameter(value, "harm_oscillator_params")
    if not _has_indexed_rd_control(oscillator):
        raise ValueError(
            "Decoder oscillator does not expose a scalar indexed R_d control; "
            "weighted table mixtures cannot be shifted as an R_d coordinate"
        )
    rd_values = getattr(oscillator, "R_d_values", None)
    minimum = float(rd_values[0].detach().cpu())
    maximum = float(rd_values[-1].detach().cpu())
    if not (0 < minimum < maximum):
        raise ValueError("Decoder oscillator has an invalid R_d table")
    normalized_delta = math.log(float(scale)) / math.log(maximum / minimum)
    raw = weight.as_tensor() if callable(getattr(weight, "as_tensor", None)) else weight
    if not isinstance(raw, torch.Tensor) or raw.ndim != 2:
        raise ValueError(
            "Scalar indexed R_d control must be a two-dimensional batch-by-frame tensor"
        )
    shifted = torch.clamp(raw + normalized_delta, 0.0, 1.0)
    if callable(getattr(weight, "new_tensor", None)):
        shifted = weight.new_tensor(shifted)
    return (shifted, *rest)


def _scale_formants(value: Any, end_filter: Any, controls: Mapping[str, float]):
    ctrl, rest = _first_parameter(value, "end_filter_params")
    raw = ctrl.as_tensor() if callable(getattr(ctrl, "as_tensor", None)) else ctrl
    ratio_names = {"f1_scale", "f2_scale", "tilt_alpha_delta"} & controls.keys()
    if ratio_names:
        scale_formants = getattr(end_filter, "scale_formants", None)
        if not callable(scale_formants):
            raise ValueError("Decoder end filter is not formant-controllable")
        raw = scale_formants(
            raw,
            f1_scale=float(controls.get("f1_scale", 1.0)),
            f2_scale=float(controls.get("f2_scale", 1.0)),
            alpha_delta=float(controls.get("tilt_alpha_delta", 0.0)),
        )
    hz_names = {"f1_hz", "f2_hz"} & controls.keys()
    if hz_names:
        set_formant_params = getattr(end_filter, "set_formant_params", None)
        if not callable(set_formant_params):
            raise ValueError("Decoder end filter does not support absolute formant targets")
        raw = set_formant_params(
            raw,
            f1=controls.get("f1_hz"),
            f2=controls.get("f2_hz"),
        )
    changed = raw
    if callable(getattr(ctrl, "new_tensor", None)):
        changed = ctrl.new_tensor(changed)
    return (changed, *rest)


def _has_indexed_rd_control(oscillator: Any) -> bool:
    """Recognize the scalar-index table protocol used by frozen GOLF presets.

    Merely exposing ``R_d_values`` is insufficient: weighted table oscillators
    use an entire probability vector and require a different transformation.
    Frozen indexed oscillators expose ``equal_energy`` and ``oversampling``;
    adapters may instead opt in explicitly with ``rd_control_mode="indexed"``.
    """

    rd_values = getattr(oscillator, "R_d_values", None)
    if rd_values is None or getattr(rd_values, "numel", lambda: 0)() < 2:
        return False
    if getattr(oscillator, "rd_control_mode", None) == "indexed":
        return True
    return hasattr(oscillator, "equal_energy") and hasattr(oscillator, "oversampling")
