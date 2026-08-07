"""Dependency-free capability and validation rules for DDSP controls.

This module describes the stable, user-facing control surface.  It deliberately
does not import PyTorch or the checkpoint-compatible research engine, so CLI and
metadata tooling can inspect and validate controls in a lightweight environment.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

SUPPORTED_MODELS = ("golf", "ddsp", "aria-golf")
"""Canonical model names understood by the control layer."""

_VARIANT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,254}\Z", re.ASCII)


@dataclass(frozen=True)
class ControlSpec:
    """Capability and numeric bounds for one public control.

    Bounds are inclusive.  ``models`` contains the canonical model names for
    which the control has an implemented execution path.
    """

    name: str
    minimum: float
    maximum: float
    default: float
    models: tuple[str, ...]
    unit: str
    description: str

    @property
    def min_value(self) -> float:
        """Alias for :attr:`minimum`, useful in generic UI code."""

        return self.minimum

    @property
    def max_value(self) -> float:
        """Alias for :attr:`maximum`, useful in generic UI code."""

        return self.maximum

    def supports(self, model: str) -> bool:
        """Return whether this control is available for *model*."""

        return model in self.models

    def normalize(self, value: object) -> float:
        """Return *value* as a finite in-range float.

        ``ValueError`` identifies the control and its accepted inclusive range.
        """

        normalized = _finite_float(value, control=self.name)
        if not self.minimum <= normalized <= self.maximum:
            raise ValueError(
                f"Control {self.name!r} must be between {self.minimum:g} and "
                f"{self.maximum:g} inclusive; got {normalized:g}"
            )
        return normalized


_ALL_CONTROL_SPECS = (
    ControlSpec(
        name="pitch_semitones",
        minimum=-36.0,
        maximum=36.0,
        default=0.0,
        models=SUPPORTED_MODELS,
        unit="semitones",
        description="Voiced-F0 shift; unvoiced frames remain unvoiced.",
    ),
    ControlSpec(
        name="output_gain_db",
        minimum=-24.0,
        maximum=12.0,
        default=0.0,
        models=SUPPORTED_MODELS,
        unit="dB",
        description="Gain applied to the synthesized waveform before output.",
    ),
    ControlSpec(
        name="noise_gain_db",
        minimum=-24.0,
        maximum=24.0,
        default=0.0,
        models=SUPPORTED_MODELS,
        unit="dB",
        description="Gain applied to the stochastic source branch.",
    ),
    ControlSpec(
        name="glottal_rd_scale",
        minimum=0.5,
        maximum=2.0,
        default=1.0,
        models=("golf", "aria-golf"),
        unit="ratio",
        description="Multiplicative shift of the glottal R_d source parameter.",
    ),
    ControlSpec(
        name="f1_scale",
        minimum=0.7,
        maximum=1.3,
        default=1.0,
        models=("aria-golf",),
        unit="ratio",
        description="Multiplicative shift of the first formant frequency.",
    ),
    ControlSpec(
        name="f2_scale",
        minimum=0.7,
        maximum=1.3,
        default=1.0,
        models=("aria-golf",),
        unit="ratio",
        description="Multiplicative shift of the second formant frequency.",
    ),
    ControlSpec(
        name="f1_hz",
        minimum=150.0,
        maximum=1300.0,
        default=500.0,
        models=("aria-golf",),
        unit="Hz",
        description=(
            "Absolute target frequency for the first formant. Overrides the frame's "
            "natural contour entirely, for evenly-spaced Hz stimulus continua; f1_scale "
            "instead preserves the natural shape and shifts it proportionally. Conflicts "
            "with f1_scale."
        ),
    ),
    ControlSpec(
        name="f2_hz",
        minimum=600.0,
        maximum=3200.0,
        default=1500.0,
        models=("aria-golf",),
        unit="Hz",
        description=(
            "Absolute target frequency for the second formant. Overrides the frame's "
            "natural contour entirely, for evenly-spaced Hz stimulus continua; f2_scale "
            "instead preserves the natural shape and shifts it proportionally. Conflicts "
            "with f2_scale."
        ),
    ),
    ControlSpec(
        name="tilt_alpha_delta",
        minimum=-0.25,
        maximum=0.25,
        default=0.0,
        models=("aria-golf",),
        unit="alpha",
        description="Additive change to the ARIS spectral-tilt coefficient.",
    ),
)

CONTROL_SPECS = MappingProxyType({spec.name: spec for spec in _ALL_CONTROL_SPECS})
"""Read-only mapping from public control names to their specifications."""


@dataclass(frozen=True)
class ControlVariant:
    """One named set of control overrides.

    ``name`` is also the output-directory name, so it must be a single safe
    ASCII slug.  ``controls`` is defensively copied and normalized to floats.
    Model-specific support is checked later by :func:`validate_controls`.
    """

    name: str
    controls: dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_variant_name(self.name))
        object.__setattr__(
            self,
            "controls",
            _normalize_variant_controls(self.controls),
        )

    @property
    def slug(self) -> str:
        """Return the safe output-directory name."""

        return self.name


def controls_for_model(model: str) -> tuple[ControlSpec, ...]:
    """Return supported controls for *model* in stable display order.

    Valid model names are ``golf``, ``ddsp``, and ``aria-golf``.
    """

    canonical = _validate_model(model)
    return tuple(spec for spec in _ALL_CONTROL_SPECS if spec.supports(canonical))


def validate_controls(
    model: str,
    mapping: Mapping[str, object],
) -> dict[str, float]:
    """Validate control overrides for *model* and return a normalized copy.

    Only keys supplied by the caller are returned; default-valued controls are
    not inserted.  Unknown controls, controls unavailable for the chosen model,
    non-finite values, and values outside inclusive bounds are rejected.
    """

    canonical = _validate_model(model)
    if not isinstance(mapping, Mapping):
        raise TypeError("controls must be a mapping from control names to numeric values")

    normalized: dict[str, float] = {}
    for name, value in mapping.items():
        spec = _control_spec(name)
        if not spec.supports(canonical):
            available = ", ".join(item.name for item in controls_for_model(canonical))
            raise ValueError(
                f"Control {name!r} is not supported by model {canonical!r}; "
                f"available controls: {available}"
            )
        normalized[name] = spec.normalize(value)
    _check_formant_control_conflicts(normalized)
    return normalized


def parse_variant(text: str) -> ControlVariant:
    """Parse ``NAME:key=value,key=value`` into a :class:`ControlVariant`.

    The name must already be a safe ASCII output-directory slug.  Whitespace
    around comma-separated assignments is ignored.  Duplicate and unknown
    controls, malformed assignments, non-finite numbers, and out-of-range
    values are rejected rather than silently repaired.
    """

    if not isinstance(text, str):
        raise TypeError("variant must be a string in NAME:key=value,key=value format")
    if text.count(":") != 1:
        raise ValueError("Variant must use NAME:key=value,key=value format with exactly one ':'")
    raw_name, raw_assignments = text.split(":", 1)
    name = _validate_variant_name(raw_name.strip())
    if not raw_assignments.strip():
        raise ValueError(f"Variant {name!r} must define at least one control")

    controls: dict[str, float] = {}
    for raw_assignment in raw_assignments.split(","):
        assignment = raw_assignment.strip()
        if not assignment or assignment.count("=") != 1:
            raise ValueError(
                f"Invalid control assignment {raw_assignment!r} in variant {name!r}; "
                "expected key=value"
            )
        raw_control, raw_value = assignment.split("=", 1)
        control = raw_control.strip()
        if not control:
            raise ValueError(f"Variant {name!r} contains an empty control name")
        if control in controls:
            raise ValueError(f"Variant {name!r} defines control {control!r} more than once")
        spec = _control_spec(control)
        value_text = raw_value.strip()
        if not value_text:
            raise ValueError(f"Variant {name!r} has no value for control {control!r}")
        controls[control] = spec.normalize(value_text)

    return ControlVariant(name=name, controls=controls)


def pitch_variant(
    semitones: object,
    *,
    name: Optional[str] = None,
) -> ControlVariant:
    """Build one pitch-only variant.

    The default name matches the established output convention, for example
    ``pitch_plus_4st``, ``pitch_minus_3p5st``, or ``pitch_zero_0st``.
    """

    value = CONTROL_SPECS["pitch_semitones"].normalize(semitones)
    variant_name = _pitch_variant_name(value) if name is None else name
    return ControlVariant(
        name=variant_name,
        controls={"pitch_semitones": value},
    )


def pitch_variants(semitones: Iterable[object]) -> tuple[ControlVariant, ...]:
    """Build a stable tuple of pitch-only variants from *semitones*.

    Empty iterables are accepted so callers can combine an optional pitch sweep
    with separately parsed variants.  Duplicate output names are rejected.
    """

    if isinstance(semitones, (str, bytes)):
        raise TypeError("semitones must be an iterable of numeric values, not text")
    try:
        values = iter(semitones)
    except TypeError as error:
        raise TypeError("semitones must be an iterable of numeric values") from error
    variants = tuple(pitch_variant(value) for value in values)

    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError(
            "Pitch values must produce unique output names; values are named to two decimal places"
        )
    return variants


_FORMANT_ABSOLUTE_VS_RATIO = {
    "f1_hz": "f1_scale",
    "f2_hz": "f2_scale",
}


def _check_formant_control_conflicts(controls: Mapping[str, float]) -> None:
    """Reject an absolute Hz target combined with a ratio scale on the same formant.

    Mirrors the f0_scale-vs-pitch_semitones conflict check in ``experiment.synthesize``:
    the two controls express incompatible intents (overriding the frame's contour vs.
    preserving its natural shape), so only one may be given per formant.
    """

    for absolute, ratio in _FORMANT_ABSOLUTE_VS_RATIO.items():
        if absolute in controls and ratio in controls:
            raise ValueError(
                f"Conflicting formant controls: {absolute!r} and {ratio!r} cannot both be "
                "set; choose an absolute target or a relative scale, not both"
            )


def _validate_model(model: object) -> str:
    if not isinstance(model, str):
        raise TypeError("model must be one of: " + ", ".join(SUPPORTED_MODELS))
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {', '.join(SUPPORTED_MODELS)}")
    return model


def _validate_variant_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("variant name must be a string")
    if _VARIANT_NAME_RE.fullmatch(name) is None:
        raise ValueError(
            "Variant name must be a safe ASCII slug: 1-255 characters, starting "
            "with a letter or digit and containing only letters, digits, '_' or '-'"
        )
    return name


def _control_spec(name: object) -> ControlSpec:
    if not isinstance(name, str):
        raise ValueError(f"Control names must be strings; got {name!r}")
    try:
        return CONTROL_SPECS[name]
    except KeyError:
        known = ", ".join(CONTROL_SPECS)
        raise ValueError(f"Unknown control {name!r}; choose from {known}") from None


def _normalize_variant_controls(controls: object) -> dict[str, float]:
    if not isinstance(controls, Mapping):
        raise TypeError("variant controls must be a mapping")
    normalized: dict[str, float] = {}
    for name, value in controls.items():
        spec = _control_spec(name)
        normalized[name] = spec.normalize(value)
    return normalized


def _finite_float(value: object, *, control: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Control {control!r} must be numeric, not a boolean")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Control {control!r} must be a finite number; got {value!r}") from error
    if not math.isfinite(normalized):
        raise ValueError(f"Control {control!r} must be finite; got {value!r}")
    return normalized


def _pitch_variant_name(semitones: float) -> str:
    magnitude = f"{abs(semitones):.2f}".rstrip("0").rstrip(".").replace(".", "p")
    sign = "plus" if semitones > 0 else "minus" if semitones < 0 else "zero"
    return f"pitch_{sign}_{magnitude}st"


__all__ = [
    "CONTROL_SPECS",
    "SUPPORTED_MODELS",
    "ControlSpec",
    "ControlVariant",
    "controls_for_model",
    "parse_variant",
    "pitch_variant",
    "pitch_variants",
    "validate_controls",
]
