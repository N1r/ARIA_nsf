"""Public, dependency-free DDSP control specifications and validation."""

from .specs import (
    CONTROL_SPECS,
    SUPPORTED_MODELS,
    ControlSpec,
    ControlVariant,
    controls_for_model,
    parse_variant,
    pitch_variant,
    pitch_variants,
    validate_controls,
)

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
