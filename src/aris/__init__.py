"""ARIS: reproducible voice analysis-by-synthesis workflows."""

from .controls import (
    CONTROL_SPECS,
    ControlSpec,
    ControlVariant,
    controls_for_model,
    parse_variant,
)
from .corpus import CorpusResult, acquire_cmu_arctic
from .experiment import create_experiment
from .manifest import (
    DatasetManifest,
    DatasetRecord,
    prepare_dataset,
    validate_manifest,
)
from .manipulation import manipulate_controls, manipulate_pitch
from .segment import split_audio

__all__ = [
    "CorpusResult",
    "CONTROL_SPECS",
    "ControlSpec",
    "ControlVariant",
    "DatasetManifest",
    "DatasetRecord",
    "acquire_cmu_arctic",
    "create_experiment",
    "controls_for_model",
    "manipulate_controls",
    "manipulate_pitch",
    "parse_variant",
    "prepare_dataset",
    "split_audio",
    "validate_manifest",
]
__version__ = "0.1.0"
