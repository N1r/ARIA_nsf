"""ARIS: reproducible voice analysis-by-synthesis workflows."""

# Defined before the submodule imports below: manipulation.py and
# controls/lightning.py import this back out of the (still-initializing)
# package to stamp their metadata, which only resolves if it is set first.
__version__ = "0.1.0"

from .controls import (
    CONTROL_SPECS,
    ControlSpec,
    ControlVariant,
    controls_for_model,
    parse_variant,
)
from .corpus import CorpusResult, acquire_cmu_arctic
from .cuda_env import auto_configure_cuda, get_cuda_info
from .doctor import Check
from .doctor import checks as doctor
from .experiment import create_experiment, synthesize, train
from .manifest import (
    DatasetManifest,
    DatasetRecord,
    prepare_dataset,
    validate_manifest,
)
from .manipulation import manipulate_controls, manipulate_pitch
from .segment import split_audio

# High-level friendly aliases for interactive and programmatic Python usage:
split = split_audio
prepare = prepare_dataset
validate = validate_manifest
init_experiment = create_experiment
manipulate = manipulate_controls


def launch_studio(*args, **kwargs):
    """Launch the interactive Gradio Studio web workspace."""
    from .studio import launch_studio as _launch_studio

    return _launch_studio(*args, **kwargs)


__all__ = [
    "CorpusResult",
    "Check",
    "CONTROL_SPECS",
    "ControlSpec",
    "ControlVariant",
    "DatasetManifest",
    "DatasetRecord",
    "acquire_cmu_arctic",
    "auto_configure_cuda",
    "create_experiment",
    "controls_for_model",
    "doctor",
    "get_cuda_info",
    "init_experiment",
    "launch_studio",
    "manipulate",
    "manipulate_controls",
    "manipulate_pitch",
    "parse_variant",
    "prepare",
    "prepare_dataset",
    "split",
    "split_audio",
    "synthesize",
    "train",
    "validate",
    "validate_manifest",
]
