"""PhonLab-DDSP: reproducible voice analysis-by-synthesis workflows."""

from .controls import (
    CONTROL_SPECS,
    ControlSpec,
    ControlVariant,
    controls_for_model,
    parse_variant,
)
from .corpus import CorpusResult, acquire_cmu_arctic
from .experiment import create_experiment
from .jobs import JobStatus, SlurmBackend
from .manifest import (
    DatasetManifest,
    DatasetRecord,
    prepare_dataset,
    validate_manifest,
)
from .manipulation import manipulate_controls, manipulate_pitch
from .metrics import build_metrics_dashboard, load_metrics
from .parameters import export_parameters, parameter_summary
from .report import build_report
from .segment import split_audio

__all__ = [
    "CorpusResult",
    "CONTROL_SPECS",
    "ControlSpec",
    "ControlVariant",
    "DatasetManifest",
    "DatasetRecord",
    "JobStatus",
    "SlurmBackend",
    "acquire_cmu_arctic",
    "build_metrics_dashboard",
    "build_report",
    "create_experiment",
    "controls_for_model",
    "export_parameters",
    "load_metrics",
    "manipulate_controls",
    "manipulate_pitch",
    "parse_variant",
    "parameter_summary",
    "prepare_dataset",
    "split_audio",
    "validate_manifest",
]
__version__ = "0.1.0"
