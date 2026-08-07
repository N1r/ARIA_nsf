"""Lightning prediction callback that renders and audits controlled audio."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torchaudio
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from .. import __version__
from .runtime import DecoderControlHook
from .specs import CONTROL_SPECS

# Pitch is excluded: it is applied through F0 conditioning, not at render time.
_RUNTIME_CONTROL_NAMES = frozenset(CONTROL_SPECS) - {"pitch_semitones"}

# Individually-in-range controls can combine to clip a rendered file well past
# what any single control's own range would suggest; warn rather than fail so
# a phonetician who wants a scaled-back result is still told about it.
_CLIPPING_WARNING_THRESHOLD = 0.001


class ControlledPredictionWriter(BasePredictionWriter):
    """Apply model controls, write PCM16 WAV files, and record render diagnostics."""

    def __init__(
        self,
        output_dir: str,
        controls_json: Optional[dict[str, float]] = None,
    ):
        super().__init__("batch")
        self.output_dir = Path(output_dir)
        parsed = {} if controls_json is None else controls_json
        if not isinstance(parsed, dict):
            raise ValueError("controls_json must be an object")
        unknown = sorted(
            str(name)
            for name in parsed
            if not isinstance(name, str) or name not in _RUNTIME_CONTROL_NAMES
        )
        if unknown:
            raise ValueError(
                "Unknown or non-runtime control(s): "
                + ", ".join(unknown)
                + ". Pitch must be applied through predict_f0_scale."
            )
        self.controls = {
            name: CONTROL_SPECS[name].normalize(value) for name, value in parsed.items()
        }
        if any(not math.isfinite(value) for value in self.controls.values()):
            raise ValueError("All controls must be finite")
        # parents=False: the parent is created by the caller, so a bad path
        # fails immediately instead of silently building a directory tree.
        self.output_dir.mkdir(parents=False, exist_ok=True)
        self.control_hook = DecoderControlHook(self.controls)
        self.files: list[dict[str, Any]] = []

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer
        self.control_hook.install(pl_module)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        del trainer, batch_indices, batch_idx, dataloader_idx
        *_, rel_path = batch
        predicted, _ = prediction
        waveform = predicted.as_tensor()
        gain_db = float(self.controls.get("output_gain_db", 0.0))
        scale = 10.0 ** (gain_db / 20.0)
        scaled = waveform * scale
        clipped_samples = int((scaled.abs() > 1.0).sum().detach().cpu())
        output_waveform = torch.clamp(scaled, -1.0, 1.0)
        relative = str(rel_path[0])
        out_path = self.output_dir / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            out_path,
            output_waveform.cpu(),
            sample_rate=pl_module.sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        samples = int(waveform.numel())
        self.files.append(
            {
                "path": relative,
                "peak_before_gain": float(waveform.abs().max().detach().cpu()),
                "peak_after_gain_unclipped": float(scaled.abs().max().detach().cpu()),
                "clipped_samples": clipped_samples,
                "samples": samples,
            }
        )
        clipped_fraction = clipped_samples / samples if samples else 0.0
        if clipped_fraction > _CLIPPING_WARNING_THRESHOLD:
            print(
                f"aris: WARNING {relative} clipped {clipped_fraction:.4%} of samples "
                f"(threshold {_CLIPPING_WARNING_THRESHOLD:.2%}); requested controls "
                "combined to exceed full scale",
                file=sys.stderr,
            )

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        try:
            self.control_hook.verify()
            clipped = sum(item["clipped_samples"] for item in self.files)
            samples = sum(item["samples"] for item in self.files)
            metadata = {
                "schema_version": "1.0",
                "aris_version": __version__,
                "controls": self.controls,
                "runtime_capabilities": sorted(self.control_hook.capabilities),
                "decoder_control_calls": self.control_hook.calls,
                "files_written": len(self.files),
                "clipped_samples": clipped,
                "samples": samples,
                "clipped_fraction": clipped / samples if samples else 0.0,
                "formant_tracking": self.control_hook.formant_tracking_summary(),
                "files": self.files,
            }
            (self.output_dir / "_render.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        finally:
            self.control_hook.remove()
