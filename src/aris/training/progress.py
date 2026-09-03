"""Compact training status output, importable without the full audio stack."""

from __future__ import annotations

from typing import Any, Optional

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:  # Lightweight documentation/test environments.
    class Callback:  # type: ignore[no-redef]
        """Fallback base; production training installs Lightning."""


class TrainingProgressPrinter(Callback):
    """Print compact step, epoch, and loss updates for non-interactive runs."""

    def __init__(self, every_n_steps: int = 50) -> None:
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        self.every_n_steps = every_n_steps

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        total = trainer.max_steps
        if step != 1 and step % self.every_n_steps != 0 and step != total:
            return
        loss = _metric_number(trainer.callback_metrics.get("train_loss"))
        loss_text = f"{loss:.4f}" if loss is not None else "n/a"
        components = []
        for key, label in (
            ("train_spectral_loss", "spectral"),
            ("train_formant_loss", "formant"),
            ("train_residual_reg", "residual"),
        ):
            value = _metric_number(trainer.callback_metrics.get(key))
            if value is not None:
                components.append(f"{label} {value:.4f}")
        detail = " | " + " | ".join(components) if components else ""
        print(
            f"[train] step {step}/{total} | epoch {trainer.current_epoch + 1} | "
            f"loss {loss_text}{detail}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        loss = _metric_number(trainer.callback_metrics.get("val_loss"))
        if loss is not None:
            print(
                f"[valid] step {trainer.global_step}/{trainer.max_steps} | "
                f"epoch {trainer.current_epoch + 1} | loss {loss:.4f}",
                flush=True,
            )


def _metric_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value.detach().cpu())
    except AttributeError:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
