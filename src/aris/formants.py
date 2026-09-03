"""Praat-based formant targets for interpretable ARIA training."""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_formants(
    audio: np.ndarray,
    sample_rate: int,
    *,
    hop_seconds: float = 0.010,
    max_formant_hz: float = 5500.0,
) -> dict[str, np.ndarray]:
    """Return Praat Burg F1--F3 and bandwidth tracks on a fixed frame grid.

    Undefined estimates are represented by zero.  The returned ``time_s``
    array records Praat's actual frame centres so downstream segmentation can
    align targets without assuming that the first analysis frame starts at 0.
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError as error:
        raise ImportError(
            "Formant extraction requires the phonetics extra: "
            "uv sync --extra phonetics"
        ) from error

    if audio.ndim != 1:
        raise ValueError("Formant extraction expects mono audio")
    if not 0 < hop_seconds <= 0.025:
        raise ValueError("Formant hop must be in (0, 0.025] seconds")
    if max_formant_hz <= 0 or max_formant_hz >= sample_rate / 2:
        raise ValueError("Formant ceiling must be positive and below Nyquist")

    sound = parselmouth.Sound(
        audio.astype(np.float64, copy=False), sampling_frequency=sample_rate
    )
    formant = call(
        sound,
        "To Formant (burg)",
        hop_seconds,
        5,
        max_formant_hz,
        0.025,
        50.0,
    )
    frame_count = int(call(formant, "Get number of frames"))
    times = np.asarray(formant.xs(), dtype=np.float64)
    if len(times) != frame_count:
        raise RuntimeError("Praat returned inconsistent formant frame metadata")

    result: dict[str, Any] = {"time_s": times.astype(np.float32)}
    for number in range(1, 4):
        frequencies = np.zeros(frame_count, dtype=np.float32)
        bandwidths = np.zeros(frame_count, dtype=np.float32)
        for index, time_s in enumerate(times):
            frequency = call(
                formant, "Get value at time", number, float(time_s), "Hertz", "Linear"
            )
            bandwidth = call(
                formant, "Get bandwidth at time", number, float(time_s), "Hertz", "Linear"
            )
            if np.isfinite(frequency) and frequency > 0:
                frequencies[index] = frequency
            if np.isfinite(bandwidth) and bandwidth > 0:
                bandwidths[index] = bandwidth
        result[f"f{number}"] = frequencies
        result[f"b{number}"] = bandwidths
    return result
