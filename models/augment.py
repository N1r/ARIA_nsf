"""
Online data augmentation for single-speaker speech synthesis.

Three augmentations:
  1. F0 scaling       — scale voiced F0 by random factor
  2. Spectral tilt    — apply first-order emphasis/de-emphasis
  3. Amplitude scale  — multiply waveform by random gain
"""

import numpy as np
from scipy.signal import lfilter


def augment_f0(f0: np.ndarray, scale: float) -> np.ndarray:
    """Scale voiced F0 frames by `scale`, keep unvoiced as 0."""
    return np.where(f0 > 0, f0 * scale, 0.0).astype(np.float32)


def apply_tilt(audio: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply spectral tilt via first-order filter.

    alpha > 0 : de-emphasis  H(z) = 1/(1 - α·z⁻¹)  → boosts low freq (darker)
    alpha < 0 : pre-emphasis H(z) = 1 - |α|·z⁻¹     → boosts high freq (brighter)

    |alpha| should be small (< 0.15) for subtle augmentation.
    """
    if abs(alpha) < 0.005:
        return audio

    if alpha > 0:
        # de-emphasis: IIR all-pole  y[n] = x[n] + alpha * y[n-1]
        # lfilter: b=[1], a=[1, -alpha]
        return lfilter([1.0], [1.0, -alpha], audio).astype(np.float32)
    else:
        # pre-emphasis: FIR  y[n] = x[n] + alpha * x[n-1]  (alpha is negative)
        # = x[n] - |alpha| * x[n-1]
        return lfilter([1.0, alpha], [1.0], audio).astype(np.float32)


class SpeechAugment:
    """
    Callable augmentation pipeline for (audio, f0) pairs.

    Parameters
    ----------
    f0_scale_range    : (min, max) scale factor for F0
    tilt_alpha_range  : (min, max) alpha for spectral tilt filter
    amp_scale_range   : (min, max) amplitude scale factor
    p_f0_scale        : probability of applying F0 augmentation
    p_tilt            : probability of applying tilt augmentation
    p_amp             : probability of applying amplitude augmentation
    """

    def __init__(
        self,
        f0_scale_range:   tuple = (0.75, 1.25),
        tilt_alpha_range: tuple = (-0.12, 0.12),
        amp_scale_range:  tuple = (0.6,  1.0),
        p_f0_scale:       float = 0.8,
        p_tilt:           float = 0.6,
        p_amp:            float = 0.5,
        rng:              np.random.Generator = None,
    ):
        self.f0_scale_range   = f0_scale_range
        self.tilt_alpha_range = tilt_alpha_range
        self.amp_scale_range  = amp_scale_range
        self.p_f0_scale       = p_f0_scale
        self.p_tilt           = p_tilt
        self.p_amp            = p_amp
        self.rng = rng or np.random.default_rng()

    def __call__(self, audio: np.ndarray, f0: np.ndarray) -> tuple:
        """
        audio : (T,) float32
        f0    : (T,) float32 Hz, 0=unvoiced
        Returns augmented (audio, f0).
        """
        # 1. Amplitude
        if self.rng.random() < self.p_amp:
            scale = self.rng.uniform(*self.amp_scale_range)
            audio = audio * scale

        # 2. Spectral tilt
        if self.rng.random() < self.p_tilt:
            alpha = self.rng.uniform(*self.tilt_alpha_range)
            audio = apply_tilt(audio, alpha)

        # 3. F0 scale (only affects conditioning, not audio)
        if self.rng.random() < self.p_f0_scale:
            scale = self.rng.uniform(*self.f0_scale_range)
            f0 = augment_f0(f0, scale)

        # clip to [-1, 1] after tilt filter might amplify signal
        audio = np.clip(audio, -1.0, 1.0)

        return audio, f0

    def describe(self) -> str:
        return (f"SpeechAugment(f0_scale={self.f0_scale_range}, "
                f"tilt_alpha={self.tilt_alpha_range}, "
                f"amp_scale={self.amp_scale_range})")
