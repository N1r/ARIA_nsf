"""
F1/F2/F3 + bandwidth + spectral tilt extraction.
All outputs are frame-level (hop = 10ms by default).

Main entry: FormantExtractor.extract(wav_path) -> dict of numpy arrays
"""

import numpy as np
import torch
import soundfile as sf
import librosa
import parselmouth
from parselmouth.praat import call
from pathlib import Path


# ─────────────────────────────────────────────
# Formant extraction (Praat via parselmouth)
# ─────────────────────────────────────────────

class FormantExtractor:
    """
    Extract F1-F3 + bandwidths frame-by-frame using Praat Burg method.

    Parameters
    ----------
    hop_ms   : frame hop in ms (default 10ms)
    n_formants: max formants to look for (5 is standard)
    max_freq : ceiling frequency for formant search
               (5500 for female, 5000 for male)
    """

    def __init__(self, hop_ms: float = 10.0, n_formants: int = 5,
                 max_freq: float = 5500.0):
        self.hop_ms    = hop_ms
        self.n_formants = n_formants
        self.max_freq   = max_freq

    def extract(self, audio: np.ndarray, sr: int) -> dict:
        """
        audio : (T,) float32/64
        Returns dict with keys f1, f2, f3, b1, b2, b3 — each shape (N_frames,)
        Values are Hz; 0 where undefined (silent / unvoiced / extraction failed).
        """
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        formant = call(snd, "To Formant (burg)",
                       self.hop_ms / 1000,   # time step (s)
                       self.n_formants,
                       self.max_freq,
                       0.025,                # window length (s)
                       50.0)                 # pre-emphasis from (Hz)

        n_frames = call(formant, "Get number of frames")
        t0       = call(formant, "Get start time")
        dt       = call(formant, "Get time step")

        out = {k: np.zeros(n_frames) for k in
               ("f1","f2","f3","b1","b2","b3","t")}

        for i in range(n_frames):
            t = t0 + i * dt
            out["t"][i] = t
            for fi, fkey, bkey in [(1,"f1","b1"),(2,"f2","b2"),(3,"f3","b3")]:
                fv = call(formant, "Get value at time", fi, t, "Hertz", "Linear")
                bv = call(formant, "Get bandwidth at time", fi, t, "Hertz", "Linear")
                out[fkey][i] = fv if (fv == fv) else 0.0   # nan → 0
                out[bkey][i] = bv if (bv == bv) else 0.0

        return out

    def extract_file(self, wav_path: str) -> dict:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        return self.extract(audio, sr)


# ─────────────────────────────────────────────
# Spectral tilt extraction
# ─────────────────────────────────────────────

def extract_tilt(audio: np.ndarray, sr: int,
                 hop_ms: float = 10.0,
                 n_fft: int = 512,
                 fmin: float = 50.0,
                 fmax: float = 4000.0) -> np.ndarray:
    """
    Estimate per-frame spectral tilt (dB/octave) via linear regression
    on the log mel-spectrogram.

    Returns alpha : (N_frames,) in range roughly [-30, 5] dB/octave.
    Negative = high-frequency roll-off (typical speech).
    """
    hop = int(sr * hop_ms / 1000)
    S   = np.abs(librosa.stft(audio.astype(np.float64),
                               n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask  = (freqs >= fmin) & (freqs <= fmax)
    freqs_log = np.log2(freqs[mask] + 1e-6)

    n_frames = S.shape[1]
    tilt = np.zeros(n_frames)
    for i in range(n_frames):
        s = 20 * np.log10(S[mask, i] + 1e-6)
        if s.std() < 0.5:          # silent frame
            continue
        # linear regression: s ≈ slope * log2(freq) + intercept
        A = np.vstack([freqs_log, np.ones_like(freqs_log)]).T
        slope, _ = np.linalg.lstsq(A, s, rcond=None)[0]
        tilt[i]  = slope            # dB/octave

    return tilt


def extract_tilt_ar(audio: np.ndarray, sr: int,
                    hop_ms: float = 10.0) -> np.ndarray:
    """
    First-order AR coefficient α as a proxy for spectral tilt.
    α ≈ R(1)/R(0) where R is the autocorrelation.
    Maps to filter 1/(1 - α·z⁻¹): α → 1 = dark, α → -1 = bright.
    """
    hop = int(sr * hop_ms / 1000)
    win = int(0.025 * sr)           # 25ms window
    audio_p = np.pad(audio, win // 2)
    n_frames = 1 + (len(audio) - 1) // hop
    alpha = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio_p[i*hop : i*hop + win]
        if len(frame) < win:
            frame = np.pad(frame, (0, win - len(frame)))
        r0 = np.dot(frame, frame)
        r1 = np.dot(frame[:-1], frame[1:])
        alpha[i] = r1 / (r0 + 1e-8)
    return alpha


# ─────────────────────────────────────────────
# Combined per-file extraction
# ─────────────────────────────────────────────

def extract_all(wav_path: str, hop_ms: float = 10.0,
                max_freq: float = 5500.0) -> dict:
    """
    Full extraction: F1-F3, B1-B3, spectral tilt (AR), F0 (via pyworld).

    Returns dict with frame-aligned arrays, all at hop_ms frame rate.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32")
    audio_f64 = audio.astype(np.float64)

    # F0
    import pyworld as pw
    f0, t_f0 = pw.dio(audio_f64, sr, f0_floor=60, f0_ceil=600,
                      frame_period=hop_ms)
    f0 = pw.stonemask(audio_f64, f0, t_f0, sr)

    # Formants
    ext   = FormantExtractor(hop_ms=hop_ms, max_freq=max_freq)
    fmts  = ext.extract(audio, sr)

    # Tilt (AR coefficient)
    alpha = extract_tilt_ar(audio, sr, hop_ms=hop_ms)

    # Align all to shortest length
    n = min(len(f0), len(alpha),
            min(len(v) for v in fmts.values()))

    return {
        "t":     np.arange(n) * hop_ms / 1000,
        "f0":    f0[:n],
        "f1":    fmts["f1"][:n],
        "f2":    fmts["f2"][:n],
        "f3":    fmts["f3"][:n],
        "b1":    fmts["b1"][:n],
        "b2":    fmts["b2"][:n],
        "b3":    fmts["b3"][:n],
        "alpha": alpha[:n],          # spectral tilt AR coef
        "sr":    sr,
    }
