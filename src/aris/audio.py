"""Small, deterministic audio utilities with an optional SoundFile backend."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

SUPPORTED_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".ogg"}


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Read audio as mono float32 in [-1, 1].

    SoundFile is used when installed. A standard-library WAV reader keeps the
    basic preparation and test workflow usable in minimal environments.
    """
    path = Path(path)
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return audio.mean(axis=1, dtype=np.float32), int(sample_rate)
    except ImportError as error:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(
                f"{path.suffix} input requires the 'audio' extra: pip install 'aris[audio]'"
            ) from error

    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        width = stream.getsampwidth()
        frames = stream.readframes(stream.getnframes())

    if width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float32) / 8388608
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"Unsupported PCM width ({width} bytes) in {path}")

    samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return samples, int(sample_rate)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono 16-bit PCM WAV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    pcm = np.round(np.clip(clean, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(int(sample_rate))
        stream.writeframes(pcm.tobytes())


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample deterministically; use polyphase filtering when SciPy is present."""
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_rate, target_rate)
        result = resample_poly(audio, target_rate // divisor, source_rate // divisor)
    except ImportError:
        # Linear interpolation is a compatibility fallback, not the recommended
        # path for measurements near the Nyquist frequency.
        output_length = int(round(len(audio) * target_rate / source_rate))
        old_positions = np.arange(len(audio), dtype=np.float64)
        new_positions = np.arange(output_length, dtype=np.float64) * source_rate / target_rate
        result = np.interp(new_positions, old_positions, audio)
    return np.asarray(result, dtype=np.float32)


def audio_statistics(audio: np.ndarray, sample_rate: int) -> dict:
    """Compute duration, peak, RMS level, clipping fraction, and DC offset."""
    audio = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(audio), initial=0.0))
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    clipping = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0
    dc = float(np.mean(audio)) if audio.size else 0.0
    return {
        "duration_s": float(len(audio) / sample_rate),
        "peak": peak,
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-12))),
        "clipped_fraction": clipping,
        "dc_offset": dc,
    }


def estimate_f0(
    audio: np.ndarray,
    sample_rate: int,
    floor_hz: float,
    ceiling_hz: float,
    method: str = "auto",
    hop_s: float = 0.005,
) -> tuple[np.ndarray, str]:
    """Estimate F0 and return ``(track_hz, backend_name)``.

    WORLD is preferred for experiment preparation. The autocorrelation backend
    is intentionally simple and is suitable for smoke tests or initial surveys.
    """
    if method in {"auto", "pyworld"}:
        try:
            try:
                import pyworld
            except ModuleNotFoundError as err:
                if "pkg_resources" in str(err):
                    import sys
                    import types

                    fake_pkg = types.ModuleType("pkg_resources")
                    fake_pkg.get_distribution = lambda name: types.SimpleNamespace(version="0.3.5")
                    sys.modules["pkg_resources"] = fake_pkg
                    import pyworld
                else:
                    raise

            x = np.asarray(audio, dtype=np.float64)
            f0, time_axis = pyworld.dio(
                x,
                sample_rate,
                f0_floor=float(floor_hz),
                f0_ceil=float(ceiling_hz),
                frame_period=hop_s * 1000,
            )
            refined = pyworld.stonemask(x, f0, time_axis, sample_rate)
            return refined.astype(np.float32), "pyworld-dio+stonemask"
        except ImportError as error:
            if method == "pyworld":
                raise RuntimeError("F0 method 'pyworld' requires: uv sync --extra world") from error
    if method not in {"auto", "autocorr"}:
        raise ValueError(f"Unknown F0 method: {method}")
    return _autocorrelation_f0(audio, sample_rate, floor_hz, ceiling_hz, hop_s), "autocorr"


def _autocorrelation_f0(audio, sample_rate, floor_hz, ceiling_hz, hop_s):
    audio = np.asarray(audio, dtype=np.float64)
    hop = max(1, int(round(sample_rate * hop_s)))
    frame_length = max(hop * 2, int(round(sample_rate * 0.04)))
    min_lag = max(1, int(sample_rate / ceiling_hz))
    max_lag = min(frame_length - 2, int(sample_rate / floor_hz))
    count = max(1, 1 + math.ceil(max(0, len(audio) - frame_length) / hop))
    f0 = np.zeros(count, dtype=np.float32)
    window = np.hanning(frame_length)
    required = (count - 1) * hop + frame_length
    padded = np.pad(audio, (0, max(0, required - len(audio))))
    frame_view = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop]
    fft_size = 1 << (2 * frame_length - 1).bit_length()
    chunk_size = 512
    for first in range(0, count, chunk_size):
        frames = np.array(frame_view[first : first + chunk_size], copy=True)
        frames = (frames - frames.mean(axis=1, keepdims=True)) * window
        spectra = np.fft.rfft(frames, n=fft_size, axis=1)
        correlation = np.fft.irfft(spectra * spectra.conj(), n=fft_size, axis=1)[:, :frame_length]
        candidates = correlation[:, min_lag : max_lag + 1]
        offsets = np.argmax(candidates, axis=1)
        lags = min_lag + offsets
        row_indices = np.arange(len(frames))
        strengths = correlation[row_indices, lags] / np.maximum(correlation[:, 0], 1e-12)
        voiced = (correlation[:, 0] >= 1e-7) & (strengths >= 0.30)
        f0[first : first + len(frames)] = np.where(voiced, sample_rate / lags, 0).astype(np.float32)
    return f0
