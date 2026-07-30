#!/usr/bin/env python
"""Extract WORLD D4C band-aperiodicity targets for the v6 aperiodicity head.

For each wav: A_target[frame, n_mag] in [0,1], 10 ms frames, on a uniform
0..Nyquist axis of n_mag points (matching AperiodicityFIRFilter's magnitude grid).
Writes <stem>.ap.npy alongside each wav.

Usage: python scripts/extract_aperiodicity.py <wav_dir> --n-mag 128 --workers 16
"""
import argparse, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import pyworld as pw
from multiprocessing import Pool
from functools import partial


def process(wav_path: Path, n_mag: int):
    try:
        x, fs = sf.read(str(wav_path), dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        f0, t = pw.dio(x, fs, frame_period=10.0)
        f0 = pw.stonemask(x, f0, t, fs)
        ap = pw.d4c(x, f0, t, fs)                 # [frames, fft//2+1] in [0,1]
        # resample each frame's half-spectrum to n_mag uniform points 0..Nyquist
        src = np.linspace(0.0, 1.0, ap.shape[1])
        dst = np.linspace(0.0, 1.0, n_mag)
        ap_n = np.stack([np.interp(dst, src, row) for row in ap]).astype(np.float32)
        np.save(str(wav_path.with_suffix(".ap.npy")), ap_n)
        return None
    except Exception as e:
        return f"{wav_path.name}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_dir")
    ap.add_argument("--n-mag", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    wavs = sorted(Path(a.wav_dir).glob("*.wav"))
    print(f"[ap] {len(wavs)} wavs -> .ap.npy (n_mag={a.n_mag})", flush=True)
    errs = []
    with Pool(a.workers) as p:
        for i, e in enumerate(p.imap_unordered(
                partial(process, n_mag=a.n_mag), wavs)):
            if e:
                errs.append(e)
            if (i + 1) % 300 == 0:
                print(f"[ap] {i+1}/{len(wavs)}", flush=True)
    print(f"[ap] DONE: {len(wavs)-len(errs)} ok, {len(errs)} errors", flush=True)
    for e in errs[:20]:
        print("  ERR", e, flush=True)
