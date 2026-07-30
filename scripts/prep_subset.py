#!/usr/bin/env python
"""
Prepare a duration-capped training subset WITH features for SingleSpeaker.

For each selected wav: resample to target_sr (soxr) if needed, then write
  <out>/<stem>.wav
  <out>/<stem>.pv         F0, raw pyworld DIO, 5 ms frame, savetxt %f  (matches scripts/wav2f0.py)
  <out>/<stem>.feat.npz   f0/f1/f2/f3/b1/b2/b3/alpha/sr at 10 ms (models.formant_extractor.extract_all)

Selection: wavs sorted by name, cumulative (native) duration until --hours reached.

Usage:
  python scripts/prep_subset.py <src_wav_dir> <out_dir> --target-sr 24000 --hours 5 --workers 16
"""
import argparse, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr
import pyworld as pw
from multiprocessing import Pool
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.formant_extractor import extract_all


def process(wav_path: Path, out_dir: Path, target_sr: int):
    try:
        stem = wav_path.stem
        out_wav = out_dir / f"{stem}.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = soxr.resample(audio, sr, target_sr).astype("float32")
            sr = target_sr
        sf.write(str(out_wav), audio, sr)

        # .pv  — raw DIO F0 @ 5ms, matching scripts/wav2f0.py defaults
        f0 = pw.dio(audio.astype("float64"), sr, f0_floor=65.0, f0_ceil=1047.0,
                    channels_in_octave=2, frame_period=5.0)[0]
        np.savetxt(str(out_dir / f"{stem}.pv"), f0, fmt="%f")

        # .feat.npz — formants/tilt/f0 @ 10ms
        feat = extract_all(str(out_wav), hop_ms=10.0)
        np.savez(str(out_dir / f"{stem}.feat.npz"), **feat)
        return None
    except Exception as e:
        return f"{wav_path.name}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--target-sr", type=int, required=True)
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(src.glob("*.wav"))

    # select first --hours of audio (native duration)
    sel, cum = [], 0.0
    cap = a.hours * 3600.0
    for w in wavs:
        if cum >= cap:
            break
        sel.append(w)
        cum += sf.info(w).duration
    print(f"[prep] {src}: selected {len(sel)}/{len(wavs)} wavs = {cum/3600:.2f} h "
          f"-> {out} @ {a.target_sr} Hz", flush=True)

    errs = []
    with Pool(a.workers) as p:
        for i, e in enumerate(p.imap_unordered(
                partial(process, out_dir=out, target_sr=a.target_sr), sel)):
            if e:
                errs.append(e)
            if (i + 1) % 200 == 0:
                print(f"[prep] {i+1}/{len(sel)} done", flush=True)
    print(f"[prep] DONE {out}: {len(sel)-len(errs)} ok, {len(errs)} errors", flush=True)
    for e in errs[:20]:
        print("  ERR", e, flush=True)
