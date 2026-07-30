"""Make fixed-duration training subsets for the ARIA data-budget ablation.
Symlinks .wav + sibling feature files (.pv f0, .npz) so the loader finds them.
Files are taken in sorted order (deterministic) until each target is reached;
subsets are nested (5min ⊂ 10min ⊂ ...), so the curve isolates *amount* of data.

Usage: python scripts/make_budget_subsets.py --wav_dir <dir> --out_root <dir> \
           --minutes 5 10 20
"""
import argparse, glob, os
from pathlib import Path
import soundfile as sf

SIBLINGS = [".pv", ".npz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--minutes", type=float, nargs="+", default=[5, 10, 20])
    args = ap.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    targets = sorted(args.minutes)
    made = {m: [] for m in targets}
    cum = 0.0
    for w in wavs:
        try:
            cum += sf.info(w).duration
        except Exception:
            continue
        cur_min = cum / 60.0
        for m in targets:
            if cur_min <= m:
                made[m].append(w)
    for m in targets:
        out = Path(args.out_root) / f"budget_{int(m)}min"
        out.mkdir(parents=True, exist_ok=True)
        dur = 0.0
        for w in made[m]:
            stem = Path(w)
            for ext in [".wav"] + SIBLINGS:
                src = stem.with_suffix(ext)
                if src.exists():
                    dst = out / src.name
                    if not dst.exists():
                        os.symlink(os.path.abspath(src), dst)
            dur += sf.info(w).duration
        print(f"{out}: {len(made[m])} files, {dur/60:.1f} min")


if __name__ == "__main__":
    main()
