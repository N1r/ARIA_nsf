"""Extract RMVPE F0 for a corpus, saved as <stem>.pv_rmvpe next to each wav (5 ms hop, fmt %f,
matching the .pv convention so ltng.data can read it via f0_suffix=.pv_rmvpe).

The repo's RMVPE(model_path) wrapper auto-selects the wrong arch for this checkpoint (it has no
intermediate `unet.tf.layers`, i.e. inter_layers=0), so we build the model ourselves and inject it
into the wrapper, reusing its mel + decode.
  python scripts/extract_rmvpe_f0.py --wav_dir <dir>
"""
import sys, glob, argparse
from pathlib import Path
import numpy as np, soundfile as sf, torch

RMVPE_DIR = "/zfsstore/user/dingyr/RMVPE"
# rmvpe.pt (181MB, used for the 1h F0) is an older variant the repo's model code can't load
# (no TimbreFilter); model.pt is the official release the repo's wrapper was written for.
RMVPE_CK = f"{RMVPE_DIR}/checkpoints/model.pt"
sys.path.insert(0, RMVPE_DIR)


def make_wrapper(dev):
    from src.inference import RMVPE
    return RMVPE(RMVPE_CK, hop_length=int(16000 * 5 / 1000))   # 5 ms @ 16 kHz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", required=True)
    p.add_argument("--suffix", default=".pv_rmvpe")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wr = make_wrapper(dev)
    wavs = sorted(glob.glob(f"{a.wav_dir}/*.wav"))
    print(f"{len(wavs)} wavs in {a.wav_dir}", flush=True)
    done = 0
    for i, wp in enumerate(wavs):
        out = Path(wp).with_suffix(a.suffix)
        if out.exists() and not a.overwrite:
            done += 1; continue
        audio, sr = sf.read(wp)
        if audio.ndim > 1: audio = audio.mean(1)
        f0 = wr.infer_from_audio(audio.astype(np.float32), sample_rate=sr, device=dev,
                                 thred=0.03, use_viterbi=False)
        f0 = np.asarray(f0, np.float32)
        np.savetxt(str(out), f0, fmt="%f")
        done += 1
        if i < 3:
            print(f"  sanity {Path(wp).stem}: frames={len(f0)} voiced_frac={(f0>0).mean():.3f} "
                  f"meanF0={f0[f0>0].mean():.1f}" if (f0>0).any() else f"  sanity {Path(wp).stem}: ALL UNVOICED", flush=True)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(wavs)}", flush=True)
    print(f"Wrote {done} {a.suffix} files", flush=True)


if __name__ == "__main__":
    main()
