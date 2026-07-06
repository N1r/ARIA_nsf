# ARIS — interactive demo

A lightweight, static demo page for **ARIS** (Analytic Resonance for Interpretable Synthesis),
an interpretable analytic source–filter neural vocoder with independent control of the parameters
phoneticians use: **F0, F1, F2, phonation (R_d), prominence/energy, and an exploratory nasalisation control**.

## View locally
```bash
python -m http.server 8000   # then open http://localhost:8000
```
No build step — plain HTML/CSS/JS + Plotly (CDN). Deployable as-is to **GitHub Pages**.

## Contents
- `index.html` · `style.css` · `app.js` — the site (manifest-driven).
- `assets/manifest.json` — describes every clip; the front-end renders from it.
- `assets/audio/*.wav`, `assets/spec/*.png` — pre-rendered manipulation continua &
  reconstruction A/B (all are model output / live syntheses).
- `build_assets.py` — regenerates assets from a trained ARIS checkpoint
  (`PYTHONPATH=<repo-root> python demo/build_assets.py`).

## Demo features (page order)
- **Control studio** — pick a phonetic control, drag the 7-step slider; spectrogram, audio and
  the F1–F2 / F0 chart update live across three syllables at once.
- **Fidelity plots** — measured-vs-commanded formant control showing decoupling.
- **Versus baselines** — ARIS vs. copy-synthesis, Praat, and WORLD on a fair global (uniform) scale,
  plus a listening studio sweeping F0 / F1 / F2 / F1+F2 for each system.
- **Reconstruction A/B** — natural vs ARIS with a UTMOS table (within-voice gap).
- **Other voices** — CSMSC + LJSpeech, same pipeline, full sentences.
- **Method** — control-surface table, architecture cards, and limitations.

Audio is 16 kHz (F024) plus 24/22 kHz cross-lingual voices (CSMSC, LJSpeech). Manipulations are real
syntheses, not pitch/formant edits.
