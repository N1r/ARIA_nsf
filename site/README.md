# ARIA — interactive demo (IEEE SLT 2026)

A lightweight, static demo page for **ARIA**, an interpretable analytic source–filter
neural vocoder with independent, near-exact control of the parameters phoneticians use:
**F0, F1, F2, spectral tilt, energy, and glottal voice quality**.

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
- `build_assets.py` — regenerates assets from a trained ARIA checkpoint
  (`PYTHONPATH=<repo-root> python demo/build_assets.py`).

## Demo features
- **Manipulation studio** — pick an utterance + a phonetic parameter, drag the slider;
  spectrogram, audio and the F1–F2 vowel position update live (F1/F2 overlaid on the spectrogram).
- **Reconstruction A/B** — reference vs ARIA with UTMOS / MCD.
- **Fidelity plots** — measured-vs-commanded formant control showing decoupling.

Audio is 16 kHz (F024 voice). Manipulations are real syntheses, not pitch/format edits.
