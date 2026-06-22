# ARIA — interactive demo (static site)

An analytic source–filter neural vocoder with an interpretable, decoupled control surface.
This folder is a **self-contained static website** — no build step, no server-side code.

## What's here

```
index.html      # the page
style.css
app.js          # interactive studio (vanilla JS, no framework)
assets/
  manifest.json     # F024 studio data (controls, steps, measured F0/F1/F2)
  voices.json       # cross-voice (CSMSC / LJSpeech) data
  manipulation_fidelity.png
  spec/  audio/     # pre-rendered spectrograms + audio (all model output)
.nojekyll       # serve files as-is on GitHub Pages (don't run Jekyll)
```

Two resources load from a CDN (so a live, online page needs internet): Google Fonts
and Plotly (`cdn.plot.ly`). Everything else is local and relative-pathed.

## Deploy to GitHub Pages

1. Create a repo and copy the **contents of this folder** into it (or into a `docs/`
   subfolder), then push.
2. Repo **Settings → Pages → Build and deployment → Deploy from a branch**, pick
   `main` and `/ (root)` (or `/docs` if you used that). Save.
3. The site goes live at `https://<user>.github.io/<repo>/` within a minute or two.

`.nojekyll` is included so the `assets/` files are served untouched.

## Preview locally

```bash
cd this-folder
python -m http.server 8000     # then open http://localhost:8000
```

(Open via a server, not `file://` — the page `fetch`es `assets/manifest.json`.)

All audio is model output; every manipulation shown is a pre-rendered synthesis.
