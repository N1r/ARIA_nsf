"""Dependency-free, portable HTML dataset reports."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .manifest import DatasetManifest, summarize


def build_report(manifest: DatasetManifest, output: Path) -> Path:
    """Write a single-file HTML audit report for a prepared dataset."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(manifest.records)
    durations = [record.duration_s for record in manifest.records]
    pitches = [record.median_f0_hz for record in manifest.records if record.median_f0_hz > 0]
    rows = []
    for record in manifest.records:
        audio_url = _relative_url(output.parent, manifest.root / record.audio_path)
        warning = []
        if record.clipped_fraction > 0:
            warning.append("clipping")
        if record.voiced_fraction == 0:
            warning.append("no voiced F0")
        rows.append(
            "<tr>"
            f"<td>{html.escape(record.id)}</td><td>{record.split}</td>"
            f"<td>{record.duration_s:.2f}</td><td>{record.rms_dbfs:.1f}</td>"
            f"<td>{record.median_f0_hz:.1f}</td><td>{record.voiced_fraction:.1%}</td>"
            f"<td>{html.escape(', '.join(warning))}</td>"
            f'<td><audio controls preload="none" src="{html.escape(audio_url)}"></audio></td>'
            "</tr>"
        )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARIS dataset report</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#18212b}}
h1,h2{{color:#183b56}} .cards{{display:flex;flex-wrap:wrap;gap:1rem}}
.card{{background:#edf6f9;border-radius:8px;padding:1rem;min-width:140px}}
.value{{font-size:1.7rem;font-weight:650}} table{{border-collapse:collapse;width:100%}}
th,td{{padding:.55rem;border-bottom:1px solid #dce3e8;text-align:left}}
th{{position:sticky;top:0;background:white}} audio{{width:220px;height:32px}}
.plot{{border:1px solid #dce3e8;border-radius:8px;margin:.5rem 0 1.5rem}}
code{{background:#eef2f4;padding:.1rem .25rem}}
</style></head><body>
<h1>ARIS dataset report</h1>
<p>Dataset fingerprint: <code>{html.escape(manifest.fingerprint)}</code></p>
<div class="cards">
<div class="card"><div class="value">{summary["files"]}</div>files</div>
<div class="card"><div class="value">{summary["duration_s"] / 60:.1f}</div>minutes</div>
<div class="card"><div class="value">{summary["splits"]["train"]}/{summary["splits"]["validation"]}/{summary["splits"]["test"]}</div>train/val/test</div>
<div class="card"><div class="value">{summary["clipped_files"]}</div>files with clipping</div>
</div>
<h2>Distributions</h2>
{_histogram_svg(durations, "Duration (seconds)")}
{_histogram_svg(pitches, "Median voiced F0 (Hz)")}
<h2>Record audit</h2>
<table><thead><tr><th>ID</th><th>Split</th><th>Seconds</th><th>RMS dBFS</th>
<th>Median F0</th><th>Voiced</th><th>Warnings</th><th>Listen</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<h2>Provenance</h2><pre>{html.escape(json.dumps(manifest.metadata, indent=2, ensure_ascii=False))}</pre>
</body></html>"""
    output.write_text(body, encoding="utf-8")
    return output


def build_synthesis_report(manifest: DatasetManifest, synthesis: Path, output: Path) -> Path:
    """Build a side-by-side listening page for held-out copy synthesis."""
    synthesis = Path(synthesis).resolve()
    output = Path(output).resolve()
    if not synthesis.is_dir():
        raise FileNotFoundError(f"Synthesis directory does not exist: {synthesis}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for record in manifest.records:
        if record.split != "test":
            continue
        prediction = synthesis / f"{record.id}.wav"
        if not prediction.is_file():
            missing.append(record.id)
            continue
        original_url = _relative_url(output.parent, manifest.root / record.audio_path)
        prediction_url = _relative_url(output.parent, prediction)
        rows.append(
            f"<tr><td>{html.escape(record.id)}</td>"
            f'<td><audio controls preload="none" src="{html.escape(original_url)}"></audio></td>'
            f'<td><audio controls preload="none" src="{html.escape(prediction_url)}"></audio></td>'
            f"<td>{record.median_f0_hz:.1f}</td></tr>"
        )
    if not rows:
        raise ValueError("No matching test-set synthesis WAV files were found")
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARIS copy-synthesis comparison</title>
<style>body{{font:15px system-ui;max-width:1000px;margin:2rem auto}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:.6rem;border-bottom:1px solid #ddd;text-align:left}}audio{{width:280px}}</style></head>
<body><h1>Held-out copy-synthesis comparison</h1>
<p>Dataset fingerprint: <code>{manifest.fingerprint}</code></p>
<p>Listen with level-matched equipment. Row order is not randomized; this page is an audit tool,
not a perceptual-test instrument.</p>
<table><thead><tr><th>ID</th><th>Original</th><th>Reconstruction</th><th>Median F0</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p>Missing reconstructions: {len(missing)}</p></body></html>"""
    output.write_text(document, encoding="utf-8")
    return output


def _relative_url(base: Path, target: Path) -> str:
    import os

    return Path(os.path.relpath(target, base)).as_posix()


def _histogram_svg(values: list[float], label: str, bins: int = 20) -> str:
    """Render a small inline SVG histogram (no plotting dependency)."""
    if not values:
        return f"<p>No values for {html.escape(label)}.</p>"
    low, high = min(values), max(values)
    if low == high:
        high = low + 1
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / (high - low) * bins))
        counts[index] += 1
    maximum = max(counts) or 1
    bars = []
    for index, count in enumerate(counts):
        x = 45 + index * 35
        height = 130 * count / maximum
        bars.append(
            f'<rect x="{x}" y="{160 - height:.1f}" width="28" height="{height:.1f}" fill="#2a9d8f">'
            f"<title>{count} files</title></rect>"
        )
    return (
        '<svg class="plot" viewBox="0 0 760 205" role="img">'
        + "".join(bars)
        + f'<text x="45" y="185">{low:.1f}</text><text x="700" y="185">{high:.1f}</text>'
        + f'<text x="380" y="200" text-anchor="middle">{html.escape(label)}</text></svg>'
    )
