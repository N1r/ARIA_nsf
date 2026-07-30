"""Read Lightning CSV logs and render portable training dashboards."""

from __future__ import annotations

import csv
import html
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

VersionSelector = Optional[Union[int, str]]

__all__ = [
    "MetricRow",
    "MetricsData",
    "NonFiniteMetric",
    "build_metrics_dashboard",
    "discover_metrics_csv",
    "find_nonfinite_metrics",
    "load_metrics",
    "metric_series",
    "read_metrics_csv",
    "render_metrics_dashboard",
    "summarize_metrics",
]

_VERSION_DIRECTORY = re.compile(r"^version_(\d+)$")
_COORDINATE_COLUMNS = frozenset({"step", "epoch", "v_num"})
_MAX_STEP = 2**63 - 1
_COLORS = (
    "#176b5b",
    "#d66b35",
    "#365f91",
    "#8b5d9b",
    "#b28a18",
    "#287c99",
    "#a54848",
    "#587d3e",
)


@dataclass(frozen=True)
class MetricRow:
    """All scalar values logged for one training step."""

    step: int
    epoch: Optional[float]
    values: Mapping[str, float]


@dataclass(frozen=True)
class NonFiniteMetric:
    """A NaN or infinity encountered in the source CSV."""

    metric: str
    value: str
    step: int
    row_number: int


@dataclass(frozen=True)
class MetricsData:
    """A selected Lightning log after sparse rows have been merged by step."""

    source: Path
    version: Optional[int]
    rows: tuple[MetricRow, ...]
    metric_names: tuple[str, ...]
    nonfinite_events: tuple[NonFiniteMetric, ...] = ()
    duplicate_steps: int = 0


def discover_metrics_csv(experiment: Path, version: VersionSelector = None) -> Path:
    """Select a Lightning ``version_N/metrics.csv`` below *experiment*.

    The highest numeric version is selected by default. ``version`` accepts either
    an integer, a numeric string, or the directory spelling (for example
    ``"version_3"``). Ambiguous duplicate version directories are rejected rather
    than selected using filesystem traversal order.
    """

    root = Path(experiment).expanduser()
    if root.is_file():
        if root.name != "metrics.csv":
            raise ValueError(f"Expected metrics.csv, got: {root}")
        selected = _normalise_version(version)
        actual = _version_from_path(root)
        if selected is not None and actual != selected:
            raise FileNotFoundError(f"Metrics version {selected} was not found in {root}")
        return root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Experiment does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Experiment is not a directory: {root}")

    candidates: dict[int, list[Path]] = {}
    for path in root.rglob("metrics.csv"):
        candidate_version = _version_from_path(path)
        if candidate_version is not None and path.is_file():
            candidates.setdefault(candidate_version, []).append(path.resolve())
    if not candidates:
        raise FileNotFoundError(
            f"No Lightning metrics CSV found below {root} (expected version_N/metrics.csv)"
        )

    selected = _normalise_version(version)
    if selected is None:
        selected = max(candidates)
    matches = sorted(candidates.get(selected, ()), key=lambda item: item.as_posix())
    if not matches:
        available = ", ".join(str(item) for item in sorted(candidates))
        raise FileNotFoundError(
            f"Metrics version {selected} was not found below {root}; "
            f"available versions: {available}"
        )
    if len(matches) > 1:
        paths = "\n".join(f"- {path}" for path in matches)
        raise ValueError(
            f"Multiple metrics.csv files claim Lightning version {selected} below {root}:\n{paths}"
        )
    return matches[0]


def read_metrics_csv(path: Path) -> MetricsData:
    """Parse one metrics CSV and merge sparse or duplicate rows by training step."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Metrics CSV does not exist: {source}")

    merged: dict[int, dict[str, object]] = {}
    seen_metrics: set[str] = set()
    nonfinite: list[NonFiniteMetric] = []
    duplicate_steps = 0

    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        try:
            raw_fields = reader.fieldnames
        except csv.Error as error:
            raise _invalid_csv_syntax(source, reader.line_num, error) from error
        if raw_fields is None:
            raise ValueError(f"Metrics CSV has no header: {source}")
        fields = [field.strip() for field in raw_fields]
        if any(not field for field in fields):
            raise ValueError(f"Metrics CSV has an empty column name: {source}")
        if len(set(fields)) != len(fields):
            raise ValueError(f"Metrics CSV has duplicate column names: {source}")
        if "step" not in fields:
            raise ValueError(f"Metrics CSV is missing required 'step' column: {source}")
        reader.fieldnames = fields

        for row_number, row in enumerate(_csv_rows(reader, source), start=2):
            if None in row:
                raise ValueError(
                    f"Metrics CSV row {row_number} has more values than columns: {source}"
                )
            if all(value is None or not value.strip() for value in row.values()):
                continue
            step = _parse_step(row.get("step"), source, row_number)
            epoch = _parse_epoch(row.get("epoch"), source, row_number)
            values: dict[str, float] = {}
            for name in fields:
                if name in _COORDINATE_COLUMNS:
                    continue
                raw_value = row.get(name)
                if raw_value is None or not raw_value.strip():
                    continue
                value = _parse_metric(raw_value, name, source, row_number)
                values[name] = value
                seen_metrics.add(name)
                if not math.isfinite(value):
                    nonfinite.append(
                        NonFiniteMetric(
                            metric=name,
                            value=_nonfinite_label(value),
                            step=step,
                            row_number=row_number,
                        )
                    )

            existing = merged.get(step)
            if existing is None:
                merged[step] = {"epoch": epoch, "values": values}
            else:
                duplicate_steps += 1
                if epoch is not None:
                    existing["epoch"] = epoch
                existing_values = existing["values"]
                if not isinstance(existing_values, dict):
                    raise AssertionError("Internal metrics merge state is invalid")
                existing_values.update(values)

    rows = tuple(
        MetricRow(
            step=step,
            epoch=_optional_float(merged[step]["epoch"]),
            values=dict(merged[step]["values"]),  # type: ignore[arg-type]
        )
        for step in sorted(merged)
    )
    metric_names = tuple(
        name for name in fields if name not in _COORDINATE_COLUMNS and name in seen_metrics
    )
    return MetricsData(
        source=source,
        version=_version_from_path(source),
        rows=rows,
        metric_names=metric_names,
        nonfinite_events=tuple(nonfinite),
        duplicate_steps=duplicate_steps,
    )


def load_metrics(experiment: Path, version: VersionSelector = None) -> MetricsData:
    """Discover and parse a selected experiment metrics log."""

    return read_metrics_csv(discover_metrics_csv(experiment, version))


def metric_series(
    data: MetricsData, metric: str, *, finite_only: bool = True
) -> tuple[tuple[int, float], ...]:
    """Return sorted ``(step, value)`` pairs for one scalar metric."""

    if metric not in data.metric_names:
        raise KeyError(f"Unknown metric {metric!r}")
    points = []
    for row in data.rows:
        value = row.values.get(metric)
        if value is None or (finite_only and not math.isfinite(value)):
            continue
        points.append((row.step, value))
    return tuple(points)


def find_nonfinite_metrics(data: MetricsData) -> list[dict[str, object]]:
    """Return JSON-safe details for every source NaN or infinity."""

    return [
        {
            "metric": event.metric,
            "value": event.value,
            "step": event.step,
            "row": event.row_number,
        }
        for event in data.nonfinite_events
    ]


def summarize_metrics(
    data: MetricsData, *, modes: Optional[Mapping[str, str]] = None
) -> dict[str, object]:
    """Summarize latest and best finite values for each scalar series.

    Best-value direction is inferred from the metric name (accuracy-like metrics
    maximize; losses and other metrics minimize) and can be overridden with
    ``modes={"metric_name": "min"}`` or ``"max"``.
    """

    requested_modes = dict(modes or {})
    invalid_modes = {
        metric: mode for metric, mode in requested_modes.items() if mode not in {"min", "max"}
    }
    if invalid_modes:
        invalid = ", ".join(f"{name}={mode!r}" for name, mode in invalid_modes.items())
        raise ValueError(f"Metric modes must be 'min' or 'max': {invalid}")

    event_counts: dict[str, int] = {}
    for event in data.nonfinite_events:
        event_counts[event.metric] = event_counts.get(event.metric, 0) + 1

    summaries: dict[str, dict[str, object]] = {}
    for metric in data.metric_names:
        points = metric_series(data, metric, finite_only=False)
        finite = tuple(point for point in points if math.isfinite(point[1]))
        latest_observed = points[-1] if points else None
        latest_finite = finite[-1] if finite else None
        mode = requested_modes.get(metric, _infer_mode(metric))
        if finite:
            chooser = min if mode == "min" else max
            best = chooser(finite, key=lambda point: point[1])
        else:
            best = None
        latest_is_finite = latest_observed is not None and math.isfinite(latest_observed[1])
        summaries[metric] = {
            "mode": mode,
            "count": len(points),
            "finite_count": len(finite),
            "nonfinite_count": event_counts.get(metric, 0),
            "latest": latest_observed[1] if latest_is_finite else None,
            "latest_step": latest_observed[0] if latest_observed is not None else None,
            "latest_finite": latest_finite[1] if latest_finite is not None else None,
            "latest_finite_step": latest_finite[0] if latest_finite is not None else None,
            "best": best[1] if best is not None else None,
            "best_step": best[0] if best is not None else None,
        }

    return {
        "source": str(data.source),
        "version": data.version,
        "rows": len(data.rows),
        "duplicate_steps": data.duplicate_steps,
        "first_step": data.rows[0].step if data.rows else None,
        "latest_step": data.rows[-1].step if data.rows else None,
        "nonfinite_count": len(data.nonfinite_events),
        "metrics": summaries,
    }


def render_metrics_dashboard(
    data: MetricsData, *, title: str = "PhonLab-DDSP training metrics"
) -> str:
    """Render a self-contained HTML document with inline SVG charts."""

    summary = summarize_metrics(data)
    metric_summary = summary["metrics"]
    if not isinstance(metric_summary, dict):
        raise AssertionError("Internal metrics summary is invalid")
    groups = _chart_groups(data.metric_names)
    charts = []
    for heading, names in groups:
        if names:
            charts.append(_line_chart(data, names, heading))
    if not charts:
        charts.append('<p class="empty">No scalar metric values have been logged yet.</p>')

    warning = _nonfinite_html(data)
    table_rows = []
    for metric in data.metric_names:
        item = metric_summary[metric]
        if not isinstance(item, dict):
            raise AssertionError("Internal metric summary is invalid")
        status = f"{item['nonfinite_count']} non-finite" if item["nonfinite_count"] else "finite"
        status_class = "bad" if item["nonfinite_count"] else "good"
        table_rows.append(
            "<tr>"
            f"<td><code>{html.escape(metric)}</code></td>"
            f"<td>{_format_optional(item['latest'])}</td>"
            f"<td>{_format_optional(item['latest_step'])}</td>"
            f"<td>{_format_optional(item['best'])}</td>"
            f"<td>{_format_optional(item['best_step'])}</td>"
            f"<td>{item['finite_count']}/{item['count']}</td>"
            f'<td class="{status_class}">{status}</td>'
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>Metric</th><th>Latest</th><th>At step</th>"
        "<th>Best</th><th>At step</th><th>Finite points</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>"
        if table_rows
        else '<p class="empty">No metric columns were found.</p>'
    )
    version = f"version_{data.version}" if data.version is not None else "unversioned"
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escaped_title}</title>
<style>
:root{{--ink:#172129;--muted:#66727a;--paper:#f6f3ed;--card:#fffdfa;--line:#d9d3c8;
--accent:#176b5b;--good:#34785d;--bad:#a13c31}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}}
header{{background:#163b36;color:white;padding:2.4rem max(1.2rem,calc((100% - 1120px)/2))}}
header h1{{font:700 clamp(1.8rem,4vw,3.2rem)/1.05 Georgia,serif;margin:0 0 .6rem}}
header p{{margin:.25rem 0;color:#d8e9e4;overflow-wrap:anywhere}}main{{max-width:1120px;margin:1.5rem auto;
padding:0 1.2rem 4rem}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem}}
.card,.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;
box-shadow:0 4px 16px #4d463b0b}}.card{{padding:1rem}}.value{{font-size:1.55rem;font-weight:700}}
.label{{color:var(--muted);font-size:.82rem}}.panel{{margin-top:1rem;padding:1rem;overflow-x:auto}}
.panel h2{{font:700 1.25rem Georgia,serif;margin:0 0 .65rem}}svg{{display:block;min-width:680px;width:100%;
height:auto}}.gridline{{stroke:#ded9d0;stroke-width:1}}.axis{{fill:#66727a;font-size:12px}}
.legend{{fill:#334047;font-size:12px}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:.58rem;border-bottom:1px solid #e2ddd4;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}code{{font:13px ui-monospace,SFMono-Regular,monospace}}
.warning{{margin-top:1rem;padding:.9rem 1rem;border-left:4px solid var(--bad);background:#fff0ec}}
.warning ul{{margin:.45rem 0 0;padding-left:1.3rem}}.good{{color:var(--good)}}.bad{{color:var(--bad);
font-weight:650}}.empty{{color:var(--muted)}}@media(max-width:720px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<header><h1>{escaped_title}</h1>
<p>{html.escape(version)} · {html.escape(str(data.source))}</p></header>
<main><div class="cards">
<div class="card"><div class="value">{len(data.rows)}</div><div class="label">merged steps</div></div>
<div class="card"><div class="value">{len(data.metric_names)}</div><div class="label">scalar series</div></div>
<div class="card"><div class="value">{_format_optional(summary["latest_step"])}</div><div class="label">latest step</div></div>
<div class="card"><div class="value">{len(data.nonfinite_events)}</div><div class="label">non-finite values</div></div>
</div>
{warning}
{"".join(charts)}
<section class="panel"><h2>Latest and best values</h2>{table}</section>
</main></body></html>"""


def build_metrics_dashboard(
    experiment: Path,
    output: Path,
    version: VersionSelector = None,
    *,
    title: str = "PhonLab-DDSP training metrics",
) -> Path:
    """Load an experiment log and write its standalone dashboard."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = load_metrics(experiment, version)
    destination.write_text(render_metrics_dashboard(data, title=title), encoding="utf-8")
    return destination


def _normalise_version(version: VersionSelector) -> Optional[int]:
    if version is None:
        return None
    if isinstance(version, bool):
        raise ValueError("Metrics version must be an integer or version_N")
    if isinstance(version, int):
        if version < 0:
            raise ValueError("Metrics version cannot be negative")
        return version
    text = str(version).strip()
    if text == "latest":
        return None
    if text.startswith("version_"):
        text = text.removeprefix("version_")
    if not text.isdigit():
        raise ValueError(f"Invalid metrics version {version!r}; expected an integer or version_N")
    return int(text)


def _version_from_path(path: Path) -> Optional[int]:
    match = _VERSION_DIRECTORY.fullmatch(path.parent.name)
    return int(match.group(1)) if match else None


def _csv_rows(reader: csv.DictReader, source: Path):
    try:
        yield from reader
    except csv.Error as error:
        raise _invalid_csv_syntax(source, reader.line_num, error) from error


def _invalid_csv_syntax(source: Path, line_number: int, error: csv.Error) -> ValueError:
    return ValueError(
        f"Metrics CSV has invalid CSV syntax near line {line_number}: {error}: {source}"
    )


def _parse_step(raw: Optional[str], source: Path, row_number: int) -> int:
    if raw is None or not raw.strip():
        raise ValueError(f"Metrics CSV row {row_number} has no step: {source}")
    try:
        numeric = Decimal(raw.strip())
    except InvalidOperation as error:
        raise ValueError(
            f"Metrics CSV row {row_number} has invalid step {raw!r}: {source}"
        ) from error
    if (
        not numeric.is_finite()
        or numeric != numeric.to_integral_value()
        or numeric < 0
        or numeric > _MAX_STEP
    ):
        raise ValueError(f"Metrics CSV row {row_number} has invalid step {raw!r}: {source}")
    return int(numeric)


def _parse_epoch(raw: Optional[str], source: Path, row_number: int) -> Optional[float]:
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"Metrics CSV row {row_number} has invalid epoch {raw!r}: {source}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"Metrics CSV row {row_number} has non-finite epoch {raw!r}: {source}")
    return value


def _parse_metric(raw: str, name: str, source: Path, row_number: int) -> float:
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(
            f"Metrics CSV row {row_number}, column {name!r} is not numeric ({raw!r}): {source}"
        ) from error


def _optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _nonfinite_label(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "+inf" if value > 0 else "-inf"


def _infer_mode(metric: str) -> str:
    tokens = set(re.findall(r"[a-z0-9]+", metric.lower()))
    maximise = {"acc", "accuracy", "auc", "dice", "f1", "iou", "precision", "r2", "recall"}
    return "max" if tokens & maximise else "min"


def _chart_groups(metric_names: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
    groups: dict[str, list[str]] = {
        "Training loss": [],
        "Validation metrics": [],
        "Learning rate": [],
        "Other metrics": [],
    }
    for metric in metric_names:
        lowered = metric.lower()
        if _is_learning_rate(lowered):
            groups["Learning rate"].append(metric)
        elif _is_validation(lowered):
            groups["Validation metrics"].append(metric)
        elif "loss" in lowered:
            groups["Training loss"].append(metric)
        else:
            groups["Other metrics"].append(metric)
    return [(heading, tuple(groups[heading])) for heading in groups]


def _is_validation(metric: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", metric))
    return bool(tokens & {"val", "valid", "validation"})


def _is_learning_rate(metric: str) -> bool:
    if "learning_rate" in metric or "learning-rate" in metric:
        return True
    tokens = set(re.findall(r"[a-z0-9]+", metric))
    return "lr" in tokens


def _line_chart(data: MetricsData, metrics: Sequence[str], heading: str) -> str:
    series = [(metric, metric_series(data, metric)) for metric in metrics]
    drawable = [(metric, points) for metric, points in series if points]
    if not drawable:
        return (
            f'<section class="panel"><h2>{html.escape(heading)}</h2>'
            '<p class="empty">No finite values are available for this chart.</p></section>'
        )

    all_points = [point for _, points in drawable for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min, y_max = _expanded_float_range(y_min)

    left, right, top, bottom = 82.0, 930.0, 24.0, 235.0
    grid = []
    for index in range(5):
        fraction = index / 4
        y = top + (bottom - top) * fraction
        value = _safe_lerp(y_max, y_min, fraction)
        grid.append(
            f'<line class="gridline" x1="{left:g}" y1="{y:.2f}" x2="{right:g}" y2="{y:.2f}"/>'
            f'<text class="axis" x="{left - 9:g}" y="{y + 4:.2f}" text-anchor="end">'
            f"{html.escape(_format_number(value))}</text>"
        )
    for index in range(5):
        fraction = index / 4
        x = left + (right - left) * fraction
        grid.append(
            f'<text class="axis" x="{x:.2f}" y="{bottom + 22:g}" text-anchor="middle">'
            f"{html.escape(_step_tick_label(x_min, x_max, index))}</text>"
        )

    lines = []
    legend = []
    for index, (metric, points) in enumerate(drawable):
        color = _COLORS[index % len(_COLORS)]
        sampled = _sample_points(points)
        coordinates = " ".join(
            f"{left + (step - x_min) / (x_max - x_min) * (right - left):.2f},"
            f"{bottom - _safe_fraction(value, y_min, y_max) * (bottom - top):.2f}"
            for step, value in sampled
        )
        if len(sampled) == 1:
            x, y = coordinates.split(",")
            lines.append(
                f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}">'
                f"<title>{html.escape(metric)}</title></circle>"
            )
        else:
            lines.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round">'
                f"<title>{html.escape(metric)}</title></polyline>"
            )
        legend_x = left + (index % 3) * 275
        legend_y = 282 + (index // 3) * 19
        legend.append(
            f'<line x1="{legend_x:g}" y1="{legend_y - 4:g}" x2="{legend_x + 18:g}" '
            f'y2="{legend_y - 4:g}" stroke="{color}" stroke-width="3"/>'
            f'<text class="legend" x="{legend_x + 24:g}" y="{legend_y:g}">'
            f"{html.escape(metric)}</text>"
        )

    height = 297 + max(0, (len(drawable) - 1) // 3) * 19
    return (
        f'<section class="panel"><h2>{html.escape(heading)}</h2>'
        f'<svg viewBox="0 0 960 {height}" role="img" '
        f'aria-label="{html.escape(heading)} over training steps">'
        + "".join(grid)
        + "".join(lines)
        + f'<text class="axis" x="{(left + right) / 2:g}" y="276" text-anchor="middle">'
        "training step</text>" + "".join(legend) + "</svg></section>"
    )


def _sample_points(
    points: Sequence[tuple[int, float]], maximum: int = 1200
) -> tuple[tuple[int, float], ...]:
    if len(points) <= maximum:
        return tuple(points)
    stride = math.ceil((len(points) - 2) / (maximum - 2))
    sampled = [points[0]]
    sampled.extend(points[index] for index in range(1, len(points) - 1, stride))
    sampled.append(points[-1])
    return tuple(sampled)


def _expanded_float_range(value: float) -> tuple[float, float]:
    if value == 0:
        return -1.0, 1.0
    toward_zero = value * 0.95
    away_from_zero = value * 1.05
    candidates = [item for item in (toward_zero, value, away_from_zero) if math.isfinite(item)]
    low, high = min(candidates), max(candidates)
    if low != high:
        return low, high
    neighbours = (
        math.nextafter(value, -math.inf),
        math.nextafter(value, math.inf),
    )
    candidates.extend(item for item in neighbours if math.isfinite(item) and item != value)
    return min(candidates), max(candidates)


def _safe_fraction(value: float, low: float, high: float) -> float:
    scale = max(abs(low), abs(high))
    if scale == 0:
        return 0.5
    scaled_low = low / scale
    denominator = high / scale - scaled_low
    if denominator == 0:
        return 0.5
    fraction = (value / scale - scaled_low) / denominator
    return min(1.0, max(0.0, fraction))


def _safe_lerp(start: float, end: float, fraction: float) -> float:
    scale = max(abs(start), abs(end))
    if scale == 0:
        return 0.0
    return ((1.0 - fraction) * (start / scale) + fraction * (end / scale)) * scale


def _step_tick_label(low: int, high: int, index: int) -> str:
    tick = (Decimal(low) * (4 - index) + Decimal(high) * index) / Decimal(4)
    label = format(tick, "f")
    return label.rstrip("0").rstrip(".") if "." in label else label


def _nonfinite_html(data: MetricsData) -> str:
    if not data.nonfinite_events:
        return ""
    shown = data.nonfinite_events[:12]
    items = "".join(
        f"<li><code>{html.escape(event.metric)}</code> was {html.escape(event.value)} "
        f"at step {event.step} (CSV row {event.row_number})</li>"
        for event in shown
    )
    remaining = len(data.nonfinite_events) - len(shown)
    suffix = f"<li>…and {remaining} more.</li>" if remaining else ""
    return (
        '<aside class="warning"><strong>Non-finite metrics detected.</strong>'
        "<ul>" + items + suffix + "</ul></aside>"
    )


def _format_optional(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return html.escape(_format_number(value))
    return html.escape(str(value))


def _format_number(value: Union[int, float]) -> str:
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1_000_000 or magnitude < 0.0001:
        return f"{value:.4g}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"
