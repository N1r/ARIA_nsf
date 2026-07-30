import json
import math
from pathlib import Path

import pytest

from phonlab_ddsp.metrics import (
    build_metrics_dashboard,
    discover_metrics_csv,
    find_nonfinite_metrics,
    load_metrics,
    metric_series,
    read_metrics_csv,
    render_metrics_dashboard,
    summarize_metrics,
)


def _write_metrics(root: Path, version: int, text: str, *, logger: str = "metrics") -> Path:
    path = root / "runs" / logger / f"version_{version}" / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_discovers_latest_or_explicit_numeric_version(tmp_path):
    older = _write_metrics(tmp_path, 2, "step,train_loss\n0,2\n")
    newest = _write_metrics(tmp_path, 10, "step,train_loss\n0,1\n")
    (tmp_path / "runs" / "metrics" / "version_99").mkdir()

    assert discover_metrics_csv(tmp_path) == newest.resolve()
    assert discover_metrics_csv(tmp_path, "latest") == newest.resolve()
    assert discover_metrics_csv(tmp_path, 2) == older.resolve()
    assert discover_metrics_csv(tmp_path, "version_2") == older.resolve()

    with pytest.raises(FileNotFoundError, match="available versions: 2, 10"):
        discover_metrics_csv(tmp_path, 3)
    with pytest.raises(ValueError, match="Invalid metrics version"):
        discover_metrics_csv(tmp_path, "second")


def test_rejects_ambiguous_duplicate_version_directories(tmp_path):
    _write_metrics(tmp_path, 1, "step,loss\n0,1\n", logger="first")
    _write_metrics(tmp_path, 1, "step,loss\n0,2\n", logger="second")

    with pytest.raises(ValueError, match="Multiple metrics.csv files claim Lightning version 1"):
        discover_metrics_csv(tmp_path)


def test_merges_sparse_duplicate_steps_and_summarizes_series(tmp_path):
    _write_metrics(
        tmp_path,
        4,
        "epoch,step,train_loss,val_loss,lr-Adam,accuracy\n"
        "0,0,1.2,,0.001,0.4\n"
        "0,1,0.9,,0.0009,\n"
        "0,1,,0.8,,0.55\n"
        "1,2,0.7,0.75,,0.5\n"
        "1,2,0.65,,,\n",
    )

    data = load_metrics(tmp_path)

    assert data.version == 4
    assert [row.step for row in data.rows] == [0, 1, 2]
    assert data.duplicate_steps == 2
    assert data.rows[1].values == {
        "train_loss": 0.9,
        "val_loss": 0.8,
        "lr-Adam": 0.0009,
        "accuracy": 0.55,
    }
    assert data.rows[2].values["train_loss"] == 0.65
    assert metric_series(data, "val_loss") == ((1, 0.8), (2, 0.75))

    summary = summarize_metrics(data)
    assert summary["rows"] == 3
    metrics = summary["metrics"]
    assert metrics["train_loss"]["latest"] == 0.65
    assert metrics["train_loss"]["best"] == 0.65
    assert metrics["train_loss"]["best_step"] == 2
    assert metrics["val_loss"]["latest_step"] == 2
    assert metrics["lr-Adam"]["latest_step"] == 1
    assert metrics["accuracy"]["mode"] == "max"
    assert metrics["accuracy"]["best"] == 0.55
    assert summarize_metrics(data, modes={"accuracy": "min"})["metrics"]["accuracy"]["best"] == 0.4


def test_nonfinite_values_are_reported_even_when_duplicate_is_overwritten(tmp_path):
    path = _write_metrics(
        tmp_path,
        0,
        "step,train_loss,val_loss\n0,1.0,0.9\n1,nan,\n1,0.8,\n2,0.7,inf\n3,0.6,-inf\n",
    )

    data = read_metrics_csv(path)
    issues = find_nonfinite_metrics(data)

    assert issues == [
        {"metric": "train_loss", "value": "nan", "step": 1, "row": 3},
        {"metric": "val_loss", "value": "+inf", "step": 2, "row": 5},
        {"metric": "val_loss", "value": "-inf", "step": 3, "row": 6},
    ]
    assert data.rows[1].values["train_loss"] == 0.8
    assert math.isinf(data.rows[-1].values["val_loss"])
    summary = summarize_metrics(data)
    assert summary["nonfinite_count"] == 3
    assert summary["metrics"]["train_loss"]["nonfinite_count"] == 1
    assert summary["metrics"]["val_loss"]["latest"] is None
    assert summary["metrics"]["val_loss"]["latest_finite"] == 0.9
    assert summary["metrics"]["val_loss"]["best"] == 0.9
    json.dumps(summary, allow_nan=False)


def test_preserves_large_step_identity_and_csv_column_order(tmp_path):
    path = _write_metrics(
        tmp_path,
        0,
        "step,first,second\n9007199254740992,,2\n9007199254740993,1,\n",
    )

    data = read_metrics_csv(path)

    assert [row.step for row in data.rows] == [9007199254740992, 9007199254740993]
    assert data.duplicate_steps == 0
    assert data.metric_names == ("first", "second")


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("epoch,loss\n0,1\n", "missing required 'step'"),
        ("step,loss\n,1\n", "row 2 has no step"),
        ("step,loss\n1.5,1\n", "invalid step"),
        ("step,loss\n1e5000,1\n", "invalid step"),
        ("step,loss\n0,not-a-number\n", "column 'loss' is not numeric"),
        ("step,loss,loss\n0,1,2\n", "duplicate column names"),
        ('step,loss\n0,"1\n', "invalid CSV syntax"),
    ],
)
def test_rejects_malformed_metric_csv(tmp_path, contents, message):
    path = _write_metrics(tmp_path, 0, contents)

    with pytest.raises(ValueError, match=message):
        read_metrics_csv(path)


def test_dashboard_is_self_contained_and_escapes_metric_names(tmp_path):
    path = _write_metrics(
        tmp_path,
        7,
        "step,train_loss,val_<loss>,lr_g,temperature\n0,2.0,1.8,0.0002,1.0\n1,1.0,1.1,0.0001,0.8\n",
    )
    data = read_metrics_csv(path)

    document = render_metrics_dashboard(data, title="Run <seven>")

    assert document.startswith("<!doctype html>")
    assert "Run &lt;seven&gt;" in document
    assert "<svg" in document
    assert "Training loss" in document
    assert "Validation metrics" in document
    assert "Learning rate" in document
    assert "Other metrics" in document
    assert "val_&lt;loss&gt;" in document
    assert "val_<loss>" not in document
    assert "<script" not in document
    assert "http://" not in document
    assert "https://" not in document

    output = build_metrics_dashboard(tmp_path, tmp_path / "reports" / "metrics.html", version=7)
    assert output.is_file()
    assert output.read_text(encoding="utf-8") == render_metrics_dashboard(data)


def test_dashboard_handles_header_only_log(tmp_path):
    path = _write_metrics(tmp_path, 0, "epoch,step,train_loss\n")

    data = read_metrics_csv(path)
    document = render_metrics_dashboard(data)

    assert data.rows == ()
    assert data.metric_names == ()
    assert "No scalar metric values have been logged yet." in document


@pytest.mark.parametrize(
    "values",
    [
        "-1e308\n1e308",
        "1.7976931348623157e308\n1.7976931348623157e308",
    ],
)
def test_dashboard_plots_extreme_finite_values_without_svg_overflow(tmp_path, values):
    path = _write_metrics(
        tmp_path,
        0,
        f"step,train_loss\n0,{values.splitlines()[0]}\n1,{values.splitlines()[1]}\n",
    )

    document = render_metrics_dashboard(read_metrics_csv(path))

    assert 'points="' in document
    assert "nan" not in document.lower()
    assert ">inf<" not in document.lower()
    assert ">-inf<" not in document.lower()


def test_short_step_range_uses_distinct_axis_labels(tmp_path):
    path = _write_metrics(
        tmp_path,
        0,
        "step,train_loss\n0,2\n1,1\n",
    )

    document = render_metrics_dashboard(read_metrics_csv(path))

    assert ">0.25</text>" in document
    assert ">0.5</text>" in document
    assert ">0.75</text>" in document
