"""Local web studio for manipulation design and A/B listening (``aris studio``).

The gradio wiring stays deliberately thin over small pure helpers so the
naming, continuum, discovery, and metadata-formatting logic is unit-testable
without gradio installed.  gradio itself is imported lazily: ``import
aris.studio`` must work in a lean environment.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .controls.specs import (
    CONTROL_SPECS,
    ControlVariant,
    controls_for_model,
    validate_controls,
)
from .experiment import _resolve_dataset_path
from .manifest import DatasetManifest
from .manipulation import manipulate_controls

_INSTALL_HINT = (
    "aris studio requires the studio extra: uv sync --extra studio (or: pip install 'aris[studio]')"
)
CLIPPED_WARN_FRACTION = 0.001
CLAMPED_WARN_FRACTION = 0.10
_SKIP_DIR_NAMES = {"studio_output", "node_modules", "__pycache__"}
_SLIDER_STEPS = {"Hz": 10.0, "dB": 0.5, "semitones": 0.5}


# ---------------------------------------------------------------------------
# Pure helpers (no gradio)
# ---------------------------------------------------------------------------


def _slug_number(value: float) -> str:
    """Format a number as a variant-name fragment: ``1.15 -> 1p15``, ``-2 -> m2``."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p").replace("+", "")


def variant_name(control: str, value: float) -> str:
    """Auto-name a single-control condition, e.g. ``f1_hz_550`` or ``f1_scale_1p15``."""
    return f"{control}_{_slug_number(value)}"


def auto_variant_name(controls: dict) -> str:
    """Auto-name a multi-control condition from its sorted control assignments."""
    if not controls:
        raise ValueError("Cannot name a variant with no controls")
    return "_".join(variant_name(name, controls[name]) for name in sorted(controls))


def continuum_values(control: str, lo: float, hi: float, steps: int) -> list:
    """Return *steps* evenly spaced in-range values from *lo* to *hi* inclusive."""
    if control not in CONTROL_SPECS:
        known = ", ".join(CONTROL_SPECS)
        raise ValueError(f"Unknown control {control!r}; choose from {known}")
    spec = CONTROL_SPECS[control]
    steps = int(steps)
    if not 2 <= steps <= 20:
        raise ValueError("Continuum steps must be between 2 and 20")
    lo = spec.normalize(lo)
    hi = spec.normalize(hi)
    if math.isclose(lo, hi):
        raise ValueError("Continuum endpoints must differ")
    return [round(lo + (hi - lo) * index / (steps - 1), 6) for index in range(steps)]


def discover_experiments(workspace: Path) -> list:
    """Find experiment directories (containing experiment.json) up to depth 3."""
    workspace = Path(workspace).resolve()
    found = []
    for pattern in (
        "experiment.json",
        "*/experiment.json",
        "*/*/experiment.json",
        "*/*/*/experiment.json",
    ):
        for metadata in workspace.glob(pattern):
            relative = metadata.parent.relative_to(workspace)
            if any(part.startswith(".") or part in _SKIP_DIR_NAMES for part in relative.parts):
                continue
            found.append(metadata.parent)
    return sorted(set(found))


def discover_checkpoints(experiment: Path) -> list:
    """List ``*.ckpt`` under ``<experiment>/runs/checkpoints``, ``last.ckpt`` first."""
    checkpoint_dir = Path(experiment) / "runs" / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    return sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: (p.name != "last.ckpt", p.name))


def format_variant_metadata(entry: dict) -> str:
    """One markdown block per manipulation.json output entry, with warnings."""
    controls = entry.get("controls", {}) or {}
    audit = entry.get("render_audit", {}) or {}
    control_text = ", ".join(f"{k}={v:g}" for k, v in sorted(controls.items())) or "无 (none)"
    clipped = float(audit.get("clipped_fraction", 0.0))
    clip_text = f"{clipped:.3%}"
    if clipped > CLIPPED_WARN_FRACTION:
        clip_text = (
            f'<span style="color:#c00">⚠️ {clip_text} — 输出削波超过 0.1%，'
            "请降低增益 (clipping; reduce gain)</span>"
        )
    name = entry.get("name") or entry.get("directory", "?")
    lines = [f"**{name}** — 控制 (controls): {control_text} · 削波比例 (clipped): {clip_text}"]
    for formant, stats in sorted((audit.get("formant_tracking") or {}).items()):
        discrepancy = float(stats.get("max_abs_hz_discrepancy", 0.0))
        clamped = float(stats.get("clamped_frame_fraction", 0.0))
        line = (
            f"- {formant}: 最大频率偏差 (max discrepancy) {discrepancy:.0f} Hz · "
            f"受限帧比例 (clamped frames) {clamped:.1%}"
        )
        if clamped > CLAMPED_WARN_FRACTION:
            line += (
                ' <span style="color:#c00">⚠️ 较多帧到达模型频率范围边缘，'
                "实际移动量小于请求值 (frames hit the model's range edge; "
                "the achieved shift is smaller than requested)</span>"
            )
        lines.append(line)
    return "\n".join(lines)


def _load_experiment(experiment: Path) -> dict:
    """Load model name and the test-split record -> original-wav map."""
    experiment = Path(experiment).resolve()
    metadata = json.loads((experiment / "experiment.json").read_text())
    model = metadata["model"]
    manifest = DatasetManifest.load(_resolve_dataset_path(metadata["dataset"], experiment))
    records = {
        record.id: str(manifest.root / record.audio_path)
        for record in manifest.records
        if record.split == "test"
    }
    return {"experiment": str(experiment), "model": model, "records": records}


def _fresh_output_dir(workspace: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(workspace) / "studio_output" / stamp
    candidate, counter = base, 1
    while candidate.exists():
        candidate = base.with_name(f"{stamp}-{counter}")
        counter += 1
    return candidate


def spectrogram_figure(original: Path, variant: Path):
    """Four aligned panels: original/variant waveform + 0-8 kHz log spectrogram."""
    import numpy as np
    import soundfile as sf
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 7))
    axes = fig.subplots(4, 1, sharex=True)
    panels = [("Original", original, axes[0], axes[1]), ("Variant", variant, axes[2], axes[3])]
    for title, path, wave_ax, spec_ax in panels:
        signal, sr = sf.read(str(path), dtype="float64", always_2d=False)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        duration = signal.size / sr
        wave_ax.plot(np.arange(signal.size) / sr, signal, linewidth=0.4, color="#2a4d69")
        wave_ax.set_ylabel("Amplitude", fontsize=8)
        wave_ax.set_title(f"{title}: {Path(path).name}", fontsize=9, loc="left")
        wave_ax.tick_params(labelsize=7)
        # 25 ms Hann window rounded to a power of two, 10 ms hop, log magnitude.
        window_size = 2 ** int(round(math.log2(max(2.0, 0.025 * sr))))
        hop = max(1, int(round(0.010 * sr)))
        if signal.size < window_size:
            signal = np.pad(signal, (0, window_size - signal.size))
        frames = 1 + (signal.size - window_size) // hop
        indices = np.arange(window_size)[None, :] + hop * np.arange(frames)[:, None]
        spectrum = np.fft.rfft(signal[indices] * np.hanning(window_size), axis=1)
        magnitude_db = 20.0 * np.log10(np.abs(spectrum).T + 1e-8)
        frequencies_khz = np.fft.rfftfreq(window_size, 1.0 / sr) / 1000.0
        keep = frequencies_khz <= 8.0
        top = float(magnitude_db.max())
        spec_ax.imshow(
            magnitude_db[keep],
            origin="lower",
            aspect="auto",
            extent=(0.0, duration, 0.0, float(frequencies_khz[keep][-1])),
            cmap="magma",
            vmin=top - 80.0,
            vmax=top,
        )
        spec_ax.set_ylabel("Freq (kHz)", fontsize=8)
        spec_ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Time (s)", fontsize=8)
    fig.tight_layout()
    return fig


def _variants_table(variants: list) -> list:
    return [[item["name"], json.dumps(item["controls"], sort_keys=True)] for item in variants]


def _import_gradio():
    try:
        import gradio
    except ImportError as error:
        raise RuntimeError(_INSTALL_HINT) from error
    return gradio


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def build_app(workspace: Path):
    """Build the single-page studio Blocks app for *workspace*."""
    gr = _import_gradio()
    workspace = Path(workspace).resolve()
    experiments = [str(path) for path in discover_experiments(workspace)]
    all_specs = list(CONTROL_SPECS.values())

    def on_experiment(experiment_path):
        if not experiment_path:
            raise gr.Error("请先选择一个实验 (select an experiment first)")
        try:
            info = _load_experiment(Path(experiment_path))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise gr.Error(f"无法加载实验 (cannot load experiment): {error}") from error
        supported = {spec.name for spec in controls_for_model(info["model"])}
        checkpoints = [str(path) for path in discover_checkpoints(Path(experiment_path))]
        slider_updates = [
            gr.update(visible=spec.name in supported, value=spec.default) for spec in all_specs
        ]
        summary = (
            f"模型 (model): **{info['model']}** · 测试句 (test records): "
            f"{len(info['records'])} · 可用控制 (controls): {len(supported)}"
        )
        return [
            info,
            gr.update(choices=checkpoints, value=checkpoints[0] if checkpoints else None),
            summary,
            gr.update(choices=sorted(supported), value=None),
            gr.update(choices=sorted(info["records"]), value=None),
            *slider_updates,
        ]

    def add_variant(info, name_text, variants, *slider_values):
        if not info:
            gr.Warning("请先选择实验 (select an experiment first)")
            return variants, _variants_table(variants), gr.update()
        values = dict(zip((spec.name for spec in all_specs), slider_values))
        controls = {
            spec.name: float(values[spec.name])
            for spec in controls_for_model(info["model"])
            if not math.isclose(float(values[spec.name]), spec.default, abs_tol=1e-9)
        }
        if not controls:
            gr.Warning("所有滑块均为默认值，无可添加的条件 (all sliders at default)")
            return variants, _variants_table(variants), gr.update()
        try:
            validated = validate_controls(info["model"], controls)
            name = (name_text or "").strip() or auto_variant_name(validated)
            ControlVariant(name, validated)
        except (TypeError, ValueError) as error:
            gr.Warning(f"无法添加条件 (cannot add variant): {error}")
            return variants, _variants_table(variants), gr.update()
        if any(item["name"] == name for item in variants):
            gr.Warning(f"条件名称重复 (duplicate variant name): {name}")
            return variants, _variants_table(variants), gr.update()
        variants = variants + [{"name": name, "controls": validated}]
        return variants, _variants_table(variants), gr.update(value="")

    def add_continuum(info, control, lo, hi, steps, variants):
        if not info:
            gr.Warning("请先选择实验 (select an experiment first)")
            return variants, _variants_table(variants)
        if not control:
            gr.Warning("请选择连续统控制参数 (choose a continuum control)")
            return variants, _variants_table(variants)
        try:
            values = continuum_values(control, float(lo), float(hi), int(steps))
            validate_controls(info["model"], {control: values[0]})
        except (TypeError, ValueError) as error:
            gr.Warning(f"无法生成连续统 (cannot build continuum): {error}")
            return variants, _variants_table(variants)
        additions = [
            {"name": variant_name(control, value), "controls": {control: value}} for value in values
        ]
        names = [item["name"] for item in variants] + [item["name"] for item in additions]
        if len(names) != len(set(names)):
            gr.Warning("连续统与现有条件名称冲突 (continuum names collide with existing variants)")
            return variants, _variants_table(variants)
        variants = variants + additions
        return variants, _variants_table(variants)

    def on_row_select(evt):
        return evt.index[0] if evt.index else None

    # A real class (not a string annotation) so gradio's get_type_hints can
    # inject the row-select event without gradio being a module-level import.
    on_row_select.__annotations__["evt"] = gr.SelectData

    def delete_selected(variants, selected):
        if selected is None or not 0 <= int(selected) < len(variants):
            gr.Warning("请先在表格中选择一行 (select a table row first)")
            return variants, _variants_table(variants), None
        variants = [item for index, item in enumerate(variants) if index != int(selected)]
        return variants, _variants_table(variants), None

    # gradio only injects a UI-tracked Progress when the parameter default is a
    # gr.Progress instance; called directly (tests, scripts) the untracked
    # default is a safe no-op.  noqa B008: this call-in-default is the
    # documented gradio injection idiom.
    def render(info, checkpoint, variants, progress=gr.Progress()):  # noqa: B008
        if not info:
            raise gr.Error("请先选择实验 (select an experiment first)")
        if not checkpoint:
            raise gr.Error("请先选择检查点 (select a checkpoint first)")
        if not variants:
            raise gr.Error("请先添加至少一个条件 (add at least one variant first)")
        progress(0.05, desc="验证 (validating)")
        control_variants = [ControlVariant(item["name"], item["controls"]) for item in variants]
        output_dir = _fresh_output_dir(workspace)
        progress(0.15, desc="渲染中，每个条件都会完整合成 (rendering)")
        try:
            manipulate_controls(
                Path(info["experiment"]), Path(checkpoint), output_dir, control_variants
            )
        except Exception as error:
            raise gr.Error(f"渲染失败 (render failed): {error}") from error
        progress(0.9, desc="整理结果 (collecting results)")
        metadata = json.loads((output_dir / "manipulation.json").read_text())
        summary = "\n\n".join(format_variant_metadata(entry) for entry in metadata["outputs"])
        directories = [entry["directory"] for entry in metadata["outputs"]]
        note = f"打开输出文件夹 (output folder): `{output_dir}`"
        progress(1.0, desc="完成 (done)")
        return (
            {"output_dir": str(output_dir), "variants": directories},
            summary,
            gr.update(choices=directories, value=directories[0] if directories else None),
            note,
        )

    def compare(info, rendered, record_id, variant):
        if not (info and rendered and record_id and variant):
            return gr.update(), gr.update(), gr.update()
        original = Path(info["records"][record_id])
        variant_wav = Path(rendered["output_dir"]) / variant / f"{record_id}.wav"
        if not variant_wav.is_file():
            raise gr.Error(f"未找到变体音频 (variant audio not found): {variant_wav}")
        return str(original), str(variant_wav), spectrogram_figure(original, variant_wav)

    with gr.Blocks(title="ARIS Studio", analytics_enabled=False) as app:
        gr.Markdown(f"# ARIS Studio\n工作区 (workspace): `{workspace}` · aris {__version__}")

        exp_state, render_state, selected_state = gr.State(None), gr.State(None), gr.State(None)
        variants_state = gr.State([])

        gr.Markdown("## 实验选择 (experiment)")
        with gr.Row():
            experiment_dd = gr.Dropdown(
                choices=experiments,
                value=experiments[0] if experiments else None,
                label="实验 (experiment)",
            )
            checkpoint_dd = gr.Dropdown(label="检查点 (checkpoint)")
        model_md = gr.Markdown()

        _FRIENDLY_LABELS = {
            "pitch_semitones": "音高调整 pitch_semitones (半音)",
            "output_gain_db": "整体响度 output_gain_db (dB)",
            "noise_gain_db": "气流杂音 noise_gain_db (dB)",
            "glottal_rd_scale": "声门波形 glottal_rd_scale (气声 vs 嘎裂紧嗓)",
            "f1_scale": "第一共振峰比例 f1_scale (舌位高低 / 开口度)",
            "f2_scale": "第二共振峰比例 f2_scale (舌位前后)",
            "f1_hz": "第一共振峰绝对值 f1_hz (Hz)",
            "f2_hz": "第二共振峰绝对值 f2_hz (Hz)",
            "tilt_alpha_delta": "谱倾斜 tilt_alpha_delta (正数柔和暗淡 / 负数明亮尖锐)",
        }
        _FRIENDLY_INFOS = {
            "pitch_semitones": "正数提高音高，负数降低音高（12半音=1个八度）",
            "output_gain_db": "整体输出音频音量放大或衰减（分贝）",
            "noise_gain_db": "调节发音中的送气与摩擦白噪声强弱",
            "glottal_rd_scale": "控制声门闭合状态：>1.0 呈现漏气声/耳语感；<1.0 呈现紧喉/嘎裂声 (creaky)",
            "f1_scale": "按比例平移F1（保留自然语调轮廓）：>1.0 下降舌位/开口增大；<1.0 提升舌位",
            "f2_scale": "按比例平移F2：>1.0 舌位向前（如/u/变/i/）；<1.0 舌位向后",
            "f1_hz": "将第一共振峰强制固定在指定绝对Hz值，用于制作感知连续统",
            "f2_hz": "将第二共振峰强制固定在指定绝对Hz值",
            "tilt_alpha_delta": "改变高频衰减斜率：>0 柔和暗淡，<0 亮丽清脆",
        }

        sliders = [
            gr.Slider(
                minimum=spec.minimum,
                maximum=spec.maximum,
                value=spec.default,
                step=_SLIDER_STEPS.get(spec.unit, 0.01),
                label=_FRIENDLY_LABELS.get(spec.name, f"{spec.name} ({spec.unit})"),
                info=_FRIENDLY_INFOS.get(spec.name, spec.description),
                visible=False,
            )
            for spec in all_specs
        ]
        with gr.Row():
            name_tb = gr.Textbox(
                label="条件名称 (variant name)", placeholder="留空自动命名 (blank = auto-name)"
            )
            add_btn = gr.Button("添加为条件 (add variant)", variant="primary")
        with gr.Row():
            continuum_dd = gr.Dropdown(label="连续统控制 (continuum control)")
            continuum_lo = gr.Number(label="起点 (from)")
            continuum_hi = gr.Number(label="终点 (to)")
            continuum_steps = gr.Slider(2, 20, value=5, step=1, label="步数 (steps)")
            continuum_btn = gr.Button("生成连续统 (build continuum)")
        variants_df = gr.Dataframe(
            headers=["名称 (name)", "控制 (controls)"],
            interactive=False,
            label="条件列表 (variants) — 点击行以选择 (click a row to select)",
        )
        with gr.Row():
            delete_btn = gr.Button("删除所选 (delete selected)")
            clear_btn = gr.Button("清空 (clear)")

        gr.Markdown("## 渲染 (render)")
        render_btn = gr.Button("渲染全部条件 (render all variants)", variant="primary")
        output_note_md = gr.Markdown()

        gr.Markdown("## 结果 (results)")
        results_md = gr.Markdown()
        gr.Markdown("### A/B 试听与语谱图 (A/B listening and spectrograms)")
        with gr.Row():
            record_dd = gr.Dropdown(label="测试句 (record)")
            variant_dd = gr.Dropdown(label="变体 (variant)")
        with gr.Row():
            original_audio = gr.Audio(label="原始 (original)", type="filepath")
            variant_audio = gr.Audio(label="变体 (variant)", type="filepath")
        comparison_plot = gr.Plot(label="波形与语谱图 (waveforms and spectrograms)")

        experiment_outputs = [exp_state, checkpoint_dd, model_md, continuum_dd, record_dd, *sliders]
        experiment_dd.change(on_experiment, [experiment_dd], experiment_outputs)
        app.load(on_experiment, [experiment_dd], experiment_outputs)
        add_btn.click(
            add_variant,
            [exp_state, name_tb, variants_state, *sliders],
            [variants_state, variants_df, name_tb],
        )
        continuum_btn.click(
            add_continuum,
            [exp_state, continuum_dd, continuum_lo, continuum_hi, continuum_steps, variants_state],
            [variants_state, variants_df],
        )
        variants_df.select(on_row_select, None, selected_state)
        delete_btn.click(
            delete_selected,
            [variants_state, selected_state],
            [variants_state, variants_df, selected_state],
        )
        clear_btn.click(lambda: ([], [], None), None, [variants_state, variants_df, selected_state])
        render_btn.click(
            render,
            [exp_state, checkpoint_dd, variants_state],
            [render_state, results_md, variant_dd, output_note_md],
        )
        for dropdown in (record_dd, variant_dd):
            dropdown.change(
                compare,
                [exp_state, render_state, record_dd, variant_dd],
                [original_audio, variant_audio, comparison_plot],
            )
    return app


def launch_studio(
    workspace: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    share: bool = False,
) -> None:
    """Build and serve the studio; blocks until interrupted."""
    app = build_app(workspace)
    app.queue(default_concurrency_limit=1)
    url_hint = f"http://{host}:{port}/"
    print(f"ARIS Studio: {url_hint} (Ctrl+C 退出 / to stop)", flush=True)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=open_browser and not share,
        show_api=False,
        quiet=True,
    )


__all__ = [
    "auto_variant_name",
    "build_app",
    "continuum_values",
    "discover_checkpoints",
    "discover_experiments",
    "format_variant_metadata",
    "launch_studio",
    "spectrogram_figure",
    "variant_name",
]
