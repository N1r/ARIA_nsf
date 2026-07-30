"""Dependency-free local web workbench for common PhonLab-DDSP tasks."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .corpus import acquire_cmu_arctic
from .doctor import checks
from .experiment import create_experiment
from .jobs import cancel_job, query_job, read_job_log, submit_job
from .launchers import create_postprocess_job
from .manifest import DatasetManifest, prepare_dataset, summarize
from .metrics import build_metrics_dashboard, load_metrics, summarize_metrics
from .parameters import export_parameters, parameter_summary
from .report import build_report
from .segment import split_audio, split_summary

MAX_REQUEST_BYTES = 64 * 1024


class Workbench:
    """Execute GUI actions without shell interpolation and serve generated reports."""

    def __init__(self):
        self.mounts: dict[str, Path] = {}
        self.lock = threading.Lock()

    def run(self, action: str, values: dict) -> dict:
        if action == "doctor":
            items = [
                {
                    "name": item.name,
                    "ok": item.ok,
                    "detail": item.detail,
                    "required_for": item.required_for,
                }
                for item in checks()
            ]
            return {"message": "环境检查完成", "checks": items}
        if action == "corpus":
            archive_text = _text(values, "archive", "")
            result = acquire_cmu_arctic(
                _path(values, "output"),
                target_duration_s=_float(values, "target_minutes", 30.0) * 60,
                max_duration_s=_float(values, "max_minutes", 60.0) * 60,
                silence_gap_s=_float(values, "silence_gap", 0.35),
                archive_source=Path(archive_text).expanduser() if archive_text else None,
            )
            metadata = result.metadata
            return {
                "message": (
                    f"完成：固定 {metadata['selection']['utterance_count']} 条、"
                    f"{metadata['selection']['selected_duration_s'] / 60:.1f} 分钟语音"
                ),
                "corpus": metadata["corpus"],
                "archive_sha256": metadata["source"]["sha256"],
                "continuous_audio": (str(result.continuous_wav) if result.continuous_wav else None),
                "metadata": str(result.selected_dir.parent / "corpus.json"),
                "suggested_segments": str(result.selected_dir.parent.parent / "segments"),
                "next_command": (
                    f"phonlab split {result.continuous_wav} SEGMENTS_OUTPUT"
                    if result.continuous_wav
                    else f"phonlab prepare {result.selected_dir} DATASET_OUTPUT"
                ),
            }
        if action == "split":
            result = split_audio(
                _path(values, "source"),
                _path(values, "output"),
                mode=_text(values, "mode", "silence"),
                segment_seconds=_float(values, "segment_seconds", 2.0),
                overlap_seconds=_float(values, "overlap_seconds", 0.0),
                silence_threshold_db=_float(values, "silence_threshold_db", -40.0),
                min_silence_seconds=_float(values, "min_silence_seconds", 0.30),
                padding_seconds=_float(values, "padding_seconds", 0.05),
                min_duration_seconds=_float(values, "min_duration_seconds", 0.25),
                max_duration_seconds=_float(values, "max_duration_seconds", 15.0),
                sample_rate=_optional_int(values, "sample_rate"),
                keep_tail=_bool(values, "keep_tail", True),
                split_f0_sidecars=_bool(values, "split_f0_sidecars", False),
                f0_hop_seconds=_float(values, "f0_hop_seconds", 0.005),
            )
            summary = split_summary(result)
            return {
                "message": f"完成：生成 {summary['segments']} 个片段",
                "summary": summary,
                "segments_audio": str(result.root / "audio"),
                "suggested_dataset": str(result.root.parent / "dataset"),
                "next_command": f"phonlab prepare {result.root / 'audio'} DATASET_OUTPUT",
            }
        if action == "prepare":
            manifest = prepare_dataset(
                _path(values, "source"),
                _path(values, "output"),
                sample_rate=_int(values, "sample_rate", 16000),
                f0_method=_text(values, "f0_method", "autocorr"),
                f0_floor=_float(values, "f0_floor", 60.0),
                f0_ceiling=_float(values, "f0_ceiling", 500.0),
                validation_ratio=_float(values, "validation_ratio", 0.1),
                test_ratio=_float(values, "test_ratio", 0.1),
                seed=_int(values, "seed", 42),
                min_duration=_float(values, "min_duration", 0.25),
                normalize_peak=_optional_float(values, "normalize_peak"),
            )
            return {
                "message": f"完成：准备 {len(manifest.records)} 条录音",
                "summary": summarize(manifest.records),
                "fingerprint": manifest.fingerprint,
                "dataset": str(manifest.root),
                "suggested_experiment": str(manifest.root.parent / "experiment"),
                "next_command": f"phonlab inspect {manifest.root}",
            }
        if action == "inspect":
            dataset = _path(values, "dataset")
            manifest = DatasetManifest.load(dataset)
            output_text = _text(values, "output", "")
            output = (
                Path(output_text).expanduser().resolve()
                if output_text
                else manifest.root / "report.html"
            )
            if manifest.root not in output.parents:
                raise ValueError("GUI report output must be inside the prepared dataset")
            report = build_report(manifest, output)
            parameters = manifest.root / "parameters.csv"
            if not parameters.exists():
                export_parameters(manifest, parameters)
            return {
                "message": "质检报告与逐条声学参数表已生成",
                "report": str(report),
                "parameters": str(parameters),
                "parameter_summary": parameter_summary(manifest),
                "report_url": self.mount(
                    manifest.root, report.relative_to(manifest.root).as_posix()
                ),
                "summary": summarize(manifest.records),
            }
        if action == "experiment":
            experiment = create_experiment(
                _path(values, "dataset"),
                _path(values, "output"),
                model=_text(values, "model", "golf"),
                batch_size=_int(values, "batch_size", 32),
                max_steps=_int(values, "max_steps", 40000),
                seed=_int(values, "seed", 42),
                f0_min=_float(values, "f0_min", 60.0),
                f0_max=_float(values, "f0_max", 500.0),
                workers=_int(values, "workers", 4),
                slurm_partition=_text(values, "partition", "gpu-short"),
                slurm_gres=_text(values, "gres", "gpu:l4:1"),
                slurm_time=_text(values, "time_limit", "04:00:00"),
                slurm_cpus=_int(values, "cpus", 8),
                slurm_memory=_text(values, "memory", "32G"),
                slurm_exclude=_text(values, "exclude", "node857"),
            )
            return {
                "message": "训练实验已建立；尚未启动训练",
                "experiment": str(experiment),
                "suggested_postprocess": str(experiment.parent / "postprocess"),
                "dry_run": f"phonlab train {experiment} --dry-run",
                "slurm": f"sbatch {experiment / 'train.slurm'}",
            }
        if action == "metrics":
            experiment = _path(values, "experiment").resolve()
            output_text = _text(values, "output", "")
            output = (
                Path(output_text).expanduser().resolve()
                if output_text
                else experiment / "metrics.html"
            )
            if experiment not in output.parents:
                raise ValueError("GUI metrics output must be inside the experiment")
            version = _text(values, "version", "") or None
            report = build_metrics_dashboard(experiment, output, version)
            data = load_metrics(experiment, version)
            return {
                "message": "训练指标报告已生成",
                "summary": summarize_metrics(data),
                "report": str(report),
                "report_url": self.mount(experiment, report.relative_to(experiment).as_posix()),
            }
        if action == "job-submit":
            if not _bool(values, "confirm", False):
                raise ValueError("请明确勾选提交确认")
            experiment = _path(values, "experiment")
            job_id = submit_job(experiment)
            return {
                "message": f"Slurm 作业已提交：{job_id}",
                "job_id": job_id,
                "status": query_job(job_id).to_dict(),
            }
        if action == "job-status":
            job_id = _text(values, "job_id", "")
            status = query_job(job_id)
            result = {
                "message": f"作业 {job_id}：{status.state}",
                "status": status.to_dict(),
            }
            experiment_text = _text(values, "experiment", "")
            if experiment_text:
                result["log"] = read_job_log(
                    Path(experiment_text).expanduser(),
                    job_id,
                    tail_lines=200,
                )
            return result
        if action == "job-cancel":
            if _text(values, "confirmation", "") != "CANCEL":
                raise ValueError("取消作业前必须输入 CANCEL")
            job_id = _text(values, "job_id", "")
            cancel_job(job_id)
            return {"message": f"已请求取消 Slurm 作业 {job_id}", "job_id": job_id}
        if action == "postprocess":
            shifts = _float_list(values, "semitones")
            bundle = create_postprocess_job(
                _path(values, "experiment"),
                _path(values, "checkpoint"),
                _path(values, "output"),
                shifts,
                partition=_text(values, "partition", "gpu-short"),
                gres=_text(values, "gres", "gpu:1"),
                time_limit=_text(values, "time_limit", "00:30:00"),
                cpus=_int(values, "cpus", 4),
                memory=_text(values, "memory", "24G"),
                exclude=_text(values, "exclude", ""),
            )
            return {
                "message": "GPU 推理与 manipulation 作业已生成；尚未提交",
                "job_bundle": str(bundle),
                "submit_hint": "在 Slurm 作业中心选择此目录并确认提交",
            }
        raise ValueError(f"Unknown GUI action: {action}")

    def mount(self, root: Path, relative: str) -> str:
        root = Path(root).resolve()
        token = hashlib.sha256(str(root).encode()).hexdigest()[:16]
        with self.lock:
            self.mounts[token] = root
        return f"/files/{token}/{relative}"

    def resolve_file(self, token: str, relative: str) -> Path:
        with self.lock:
            root = self.mounts.get(token)
        if root is None:
            raise FileNotFoundError("Unknown report mount")
        target = (root / unquote(relative)).resolve()
        if target != root and root not in target.parents:
            raise PermissionError("Path escapes the report directory")
        if not target.is_file():
            raise FileNotFoundError(target)
        return target


def serve_gui(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "The GUI can read and write local paths, so it only binds to loopback. "
            "Use an SSH tunnel for remote access."
        )
    workbench = Workbench()
    server = ThreadingHTTPServer((host, port), _handler(workbench))
    actual_port = server.server_address[1]
    display_host = f"[{host}]" if ":" in host else host
    url = f"http://{display_host}:{actual_port}/"
    print(f"PhonLab-DDSP GUI: {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGUI stopped.")
    finally:
        server.server_close()


def _handler(workbench: Workbench):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PhonLabWorkbench/0.1"

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._bytes(HTTPStatus.OK, GUI_HTML.encode(), "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/files/"):
                parts = parsed.path.split("/", 3)
                if len(parts) != 4:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    path = workbench.resolve_file(parts[2], parts[3])
                    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    self._bytes(HTTPStatus.OK, path.read_bytes(), content_type)
                except (FileNotFoundError, PermissionError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in {"127.0.0.1", "localhost", "::1"}:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                values = json.loads(self.rfile.read(size))
                if not isinstance(values, dict):
                    raise ValueError("JSON body must be an object")
                result = workbench.run(parsed.path.removeprefix("/api/"), values)
                self._json(HTTPStatus.OK, {"ok": True, **result})
            except (
                FileExistsError,
                FileNotFoundError,
                json.JSONDecodeError,
                PermissionError,
                RuntimeError,
                ValueError,
            ) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(error)})

        def log_message(self, message, *args):
            print(f"[GUI] {self.address_string()} {message % args}")

        def _json(self, status: HTTPStatus, value: dict):
            self._bytes(
                status,
                json.dumps(value, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )

        def _bytes(self, status: HTTPStatus, content: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; media-src 'self'; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(content)

    return Handler


def _text(values: dict, name: str, default: str) -> str:
    value = values.get(name, default)
    return str(value).strip()


def _path(values: dict, name: str) -> Path:
    value = _text(values, name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return Path(value).expanduser()


def _float(values: dict, name: str, default: float) -> float:
    value = _text(values, name, "")
    return default if value == "" else float(value)


def _optional_float(values: dict, name: str):
    value = _text(values, name, "")
    return None if value == "" else float(value)


def _int(values: dict, name: str, default: int) -> int:
    value = _text(values, name, "")
    return default if value == "" else int(value)


def _optional_int(values: dict, name: str):
    value = _text(values, name, "")
    return None if value == "" else int(value)


def _bool(values: dict, name: str, default: bool) -> bool:
    value = values.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _float_list(values: dict, name: str) -> list[float]:
    raw = _text(values, name, "")
    if not raw:
        raise ValueError(f"{name} is required")
    parts = raw.replace(",", " ").split()
    return [float(value) for value in parts]


GUI_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhonLab-DDSP 工作台</title>
<style>
:root{--ink:#172129;--muted:#66727a;--paper:#f6f3ed;--card:#fffdfa;--line:#d9d3c8;
--accent:#176b5b;--accent2:#d66b35;--good:#e1f2eb;--bad:#fff0ec}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Noto Sans SC",sans-serif}
header{background:#163b36;color:white;padding:2.6rem max(1.2rem,calc((100% - 1120px)/2))}
header h1{font:700 clamp(2rem,5vw,3.8rem)/1.05 Georgia,serif;margin:0 0 .6rem}
header p{max-width:760px;margin:0;color:#d8e9e4;font-size:1.05rem}
main{max-width:1120px;margin:2rem auto;padding:0 1.2rem 4rem}
.toolbar{display:flex;gap:.8rem;align-items:center;flex-wrap:wrap;margin-bottom:1.2rem}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.3rem;
box-shadow:0 4px 18px #4d463b0b}.card h2{margin:0 0 .2rem;font:700 1.4rem Georgia,serif}
.card>p{color:var(--muted);margin:.2rem 0 1rem}.fields{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}
label{display:flex;flex-direction:column;gap:.28rem;font-size:.82rem;color:#4b565c}
label.wide{grid-column:1/-1}input,select{width:100%;border:1px solid #c9c2b7;border-radius:8px;
background:white;color:var(--ink);padding:.62rem .68rem;font:inherit}
input:focus,select:focus{outline:2px solid #78b5aa;outline-offset:1px}
.check{flex-direction:row;align-items:center;margin-top:.45rem}.check input{width:auto}
button{border:0;border-radius:9px;padding:.67rem 1rem;background:var(--accent);color:white;
font-weight:650;cursor:pointer}button:hover{filter:brightness(1.08)}button.secondary{background:#42525a}
.submit{margin-top:1rem}.result{display:none;margin-top:1rem;padding:.8rem;border-radius:9px;
white-space:pre-wrap;overflow-wrap:anywhere;background:var(--good)}.result.error{background:var(--bad);color:#8a291b}
.status{color:var(--muted)}code{background:#ebe7df;padding:.12rem .32rem;border-radius:4px}
details{margin-top:.8rem}summary{cursor:pointer;color:var(--accent)}
@media(max-width:760px){.grid{grid-template-columns:1fr}.fields{grid-template-columns:1fr}
label.wide{grid-column:auto}header{padding:2rem 1.2rem}}
</style>
</head>
<body>
<header><h1>PhonLab-DDSP 工作台</h1>
<p>从示例语料或自己的录音出发，完成切分、参数提取、质检、Slurm 训练、loss 检查、checkpoint 推理和可试听的音高操控。</p></header>
<main>
<div class="toolbar">
  <button class="secondary" onclick="doctor()">检查环境</button>
  <span class="status" id="global-status">服务仅在本机运行；训练只会在勾选确认后提交至 Slurm。</span>
</div>
<div class="grid">
<section class="card">
<h2>0 · 获取可复现示例语料</h2>
<p>下载许可清楚的 CMU ARCTIC 单说话人英语语料，并按官方顺序固定 30–60 分钟子集。</p>
<form data-action="corpus"><div class="fields">
<label class="wide">新输出目录<input name="output" required placeholder="/path/to/cmu-arctic-slt"></label>
<label class="wide">已有官方压缩包（可选）<input name="archive" placeholder="/path/to/cmu_us_slt_arctic-0.95-release.tar.bz2"></label>
<label>目标时长（分钟）<input name="target_minutes" type="number" step=".5" value="30"></label>
<label>最长时长（分钟）<input name="max_minutes" type="number" step=".5" value="60"></label>
<label>话语间静音（秒）<input name="silence_gap" type="number" step=".05" value=".35"></label>
</div><button class="submit">获取并校验语料</button><div class="result"></div></form>
</section>

<section class="card">
<h2>1 · 音频切分</h2><p>静音边界适合话语切分；固定窗口适合等长训练样本。</p>
<form data-action="split"><div class="fields">
<label class="wide">源音频文件或目录<input name="source" required placeholder="/path/to/recordings"></label>
<label class="wide">新输出目录<input name="output" required placeholder="/path/to/segments"></label>
<label>切分模式<select name="mode"><option value="silence">静音边界</option><option value="fixed">固定时长</option></select></label>
<label>输出采样率（留空则保留）<input name="sample_rate" type="number" placeholder="16000"></label>
<label>固定片段秒数<input name="segment_seconds" type="number" step=".05" value="2"></label>
<label>重叠秒数<input name="overlap_seconds" type="number" step=".05" value="0"></label>
<label>静音阈值 dBFS<input name="silence_threshold_db" type="number" step="1" value="-40"></label>
<label>最短静音秒数<input name="min_silence_seconds" type="number" step=".05" value=".30"></label>
<label>边界留白秒数<input name="padding_seconds" type="number" step=".01" value=".05"></label>
<label>最短片段秒数<input name="min_duration_seconds" type="number" step=".05" value=".25"></label>
<label>最长片段秒数<input name="max_duration_seconds" type="number" step=".5" value="15"></label>
<label class="check"><input name="keep_tail" type="checkbox" checked>保留固定模式末尾片段</label>
<label class="check"><input name="split_f0_sidecars" type="checkbox">同步切分同名 .pv F0</label>
<label>F0 帧移秒数<input name="f0_hop_seconds" type="number" step=".001" value=".005"></label>
</div><button class="submit">开始切分</button><div class="result"></div></form>
</section>

<section class="card">
<h2>2 · 准备数据集</h2><p>统一音频、提取或复用 F0，并生成确定性数据划分和哈希。</p>
<form data-action="prepare"><div class="fields">
<label class="wide">录音目录<input name="source" required placeholder="/path/to/segments/audio"></label>
<label class="wide">新数据集目录<input name="output" required placeholder="/path/to/dataset"></label>
<label>F0 方法<select name="f0_method"><option value="autocorr">自动相关（易安装）</option><option value="sidecar">同名 .pv</option><option value="auto">自动选择</option><option value="pyworld">pyworld</option></select></label>
<label>采样率<input name="sample_rate" type="number" value="16000"></label>
<label>F0 下限 Hz<input name="f0_floor" type="number" value="60"></label>
<label>F0 上限 Hz<input name="f0_ceiling" type="number" value="500"></label>
<label>验证集比例<input name="validation_ratio" type="number" step=".01" value=".1"></label>
<label>测试集比例<input name="test_ratio" type="number" step=".01" value=".1"></label>
<label>随机种子<input name="seed" type="number" value="42"></label>
<label>最短录音秒数<input name="min_duration" type="number" step=".05" value=".25"></label>
<label>峰值归一化（默认关闭）<input name="normalize_peak" type="number" step=".05" placeholder="0.95"></label>
</div><button class="submit">准备数据</button><div class="result"></div></form>
</section>

<section class="card">
<h2>3 · 质检与试听</h2><p>生成包含时长、F0、削波、数据来源和逐条试听的离线报告。</p>
<form data-action="inspect"><div class="fields">
<label class="wide">已准备的数据集<input name="dataset" required placeholder="/path/to/dataset"></label>
<label class="wide">报告路径（留空使用 dataset/report.html）<input name="output" placeholder="/path/to/report.html"></label>
</div><button class="submit">生成报告</button><div class="result"></div></form>
</section>

<section class="card">
<h2>4 · 建立训练实验</h2><p>生成配置、provenance 和 Slurm 启动器；GPU 训练需在终端提交。</p>
<form data-action="experiment"><div class="fields">
<label class="wide">已准备的数据集<input name="dataset" required placeholder="/path/to/dataset"></label>
<label class="wide">新实验目录<input name="output" required placeholder="/path/to/experiment"></label>
<label>模型<select name="model"><option value="golf">GOLF</option><option value="ddsp">DDSP</option><option value="aria-golf">ARIA-GOLF</option></select></label>
<label>Batch size<input name="batch_size" type="number" value="32"></label>
<label>训练步数<input name="max_steps" type="number" value="40000"></label>
<label>Data workers<input name="workers" type="number" value="4"></label>
<label>F0 下限 Hz<input name="f0_min" type="number" value="60"></label>
<label>F0 上限 Hz<input name="f0_max" type="number" value="500"></label>
<label>随机种子<input name="seed" type="number" value="42"></label>
<label>Slurm partition<input name="partition" value="gpu-short"></label>
<label>GPU GRES<input name="gres" value="gpu:l4:1"></label>
<label>时间上限<input name="time_limit" value="04:00:00"></label>
<label>CPU<input name="cpus" type="number" value="8"></label>
<label>内存<input name="memory" value="32G"></label>
<label>排除节点（可留空）<input name="exclude" value="node857"></label>
</div><button class="submit">生成实验</button><div class="result"></div></form>
</section>

<section class="card">
<h2>5 · Loss 与训练指标</h2><p>读取 Lightning CSV，绘制 train/validation loss、学习率并标出 NaN/Inf。</p>
<form data-action="metrics"><div class="fields">
<label class="wide">实验目录<input name="experiment" required placeholder="/path/to/experiment"></label>
<label class="wide">报告路径（留空使用 experiment/metrics.html）<input name="output" placeholder="/path/to/experiment/metrics.html"></label>
<label>日志版本（留空取最新）<input name="version" placeholder="latest"></label>
</div><button class="submit">生成指标报告</button><div class="result"></div></form>
</section>

<section class="card">
<h2>6 · Slurm 作业中心</h2><p>所有调度命令均使用参数数组；提交和取消需要显式确认。</p>
<form data-action="job-submit"><div class="fields">
<label class="wide">训练实验或作业包目录<input name="experiment" required placeholder="/path/to/job-bundle"></label>
<label class="check wide"><input name="confirm" type="checkbox">确认向 Slurm 提交该作业</label>
</div><button class="submit">提交作业</button><div class="result"></div></form>
<details><summary>查询状态与末尾200行日志</summary>
<form data-action="job-status"><div class="fields">
<label>Job ID<input name="job_id" required placeholder="4553909"></label>
<label>实验/作业包目录（可选）<input name="experiment" placeholder="/path/to/job-bundle"></label>
</div><button class="submit">刷新状态</button><div class="result"></div></form></details>
<details><summary>取消作业</summary>
<form data-action="job-cancel"><div class="fields">
<label>Job ID<input name="job_id" required></label>
<label>输入 CANCEL 确认<input name="confirmation" required></label>
</div><button class="submit secondary">取消作业</button><div class="result"></div></form></details>
</section>

<section class="card">
<h2>7 · 推理与 Manipulation</h2><p>生成GPU后处理作业：重建测试集、±半音F0操控、试听页和指标报告。</p>
<form data-action="postprocess"><div class="fields">
<label class="wide">实验目录<input name="experiment" required placeholder="/path/to/experiment"></label>
<label class="wide">Checkpoint<input name="checkpoint" required placeholder="/path/to/last.ckpt"></label>
<label class="wide">新输出目录<input name="output" required placeholder="/path/to/postprocess-output"></label>
<label>半音偏移（空格或逗号）<input name="semitones" value="-4, 4"></label>
<label>Partition<input name="partition" value="gpu-short"></label>
<label>GRES<input name="gres" value="gpu:l4:1"></label>
<label>时间<input name="time_limit" value="00:30:00"></label>
<label>CPU<input name="cpus" type="number" value="4"></label>
<label>内存<input name="memory" value="24G"></label>
<label>排除节点（可选）<input name="exclude" placeholder="node857"></label>
</div><button class="submit">生成后处理作业</button><div class="result"></div></form>
</section>
</div>
</main>
<script>
function values(form){
  const out={}; new FormData(form).forEach((v,k)=>out[k]=v);
  form.querySelectorAll('input[type=checkbox]').forEach(x=>out[x.name]=x.checked);
  return out;
}
function fill(action,name,value){
  if(!value)return;
  document.querySelectorAll('form[data-action="'+action+'"] [name="'+name+'"]').forEach(x=>{
    if(!x.value)x.value=value;
  });
}
function cascade(data){
  fill('split','source',data.continuous_audio);
  fill('split','output',data.suggested_segments);
  fill('prepare','source',data.segments_audio);
  fill('prepare','output',data.suggested_dataset);
  ['inspect','experiment'].forEach(a=>fill(a,'dataset',data.dataset));
  fill('experiment','output',data.suggested_experiment);
  ['metrics','job-submit','postprocess'].forEach(a=>fill(a,'experiment',data.experiment));
  fill('postprocess','output',data.suggested_postprocess);
  fill('job-submit','experiment',data.job_bundle);
  ['job-status','job-cancel'].forEach(a=>fill(a,'job_id',data.job_id));
}
function render(box,data){
  box.classList.toggle('error',!data.ok); box.style.display='block';
  if(!data.ok){box.textContent='错误：'+data.error;return}
  cascade(data);
  const copy={...data}; delete copy.ok; const url=copy.report_url; delete copy.report_url;
  box.textContent=JSON.stringify(copy,null,2);
  if(url){const a=document.createElement('a');a.href=url;a.target='_blank';a.textContent='\n打开质检报告 ↗';box.appendChild(a)}
}
document.querySelectorAll('form[data-action]').forEach(form=>form.addEventListener('submit',async e=>{
  e.preventDefault(); const button=form.querySelector('button'); const box=form.querySelector('.result');
  const label=button.textContent;
  button.disabled=true; button.textContent='处理中…'; box.style.display='none';
  try{
    const response=await fetch('/api/'+form.dataset.action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values(form))});
    render(box,await response.json());
  }catch(error){render(box,{ok:false,error:String(error)})}
  finally{button.disabled=false;button.textContent=label}
}));
async function doctor(){
  const status=document.getElementById('global-status');status.textContent='检查中…';
  try{const r=await fetch('/api/doctor',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  const d=await r.json();status.textContent=d.ok?d.checks.map(x=>(x.ok?'✓ ':'– ')+x.name).join(' · '):d.error}
  catch(e){status.textContent=String(e)}
}
</script>
</body></html>
"""
