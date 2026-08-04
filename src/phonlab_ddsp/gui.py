"""Dependency-free local web workbench for common PhonLab-DDSP tasks."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

from .controls.specs import controls_for_model, parse_variant
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
from .webui_assets import FAVICON_SVG, GUI_HTML
from .webui_results import (
    create_export_zip,
    export_condition,
    export_wav,
    load_result_catalog,
)
from .webui_workspace import scan_workspace

MAX_REQUEST_BYTES = 64 * 1024


class Workbench:
    """Execute GUI actions without shell interpolation and serve generated reports."""

    def __init__(self, workspace_root: Optional[Path] = None):
        workspace = Path.cwd() if workspace_root is None else Path(workspace_root).expanduser()
        self.workspace_root = Path(os.path.abspath(os.fspath(workspace)))
        if self.workspace_root.parent == self.workspace_root:
            raise ValueError("WebUI workspace may not be a filesystem root")
        current = Path(self.workspace_root.anchor)
        parts = (
            self.workspace_root.parts[1:]
            if self.workspace_root.anchor
            else self.workspace_root.parts
        )
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"WebUI workspace contains a symbolic link: {current}")
        if not self.workspace_root.is_dir():
            raise FileNotFoundError(f"WebUI workspace does not exist: {self.workspace_root}")
        self.mounts: dict[str, Path] = {}
        self.lock = threading.Lock()

    def run(self, action: str, values: dict) -> dict:
        if action == "workspace-scan":
            result = scan_workspace(self.workspace_root)
            return {
                "message": f"已发现 {result['entry_count']} 个仓库内工作流路径",
                **result,
            }
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
        if action == "control-list":
            experiment = _path(values, "experiment")
            metadata = json.loads((experiment / "experiment.json").read_text())
            controls = controls_for_model(metadata["model"])
            return {
                "message": (
                    f"{metadata['model']} 声明支持 {len(controls)} 个控制参数；"
                    "checkpoint 将在 GPU 推理时再次检查"
                ),
                "model": metadata["model"],
                "controls": [asdict(item) for item in controls],
            }
        if action == "postprocess":
            shifts = _float_list(values, "semitones")
            variants = _variant_list(values, "variants")
            bundle = create_postprocess_job(
                _path(values, "experiment"),
                _path(values, "checkpoint"),
                _path(values, "output"),
                shifts,
                control_variants=variants,
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
        if action == "results-load":
            catalog = load_result_catalog(self.workspace_root, _path(values, "output"))
            result = catalog.to_dict()
            for report_name, relative in list(result["reports"].items()):
                file_url = self.mount(catalog.result_root, relative)
                result["reports"][report_name] = {
                    "name": report_name,
                    "path": relative,
                    "file_url": file_url,
                    "url": file_url,
                }
            for item in result["items"]:
                for audio in item["audio"]:
                    file_url = self.mount(catalog.result_root, audio["path"])
                    audio["name"] = Path(audio["path"]).name
                    audio["file_url"] = file_url
                    audio["download_url"] = f"{file_url}?download=1"
            summaries = [result["baseline"]["clipping"]]
            summaries.extend(condition["clipping"] for condition in result["conditions"])
            clipped_samples = sum(item["clipped_samples"] for item in summaries)
            samples = sum(item["samples"] for item in summaries)
            result["clipping"] = {
                "clipped_samples": clipped_samples,
                "samples": samples,
                "clipped_fraction": clipped_samples / samples if samples else 0.0,
                "files_with_clipping": sum(item["files_with_clipping"] for item in summaries),
            }
            result["wav_count"] = len(result["items"]) * (1 + len(result["conditions"]))
            return {
                "message": (
                    f"已加载 {len(result['items'])} 条语料、"
                    f"{len(result['conditions'])} 个 manipulation 条件"
                ),
                **result,
            }
        if action == "results-export":
            catalog = load_result_catalog(self.workspace_root, _path(values, "output"))
            condition = _text(values, "condition", "")
            if not condition:
                raise ValueError("condition is required")
            scope = _text(values, "scope", "wav")
            destination = _path(values, "destination")
            if scope == "wav":
                item_id = _text(values, "item", "") or _text(values, "item_id", "")
                if not item_id:
                    raise ValueError("item is required for a single-WAV export")
                receipt = export_wav(catalog, condition, item_id, destination)
            elif scope == "condition":
                receipt = export_condition(catalog, condition, destination)
            else:
                raise ValueError("scope must be 'wav' or 'condition'")
            return {"message": "WAV 与 provenance 已另存", **receipt.to_dict()}
        if action == "results-zip":
            catalog = load_result_catalog(self.workspace_root, _path(values, "output"))
            condition = _text(values, "condition", "")
            if not condition:
                raise ValueError("condition is required")
            scope = _text(values, "scope", "condition")
            if scope not in {"wav", "condition"}:
                raise ValueError("scope must be 'wav' or 'condition'")
            item_id = None
            if scope == "wav":
                item_id = _text(values, "item", "") or _text(values, "item_id", "")
                if not item_id:
                    raise ValueError("item is required for a single-WAV ZIP")
            archive = create_export_zip(catalog, condition, item_id=item_id)
            file_url = self.mount(archive.parent, archive.name)
            return {
                "message": "包含 WAV 与 provenance 的 ZIP 已生成",
                "archive": str(archive),
                "file_url": file_url,
                "download_url": f"{file_url}?download=1",
            }
        raise ValueError(f"Unknown GUI action: {action}")

    def mount(self, root: Path, relative: str) -> str:
        root = Path(root).resolve()
        token = hashlib.sha256(str(root).encode()).hexdigest()[:16]
        with self.lock:
            self.mounts[token] = root
        return f"/files/{token}/{quote(relative, safe='/')}"

    def resolve_file(self, token: str, relative: str) -> Path:
        with self.lock:
            root = self.mounts.get(token)
        if root is None:
            raise FileNotFoundError("Unknown report mount")
        decoded = unquote(relative)
        lexical_target = root / decoded
        current = root
        for part in Path(decoded).parts:
            current = current / part
            if current.is_symlink():
                raise PermissionError("Symbolic links are not served by the WebUI")
        target = lexical_target.resolve()
        if target != root and root not in target.parents:
            raise PermissionError("Path escapes the report directory")
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError(target)
        return target


def serve_gui(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    workspace_root: Optional[Path] = None,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "The GUI can read and write local paths, so it only binds to loopback. "
            "Use an SSH tunnel for remote access."
        )
    workbench = Workbench(workspace_root)
    server = ThreadingHTTPServer((host, port), _handler(workbench))
    actual_port = server.server_address[1]
    display_host = f"[{host}]" if ":" in host else host
    url = f"http://{display_host}:{actual_port}/"
    print(f"PhonLab-DDSP GUI: {url}")
    print(f"Workspace: {workbench.workspace_root}")
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
        server_version = "PhonLabWorkbench/0.2"

        def do_GET(self):
            self._serve_get(head_only=False)

        def do_HEAD(self):
            self._serve_get(head_only=True)

        def _serve_get(self, *, head_only: bool):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._bytes(
                    HTTPStatus.OK,
                    GUI_HTML.encode(),
                    "text/html; charset=utf-8",
                    head_only=head_only,
                )
                return
            if parsed.path == "/favicon.svg":
                self._bytes(
                    HTTPStatus.OK,
                    FAVICON_SVG.encode(),
                    "image/svg+xml; charset=utf-8",
                    head_only=head_only,
                )
                return
            if parsed.path.startswith("/files/"):
                parts = parsed.path.split("/", 3)
                if len(parts) != 4:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    path = workbench.resolve_file(parts[2], parts[3])
                    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    download = parse_qs(parsed.query).get("download", [""])[-1].lower()
                    self._file(
                        path,
                        content_type,
                        download=download in {"1", "true", "yes"},
                        head_only=head_only,
                    )
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

        def _bytes(
            self,
            status: HTTPStatus,
            content: bytes,
            content_type: str,
            *,
            head_only: bool = False,
        ):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self._security_headers()
            self.end_headers()
            if not head_only:
                self.wfile.write(content)

        def _file(
            self,
            path: Path,
            content_type: str,
            *,
            download: bool,
            head_only: bool,
        ):
            size = path.stat().st_size
            try:
                byte_range = _parse_byte_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.send_header("Accept-Ranges", "bytes")
                self._security_headers()
                self.end_headers()
                return

            if byte_range is None:
                start, end = 0, max(size - 1, 0)
                status = HTTPStatus.OK
            else:
                start, end = byte_range
                status = HTTPStatus.PARTIAL_CONTENT
            length = 0 if size == 0 else end - start + 1

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if byte_range is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:
                ascii_name = "".join(
                    character
                    if 32 <= ord(character) < 127 and character not in {'"', "\\"}
                    else "_"
                    for character in path.name
                )
                encoded_name = quote(path.name, safe="")
                self.send_header(
                    "Content-Disposition",
                    f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded_name}",
                )
            self._security_headers()
            self.end_headers()
            if head_only or length == 0:
                return

            remaining = length
            with path.open("rb") as stream:
                stream.seek(start)
                while remaining:
                    chunk = stream.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _security_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; media-src 'self'; connect-src 'self'",
            )

    return Handler


def _parse_byte_range(header: Optional[str], size: int) -> Optional[tuple[int, int]]:
    """Parse one RFC 7233 byte range for local audio seeking."""

    if not header:
        return None
    if size <= 0 or not header.startswith("bytes=") or "," in header:
        raise ValueError("Unsupported byte range")
    value = header.removeprefix("bytes=").strip()
    if value.count("-") != 1:
        raise ValueError("Invalid byte range")
    raw_start, raw_end = (part.strip() for part in value.split("-", 1))
    if not raw_start:
        if not raw_end.isdigit() or int(raw_end) <= 0:
            raise ValueError("Invalid suffix byte range")
        suffix = min(int(raw_end), size)
        return size - suffix, size - 1
    if not raw_start.isdigit():
        raise ValueError("Invalid byte range start")
    start = int(raw_start)
    if start >= size:
        raise ValueError("Byte range starts after the file")
    if raw_end:
        if not raw_end.isdigit():
            raise ValueError("Invalid byte range end")
        end = min(int(raw_end), size - 1)
        if end < start:
            raise ValueError("Byte range end precedes start")
    else:
        end = size - 1
    return start, end


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
        return []
    parts = raw.replace(",", " ").split()
    return [float(value) for value in parts]


def _variant_list(values: dict, name: str):
    raw = _text(values, name, "")
    if not raw:
        return []
    entries = [entry.strip() for entry in raw.replace(";", "\n").splitlines()]
    return [parse_variant(entry) for entry in entries if entry]
