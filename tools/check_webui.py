#!/usr/bin/env python3
"""CPU-only acceptance check for the local PhonLab WebUI results workflow.

The checker starts the real dependency-free HTTP handler on an ephemeral
loopback port.  It never invokes training, inference, Slurm, or a non-loopback
URL.  All inputs, generated ZIP files, optional exports, and report output are
confined to the selected workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import zipfile
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from phonlab_ddsp.gui import Workbench, _handler

SUCCESS_TOKEN = "PHONLAB_WEBUI_OK"
REPORT_SCHEMA_VERSION = "1.0"
DEFAULT_RESULT = Path("artifacts/cmu_arctic_slt_demo/control_postprocess_v2")
HTTP_TIMEOUT_SECONDS = 15.0


def check_webui(
    workspace: Optional[Path] = None,
    result: Optional[Path] = None,
    *,
    check_export: bool = False,
) -> Dict[str, Any]:
    """Exercise the WebUI catalog, HTTP media, ZIP, and optional export APIs.

    ``result`` may be absolute or relative to ``workspace``.  The returned
    dictionary is deliberately compact: it records counts and the one selected
    WAV, but does not copy the potentially large catalog into the report.
    """

    raw_workspace = _default_workspace() if workspace is None else Path(workspace)
    raw_result = DEFAULT_RESULT if result is None else Path(result)
    report: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool": "check_webui",
        "success": False,
        "success_token": None,
        "inputs": {
            "workspace": str(raw_workspace),
            "result": str(raw_result),
            "check_export": bool(check_export),
        },
        "checks": [],
        "issues": [],
    }

    try:
        workspace_root = _existing_workspace(raw_workspace)
        result_root = _existing_result(workspace_root, raw_result)
    except (OSError, ValueError) as error:
        _failure(report, "inputs", "inputs.invalid", str(error))
        return _finish(report)

    report["inputs"]["workspace"] = str(workspace_root)
    report["inputs"]["result"] = str(result_root)
    try:
        workbench = Workbench(workspace_root=workspace_root)
    except (OSError, TypeError, ValueError) as error:
        _failure(report, "workbench", "workbench.init", str(error))
        return _finish(report)

    archive_to_remove: Optional[Path] = None
    try:
        with _loopback_server(workbench) as (base_url, server_details):
            _success(report, "server", server_details)

            try:
                homepage = _http_request(base_url, "/")
                _validate_homepage(homepage)
                _success(
                    report,
                    "homepage",
                    {
                        "status": homepage[0],
                        "content_type": homepage[1].get("Content-Type", ""),
                        "bytes": len(homepage[2]),
                    },
                )
            except (OSError, RuntimeError, ValueError) as error:
                _failure(report, "homepage", "http.homepage", str(error))

            try:
                catalog = workbench.run("results-load", {"output": str(result_root)})
                selection, catalog_details = _validate_catalog(catalog, result_root)
                _success(report, "catalog", catalog_details)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                _failure(report, "catalog", "results.load", str(error))
                return _finish(report)

            try:
                range_details = _check_wav_range(base_url, selection)
                _success(report, "wav_range", range_details)
            except (OSError, RuntimeError, ValueError) as error:
                _failure(report, "wav_range", "http.wav_range", str(error))

            try:
                download_details = _check_wav_download(base_url, selection)
                _success(report, "wav_download", download_details)
            except (OSError, RuntimeError, ValueError) as error:
                _failure(report, "wav_download", "http.wav_download", str(error))

            zip_response: Optional[Mapping[str, Any]] = None
            try:
                zip_response = workbench.run(
                    "results-zip",
                    {
                        "output": str(result_root),
                        "condition": selection["condition"],
                        "item": selection["item_id"],
                        "scope": "wav",
                    },
                )
                archive_to_remove = _response_workspace_file(
                    zip_response,
                    "archive",
                    workspace_root,
                    suffix=".zip",
                )
                cache_root = workspace_root / ".cache"
                if not _is_beneath(archive_to_remove, cache_root):
                    raise ValueError("results-zip archive is not inside workspace/.cache")
                zip_details = _validate_zip(
                    archive_to_remove,
                    selection,
                    zip_response,
                    base_url,
                )
                _success(report, "results_zip", zip_details)
            except (OSError, RuntimeError, TypeError, ValueError, zipfile.BadZipFile) as error:
                _failure(report, "results_zip", "results.zip", str(error))

            if check_export:
                try:
                    export_details = _check_safe_export(
                        workbench,
                        workspace_root,
                        result_root,
                        selection,
                    )
                    _success(report, "results_export", export_details)
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    _failure(report, "results_export", "results.export", str(error))
    except (OSError, RuntimeError, ValueError) as error:
        _failure(report, "server", "http.server", str(error))
    finally:
        if archive_to_remove is not None:
            _remove_created_archive(archive_to_remove, workspace_root, report)

    return _finish(report)


def _default_workspace() -> Path:
    return Path(__file__).resolve().parents[1]


def _existing_workspace(raw_workspace: Path) -> Path:
    lexical = Path(os.path.abspath(str(raw_workspace.expanduser())))
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError(f"Workspace must be an existing non-symlink directory: {lexical}")
    return lexical.resolve()


def _existing_result(workspace: Path, raw_result: Path) -> Path:
    supplied = raw_result.expanduser()
    candidate = supplied if supplied.is_absolute() else workspace / supplied
    lexical = Path(os.path.abspath(str(candidate)))
    if lexical.is_symlink() or not lexical.is_dir():
        raise ValueError(f"Result must be an existing non-symlink directory: {lexical}")
    result = lexical.resolve()
    if not _is_beneath(result, workspace):
        raise ValueError(f"Result must stay inside workspace {workspace}: {result}")
    return result


@contextmanager
def _loopback_server(workbench: Workbench) -> Iterator[Tuple[str, Dict[str, Any]]]:
    base_handler = _handler(workbench)

    class QuietHandler(base_handler):
        def log_message(self, message, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    server.daemon_threads = True
    host, port = server.server_address[:2]
    if host != "127.0.0.1":
        server.server_close()
        raise RuntimeError(f"Acceptance server did not bind to loopback: {host}")
    thread = threading.Thread(
        target=server.serve_forever,
        name="phonlab-webui-acceptance",
        daemon=True,
    )
    thread.start()
    try:
        yield (
            f"http://127.0.0.1:{port}",
            {"host": "127.0.0.1", "port": int(port), "ephemeral": True},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("Loopback WebUI server did not stop cleanly")


def _http_request(
    base_url: str,
    target: str,
    *,
    method: str = "GET",
    headers: Optional[Mapping[str, str]] = None,
) -> Tuple[int, Mapping[str, str], bytes]:
    url = _loopback_url(base_url, target)
    request = Request(url, method=method, headers=dict(headers or {}))
    with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        body = response.read()
        return int(response.status), response.headers, body


def _loopback_url(base_url: str, target: str) -> str:
    base = urlsplit(base_url)
    combined = urlsplit(urljoin(base_url + "/", str(target)))
    if (
        base.scheme != "http"
        or base.hostname != "127.0.0.1"
        or combined.scheme != "http"
        or combined.hostname != "127.0.0.1"
        or combined.port != base.port
        or combined.username is not None
        or combined.password is not None
    ):
        raise ValueError(f"Refusing non-acceptance-server URL: {target!r}")
    return urlunsplit(combined)


def _validate_homepage(response: Tuple[int, Mapping[str, str], bytes]) -> None:
    status, headers, body = response
    content_type = headers.get("Content-Type", "").lower()
    lowered = body.lower()
    if status != 200:
        raise ValueError(f"Homepage returned HTTP {status}, expected 200")
    if "text/html" not in content_type:
        raise ValueError(f"Homepage Content-Type is not HTML: {content_type!r}")
    if b"<!doctype html" not in lowered or b"phonlab-ddsp" not in lowered:
        raise ValueError("Homepage does not contain the PhonLab WebUI document markers")


def _validate_catalog(
    catalog: Mapping[str, Any],
    result_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(catalog, dict):
        raise ValueError("results-load did not return an object")
    items = catalog.get("items")
    conditions = catalog.get("conditions")
    if not isinstance(items, list) or not items:
        raise ValueError("results-load catalog has no items")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("results-load catalog has no manipulation conditions")

    condition_names = []
    for condition in conditions:
        if not isinstance(condition, dict) or not isinstance(condition.get("name"), str):
            raise ValueError("results-load condition entries must have string names")
        condition_names.append(condition["name"])
    selected_condition = condition_names[0]

    selected_item = None
    selected_audio = None
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("item_id") or item.get("id")
        audio_entries = item.get("audio")
        if not isinstance(item_id, str) or not isinstance(audio_entries, list):
            continue
        for audio in audio_entries:
            if isinstance(audio, dict) and audio.get("condition") == selected_condition:
                selected_item = item_id
                selected_audio = audio
                break
        if selected_audio is not None:
            break
    if selected_audio is None or selected_item is None:
        raise ValueError("No catalog item contains audio for the first manipulation condition")

    relative = selected_audio.get("path")
    file_url = selected_audio.get("file_url")
    download_url = selected_audio.get("download_url")
    if not isinstance(relative, str) or not relative.lower().endswith(".wav"):
        raise ValueError("Selected catalog audio has no WAV path")
    if not isinstance(file_url, str) or not file_url.startswith("/files/"):
        raise ValueError("Selected catalog audio has no mounted file_url")
    if not isinstance(download_url, str) or not download_url.startswith("/files/"):
        raise ValueError("Selected catalog audio has no mounted download_url")
    download_query = parse_qs(urlsplit(download_url).query)
    if download_query.get("download") != ["1"]:
        raise ValueError("Catalog download_url does not use ?download=1")

    source = _safe_result_file(result_root, relative, suffix=".wav")
    size = source.stat().st_size
    if size < 12:
        raise ValueError(f"Selected WAV is too small: {source}")
    with source.open("rb") as stream:
        wav_prefix = stream.read(12)
    if wav_prefix[:4] not in {b"RIFF", b"RF64"} or wav_prefix[8:12] != b"WAVE":
        raise ValueError(f"Selected catalog file is not a RIFF/RF64 WAV: {source}")

    expected_wavs = len(items) * (len(conditions) + 1)
    reported_wavs = catalog.get("wav_count")
    if reported_wavs != expected_wavs:
        raise ValueError(f"Catalog wav_count is inconsistent: {reported_wavs!r} != {expected_wavs}")
    clipping = catalog.get("clipping")
    if not isinstance(clipping, dict):
        raise ValueError("Catalog has no aggregate clipping object")

    selection = {
        "item_id": selected_item,
        "condition": selected_condition,
        "relative_path": relative,
        "source": source,
        "file_url": file_url,
        "download_url": download_url,
    }
    details = {
        "items": len(items),
        "conditions": len(conditions),
        "wav_count": expected_wavs,
        "reports": len(catalog.get("reports", {})),
        "selected_item": selected_item,
        "selected_condition": selected_condition,
        "selected_wav": relative,
        "clipped_samples": clipping.get("clipped_samples"),
    }
    return selection, details


def _check_wav_range(base_url: str, selection: Mapping[str, Any]) -> Dict[str, Any]:
    source = selection["source"]
    size = source.stat().st_size
    end = min(63, size - 1)
    status, headers, body = _http_request(
        base_url,
        selection["file_url"],
        headers={"Range": f"bytes=0-{end}"},
    )
    with source.open("rb") as stream:
        expected = stream.read(end + 1)
    if status != 206:
        raise ValueError(f"WAV Range returned HTTP {status}, expected 206")
    if headers.get("Accept-Ranges", "").lower() != "bytes":
        raise ValueError("WAV Range response does not advertise byte ranges")
    expected_content_range = f"bytes 0-{end}/{size}"
    if headers.get("Content-Range") != expected_content_range:
        raise ValueError(
            f"Unexpected Content-Range: {headers.get('Content-Range')!r}; "
            f"expected {expected_content_range!r}"
        )
    if body != expected:
        raise ValueError("WAV Range body does not match the mounted result file")
    return {
        "status": status,
        "requested": f"bytes=0-{end}",
        "content_range": expected_content_range,
        "bytes": len(body),
    }


def _check_wav_download(base_url: str, selection: Mapping[str, Any]) -> Dict[str, Any]:
    status, headers, body = _http_request(
        base_url,
        selection["download_url"],
        headers={"Range": "bytes=0-0"},
    )
    disposition = headers.get("Content-Disposition", "")
    if status != 206 or len(body) != 1:
        raise ValueError(f"WAV download Range returned HTTP {status} with {len(body)} bytes")
    if not disposition.lower().startswith("attachment;") or "filename=" not in disposition:
        raise ValueError(f"WAV download has no attachment disposition: {disposition!r}")
    return {"status": status, "content_disposition": disposition, "bytes": len(body)}


def _validate_zip(
    archive_path: Path,
    selection: Mapping[str, Any],
    response: Mapping[str, Any],
    base_url: str,
) -> Dict[str, Any]:
    download_url = response.get("download_url")
    if not isinstance(download_url, str):
        raise ValueError("results-zip did not return download_url")
    query = parse_qs(urlsplit(download_url).query)
    if query.get("download") != ["1"]:
        raise ValueError("results-zip download_url does not use ?download=1")
    status, headers, body = _http_request(base_url, download_url, method="HEAD")
    if status != 200 or body:
        raise ValueError(f"ZIP HEAD returned HTTP {status} or an unexpected body")
    if int(headers.get("Content-Length", "-1")) != archive_path.stat().st_size:
        raise ValueError("ZIP HTTP Content-Length does not match the archive")
    if not headers.get("Content-Disposition", "").lower().startswith("attachment;"):
        raise ValueError("ZIP download URL has no attachment disposition")

    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = archive.infolist()
        if not infos:
            raise ValueError("results-zip archive is empty")
        for info in infos:
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in info.filename:
                raise ValueError(f"Unsafe path in results ZIP: {info.filename!r}")
            if info.flag_bits & 0x1:
                raise ValueError(f"Encrypted entry in results ZIP: {info.filename!r}")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Corrupt results ZIP member: {corrupt}")
        names = {info.filename for info in infos if not info.is_dir()}
        wav_names = sorted(name for name in names if name.lower().endswith(".wav"))
        if len(wav_names) != 1 or "provenance.json" not in names:
            raise ValueError("Single-WAV results ZIP must contain one WAV and provenance.json")
        provenance = json.loads(archive.read("provenance.json").decode("utf-8"))
        selected_ids = provenance.get("selection", {}).get("item_ids")
        if selected_ids != [selection["item_id"]]:
            raise ValueError("ZIP provenance does not record the selected item")
        with archive.open(wav_names[0], "r") as stream:
            archived_digest = _stream_sha256(stream)
    source_digest = _file_sha256(selection["source"])
    if archived_digest != source_digest:
        raise ValueError("ZIP WAV bytes do not match the selected result WAV")
    return {
        "status": status,
        "archive": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "entries": len(infos),
        "wav_entries": len(wav_names),
        "integrity": "ok",
    }


def _check_safe_export(
    workbench: Workbench,
    workspace_root: Path,
    result_root: Path,
    selection: Mapping[str, Any],
) -> Dict[str, Any]:
    cache = workspace_root / ".cache"
    if cache.is_symlink():
        raise ValueError(f"Workspace cache must not be a symlink: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    if not _is_beneath(cache.resolve(), workspace_root):
        raise ValueError("Workspace cache resolved outside the workspace")

    details: Dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="webui-acceptance-", dir=cache) as temporary:
        temporary_root = Path(temporary).resolve()
        destination = temporary_root / "export"
        response = workbench.run(
            "results-export",
            {
                "output": str(result_root),
                "condition": selection["condition"],
                "item": selection["item_id"],
                "scope": "wav",
                "destination": str(destination),
            },
        )
        returned_destination = _response_workspace_file(
            response,
            "destination",
            workspace_root,
            directory=True,
        )
        if returned_destination != destination.resolve():
            raise ValueError("results-export returned a different destination")
        if not _is_beneath(returned_destination, temporary_root):
            raise ValueError("results-export destination escaped its acceptance temp directory")
        provenance_path = _response_workspace_file(
            response,
            "provenance_path",
            workspace_root,
            suffix=".json",
        )
        files = response.get("files")
        if response.get("file_count") != 1 or not isinstance(files, list) or len(files) != 1:
            raise ValueError("Single-WAV results-export did not return exactly one file")
        exported = _workspace_file(Path(files[0]), workspace_root, suffix=".wav")
        if not _is_beneath(exported, returned_destination):
            raise ValueError("Exported WAV is outside the returned destination")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("selection", {}).get("item_ids") != [selection["item_id"]]:
            raise ValueError("Export provenance does not record the selected item")
        if _file_sha256(exported) != _file_sha256(selection["source"]):
            raise ValueError("Exported WAV bytes do not match the selected result WAV")
        details = {
            "destination": str(returned_destination),
            "files": 1,
            "provenance": str(provenance_path),
            "bytes": exported.stat().st_size,
            "temporary": True,
        }
    details["cleaned"] = not Path(details["destination"]).exists()
    if not details["cleaned"]:
        raise RuntimeError("Temporary results export was not cleaned up")
    return details


def _response_workspace_file(
    response: Mapping[str, Any],
    key: str,
    workspace: Path,
    *,
    suffix: Optional[str] = None,
    directory: bool = False,
) -> Path:
    if not isinstance(response, dict) or not isinstance(response.get(key), str):
        raise ValueError(f"WebUI response has no string {key!r}")
    return _workspace_file(Path(response[key]), workspace, suffix=suffix, directory=directory)


def _workspace_file(
    raw_path: Path,
    workspace: Path,
    *,
    suffix: Optional[str] = None,
    directory: bool = False,
) -> Path:
    candidate = raw_path if raw_path.is_absolute() else workspace / raw_path
    path = candidate.resolve()
    if not _is_beneath(path, workspace):
        raise ValueError(f"WebUI path escaped the workspace: {path}")
    if path.is_symlink() or (not path.is_dir() if directory else not path.is_file()):
        kind = "directory" if directory else "file"
        raise ValueError(f"WebUI response is not a regular {kind}: {path}")
    if suffix is not None and path.suffix.lower() != suffix.lower():
        raise ValueError(f"WebUI response has the wrong suffix: {path}")
    return path


def _safe_result_file(result_root: Path, relative: str, *, suffix: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        raise ValueError(f"Unsafe catalog path: {relative!r}")
    return _workspace_file(result_root.joinpath(*pure.parts), result_root, suffix=suffix)


def _is_beneath(path: Path, root: Path) -> bool:
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path == root or root in path.parents


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return _stream_sha256(stream)


def _stream_sha256(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _remove_created_archive(
    archive: Path,
    workspace: Path,
    report: Dict[str, Any],
) -> None:
    cache = workspace / ".cache"
    try:
        if archive.is_file() and not archive.is_symlink() and _is_beneath(archive, cache):
            archive.unlink()
            report["cleanup"] = {"archive_removed": str(archive)}
    except OSError as error:
        _failure(report, "cleanup", "cleanup.archive", str(error))


def _success(report: Dict[str, Any], name: str, details: Mapping[str, Any]) -> None:
    report["checks"].append({"name": name, "ok": True, **dict(details)})


def _failure(report: Dict[str, Any], name: str, code: str, message: str) -> None:
    report["checks"].append({"name": name, "ok": False, "detail": message})
    report["issues"].append({"code": code, "message": message})


def _finish(report: Dict[str, Any]) -> Dict[str, Any]:
    report["success"] = not report["issues"]
    report["success_token"] = SUCCESS_TOKEN if report["success"] else None
    return report


def _safe_report_output(raw_output: Path, workspace: Path) -> Path:
    supplied = raw_output.expanduser()
    candidate = supplied if supplied.is_absolute() else workspace / supplied
    output = Path(os.path.abspath(str(candidate)))
    if not _is_beneath(output, workspace):
        raise ValueError(f"JSON output must stay inside workspace {workspace}: {output}")
    if output.exists() and output.is_dir():
        raise IsADirectoryError(f"JSON output is a directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve() != output.parent or output.is_symlink():
        raise ValueError(f"JSON output path contains a symbolic link: {output}")
    return output


def _write_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CPU-only, loopback-only acceptance check of the PhonLab WebUI "
            "results catalog, WAV serving, download, and ZIP workflow"
        )
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=_default_workspace(),
        help="workspace root (default: repository containing this tool)",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_RESULT,
        help=(
            "result tree, absolute or workspace-relative "
            "(default: artifacts/cmu_arctic_slt_demo/control_postprocess_v2)"
        ),
    )
    parser.add_argument(
        "--check-export",
        "--export",
        dest="check_export",
        action="store_true",
        help="also create and remove a single-WAV export under workspace/.cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path inside the workspace",
    )
    args = parser.parse_args(argv)

    report = check_webui(
        args.workspace,
        args.result,
        check_export=args.check_export,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        try:
            workspace = _existing_workspace(args.workspace)
            output = _safe_report_output(args.output, workspace)
            _write_json_atomic(output, report)
        except (OSError, ValueError) as error:
            print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
            print(f"Could not write JSON report: {error}", file=sys.stderr)
            return 2
    if report["success"]:
        print(SUCCESS_TOKEN)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
