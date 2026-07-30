"""Safe, dependency-free access to Slurm jobs created by PhonLab-DDSP.

The functions in this module deliberately expose only a small Slurm surface:
submit an experiment's ``train.slurm``, inspect one numeric job ID, read that
job's bounded log tail, and explicitly cancel one numeric job ID.  Commands are
always passed to a subprocess runner as argument vectors; shell parsing is
never involved.
"""

from __future__ import annotations

import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

Runner = Callable[..., Any]
PathLike = Union[str, Path]
JobId = Union[str, int]

DEFAULT_COMMAND_TIMEOUT = 10.0
DEFAULT_LOG_BYTES = 64 * 1024
MAX_LOG_BYTES = 1024 * 1024
MAX_LOG_LINES = 10_000
_MAX_JOB_ID_DIGITS = 32
_FIELD_SEPARATOR = "\x1f"
_JOB_ID_RE = re.compile(r"[0-9]+")
_CLUSTER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_PARSABLE_JOB_ID_RE = re.compile(r"([0-9]+)(?:;([A-Za-z0-9][A-Za-z0-9._-]{0,63}))?")
_HUMAN_JOB_ID_RE = re.compile(r"Submitted\s+batch\s+job\s+([0-9]+)", re.IGNORECASE)


class SlurmError(RuntimeError):
    """Base class for an expected Slurm integration error."""


class SlurmUnavailableError(SlurmError):
    """Raised when a requested Slurm executable is not installed."""


class SlurmCommandError(SlurmError):
    """Raised when a Slurm command starts but does not complete successfully."""

    def __init__(
        self,
        command: str,
        message: str,
        *,
        returncode: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.returncode = returncode


class SubmittedJob(str):
    """Numeric job ID string that also retains an optional Slurm cluster.

    It intentionally remains a ``str`` for compatibility with callers that
    persist or display numeric job IDs.  Pass the object itself to
    :func:`query_job` or :func:`cancel_job` to retain federation routing.
    """

    cluster: Optional[str]

    def __new__(cls, job_id: str, cluster: Optional[str] = None) -> SubmittedJob:
        instance = str.__new__(cls, job_id)
        instance.cluster = cluster
        return instance

    @property
    def job_id(self) -> str:
        return str(self)

    def to_dict(self) -> dict[str, Optional[str]]:
        return {"job_id": self.job_id, "cluster": self.cluster}

    def as_dict(self) -> dict[str, Optional[str]]:
        return self.to_dict()


JobHandle = SubmittedJob


class JobCategory(str, Enum):
    """Small, stable categories independent of Slurm version-specific states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class JobStatus:
    """A stable representation of one Slurm job.

    ``state`` retains Slurm's normalized state name.  ``category`` is the
    coarser value applications should normally use for control flow.
    """

    job_id: str
    state: str
    category: JobCategory
    cluster: Optional[str] = None
    name: Optional[str] = None
    reason: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    elapsed: Optional[str] = None
    exit_code: Optional[str] = None
    node_list: Optional[str] = None
    source: Optional[str] = None
    detail: Optional[str] = None

    @property
    def terminal(self) -> bool:
        return self.category in {
            JobCategory.SUCCEEDED,
            JobCategory.FAILED,
            JobCategory.CANCELLED,
        }

    @property
    def is_terminal(self) -> bool:
        """Compatibility-friendly spelling for :attr:`terminal`."""

        return self.terminal

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable status dictionary with stable keys."""

        return {
            "job_id": self.job_id,
            "state": self.state,
            "category": self.category.value,
            "cluster": self.cluster,
            "name": self.name,
            "reason": self.reason,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed": self.elapsed,
            "exit_code": self.exit_code,
            "node_list": self.node_list,
            "source": self.source,
            "detail": self.detail,
            "terminal": self.terminal,
        }

    def as_dict(self) -> dict[str, object]:
        """Alias for :meth:`to_dict`."""

        return self.to_dict()


# Descriptive alias for callers that prefer a Slurm-specific type name.
SlurmJob = JobStatus
SlurmStatus = JobStatus


class SlurmBackend:
    """Stateful facade useful for injecting one subprocess runner in tests."""

    def __init__(
        self,
        runner: Optional[Runner] = None,
        *,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
    ) -> None:
        self.runner = runner
        self.command_timeout = _validate_timeout(command_timeout)

    def submit(self, experiment: PathLike) -> SubmittedJob:
        return submit_job(
            experiment,
            runner=self.runner,
            timeout=self.command_timeout,
        )

    def query(self, job_id: JobId, *, cluster: Optional[str] = None) -> JobStatus:
        return query_job(
            job_id,
            cluster=cluster,
            runner=self.runner,
            timeout=self.command_timeout,
        )

    def read_log(
        self,
        experiment: PathLike,
        job_id: JobId,
        *,
        max_bytes: int = DEFAULT_LOG_BYTES,
        tail_lines: Optional[int] = None,
        missing_ok: bool = True,
    ) -> str:
        return read_job_log(
            experiment,
            job_id,
            max_bytes=max_bytes,
            tail_lines=tail_lines,
            missing_ok=missing_ok,
        )

    def cancel(self, job_id: JobId, *, cluster: Optional[str] = None) -> bool:
        return cancel_job(
            job_id,
            cluster=cluster,
            runner=self.runner,
            timeout=self.command_timeout,
        )


def validate_experiment(experiment: PathLike) -> Path:
    """Validate and return a resolved experiment directory.

    A valid experiment is a real directory containing a non-empty regular file
    named exactly ``train.slurm``.  The script may not be a symlink, which keeps
    both submission and log-path inspection anchored to the selected
    experiment.
    """

    directory, _ = _validated_experiment(experiment)
    return directory


def train_slurm_path(experiment: PathLike) -> Path:
    """Validate an experiment and return its resolved ``train.slurm`` path."""

    _, script = _validated_experiment(experiment)
    return script


def submit_job(
    experiment: PathLike,
    *,
    runner: Optional[Runner] = None,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
) -> SubmittedJob:
    """Submit ``train.slurm`` and return Slurm's validated numeric job ID."""

    directory, script = _validated_experiment(experiment)
    completed = _invoke(
        ["sbatch", "--parsable", str(script)],
        runner=runner,
        cwd=directory,
        timeout=timeout,
    )
    _require_success(completed, "sbatch")
    output = _output_text(completed, "stdout").strip()
    match = _PARSABLE_JOB_ID_RE.fullmatch(output)
    cluster: Optional[str] = None
    if match is None:
        match = _HUMAN_JOB_ID_RE.fullmatch(output)
    else:
        cluster = match.group(2)
    if match is None:
        raise SlurmCommandError(
            "sbatch",
            "sbatch succeeded but returned an unrecognized job ID",
            returncode=_returncode(completed),
        )
    return SubmittedJob(_validate_job_id(match.group(1)), cluster)


def query_job(
    job_id: JobId,
    *,
    cluster: Optional[str] = None,
    runner: Optional[Runner] = None,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
) -> JobStatus:
    """Query ``squeue`` first, then ``sacct`` for a completed or missing job.

    Inspection is intentionally non-throwing for scheduler availability and
    command failures.  Such cases become ``unavailable`` or ``unknown`` status
    records with an explanatory ``detail`` value.
    """

    validated_id, validated_cluster = _validated_job_reference(job_id, cluster)
    errors: list[str] = []
    commands_found = 0

    squeue_args = [
        "squeue",
        *_cluster_arguments(validated_cluster),
        "--noheader",
        "--jobs",
        validated_id,
        f"--format=%i{_FIELD_SEPARATOR}%T{_FIELD_SEPARATOR}%j"
        f"{_FIELD_SEPARATOR}%r{_FIELD_SEPARATOR}%S{_FIELD_SEPARATOR}%M"
        f"{_FIELD_SEPARATOR}%N",
    ]
    squeue = _query_command(
        squeue_args,
        runner=runner,
        timeout=timeout,
        errors=errors,
    )
    if squeue is not None:
        commands_found += 1
        if _returncode(squeue) == 0:
            status = _parse_squeue(
                _output_text(squeue, "stdout"),
                validated_id,
                validated_cluster,
            )
            if status is not None:
                return status

    sacct_args = [
        "sacct",
        *_cluster_arguments(validated_cluster),
        "--noheader",
        "--parsable2",
        "--delimiter",
        _FIELD_SEPARATOR,
        "--allocations",
        "--jobs",
        validated_id,
        "--format=JobIDRaw,State,JobName,Reason,Start,Elapsed,End,ExitCode,NodeList",
    ]
    sacct = _query_command(
        sacct_args,
        runner=runner,
        timeout=timeout,
        errors=errors,
    )
    if sacct is not None:
        commands_found += 1
        if _returncode(sacct) == 0:
            status = _parse_sacct(
                _output_text(sacct, "stdout"),
                validated_id,
                validated_cluster,
            )
            if status is not None:
                return status

    if commands_found == 0:
        return JobStatus(
            job_id=validated_id,
            state="UNAVAILABLE",
            category=JobCategory.UNAVAILABLE,
            cluster=validated_cluster,
            detail=_join_details(errors) or "squeue and sacct are not available",
        )
    return JobStatus(
        job_id=validated_id,
        state="UNKNOWN",
        category=JobCategory.UNKNOWN,
        cluster=validated_cluster,
        detail=_join_details(errors) or "job was not found in squeue or sacct",
    )


def cancel_job(
    job_id: JobId,
    *,
    cluster: Optional[str] = None,
    runner: Optional[Runner] = None,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
) -> bool:
    """Explicitly cancel one validated numeric job ID."""

    validated_id, validated_cluster = _validated_job_reference(job_id, cluster)
    completed = _invoke(
        ["scancel", *_cluster_arguments(validated_cluster), validated_id],
        runner=runner,
        timeout=timeout,
    )
    _require_success(completed, "scancel")
    return True


def job_log_path(experiment: PathLike, job_id: JobId) -> Path:
    """Resolve a job log path without allowing it to escape the experiment.

    The first applicable ``#SBATCH --output``/``-o`` directive is honored.
    ``%j``, ``%J``, and ``%A`` are expanded to the validated numeric job ID.
    When the script has no output directive, Slurm's ``slurm-%j.out`` default
    is used.
    """

    directory, script = _validated_experiment(experiment)
    validated_id = _validate_job_id(job_id)
    pattern = _slurm_output_pattern(script) or "slurm-%j.out"
    expanded = _expand_log_pattern(pattern, validated_id)
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = directory / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != directory and directory not in resolved.parents:
        raise ValueError("#SBATCH --output must resolve inside the experiment directory")
    return resolved


def read_job_log(
    experiment: PathLike,
    job_id: JobId,
    *,
    max_bytes: int = DEFAULT_LOG_BYTES,
    tail_lines: Optional[int] = None,
    missing_ok: bool = True,
) -> str:
    """Read at most ``max_bytes`` from the end of a job log.

    ``tail_lines`` can impose an additional line limit.  Missing logs return an
    empty string by default because queued jobs often have not created a log
    yet; pass ``missing_ok=False`` when absence should be exceptional.
    """

    byte_limit = _bounded_positive_int("max_bytes", max_bytes, MAX_LOG_BYTES)
    if tail_lines is not None:
        line_limit = _bounded_positive_int("tail_lines", tail_lines, MAX_LOG_LINES)
    else:
        line_limit = None
    path = job_log_path(experiment, job_id)
    if not path.exists():
        if missing_ok:
            return ""
        raise FileNotFoundError(f"Slurm log does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Slurm log is not a regular file: {path}")

    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - byte_limit))
        data = stream.read(byte_limit)
    text = data.decode("utf-8", errors="replace")
    if line_limit is not None:
        text = "".join(text.splitlines(keepends=True)[-line_limit:])
    return text


def _validated_experiment(experiment: PathLike) -> tuple[Path, Path]:
    raw_directory = Path(experiment).expanduser()
    try:
        directory = raw_directory.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Experiment directory does not exist: {raw_directory}") from error
    if not directory.is_dir():
        raise NotADirectoryError(f"Experiment path is not a directory: {directory}")

    candidate = directory / "train.slurm"
    if candidate.is_symlink():
        raise ValueError(f"train.slurm must not be a symlink: {candidate}")
    try:
        script = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Experiment has no train.slurm: {directory}") from error
    if script.parent != directory or not script.is_file():
        raise ValueError(f"train.slurm is not a regular file inside the experiment: {script}")
    if script.stat().st_size == 0:
        raise ValueError(f"train.slurm is empty: {script}")
    return directory, script


def _validate_job_id(job_id: JobId) -> str:
    if isinstance(job_id, bool):
        raise ValueError("Slurm job ID must contain only decimal digits")
    value = str(job_id)
    if len(value) > _MAX_JOB_ID_DIGITS or _JOB_ID_RE.fullmatch(value) is None or int(value) <= 0:
        raise ValueError("Slurm job ID must be a positive decimal integer")
    return str(int(value))


def _validated_job_reference(
    job_id: JobId,
    cluster: Optional[str],
) -> tuple[str, Optional[str]]:
    embedded_cluster = job_id.cluster if isinstance(job_id, SubmittedJob) else None
    if cluster is not None and embedded_cluster is not None and cluster != embedded_cluster:
        raise ValueError("Explicit Slurm cluster does not match the submitted job handle")
    selected_cluster = cluster if cluster is not None else embedded_cluster
    return _validate_job_id(job_id), _validate_cluster(selected_cluster)


def _validate_cluster(cluster: Optional[str]) -> Optional[str]:
    if cluster is None:
        return None
    if not isinstance(cluster, str) or _CLUSTER_RE.fullmatch(cluster) is None:
        raise ValueError("Slurm cluster name contains unsupported characters")
    return cluster


def _cluster_arguments(cluster: Optional[str]) -> list[str]:
    return [] if cluster is None else ["--clusters", cluster]


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool):
        raise ValueError("command timeout must be a positive number")
    try:
        value = float(timeout)
    except (TypeError, ValueError) as error:
        raise ValueError("command timeout must be a positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("command timeout must be a positive number")
    return value


def _bounded_positive_int(name: str, value: int, upper_bound: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper_bound:
        raise ValueError(f"{name} must be an integer between 1 and {upper_bound}")
    return value


def _invoke(
    argv: Sequence[str],
    *,
    runner: Optional[Runner],
    timeout: float,
    cwd: Optional[Path] = None,
) -> Any:
    command_timeout = _validate_timeout(timeout)
    selected_runner = runner if runner is not None else subprocess.run
    try:
        return selected_runner(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=command_timeout,
            shell=False,
        )
    except FileNotFoundError as error:
        raise SlurmUnavailableError(f"Slurm command is not available: {argv[0]}") from error
    except OSError as error:
        raise SlurmCommandError(
            argv[0],
            f"{argv[0]} could not be started: {_clean_detail(error)}",
        ) from error
    except subprocess.TimeoutExpired as error:
        raise SlurmCommandError(
            argv[0],
            f"{argv[0]} timed out after {command_timeout:g} seconds",
        ) from error
    except subprocess.CalledProcessError as error:
        detail = _clean_detail(error.stderr) or _clean_detail(error.stdout)
        message = f"{argv[0]} failed with exit code {error.returncode}"
        if detail:
            message += f": {detail}"
        raise SlurmCommandError(
            argv[0],
            message,
            returncode=error.returncode,
        ) from error


def _require_success(completed: Any, command: str) -> None:
    returncode = _returncode(completed)
    if returncode == 0:
        return
    detail = _clean_detail(_output_text(completed, "stderr"))
    message = f"{command} failed with exit code {returncode}"
    if detail:
        message += f": {detail}"
    raise SlurmCommandError(command, message, returncode=returncode)


def _returncode(completed: Any) -> int:
    try:
        return int(completed.returncode)
    except (AttributeError, TypeError, ValueError) as error:
        raise SlurmCommandError(
            "runner",
            "subprocess runner returned an invalid result",
        ) from error


def _output_text(completed: Any, name: str) -> str:
    value = getattr(completed, name, "")
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _clean_detail(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())[:500]


def _query_command(
    argv: Sequence[str],
    *,
    runner: Optional[Runner],
    timeout: float,
    errors: list[str],
) -> Optional[Any]:
    try:
        completed = _invoke(argv, runner=runner, timeout=timeout)
    except SlurmError as error:
        errors.append(str(error))
        return None
    if _returncode(completed) != 0:
        detail = _clean_detail(_output_text(completed, "stderr"))
        message = f"{argv[0]} exited with code {_returncode(completed)}"
        errors.append(f"{message}: {detail}" if detail else message)
    return completed


def _parse_squeue(
    output: str,
    job_id: str,
    cluster: Optional[str],
) -> Optional[JobStatus]:
    for raw_line in output.splitlines():
        separator = _FIELD_SEPARATOR if _FIELD_SEPARATOR in raw_line else "|"
        fields = [field.strip() for field in raw_line.split(separator, 6)]
        if len(fields) != 7 or fields[0] != job_id:
            continue
        state = _normalize_state(fields[1])
        return JobStatus(
            job_id=job_id,
            state=state,
            category=_category_for_state(state),
            cluster=cluster,
            name=_optional_field(fields[2]),
            reason=_optional_field(fields[3]),
            started_at=_optional_field(fields[4]),
            elapsed=_optional_field(fields[5]),
            node_list=_optional_field(fields[6]),
            source="squeue",
        )
    return None


def _parse_sacct(
    output: str,
    job_id: str,
    cluster: Optional[str],
) -> Optional[JobStatus]:
    for raw_line in output.splitlines():
        separator = _FIELD_SEPARATOR if _FIELD_SEPARATOR in raw_line else "|"
        fields = [field.strip() for field in raw_line.split(separator)]
        # --parsable2 has no trailing delimiter.  Accept one anyway for
        # compatibility with --parsable fixtures, but preserve an empty ninth
        # field because it is a legitimate empty NodeList.
        if len(fields) >= 10 and fields[-1] == "":
            fields.pop()
        if not fields or fields[0] != job_id:
            continue
        if len(fields) >= 9:
            _, raw_state, name, reason, started, elapsed, ended, exit_code, nodes = fields[:9]
        elif len(fields) >= 8:
            _, raw_state, name, started, elapsed, ended, exit_code, nodes = fields[:8]
            reason = ""
        else:
            continue
        state = _normalize_state(raw_state)
        return JobStatus(
            job_id=job_id,
            state=state,
            category=_category_for_state(state),
            cluster=cluster,
            name=_optional_field(name),
            reason=_optional_field(reason),
            started_at=_optional_field(started),
            ended_at=_optional_field(ended),
            elapsed=_optional_field(elapsed),
            exit_code=_optional_field(exit_code),
            node_list=_optional_field(nodes),
            source="sacct",
        )
    return None


def _normalize_state(raw_state: str) -> str:
    value = raw_state.strip().upper()
    if not value:
        return "UNKNOWN"
    return value.split(maxsplit=1)[0].rstrip("+")


def _category_for_state(state: str) -> JobCategory:
    if state in {
        "PENDING",
        "CONFIGURING",
        "REQUEUE_FED",
        "REQUEUED",
        "REQUEUE_HOLD",
        "RESV_DEL_HOLD",
        "SPECIAL_EXIT",
        "SUSPENDED",
    }:
        return JobCategory.QUEUED
    if state in {
        "RUNNING",
        "COMPLETING",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
    }:
        return JobCategory.RUNNING
    if state == "COMPLETED":
        return JobCategory.SUCCEEDED
    if state in {"CANCELLED", "PREEMPTED", "REVOKED"}:
        return JobCategory.CANCELLED
    if state in {
        "BOOT_FAIL",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "TIMEOUT",
    }:
        return JobCategory.FAILED
    if state == "UNAVAILABLE":
        return JobCategory.UNAVAILABLE
    return JobCategory.UNKNOWN


def _optional_field(value: str) -> Optional[str]:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"n/a", "none", "(null)", "unknown"}:
        return None
    return stripped


def _join_details(errors: Sequence[str]) -> Optional[str]:
    if not errors:
        return None
    return "; ".join(dict.fromkeys(errors))[:1000]


def _slurm_output_pattern(script: Path) -> Optional[str]:
    with script.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number > 1000:
                break
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#!")
                or (stripped.startswith("#") and not stripped.startswith("#SBATCH"))
            ):
                continue
            if not stripped.startswith("#SBATCH"):
                break
            payload = stripped[len("#SBATCH") :].strip()
            try:
                tokens = shlex.split(payload, comments=True, posix=True)
            except ValueError as error:
                raise ValueError(f"Invalid #SBATCH directive in {script.name}") from error
            for index, token in enumerate(tokens):
                value: Optional[str] = None
                if token.startswith("--output="):
                    value = token.partition("=")[2]
                elif token == "--output" or token == "-o":
                    if index + 1 >= len(tokens):
                        raise ValueError(f"Missing #SBATCH output path in {script.name}")
                    value = tokens[index + 1]
                elif token.startswith("-o") and len(token) > 2:
                    value = token[2:]
                if value is not None:
                    if not value or "\x00" in value:
                        raise ValueError(f"Invalid #SBATCH output path in {script.name}")
                    return value
    return None


def _expand_log_pattern(pattern: str, job_id: str) -> str:
    sentinel = "\x00PERCENT\x00"
    expanded = pattern.replace("%%", sentinel)

    def replace_job_id(match: re.Match[str]) -> str:
        width_text = match.group(1)
        if not width_text:
            return job_id
        significant_width = width_text.lstrip("0")
        if len(significant_width) > 2:
            width = 10
        else:
            width = min(int(significant_width or "0"), 10)
        return job_id.zfill(width)

    expanded = re.sub(r"%([0-9]*)(?:j|J|A)", replace_job_id, expanded)
    unsupported = re.search(r"%[0-9]*[A-Za-z]", expanded)
    if unsupported is not None:
        raise ValueError(
            f"Cannot resolve #SBATCH output placeholder {unsupported.group(0)!r} safely"
        )
    return expanded.replace(sentinel, "%")


# Useful, unsurprising aliases for GUI/CLI integrations.
submit_slurm_job = submit_job
query_job_status = query_job
get_job_status = query_job
resolve_job_log = job_log_path
tail_job_log = read_job_log
