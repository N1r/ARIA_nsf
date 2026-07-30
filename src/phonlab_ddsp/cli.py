"""Command-line interface for the complete PhonLab-DDSP workflow."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .corpus import acquire_cmu_arctic
from .doctor import as_json, checks
from .experiment import MODELS, create_experiment, synthesize, train
from .jobs import cancel_job, query_job, read_job_log, submit_job
from .launchers import create_postprocess_job
from .manifest import DatasetManifest, prepare_dataset, summarize, validate_manifest
from .manipulation import (
    build_manipulation_report,
    manipulate_pitch,
    semitones_to_scale,
)
from .metrics import build_metrics_dashboard, load_metrics, summarize_metrics
from .parameters import export_parameters, parameter_summary
from .report import build_report, build_synthesis_report
from .segment import split_audio, split_summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="phonlab",
        description="Prepare, audit, train, and visualise reproducible DDSP experiments.",
    )
    root.add_argument("--version", action="version", version="phonlab-ddsp 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check audio and training dependencies")
    doctor.add_argument("--json", action="store_true")

    corpus = commands.add_parser(
        "fetch-corpus",
        help="download and select a reproducible 30-60 minute CMU ARCTIC corpus",
    )
    corpus.add_argument("output", type=Path)
    corpus.add_argument(
        "--archive",
        type=Path,
        help="reuse an already downloaded official archive instead of downloading it",
    )
    corpus.add_argument("--target-minutes", type=float, default=30.0)
    corpus.add_argument("--max-minutes", type=float, default=60.0)
    corpus.add_argument("--silence-gap", type=float, default=0.35)

    split = commands.add_parser(
        "split-audio",
        aliases=["split"],
        help="split recordings at silence boundaries or into fixed windows",
    )
    split.add_argument("source", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--mode", choices=["silence", "fixed"], default="silence")
    split.add_argument("--segment-seconds", type=float, default=2.0)
    split.add_argument("--overlap-seconds", type=float, default=0.0)
    split.add_argument("--silence-threshold-db", type=float, default=-40.0)
    split.add_argument("--min-silence-seconds", type=float, default=0.30)
    split.add_argument("--padding-seconds", type=float, default=0.05)
    split.add_argument("--min-duration-seconds", type=float, default=0.25)
    split.add_argument("--max-duration-seconds", type=float, default=15.0)
    split.add_argument(
        "--sample-rate",
        type=int,
        help="optional output rate; by default each source rate is preserved",
    )
    split.add_argument(
        "--discard-tail",
        action="store_true",
        help="in fixed mode, discard the final partial window",
    )
    split.add_argument(
        "--split-f0-sidecars",
        action="store_true",
        help="split a matching .pv F0 track alongside every source WAV",
    )
    split.add_argument(
        "--f0-hop-seconds",
        type=float,
        default=0.005,
        help="frame shift of .pv tracks (default: 0.005)",
    )

    prepare = commands.add_parser("prepare", help="prepare audio and a deterministic manifest")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--sample-rate", type=int, default=16000)
    prepare.add_argument(
        "--f0-method",
        choices=["auto", "pyworld", "autocorr", "sidecar"],
        default="auto",
        help="'sidecar' reuses a .pv file beside each source WAV",
    )
    prepare.add_argument("--f0-floor", type=float, default=60)
    prepare.add_argument("--f0-ceiling", type=float, default=500)
    prepare.add_argument("--validation-ratio", type=float, default=0.1)
    prepare.add_argument("--test-ratio", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--min-duration", type=float, default=0.25)
    prepare.add_argument("--normalize-peak", type=float)

    parameters = commands.add_parser(
        "parameters",
        help="export a stable per-record acoustic parameter table",
    )
    parameters.add_argument("dataset", type=Path)
    parameters.add_argument("--output", type=Path)

    validate = commands.add_parser("validate", help="validate a prepared dataset")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--json", action="store_true")

    inspect = commands.add_parser(
        "inspect", aliases=["visualize"], help="build an HTML audit report"
    )
    inspect.add_argument("dataset", type=Path)
    inspect.add_argument("--output", type=Path)

    init = commands.add_parser("init-experiment", help="create a portable training experiment")
    init.add_argument("dataset", type=Path)
    init.add_argument("output", type=Path)
    init.add_argument("--model", choices=sorted(MODELS), default="golf")
    init.add_argument("--batch-size", type=int, default=32)
    init.add_argument("--max-steps", type=int, default=40000)
    init.add_argument("--seed", type=int, default=42)
    init.add_argument("--f0-min", type=float, default=60)
    init.add_argument("--f0-max", type=float, default=500)
    init.add_argument("--workers", type=int, default=4)
    init.add_argument("--partition", default="gpu-short")
    init.add_argument("--gres", default="gpu:l4:1")
    init.add_argument("--time", default="04:00:00")
    init.add_argument("--cpus", type=int, default=8)
    init.add_argument("--memory", default="32G")
    init.add_argument("--exclude", default="node857")

    training = commands.add_parser("train", help="verify provenance and run an experiment")
    training.add_argument("experiment", type=Path)
    training.add_argument("--dry-run", action="store_true")

    metrics = commands.add_parser(
        "metrics",
        help="summarize training metrics and build a loss dashboard",
    )
    metrics.add_argument("experiment", type=Path)
    metrics.add_argument("--output", type=Path)
    metrics.add_argument("--version")
    metrics.add_argument("--json", action="store_true")

    synthesis = commands.add_parser("synthesize", help="reconstruct held-out audio")
    synthesis.add_argument("experiment", type=Path)
    synthesis.add_argument("checkpoint", type=Path)
    synthesis.add_argument("output", type=Path)
    synthesis.add_argument("--dry-run", action="store_true")
    synthesis.add_argument(
        "--semitones",
        type=float,
        default=0.0,
        help="shift voiced F0 conditioning while preserving unvoiced frames",
    )

    manipulation = commands.add_parser(
        "manipulate",
        help="render auditable pitch manipulations from a checkpoint",
    )
    manipulation.add_argument("experiment", type=Path)
    manipulation.add_argument("checkpoint", type=Path)
    manipulation.add_argument("output", type=Path)
    manipulation.add_argument("--semitones", type=float, nargs="+", required=True)
    manipulation.add_argument("--baseline", type=Path)
    manipulation.add_argument("--report", type=Path)
    manipulation.add_argument("--dry-run", action="store_true")

    postprocess = commands.add_parser(
        "init-postprocess",
        help="create a Slurm job for reconstruction, manipulation, and reports",
    )
    postprocess.add_argument("experiment", type=Path)
    postprocess.add_argument("checkpoint", type=Path)
    postprocess.add_argument("output", type=Path)
    postprocess.add_argument("--semitones", type=float, nargs="+", default=[-4, 4])
    postprocess.add_argument("--partition", default="gpu-short")
    postprocess.add_argument("--gres", default="gpu:1")
    postprocess.add_argument("--time", default="00:30:00")
    postprocess.add_argument("--cpus", type=int, default=4)
    postprocess.add_argument("--memory", default="24G")
    postprocess.add_argument("--exclude", default="")

    submit = commands.add_parser("submit-job", help="submit a generated Slurm job")
    submit.add_argument("job_bundle", type=Path)
    submit.add_argument(
        "--confirm",
        action="store_true",
        help="required safety confirmation before calling sbatch",
    )

    status = commands.add_parser("job-status", help="query one numeric Slurm job ID")
    status.add_argument("job_id")
    status.add_argument("--json", action="store_true")

    job_log = commands.add_parser("job-log", help="show a bounded Slurm log tail")
    job_log.add_argument("job_bundle", type=Path)
    job_log.add_argument("job_id")
    job_log.add_argument("--tail-lines", type=int, default=200)

    cancel = commands.add_parser("cancel-job", help="cancel one numeric Slurm job ID")
    cancel.add_argument("job_id")
    cancel.add_argument(
        "--confirm",
        action="store_true",
        help="required safety confirmation before calling scancel",
    )

    compare = commands.add_parser("compare", help="build original/reconstruction listening page")
    compare.add_argument("dataset", type=Path)
    compare.add_argument("synthesis", type=Path)
    compare.add_argument("--output", type=Path, default=Path("comparison.html"))

    gui = commands.add_parser("gui", help="launch the local browser workbench")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-browser", action="store_true")
    return root


def main(argv=None) -> int:
    argument_parser = parser()
    args, extra = argument_parser.parse_known_args(argv)
    if extra and args.command != "train":
        argument_parser.error("unrecognized arguments: " + " ".join(extra))
    try:
        if args.command == "doctor":
            if args.json:
                print(as_json())
            else:
                for item in checks():
                    symbol = "OK" if item.ok else "--"
                    print(f"[{symbol:2}] {item.name:12} {item.detail} ({item.required_for})")
            return 0
        if args.command == "fetch-corpus":
            result = acquire_cmu_arctic(
                args.output,
                target_duration_s=args.target_minutes * 60,
                max_duration_s=args.max_minutes * 60,
                silence_gap_s=args.silence_gap,
                archive_source=args.archive,
            )
            metadata = result.metadata
            print(
                json.dumps(
                    {
                        "corpus": metadata["corpus"],
                        "utterances": metadata["selection"]["utterance_count"],
                        "speech_duration_s": metadata["selection"]["selected_duration_s"],
                        "selected_audio": str(result.selected_dir),
                        "continuous_audio": (
                            str(result.continuous_wav) if result.continuous_wav else None
                        ),
                        "archive_sha256": metadata["source"]["sha256"],
                        "metadata": str(Path(args.output).resolve() / "corpus.json"),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command in {"split-audio", "split"}:
            result = split_audio(
                args.source,
                args.output,
                mode=args.mode,
                segment_seconds=args.segment_seconds,
                overlap_seconds=args.overlap_seconds,
                silence_threshold_db=args.silence_threshold_db,
                min_silence_seconds=args.min_silence_seconds,
                padding_seconds=args.padding_seconds,
                min_duration_seconds=args.min_duration_seconds,
                max_duration_seconds=args.max_duration_seconds,
                sample_rate=args.sample_rate,
                keep_tail=not args.discard_tail,
                split_f0_sidecars=args.split_f0_sidecars,
                f0_hop_seconds=args.f0_hop_seconds,
            )
            print(json.dumps(split_summary(result), indent=2))
            return 0
        if args.command == "prepare":
            manifest = prepare_dataset(
                args.source,
                args.output,
                sample_rate=args.sample_rate,
                f0_method=args.f0_method,
                f0_floor=args.f0_floor,
                f0_ceiling=args.f0_ceiling,
                validation_ratio=args.validation_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                min_duration=args.min_duration,
                normalize_peak=args.normalize_peak,
            )
            print(json.dumps(summarize(manifest.records), indent=2))
            print(f"Dataset: {manifest.root}")
            print(f"Fingerprint: {manifest.fingerprint}")
            return 0
        if args.command == "parameters":
            destination = args.output or Path(args.dataset).resolve() / "parameters.csv"
            print(export_parameters(args.dataset, destination))
            print(json.dumps(parameter_summary(args.dataset), indent=2))
            return 0
        if args.command == "validate":
            manifest = DatasetManifest.load(args.dataset)
            errors = validate_manifest(manifest)
            result = {"ok": not errors, "errors": errors, **summarize(manifest.records)}
            print(json.dumps(result, indent=2) if args.json else _validation_text(result))
            return 0 if not errors else 2
        if args.command in {"inspect", "visualize"}:
            manifest = DatasetManifest.load(args.dataset)
            destination = args.output or manifest.root / "report.html"
            print(build_report(manifest, destination))
            return 0
        if args.command == "init-experiment":
            path = create_experiment(
                args.dataset,
                args.output,
                model=args.model,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                seed=args.seed,
                f0_min=args.f0_min,
                f0_max=args.f0_max,
                workers=args.workers,
                slurm_partition=args.partition,
                slurm_gres=args.gres,
                slurm_time=args.time,
                slurm_cpus=args.cpus,
                slurm_memory=args.memory,
                slurm_exclude=args.exclude,
            )
            print(f"Experiment created: {path}")
            print(f"Dry run: phonlab train {shlex.quote(str(path))} --dry-run")
            return 0
        if args.command == "train":
            command = train(args.experiment, extra, dry_run=args.dry_run)
            if args.dry_run:
                print(" ".join(shlex.quote(part) for part in command))
            return 0
        if args.command == "metrics":
            data = load_metrics(args.experiment, args.version)
            summary = summarize_metrics(data)
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                destination = args.output or Path(args.experiment) / "metrics.html"
                print(build_metrics_dashboard(args.experiment, destination, args.version))
                print(
                    f"Metrics: {len(data.rows)} steps, "
                    f"{len(data.metric_names)} series, "
                    f"{len(data.nonfinite_events)} non-finite values"
                )
            return 0 if not data.nonfinite_events else 2
        if args.command == "synthesize":
            command = synthesize(
                args.experiment,
                args.checkpoint,
                args.output,
                dry_run=args.dry_run,
                f0_scale=semitones_to_scale(args.semitones),
            )
            if args.dry_run:
                print(" ".join(shlex.quote(part) for part in command))
            return 0
        if args.command == "manipulate":
            commands = manipulate_pitch(
                args.experiment,
                args.checkpoint,
                args.output,
                args.semitones,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                for command in commands:
                    print(" ".join(shlex.quote(part) for part in command))
            elif args.baseline:
                destination = args.report or args.output / "comparison.html"
                print(
                    build_manipulation_report(
                        args.experiment,
                        args.baseline,
                        args.output,
                        destination,
                    )
                )
            else:
                print(f"Manipulations: {args.output}")
            return 0
        if args.command == "init-postprocess":
            bundle = create_postprocess_job(
                args.experiment,
                args.checkpoint,
                args.output,
                args.semitones,
                partition=args.partition,
                gres=args.gres,
                time_limit=args.time,
                cpus=args.cpus,
                memory=args.memory,
                exclude=args.exclude,
            )
            print(f"Post-processing job: {bundle}")
            print(f"Submit: sbatch {shlex.quote(str(bundle / 'train.slurm'))}")
            return 0
        if args.command == "submit-job":
            if not args.confirm:
                raise ValueError("submit-job requires --confirm")
            job = submit_job(args.job_bundle)
            print(json.dumps(job.to_dict(), indent=2))
            return 0
        if args.command == "job-status":
            status = query_job(args.job_id)
            payload = status.to_dict()
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"{payload['job_id']}: {payload['state']} "
                    f"({payload['category']}); {payload['node_list'] or payload['reason'] or ''}"
                )
            return 0
        if args.command == "job-log":
            print(
                read_job_log(
                    args.job_bundle,
                    args.job_id,
                    tail_lines=args.tail_lines,
                ),
                end="",
            )
            return 0
        if args.command == "cancel-job":
            if not args.confirm:
                raise ValueError("cancel-job requires --confirm")
            cancel_job(args.job_id)
            print(f"Cancellation requested for Slurm job {args.job_id}")
            return 0
        if args.command == "compare":
            manifest = DatasetManifest.load(args.dataset)
            print(build_synthesis_report(manifest, args.synthesis, args.output))
            return 0
        if args.command == "gui":
            from .gui import serve_gui

            serve_gui(args.host, args.port, open_browser=not args.no_browser)
            return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 1


def _validation_text(result):
    if result["ok"]:
        return (
            f"OK: {result['files']} files, {result['duration_s'] / 60:.1f} minutes, "
            f"sample rates {result['sample_rates']}"
        )
    return "INVALID:\n" + "\n".join(f"- {error}" for error in result["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
