"""Command-line interface for the complete ARIS workflow."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .controls.specs import controls_for_model, parse_variant, pitch_variants
from .corpus import acquire_cmu_arctic
from .doctor import as_json, checks
from .experiment import MODELS, create_experiment, synthesize, train
from .manifest import DatasetManifest, prepare_dataset, summarize, validate_manifest
from .manipulation import (
    build_manipulation_report,
    manipulate_controls,
    semitones_to_scale,
)
from .segment import split_audio, split_summary


def parser() -> argparse.ArgumentParser:
    """Build the argument parser for all ``aris`` subcommands."""
    root = argparse.ArgumentParser(
        prog="aris",
        description="Prepare data, train differentiable vocoders, and generate manipulated speech stimuli.",
    )
    root.add_argument(
        "--version", action="version", version="aris 0.1.0", help="show the version and exit"
    )
    commands = root.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="check audio and training dependencies")
    doctor.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of text"
    )

    corpus = commands.add_parser(
        "fetch-corpus",
        help="download and select a reproducible 30-60 minute CMU ARCTIC corpus",
    )
    corpus.add_argument("output", type=Path, help="directory to write the selected corpus into")
    corpus.add_argument(
        "--archive",
        type=Path,
        help="reuse an already downloaded official archive instead of downloading it",
    )
    corpus.add_argument(
        "--target-minutes",
        type=float,
        default=30.0,
        help="desired total speech duration in minutes (default: %(default)s)",
    )
    corpus.add_argument(
        "--max-minutes",
        type=float,
        default=60.0,
        help="hard cap on total speech duration in minutes (default: %(default)s)",
    )
    corpus.add_argument(
        "--silence-gap",
        type=float,
        default=0.35,
        help="minimum silence gap in seconds used to choose utterance boundaries (default: %(default)s)",
    )

    split = commands.add_parser(
        "split-audio",
        aliases=["split"],
        help="split recordings at silence boundaries or into fixed windows",
    )
    split.add_argument("source", type=Path, help="input audio file or directory")
    split.add_argument("output", type=Path, help="directory to write split segments into")
    split.add_argument(
        "--mode",
        choices=["silence", "fixed"],
        default="silence",
        help="split at detected silence boundaries or into fixed-length windows (default: %(default)s)",
    )
    split.add_argument(
        "--segment-seconds",
        type=float,
        default=2.0,
        help="window length in seconds for --mode fixed (default: %(default)s)",
    )
    split.add_argument(
        "--overlap-seconds",
        type=float,
        default=0.0,
        help="overlap between consecutive windows in seconds for --mode fixed (default: %(default)s)",
    )
    split.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-40.0,
        help="level below which audio counts as silence, in dBFS (default: %(default)s)",
    )
    split.add_argument(
        "--min-silence-seconds",
        type=float,
        default=0.30,
        help="minimum silence duration in seconds to count as a split boundary (default: %(default)s)",
    )
    split.add_argument(
        "--padding-seconds",
        type=float,
        default=0.05,
        help="audio kept on each side of a silence boundary, in seconds (default: %(default)s)",
    )
    split.add_argument(
        "--min-duration-seconds",
        type=float,
        default=0.25,
        help="discard segments shorter than this, in seconds (default: %(default)s)",
    )
    split.add_argument(
        "--max-duration-seconds",
        type=float,
        default=15.0,
        help="discard segments longer than this, in seconds (default: %(default)s)",
    )
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
    prepare.add_argument("source", type=Path, help="input audio directory")
    prepare.add_argument("output", type=Path, help="directory to write the prepared dataset into")
    prepare.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="target sample rate in Hz (default: %(default)s)",
    )
    prepare.add_argument(
        "--f0-method",
        choices=["auto", "pyworld", "autocorr", "sidecar"],
        default="auto",
        help="'sidecar' reuses a .pv file beside each source WAV",
    )
    prepare.add_argument(
        "--f0-floor",
        type=float,
        default=60,
        help="lowest F0 to track, in Hz (default: %(default)s)",
    )
    prepare.add_argument(
        "--f0-ceiling",
        type=float,
        default=500,
        help="highest F0 to track, in Hz (default: %(default)s)",
    )
    prepare.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="fraction of utterances held out for validation (default: %(default)s)",
    )
    prepare.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="fraction of utterances held out for testing (default: %(default)s)",
    )
    prepare.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for the train/validation/test split (default: %(default)s)",
    )
    prepare.add_argument(
        "--min-duration",
        type=float,
        default=0.25,
        help="discard utterances shorter than this, in seconds (default: %(default)s)",
    )
    prepare.add_argument(
        "--normalize-peak",
        type=float,
        help="peak-normalize audio to this level in dBFS; omit to leave levels unchanged",
    )

    validate = commands.add_parser("validate", help="validate a prepared dataset")
    validate.add_argument("dataset", type=Path, help="prepared dataset directory to validate")
    validate.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of text"
    )

    init = commands.add_parser("init-experiment", help="create a portable training experiment")
    init.add_argument(
        "dataset", type=Path, help="prepared dataset directory to bind the experiment to"
    )
    init.add_argument(
        "output", type=Path, help="directory to create the experiment in (must not already exist)"
    )
    init.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="golf",
        help="decoder architecture preset to train (default: %(default)s)",
    )
    init.add_argument(
        "--batch-size", type=int, default=32, help="training batch size (default: %(default)s)"
    )
    init.add_argument(
        "--max-steps",
        type=int,
        default=40000,
        help="number of training steps to run (default: %(default)s)",
    )
    init.add_argument(
        "--seed", type=int, default=42, help="random seed for training (default: %(default)s)"
    )
    init.add_argument(
        "--f0-min",
        type=float,
        default=60,
        help="lowest F0 the encoder should expect, in Hz (default: %(default)s)",
    )
    init.add_argument(
        "--f0-max",
        type=float,
        default=500,
        help="highest F0 the encoder should expect, in Hz (default: %(default)s)",
    )
    init.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of data-loading worker processes (default: %(default)s)",
    )
    init.add_argument(
        "--partition",
        default="gpu",
        help="Slurm partition for the generated train.slurm script (default: %(default)s)",
    )
    init.add_argument(
        "--gres",
        default="gpu:1",
        help="Slurm --gres value for the generated train.slurm script (default: %(default)s)",
    )
    init.add_argument(
        "--time",
        default="04:00:00",
        help="Slurm wall-clock time limit for the generated train.slurm script (default: %(default)s)",
    )
    init.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="Slurm CPUs per task for the generated train.slurm script (default: %(default)s)",
    )
    init.add_argument(
        "--memory",
        default="32G",
        help="Slurm memory request for the generated train.slurm script (default: %(default)s)",
    )
    init.add_argument(
        "--exclude", default="", help="comma-separated Slurm node names to exclude, if any"
    )

    training = commands.add_parser("train", help="check dataset integrity and start training")
    training.add_argument("experiment", type=Path, help="experiment directory from init-experiment")
    training.add_argument(
        "--dry-run", action="store_true", help="print the training command instead of running it"
    )

    controls = commands.add_parser(
        "controls",
        help="list manipulation parameters declared by an experiment model",
    )
    controls.add_argument("experiment", type=Path, help="experiment directory from init-experiment")
    controls.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of text"
    )

    synthesis = commands.add_parser("synthesize", help="reconstruct held-out audio")
    synthesis.add_argument(
        "experiment", type=Path, help="experiment directory from init-experiment"
    )
    synthesis.add_argument(
        "checkpoint", type=Path, help="trained checkpoint file, e.g. runs/checkpoints/last.ckpt"
    )
    synthesis.add_argument(
        "output",
        type=Path,
        help="directory to write synthesized audio into (must not already exist)",
    )
    synthesis.add_argument(
        "--dry-run", action="store_true", help="print the synthesis command instead of running it"
    )
    synthesis.add_argument(
        "--semitones",
        type=float,
        default=0.0,
        help="shift voiced F0 conditioning while preserving unvoiced frames",
    )
    synthesis.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "apply one non-F0 DDSP control; repeat for combinations, e.g. "
            "--control f1_scale=1.1 --control pitch_semitones=2 (equivalent to manipulate's "
            "--variant name:f1_scale=1.1,pitch_semitones=2)"
        ),
    )

    manipulation = commands.add_parser(
        "manipulate",
        help="render auditable named DDSP-control variants from a checkpoint",
    )
    manipulation.add_argument(
        "experiment", type=Path, help="experiment directory from init-experiment"
    )
    manipulation.add_argument(
        "checkpoint", type=Path, help="trained checkpoint file, e.g. runs/checkpoints/last.ckpt"
    )
    manipulation.add_argument(
        "output",
        type=Path,
        help="directory to write rendered variants into (must not already exist)",
    )
    manipulation.add_argument(
        "--semitones",
        type=float,
        nargs="+",
        help="backward-compatible pitch sweep, in semitones",
    )
    manipulation.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME:CONTROL=VALUE,...",
        help=(
            "named multi-parameter variant; repeat to render several conditions, e.g. "
            "--variant boosted:f1_scale=1.1,pitch_semitones=2 (equivalent to synthesize's "
            "--control f1_scale=1.1 --control pitch_semitones=2)"
        ),
    )
    manipulation.add_argument(
        "--baseline",
        type=Path,
        help="unmodified reconstruction directory to compare variants against in --report",
    )
    manipulation.add_argument(
        "--report",
        type=Path,
        help="path for the generated HTML comparison report (default: <output>/comparison.html)",
    )
    manipulation.add_argument(
        "--dry-run", action="store_true", help="print the render commands instead of running them"
    )

    studio = commands.add_parser(
        "studio",
        help="launch a local web studio for manipulation design and A/B listening",
    )
    studio.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="host address to bind the studio server to (default: %(default)s)",
    )
    studio.add_argument(
        "--port",
        type=int,
        default=8765,
        help="local port to serve the studio on (default: %(default)s)",
    )
    studio.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="directory scanned for experiments (default: current directory)",
    )
    studio.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a web browser automatically",
    )
    studio.add_argument(
        "--share",
        action="store_true",
        help="create a publicly shareable Gradio link (recommended for Google Colab and remote hosts)",
    )
    return root


def main(argv=None) -> int:
    """Dispatch a parsed command and return a process exit code."""
    from .cuda_env import auto_configure_cuda

    auto_configure_cuda()

    argument_parser = parser()
    args, extra = argument_parser.parse_known_args(argv)
    # Only `train` forwards unrecognized arguments (to the training engine).
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
            print(
                json.dumps(
                    summarize(manifest.records, skipped=manifest.metadata.get("skipped")),
                    indent=2,
                )
            )
            print(f"Dataset: {manifest.root}")
            print(f"Fingerprint: {manifest.fingerprint}")
            return 0
        if args.command == "validate":
            manifest = DatasetManifest.load(args.dataset)
            errors = validate_manifest(manifest)
            result = {"ok": not errors, "errors": errors, **summarize(manifest.records)}
            print(json.dumps(result, indent=2) if args.json else _validation_text(result))
            return 0 if not errors else 2
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
            print(f"Dry run: aris train {shlex.quote(str(path))} --dry-run")
            return 0
        if args.command == "train":
            command = train(args.experiment, extra, dry_run=args.dry_run)
            if args.dry_run:
                print(" ".join(shlex.quote(part) for part in command))
            return 0
        if args.command == "controls":
            metadata = json.loads((Path(args.experiment).resolve() / "experiment.json").read_text())
            model = metadata.get("model")
            if not isinstance(model, str):
                raise ValueError(f"experiment.json has no valid 'model' field (found {model!r})")
            available = controls_for_model(model)
            payload = {
                "model": model,
                "controls": [asdict(spec) for spec in available],
            }
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(f"Model: {model}")
                for spec in available:
                    print(
                        f"{spec.name:22} {spec.minimum:g}..{spec.maximum:g} "
                        f"{spec.unit:10} default={spec.default:g}  {spec.description}"
                    )
            return 0
        if args.command == "synthesize":
            controls = _control_assignments(args.control)
            command = synthesize(
                args.experiment,
                args.checkpoint,
                args.output,
                dry_run=args.dry_run,
                f0_scale=semitones_to_scale(args.semitones),
                controls=controls,
            )
            if args.dry_run:
                print(" ".join(shlex.quote(part) for part in command))
            return 0
        if args.command == "manipulate":
            variants = list(pitch_variants(args.semitones or []))
            variants.extend(parse_variant(value) for value in args.variant)
            commands = manipulate_controls(
                args.experiment,
                args.checkpoint,
                args.output,
                variants,
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
        if args.command == "studio":
            # Imported lazily: gradio is an optional extra and must not be
            # required for any other subcommand.
            from .studio import launch_studio

            launch_studio(
                args.workspace,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
                share=args.share,
            )
            return 0
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        if isinstance(error, subprocess.CalledProcessError):
            message = (
                f"aris {args.command} failed: the underlying engine exited with "
                f"status {error.returncode} (see output above for details)"
            )
        else:
            message = str(error)
        print(f"error: {message}", file=sys.stderr)
        return 2
    return 1


def _validation_text(result):
    if result["ok"]:
        return (
            f"OK: {result['files']} files, {result['duration_s'] / 60:.1f} minutes, "
            f"sample rates {result['sample_rates']}"
        )
    return "INVALID:\n" + "\n".join(f"- {error}" for error in result["errors"])


def _control_assignments(values: list[str]) -> dict[str, float]:
    """Parse repeated ``NAME=VALUE`` options into a control mapping."""
    if not values:
        return {}
    return parse_variant("synthesis:" + ",".join(values)).controls


if __name__ == "__main__":
    raise SystemExit(main())
