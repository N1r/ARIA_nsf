import subprocess
import tempfile
import unittest
from pathlib import Path

from phonlab_ddsp.jobs import (
    JobCategory,
    SlurmBackend,
    SlurmCommandError,
    SlurmUnavailableError,
    SubmittedJob,
    cancel_job,
    job_log_path,
    query_job,
    read_job_log,
    submit_job,
    validate_experiment,
)


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _completed(argv, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)


class SlurmJobsTest(unittest.TestCase):
    def _experiment(self, root: Path, output="slurm-%j.log") -> Path:
        experiment = root / "experiment"
        experiment.mkdir()
        (experiment / "train.slurm").write_text(
            f"#!/usr/bin/env bash\n#SBATCH --output={output}\nset -euo pipefail\n./train.sh\n",
            encoding="utf-8",
        )
        return experiment

    def test_validate_experiment_requires_directory_and_nonempty_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                validate_experiment(root / "missing")
            experiment = root / "experiment"
            experiment.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "no train.slurm"):
                validate_experiment(experiment)
            (experiment / "train.slurm").touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                validate_experiment(experiment)

    def test_validate_experiment_rejects_script_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            target = root / "outside.slurm"
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            (experiment / "train.slurm").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                validate_experiment(experiment)

    def test_submit_uses_argv_without_shell_and_parses_cluster_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            runner = FakeRunner([_completed([], stdout="48201;alpha\n")])

            job_id = submit_job(experiment, runner=runner)

            self.assertEqual(job_id, "48201")
            self.assertIsInstance(job_id, SubmittedJob)
            self.assertEqual(job_id.cluster, "alpha")
            self.assertEqual(job_id.to_dict(), {"job_id": "48201", "cluster": "alpha"})
            argv, kwargs = runner.calls[0]
            self.assertEqual(argv, ["sbatch", "--parsable", str(experiment / "train.slurm")])
            self.assertIs(kwargs["shell"], False)
            self.assertEqual(kwargs["cwd"], str(experiment))

    def test_submit_reports_missing_command_and_bad_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            missing = FakeRunner([FileNotFoundError("sbatch")])
            with self.assertRaises(SlurmUnavailableError):
                submit_job(experiment, runner=missing)

            malformed = FakeRunner([_completed([], stdout="not a job id")])
            with self.assertRaisesRegex(SlurmCommandError, "unrecognized"):
                submit_job(experiment, runner=malformed)

    def test_query_reads_active_squeue_job_and_serializes_stably(self):
        runner = FakeRunner(
            [
                _completed(
                    [],
                    stdout="48201|RUNNING|voice-model|None|2026-07-30T12:00:00|01:02|node01\n",
                )
            ]
        )

        status = query_job("48201", runner=runner)

        self.assertEqual(status.category, JobCategory.RUNNING)
        self.assertEqual(status.source, "squeue")
        self.assertEqual(status.node_list, "node01")
        self.assertFalse(status.terminal)
        self.assertEqual(status.to_dict()["category"], "running")
        self.assertEqual(len(runner.calls), 1)
        self.assertIs(runner.calls[0][1]["shell"], False)
        self.assertIn("%r", runner.calls[0][0][-1])
        self.assertNotIn("%R", runner.calls[0][0][-1])

    def test_query_delimiter_does_not_confuse_pipe_in_job_name(self):
        separator = "\x1f"
        runner = FakeRunner(
            [
                _completed(
                    [],
                    stdout=separator.join(
                        [
                            "48201",
                            "RUNNING",
                            "voice|model",
                            "None",
                            "2026-07-30T12:00:00",
                            "01:02",
                            "node01",
                        ]
                    ),
                )
            ]
        )

        status = query_job("48201", runner=runner)

        self.assertEqual(status.name, "voice|model")
        self.assertEqual(status.reason, None)

    def test_submitted_cluster_routes_query_and_cancel(self):
        handle = SubmittedJob("48201", "alpha")
        runner = FakeRunner(
            [
                _completed(
                    [],
                    stdout="48201|PENDING|voice-model|Priority|N/A|0:00|(null)\n",
                ),
                _completed([]),
            ]
        )

        status = query_job(handle, runner=runner)
        self.assertEqual(status.cluster, "alpha")
        self.assertEqual(status.category, JobCategory.QUEUED)
        self.assertEqual(runner.calls[0][0][:4], ["squeue", "--clusters", "alpha", "--noheader"])
        self.assertTrue(cancel_job(handle, runner=runner))
        self.assertEqual(runner.calls[1][0], ["scancel", "--clusters", "alpha", "48201"])

    def test_query_falls_back_to_sacct_for_completed_job(self):
        runner = FakeRunner(
            [
                _completed([], stdout=""),
                _completed(
                    [],
                    stdout=(
                        "48201|COMPLETED|voice-model|None|2026-07-30T12:00:00|"
                        "00:05:00|2026-07-30T12:05:00|0:0|node01|\n"
                    ),
                ),
            ]
        )

        status = query_job(48201, runner=runner)

        self.assertEqual(status.state, "COMPLETED")
        self.assertEqual(status.category, JobCategory.SUCCEEDED)
        self.assertEqual(status.source, "sacct")
        self.assertEqual(status.exit_code, "0:0")
        self.assertTrue(status.is_terminal)

    def test_query_accepts_sacct_layout_without_reason_column(self):
        runner = FakeRunner(
            [
                _completed([], stdout=""),
                _completed(
                    [],
                    stdout=(
                        "48201|TIMEOUT|voice-model|2026-07-30T12:00:00|"
                        "01:00:00|2026-07-30T13:00:00|0:15|node01\n"
                    ),
                ),
            ]
        )

        status = query_job("48201", runner=runner)

        self.assertEqual(status.started_at, "2026-07-30T12:00:00")
        self.assertEqual(status.elapsed, "01:00:00")
        self.assertEqual(status.category, JobCategory.FAILED)

    def test_query_preserves_empty_sacct_node_list(self):
        separator = "\x1f"
        runner = FakeRunner(
            [
                _completed([], stdout=""),
                _completed(
                    [],
                    stdout=separator.join(
                        [
                            "48201",
                            "PENDING",
                            "voice|model",
                            "Priority",
                            "Unknown",
                            "00:00:00",
                            "Unknown",
                            "0:0",
                            "",
                        ]
                    ),
                ),
            ]
        )

        status = query_job("00048201", runner=runner)

        self.assertEqual(status.job_id, "48201")
        self.assertEqual(status.name, "voice|model")
        self.assertEqual(status.reason, "Priority")
        self.assertIsNone(status.node_list)
        self.assertEqual(status.category, JobCategory.QUEUED)

    def test_query_is_graceful_when_slurm_is_absent_or_job_unknown(self):
        missing = FakeRunner([FileNotFoundError("squeue"), FileNotFoundError("sacct")])
        unavailable = query_job("123", runner=missing)
        self.assertEqual(unavailable.category, JobCategory.UNAVAILABLE)
        self.assertIn("not available", unavailable.detail)

        unknown_runner = FakeRunner([_completed([], stdout=""), _completed([], stdout="")])
        unknown = query_job("123", runner=unknown_runner)
        self.assertEqual(unknown.category, JobCategory.UNKNOWN)
        self.assertIn("not found", unknown.detail)

    def test_cancel_validates_id_before_invoking_runner(self):
        runner = FakeRunner([_completed([])])
        with self.assertRaisesRegex(ValueError, "positive decimal"):
            cancel_job("123; touch /tmp/not-safe", runner=runner)
        self.assertEqual(runner.calls, [])

        self.assertTrue(cancel_job("123", runner=runner))
        argv, kwargs = runner.calls[0]
        self.assertEqual(argv, ["scancel", "123"])
        self.assertIs(kwargs["shell"], False)

    def test_log_path_honors_output_directive_and_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = self._experiment(root, "logs/job-%j.txt")
            self.assertEqual(
                job_log_path(experiment, "789"),
                experiment / "logs" / "job-789.txt",
            )

            (experiment / "train.slurm").write_text(
                "#!/bin/sh\n#SBATCH --output=../outside-%j.log\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inside the experiment"):
                job_log_path(experiment, "789")

    def test_log_path_uses_first_output_directive(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            (experiment / "train.slurm").write_text(
                "#!/bin/sh\n"
                "#SBATCH --output=first-%j.log\n"
                "#SBATCH --output=second-%j.log\n"
                "./train.sh\n",
                encoding="utf-8",
            )
            self.assertEqual(
                job_log_path(experiment, "42"),
                experiment / "first-42.log",
            )

    def test_log_path_supports_quoted_short_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            (experiment / "train.slurm").write_text(
                '#!/bin/sh\n#SBATCH -o "logs/job %j.log"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                job_log_path(experiment, "42"),
                experiment / "logs" / "job 42.log",
            )

    def test_log_path_supports_slurm_zero_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary), "slurm-%8j.log")
            self.assertEqual(
                job_log_path(experiment, "42"),
                experiment / "slurm-00000042.log",
            )
            (experiment / "train.slurm").write_text(
                "#!/bin/sh\n#SBATCH --output=slurm-%11j.log\n",
                encoding="utf-8",
            )
            self.assertEqual(
                job_log_path(experiment, "42"),
                experiment / "slurm-0000000042.log",
            )

    def test_log_path_uses_slurm_default_when_directive_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            (experiment / "train.slurm").write_text(
                "#!/bin/sh\nset -eu\n./train.sh\n",
                encoding="utf-8",
            )
            self.assertEqual(
                job_log_path(experiment, "42"),
                experiment / "slurm-42.out",
            )

    def test_read_log_is_byte_bounded_and_can_tail_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            log = experiment / "slurm-55.log"
            log.write_text("first\nsecond\nthird\nfourth\n", encoding="utf-8")

            self.assertEqual(
                read_job_log(experiment, "55", max_bytes=13),
                "third\nfourth\n",
            )
            self.assertEqual(
                read_job_log(experiment, "55", max_bytes=100, tail_lines=2),
                "third\nfourth\n",
            )
            self.assertEqual(read_job_log(experiment, "56"), "")
            with self.assertRaises(FileNotFoundError):
                read_job_log(experiment, "56", missing_ok=False)
            with self.assertRaisesRegex(ValueError, "max_bytes"):
                read_job_log(experiment, "55", max_bytes=2 * 1024 * 1024)

    def test_backend_reuses_injected_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self._experiment(Path(temporary))
            runner = FakeRunner([_completed([], stdout="9001\n"), _completed([])])
            backend = SlurmBackend(runner)
            self.assertEqual(backend.submit(experiment), "9001")
            self.assertTrue(backend.cancel("9001"))
            self.assertEqual([call[0][0] for call in runner.calls], ["sbatch", "scancel"])


if __name__ == "__main__":
    unittest.main()
