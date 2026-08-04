import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from phonlab_ddsp.webui_results import (
    create_export_zip,
    discover_result_catalog,
    export_condition,
    export_wav,
    load_result_catalog,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_render(path: Path, controls, audio, clipping=None) -> None:
    clipping = clipping or {}
    rows = []
    total_clipped = 0
    total_samples = 0
    for relative, payload in audio.items():
        wav = path.parent.joinpath(*relative.split("/"))
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(payload)
        clipped = clipping.get(relative, 0)
        samples = 100
        total_clipped += clipped
        total_samples += samples
        rows.append(
            {
                "path": relative,
                "peak_before_gain": 0.5,
                "peak_after_gain_unclipped": 0.5,
                "clipped_samples": clipped,
                "samples": samples,
            }
        )
    _write_json(
        path,
        {
            "schema_version": "1.0",
            "controls": controls,
            "runtime_capabilities": ["output_gain_db"],
            "decoder_control_calls": len(audio) if controls else 0,
            "files_written": len(rows),
            "clipped_samples": total_clipped,
            "samples": total_samples,
            "clipped_fraction": total_clipped / total_samples,
            "files": rows,
        },
    )


def _fixture(workspace: Path) -> Path:
    root = workspace / "results" / "postprocess"
    baseline_audio = {
        "speaker/item-a.wav": b"RIFF-baseline-a",
        "speaker/item-b.wav": b"RIFF-baseline-b",
    }
    _write_render(root / "reconstruction" / "_render.json", {}, baseline_audio)
    _write_render(
        root / "manipulations" / "lower" / "_render.json",
        {},
        {name: payload + b"-lower" for name, payload in baseline_audio.items()},
    )
    _write_render(
        root / "manipulations" / "louder" / "_render.json",
        {"output_gain_db": 3.0},
        {name: payload + b"-louder" for name, payload in baseline_audio.items()},
        {"speaker/item-b.wav": 4},
    )
    _write_json(
        root / "manipulations" / "manipulation.json",
        {
            "schema_version": "2.0",
            "created_at": "2026-08-03T12:00:00+00:00",
            "operation": "ddsp-multi-parameter-control",
            "model": "golf",
            "dataset_fingerprint": "b" * 64,
            "checkpoint_sha256": "a" * 64,
            "experiment": "/external/private/experiment",
            "checkpoint": "/external/private/last.ckpt",
            "outputs": [
                {
                    "name": "lower",
                    "directory": "lower",
                    "controls": {"pitch_semitones": -4.0},
                    "label": "Pitch -4 st",
                    "render_audit": {"metadata": "lower/_render.json"},
                },
                {
                    "name": "louder",
                    "directory": "louder",
                    "controls": {"output_gain_db": 3.0},
                    "label": "Output +3 dB",
                    "render_audit": {"metadata": "louder/_render.json"},
                },
            ],
        },
    )
    for report in ("reconstruction.html", "manipulation.html", "metrics.html"):
        (root / report).write_text(f"<html>{report}</html>", encoding="utf-8")
    return root


class WebUiResultCatalogTest(unittest.TestCase):
    def test_catalog_is_json_serializable_and_uniformly_lists_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = _fixture(workspace)
            catalog = discover_result_catalog(workspace, root.relative_to(workspace))
            payload = catalog.to_dict()

            json.dumps(payload)
            self.assertEqual(payload["result_relative_path"], "results/postprocess")
            self.assertEqual(len(payload["conditions"]), 2)
            self.assertEqual(len(payload["items"]), 2)
            self.assertEqual(
                payload["reports"],
                {
                    "reconstruction": "reconstruction.html",
                    "manipulation": "manipulation.html",
                    "metrics": "metrics.html",
                },
            )
            item = payload["items"][0]
            self.assertEqual(item["id"], "speaker/item-a.wav")
            self.assertEqual(item["item_id"], item["id"])
            self.assertEqual(
                [audio["condition"] for audio in item["audio"]],
                ["baseline", "lower", "louder"],
            )
            self.assertEqual(item["baseline"], item["audio"][0])
            self.assertEqual(item["variants"], item["audio"][1:])
            self.assertEqual(
                item["audio"][1]["path"],
                "manipulations/lower/speaker/item-a.wav",
            )
            self.assertNotIn("checkpoint", payload["provenance"])
            louder = next(item for item in payload["conditions"] if item["name"] == "louder")
            self.assertEqual(louder["clipping"]["clipped_samples"], 4)
            self.assertEqual(louder["clipping"]["files_with_clipping"], 1)
            second = payload["items"][1]["audio"][2]
            self.assertEqual(second["clipped_samples"], 4)
            self.assertEqual(second["clipped_fraction"], 0.04)

    def test_loads_realistic_pitch_controls_not_repeated_by_runtime_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = _fixture(workspace)
            catalog = load_result_catalog(workspace, root)
            lower = next(condition for condition in catalog.conditions if condition.name == "lower")
            self.assertEqual(lower.controls, {"pitch_semitones": -4.0})

    def test_rejects_result_outside_workspace_and_symlinked_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            workspace = parent / "workspace"
            workspace.mkdir()
            outside = parent / "outside"
            root = _fixture(outside)
            with self.assertRaisesRegex(ValueError, "below"):
                load_result_catalog(workspace, root)

            linked = workspace / "linked-results"
            linked.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symbolic-link"):
                load_result_catalog(workspace, linked)

    def test_rejects_traversal_symlinks_and_inconsistent_audits(self):
        mutations = {
            "variant traversal": lambda root: _mutate_metadata(
                root, lambda value: value["outputs"][0].__setitem__("directory", "../lower")
            ),
            "render traversal": lambda root: _mutate_metadata(
                root,
                lambda value: value["outputs"][0]["render_audit"].__setitem__(
                    "metadata", "../lower/_render.json"
                ),
            ),
            "wav traversal": lambda root: _mutate_render_path(
                root / "reconstruction" / "_render.json", "../item-a.wav"
            ),
            "windows path": lambda root: _mutate_render_path(
                root / "reconstruction" / "_render.json", "speaker\\item-a.wav"
            ),
            "aggregate clipping": lambda root: _mutate_render_value(
                root / "manipulations" / "louder" / "_render.json",
                "clipped_samples",
                3,
            ),
            "missing variant": lambda root: (
                root / "manipulations" / "lower" / "speaker" / "item-b.wav"
            ).unlink(),
            "symlink wav": _replace_wav_with_symlink,
            "symlink report": _replace_report_with_symlink,
            "symlink condition": _replace_condition_with_symlink,
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary)
                root = _fixture(workspace)
                mutation(root)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    load_result_catalog(workspace, root)

    def test_rejects_duplicate_and_mismatched_wav_sets_and_file_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = _fixture(workspace)
            render_path = root / "manipulations" / "lower" / "_render.json"
            render = json.loads(render_path.read_text())
            render["files"][1]["path"] = render["files"][0]["path"]
            _write_json(render_path, render)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_result_catalog(workspace, root)

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = _fixture(workspace)
            with self.assertRaisesRegex(ValueError, "file limit"):
                load_result_catalog(workspace, root, max_files_per_condition=1)


class WebUiResultExportTest(unittest.TestCase):
    def test_exports_one_wav_and_a_complete_condition_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = _fixture(workspace)
            catalog = load_result_catalog(workspace, root)

            single = export_wav(
                catalog,
                "lower",
                "speaker/item-a.wav",
                "exports/single",
            )
            self.assertEqual(len(single.files), 1)
            exported = workspace / "exports/single/audio/lower/speaker/item-a.wav"
            self.assertTrue(exported.is_file())
            provenance = json.loads(single.provenance_path.read_text())
            self.assertEqual(provenance["selection"]["kind"], "wav")
            self.assertEqual(provenance["source"]["condition"]["name"], "lower")
            self.assertEqual(
                provenance["files"][0]["sha256"],
                hashlib.sha256(exported.read_bytes()).hexdigest(),
            )
            with self.assertRaises(FileExistsError):
                export_wav(
                    catalog,
                    "lower",
                    "speaker/item-a.wav",
                    "exports/single",
                )

            complete = export_condition(catalog, "louder", "exports/louder")
            self.assertEqual(len(complete.files), 2)
            self.assertTrue(
                (workspace / "exports/louder/audio/louder/speaker/item-b.wav").is_file()
            )
            complete_provenance = json.loads(complete.provenance_path.read_text())
            self.assertEqual(complete_provenance["selection"]["kind"], "condition")
            self.assertEqual(len(complete_provenance["files"]), 2)

    def test_export_revalidates_sources_and_rejects_unknown_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = _fixture(workspace)
            catalog = load_result_catalog(workspace, root)
            with self.assertRaisesRegex(ValueError, "Unknown catalog condition"):
                export_condition(catalog, "invented", "exports/unknown")
            with self.assertRaisesRegex(ValueError, "Unknown catalog item"):
                export_wav(catalog, "lower", "not-listed.wav", "exports/unknown")

            source = root / "manipulations/lower/speaker/item-a.wav"
            source.unlink()
            source.symlink_to(root / "reconstruction/speaker/item-a.wav")
            with self.assertRaisesRegex(ValueError, "Symbolic-link"):
                export_wav(catalog, "lower", "speaker/item-a.wav", "exports/changed")

    def test_export_destination_must_be_new_safe_and_inside_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            workspace = parent / "workspace"
            workspace.mkdir()
            root = _fixture(workspace)
            catalog = load_result_catalog(workspace, root)

            existing = workspace / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                export_condition(catalog, "lower", existing)
            with self.assertRaisesRegex(ValueError, "below"):
                export_condition(catalog, "lower", parent / "outside-export")
            with self.assertRaisesRegex(ValueError, "outside the source"):
                export_condition(catalog, "lower", root / "saved")

            real_parent = workspace / "real-parent"
            real_parent.mkdir()
            linked_parent = workspace / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symbolic-link"):
                export_condition(catalog, "lower", linked_parent / "saved")

    def test_condition_export_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            catalog = load_result_catalog(workspace, _fixture(workspace))
            with self.assertRaisesRegex(ValueError, "export limit"):
                export_condition(catalog, "lower", "exports/all", max_files=1)

    def test_zip_is_atomic_bounded_and_contains_audio_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            catalog = load_result_catalog(workspace, _fixture(workspace))
            archive = create_export_zip(catalog, "lower")
            self.assertEqual(archive.parent, workspace / ".cache/webui_exports")
            self.assertTrue(archive.is_file())
            self.assertFalse(any(archive.parent.glob(".webui-export-*.tmp")))
            with zipfile.ZipFile(archive) as reader:
                names = reader.namelist()
                self.assertEqual(
                    names,
                    [
                        "audio/lower/speaker/item-a.wav",
                        "audio/lower/speaker/item-b.wav",
                        "provenance.json",
                    ],
                )
                provenance = json.loads(reader.read("provenance.json"))
                self.assertEqual(provenance["selection"]["kind"], "condition")
                self.assertEqual(len(provenance["files"]), 2)

            single = create_export_zip(
                catalog,
                "baseline",
                item_id="speaker/item-b.wav",
                cache_root="downloads",
                max_files=1,
            )
            self.assertEqual(single.parent, workspace / "downloads")
            with zipfile.ZipFile(single) as reader:
                self.assertEqual(
                    reader.namelist(),
                    ["audio/baseline/speaker/item-b.wav", "provenance.json"],
                )
            with self.assertRaisesRegex(ValueError, "ZIP limit"):
                create_export_zip(catalog, "lower", max_files=1)

    def test_zip_rejects_outside_or_symlinked_cache_and_cleans_temp_on_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            workspace = parent / "workspace"
            workspace.mkdir()
            catalog = load_result_catalog(workspace, _fixture(workspace))
            with self.assertRaisesRegex(ValueError, "below"):
                create_export_zip(catalog, "lower", cache_root=parent / "outside")

            real_cache = workspace / "real-cache"
            real_cache.mkdir()
            linked_cache = workspace / "linked-cache"
            linked_cache.symlink_to(real_cache, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symbolic-link"):
                create_export_zip(catalog, "lower", cache_root=linked_cache)

            cache = workspace / "safe-cache"
            with patch(
                "phonlab_ddsp.webui_results._copy_into_zip",
                side_effect=RuntimeError("copy failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "copy failed"):
                    create_export_zip(catalog, "lower", cache_root=cache)
            self.assertEqual(list(cache.iterdir()), [])


def _mutate_metadata(root: Path, mutation) -> None:
    path = root / "manipulations/manipulation.json"
    payload = json.loads(path.read_text())
    mutation(payload)
    _write_json(path, payload)


def _mutate_render_path(path: Path, value: str) -> None:
    payload = json.loads(path.read_text())
    payload["files"][0]["path"] = value
    _write_json(path, payload)


def _mutate_render_value(path: Path, field: str, value) -> None:
    payload = json.loads(path.read_text())
    payload[field] = value
    _write_json(path, payload)


def _replace_wav_with_symlink(root: Path) -> None:
    path = root / "manipulations/lower/speaker/item-a.wav"
    path.unlink()
    path.symlink_to(root / "reconstruction/speaker/item-a.wav")


def _replace_report_with_symlink(root: Path) -> None:
    path = root / "metrics.html"
    path.unlink()
    path.symlink_to(root / "reconstruction.html")


def _replace_condition_with_symlink(root: Path) -> None:
    path = root / "manipulations/lower"
    moved = root / "real-lower"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=True)


if __name__ == "__main__":
    unittest.main()
