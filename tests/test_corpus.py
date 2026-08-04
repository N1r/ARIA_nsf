import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path

from aris.corpus import (
    CORPUS_METADATA,
    acquire_cmu_arctic,
    corpus_wavs_in_order,
    create_continuous_wav,
    download_with_resume,
    load_corpus_result,
    safe_extract_tar,
    select_utterances,
    verify_archive,
)


def _write_wav(path: Path, frames: int, sample_rate: int = 100, value: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(value.to_bytes(2, "little", signed=True) * frames)


def _fixture_corpus(root: Path) -> Path:
    corpus = root / "cmu_us_slt_arctic"
    (corpus / "etc").mkdir(parents=True)
    (corpus / "etc" / "txt.done.data").write_text(
        '( utt_b "second alphabetically, first in corpus" )\n'
        '( utt_a "first alphabetically, second in corpus" )\n'
        '( utt_c "third" )\n',
        encoding="utf-8",
    )
    (corpus / "COPYING").write_text("Fixture distribution terms\n", encoding="utf-8")
    _write_wav(corpus / "wav" / "utt_a.wav", 60, value=100)
    _write_wav(corpus / "wav" / "utt_b.wav", 60, value=200)
    _write_wav(corpus / "wav" / "utt_c.wav", 60, value=300)
    return corpus


def _archive_tree(source: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:bz2") as bundle:
        for path in sorted(source.rglob("*")):
            bundle.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)


def _single_member_archive(archive: Path, name: str, *, kind: str = "file") -> None:
    with tarfile.open(archive, "w:bz2") as bundle:
        member = tarfile.TarInfo(name)
        if kind == "file":
            payload = b"contents"
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
        elif kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "elsewhere"
            bundle.addfile(member)
        elif kind == "device":
            member.type = tarfile.CHRTYPE
            bundle.addfile(member)
        else:
            raise AssertionError(kind)


class _BytesResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int, headers: dict):
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class DownloadTest(unittest.TestCase):
    def test_resumes_part_with_range_and_verifies_exact_file(self):
        payload = b"verified archive payload" * 20
        digest = hashlib.sha256(payload).hexdigest()
        requests = []

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "archive.tar.bz2"
            partial = destination.with_name(destination.name + ".part")
            offset = 73
            partial.write_bytes(payload[:offset])

            def opener(request, timeout):
                requests.append((request, timeout))
                return _BytesResponse(
                    payload[offset:],
                    status=206,
                    headers={"Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}"},
                )

            result = download_with_resume(
                "https://example.invalid/archive",
                destination,
                expected_size=len(payload),
                expected_sha256=digest,
                opener=opener,
                chunk_size=17,
            )

            self.assertEqual(result.read_bytes(), payload)
            self.assertFalse(partial.exists())
            self.assertEqual(requests[0][0].get_header("Range"), f"bytes={offset}-")
            self.assertEqual(verify_archive(result, len(payload), digest), digest)

    def test_server_ignoring_range_keeps_partial_unchanged(self):
        payload = b"0123456789"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "archive"
            partial = destination.with_name(destination.name + ".part")
            partial.write_bytes(payload[:4])

            def opener(request, timeout):
                return _BytesResponse(payload, status=200, headers={})

            with self.assertRaisesRegex(RuntimeError, "did not honour"):
                download_with_resume(
                    "https://example.invalid/archive",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    opener=opener,
                )
            self.assertEqual(partial.read_bytes(), payload[:4])
            self.assertFalse(destination.exists())

    def test_empty_existing_part_is_reused_for_full_download(self):
        payload = b"fresh bytes"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "archive"
            partial = destination.with_name(destination.name + ".part")
            partial.touch()

            def opener(request, timeout):
                self.assertIsNone(request.get_header("Range"))
                return _BytesResponse(payload, status=200, headers={})

            download_with_resume(
                "https://example.invalid/archive",
                destination,
                expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                opener=opener,
            )
            self.assertEqual(destination.read_bytes(), payload)

    def test_verification_rejects_size_and_hash_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "archive"
            archive.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_archive(archive, 4, hashlib.sha256(b"abc").hexdigest())
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_archive(archive, 3, "0" * 64)


class SafeExtractionTest(unittest.TestCase):
    def test_extracts_regular_members_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "good.tar.bz2"
            _single_member_archive(archive, "corpus/wav/item.wav")
            destination = root / "output"

            self.assertEqual(safe_extract_tar(archive, destination), destination)
            self.assertEqual((destination / "corpus/wav/item.wav").read_bytes(), b"contents")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                safe_extract_tar(archive, destination)
            self.assertEqual((destination / "corpus/wav/item.wav").read_bytes(), b"contents")

    def test_stale_sibling_staging_is_preserved_and_a_new_attempt_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "good.tar.bz2"
            _single_member_archive(archive, "corpus/item.txt")
            destination = root / "output"
            stale = root / ".output.extracting"
            stale.mkdir()
            sentinel = stale / "interrupted"
            sentinel.write_text("keep", encoding="utf-8")

            safe_extract_tar(archive, destination)

            self.assertEqual((destination / "corpus/item.txt").read_bytes(), b"contents")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_rejects_traversal_links_and_devices_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [
                ("traversal", "../escaped", "file"),
                ("absolute", "/escaped", "file"),
                ("symlink", "corpus/link", "symlink"),
                ("device", "corpus/device", "device"),
            ]
            for label, name, kind in cases:
                with self.subTest(label=label):
                    archive = root / f"{label}.tar.bz2"
                    destination = root / f"{label}-output"
                    _single_member_archive(archive, name, kind=kind)
                    with self.assertRaises(ValueError):
                        safe_extract_tar(archive, destination)
                    self.assertFalse(destination.exists())
            self.assertFalse((root / "escaped").exists())

    def test_rejects_a_symlinked_destination_ancestor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "good.tar.bz2"
            _single_member_archive(archive, "corpus/item.txt")
            real_parent = root / "real"
            real_parent.mkdir()
            linked_parent = root / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "Symbolic-link"):
                safe_extract_tar(archive, linked_parent / "output")
            self.assertFalse((real_parent / "output").exists())


class SubsetTest(unittest.TestCase):
    def test_uses_corpus_order_until_target_without_crossing_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = _fixture_corpus(Path(temporary))
            self.assertEqual(
                [path.stem for path in corpus_wavs_in_order(corpus)],
                ["utt_b", "utt_a", "utt_c"],
            )

            selection = select_utterances(corpus, target_duration_s=1.0, max_duration_s=1.2)
            self.assertEqual([item.id for item in selection.utterances], ["utt_b", "utt_a"])
            self.assertAlmostEqual(selection.duration_s, 1.2)
            self.assertTrue(selection.reached_target)

            capped = select_utterances(corpus, target_duration_s=0.9, max_duration_s=0.9)
            self.assertEqual([item.id for item in capped.utterances], ["utt_b"])
            self.assertAlmostEqual(capped.duration_s, 0.6)
            self.assertFalse(capped.reached_target)

    def test_continuous_wav_inserts_only_configured_inter_item_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.wav"
            second = root / "second.wav"
            _write_wav(first, 60, value=100)
            _write_wav(second, 40, value=200)
            output = root / "continuous.wav"

            create_continuous_wav([first, second], output, silence_gap_s=0.25)
            with wave.open(str(output), "rb") as stream:
                self.assertEqual(stream.getnframes(), 125)
                samples = stream.readframes(125)
            self.assertEqual(samples[60 * 2 : 85 * 2], b"\x00" * 25 * 2)
            self.assertEqual(samples[85 * 2 : 87 * 2], (200).to_bytes(2, "little", signed=True) * 2)
            with self.assertRaises(FileExistsError):
                create_continuous_wav([first, second], output, silence_gap_s=0.25)

    def test_continuous_wav_preflights_format_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.wav"
            incompatible = root / "incompatible.wav"
            _write_wav(first, 60, sample_rate=100)
            _write_wav(incompatible, 60, sample_rate=200)
            output = root / "continuous.wav"

            with self.assertRaisesRegex(ValueError, "Incompatible WAV format"):
                create_continuous_wav([first, incompatible], output)
            self.assertFalse(output.exists())


class AcquisitionTest(unittest.TestCase):
    def _fixture_settings(self, root: Path):
        fixture = root / "fixture"
        fixture.mkdir()
        _fixture_corpus(fixture)
        source_archive = root / "source.tar.bz2"
        _archive_tree(fixture, source_archive)
        digest = hashlib.sha256(source_archive.read_bytes()).hexdigest()
        return source_archive, {
            "target_duration_s": 1.0,
            "max_duration_s": 1.3,
            "silence_gap_s": 0.25,
            "url": "fixture://cmu-arctic",
            "expected_sha256": digest,
            "expected_size": source_archive.stat().st_size,
            "archive_source": source_archive,
        }

    def test_archive_source_builds_and_idempotently_loads_verified_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_archive, settings = self._fixture_settings(root)
            digest = settings["expected_sha256"]
            output = root / "result"

            result = acquire_cmu_arctic(output, **settings)

            self.assertEqual(result.archive.parent, output)
            self.assertTrue(result.extracted.is_relative_to(output))
            self.assertEqual(result.selected_dir, output / "selected")
            self.assertEqual(result.continuous_wav, output / "continuous.wav")
            self.assertEqual(
                [item["id"] for item in result.metadata["selection"]["utterances"]],
                ["utt_b", "utt_a"],
            )
            self.assertEqual(result.metadata["source"]["sha256"], digest)
            self.assertEqual(result.metadata["license"]["identifier"], "LicenseRef-CMU-ARCTIC")
            self.assertTrue((output / CORPUS_METADATA).is_file())

            loaded = acquire_cmu_arctic(output, **settings)
            self.assertEqual(loaded, result)

            with self.assertRaisesRegex(FileExistsError, "different acquisition settings"):
                acquire_cmu_arctic(output, **{**settings, "silence_gap_s": None})

    def test_recovers_local_partial_and_stale_extraction_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_archive, settings = self._fixture_settings(root)
            output = root / "result"
            output.mkdir()
            partial = output / "cmu_us_slt_arctic-0.95-release.tar.bz2.part"
            partial.write_bytes(source_archive.read_bytes()[:31])
            stale = output / ".extracted.extracting"
            stale.mkdir()
            sentinel = stale / "partial-member"
            sentinel.write_text("preserved", encoding="utf-8")

            result = acquire_cmu_arctic(output, **settings)

            self.assertTrue(result.archive.is_file())
            self.assertFalse(partial.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserved")
            self.assertEqual(result.metadata["selection"]["utterance_count"], 2)

    def test_recovers_atomic_phase_outputs_when_metadata_publish_was_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, settings = self._fixture_settings(root)
            output = root / "result"
            first = acquire_cmu_arctic(output, **settings)
            metadata_path = output / CORPUS_METADATA
            metadata_path.unlink()
            stale_metadata = output / ".corpus.json.preparing"
            stale_metadata.write_text("{interrupted", encoding="utf-8")

            recovered = acquire_cmu_arctic(output, **settings)

            self.assertEqual(recovered.metadata, first.metadata)
            self.assertEqual(stale_metadata.read_text(encoding="utf-8"), "{interrupted")
            self.assertTrue(metadata_path.is_file())

    def test_loader_rejects_reordered_selection_and_partial_continuous_tuple(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, settings = self._fixture_settings(root)
            output = root / "result"
            acquire_cmu_arctic(output, **settings)
            metadata_path = output / CORPUS_METADATA
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["selection"]["utterances"].reverse()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "corpus order"):
                load_corpus_result(output)

            metadata["selection"]["utterances"].reverse()
            metadata["outputs"]["continuous_sha256"] = None
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "all present or all null"):
                load_corpus_result(output)

    def test_refuses_unknown_nonempty_output_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            output.mkdir()
            sentinel = output / "keep-me"
            sentinel.write_text("untouched", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty incomplete"):
                acquire_cmu_arctic(output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")


if __name__ == "__main__":
    unittest.main()
