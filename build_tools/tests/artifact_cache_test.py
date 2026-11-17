#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Static LLVM lookup tests; no build, network, or AWS credentials required."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))
import artifact_cache


class StaticCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="therock-static-cache-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.metadata = self.root / "metadata with spaces"
        self.metadata.mkdir()
        self.index_path = self.root / "index.json"
        self.plan_path = self.root / "plan.json"
        self.fingerprint = "a" * 64
        # Deliberately different from the fingerprint: these are archive hashes.
        self.checksums = {}
        for position, artifact in enumerate(artifact_cache.LLVM_ARTIFACTS):
            (self.metadata / f"{artifact}.fprint").write_text(self.fingerprint + "\n")
            checksum = str(position + 1) * 64
            self.checksums[artifact] = checksum
            (self.metadata / f"{artifact}.tar.zst.sha256sum").write_text(
                checksum + "\n"
            )
        self.empty = artifact_cache.CacheIndex(schema_version=1, entries=[])
        self.index = self.record_artifacts_from_ci_run()

    def record_artifacts_from_ci_run(
        self, index: artifact_cache.CacheIndex | None = None, suffix: str = ".tar.zst"
    ) -> artifact_cache.CacheIndex:
        return artifact_cache.record_artifacts_from_ci_run(
            index=self.empty if index is None else index,
            metadata_dir=self.metadata,
            platform="linux",
            build_variant="ci-x86_64-release",
            repository="ROCm/TheRock",
            run_id="12345",
            root="s3://test-bucket/12345-linux",
            archive_suffix=suffix,
        )

    def lookup_artifacts_by_fingerprint(
        self,
        index: artifact_cache.CacheIndex | None = None,
        platform: str = "linux",
        variant: str = "ci-x86_64-release",
    ) -> artifact_cache.LookupPlan:
        return artifact_cache.lookup_artifacts_by_fingerprint(
            index=self.index if index is None else index,
            fingerprints_dir=self.metadata,
            platform=platform,
            build_variant=variant,
        )

    def assert_build(self, plan: artifact_cache.LookupPlan) -> None:
        self.assertEqual(plan.status, "miss")
        self.assertEqual(plan.action, "build")
        self.assertEqual(plan.copy_requests, [])

    def cli(self, *args: str | Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, artifact_cache.__file__, *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def lookup_cli(self) -> subprocess.CompletedProcess[str]:
        return self.cli(
            "lookup",
            "--index",
            self.index_path,
            "--fingerprints-dir",
            self.metadata,
            "--platform",
            "linux",
            "--build-variant",
            "ci-x86_64-release",
            "--output",
            self.plan_path,
        )

    def record_cli(self, run_id: str = "12345") -> subprocess.CompletedProcess[str]:
        return self.cli(
            "record",
            "--index",
            self.index_path,
            "--metadata-dir",
            self.metadata,
            "--platform",
            "linux",
            "--build-variant",
            "ci-x86_64-release",
            "--repository",
            "ROCm/TheRock",
            "--run-id",
            run_id,
            "--source-root",
            f"s3://test-bucket/{run_id}-linux",
        )

    def test_complete_hit_selects_matching_entries_and_preserves_producers(
        self,
    ) -> None:
        index = copy.deepcopy(self.index)
        index.entries[0].producer.run_id = "99999"
        index.entries[0].source.root = "s3://another-bucket/99999-linux"
        expected = {entry.artifact: entry for entry in index.entries}
        historical = copy.deepcopy(self.index.entries)
        for entry in historical:
            entry.fingerprint = "b" * 64
        index.entries = historical + index.entries
        plan = self.lookup_artifacts_by_fingerprint(index)
        self.assertEqual(plan.status, "hit")
        self.assertEqual(plan.action, "populate")
        self.assertEqual(len(plan.copy_requests), len(artifact_cache.LLVM_ARTIFACTS))
        for request in plan.copy_requests:
            artifact = request.artifact
            self.assertEqual(request.fingerprint, self.fingerprint)
            self.assertEqual(request.source.sha256, self.checksums[artifact])
            self.assertEqual(request.source, expected[artifact].source)
            self.assertEqual(request.producer, expected[artifact].producer)
            self.assertEqual(request.fingerprint_file, artifact + ".fprint")
            self.assertEqual(request.checksum_file, artifact + ".tar.zst.sha256sum")
        self.assertEqual(plan, self.lookup_artifacts_by_fingerprint(index))

    def test_required_components_match_the_actual_cmake_artifact_declaration(
        self,
    ) -> None:
        source = (
            Path(artifact_cache.__file__).resolve().parents[1]
            / "compiler/CMakeLists.txt"
        )
        declaration = re.search(
            r"therock_provide_artifact\(amd-llvm\b(.*?)\)",
            source.read_text(),
            flags=re.S,
        )
        self.assertIsNotNone(declaration)
        components = (
            declaration.group(1)
            .split("COMPONENTS", 1)[1]
            .split("SUBPROJECT_DEPS", 1)[0]
            .split()
        )
        self.assertEqual(set(components), set(artifact_cache.LLVM_COMPONENTS))

    def test_unmatched_requests_miss(self) -> None:
        for case, arguments in (
            ("empty index", {"index": self.empty}),
            ("different platform", {"platform": "windows"}),
            ("different architecture label", {"variant": "ci-aarch64-release"}),
            ("different build type label", {"variant": "ci-x86_64-debug"}),
        ):
            with self.subTest(case=case):
                self.assert_build(self.lookup_artifacts_by_fingerprint(**arguments))
        for artifact in artifact_cache.LLVM_ARTIFACTS:
            (self.metadata / f"{artifact}.fprint").write_text("b" * 64)
        with self.subTest(case="changed fingerprints"):
            self.assert_build(self.lookup_artifacts_by_fingerprint())

    def test_incomplete_bundles_prevent_partial_copy(self) -> None:
        for position, artifact in enumerate(artifact_cache.LLVM_ARTIFACTS):
            with self.subTest(missing="index entry", artifact=artifact):
                index = copy.deepcopy(self.index)
                index.entries.pop(position)
                plan = self.lookup_artifacts_by_fingerprint(index)
                self.assert_build(plan)
                self.assertEqual(
                    sum(row.status == "hit" for row in plan.artifacts),
                    len(artifact_cache.LLVM_ARTIFACTS) - 1,
                )
            with self.subTest(missing="expected fingerprint", artifact=artifact):
                path = self.metadata / f"{artifact}.fprint"
                original = path.read_bytes()
                path.unlink()
                try:
                    plan = self.lookup_artifacts_by_fingerprint()
                    self.assert_build(plan)
                    self.assertIn(
                        "missing_expected_fingerprint",
                        [row.reason for row in plan.artifacts],
                    )
                finally:
                    path.write_bytes(original)

    def test_empty_invalid_and_non_utf8_expected_fingerprints_miss(self) -> None:
        path = self.metadata / "amd-llvm_lib_generic.fprint"
        for value in (b"", b" \n", b"not-a-hash", b"a" * 63, b"\xff"):
            with self.subTest(value=value):
                path.write_bytes(value)
                self.assert_build(self.lookup_artifacts_by_fingerprint())

    def test_inconsistent_component_fingerprints_prevent_mixed_bundle(self) -> None:
        (self.metadata / "amd-llvm_lib_generic.fprint").write_text("b" * 64)
        index = copy.deepcopy(self.index)
        next(
            entry for entry in index.entries if entry.artifact == "amd-llvm_lib_generic"
        ).fingerprint = ("b" * 64)
        plan = self.lookup_artifacts_by_fingerprint(index)
        self.assert_build(plan)
        self.assertTrue(
            all(
                row.reason == "inconsistent_expected_bundle_fingerprints"
                for row in plan.artifacts
            )
        )

    def test_duplicate_matches_are_rejected(self) -> None:
        index = copy.deepcopy(self.index)
        index.entries.append(copy.deepcopy(index.entries[0]))
        artifact_cache.write_json(self.index_path, index)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            artifact_cache.load_index(self.index_path)

    def test_invalid_index_entries_are_rejected(self) -> None:
        cases = [
            ("entry", "fingerprint", "not-a-hash"),
            ("entry", "platform", ""),
            ("entry", "build_variant", None),
            ("entry", "artifact", "hipify_run_generic"),
            ("source", "root", "s3://"),
            ("source", "root", "https://user:password@example.invalid/path"),
            ("source", "root", "https://example.invalid/path?token=value"),
            ("source", "root", "ftp://example.invalid/prefix"),
            ("source", "archive", "amd-llvm_lib_generic.tar.zst"),
            ("source", "archive", "../amd-llvm_dbg_generic.tar.zst"),
            ("source", "archive", "amd-llvm_dbg_generic.zip"),
        ]
        for section, field, value in cases:
            with self.subTest(section=section, field=field, value=value):
                index = artifact_cache.to_json(self.index)
                entry = index["entries"][0]
                target = entry if section == "entry" else entry[section]
                target[field] = value
                self.index_path.write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    artifact_cache.load_index(self.index_path)

    def test_record_accepts_supported_archive_and_checksum_formats(self) -> None:
        for suffix in (".tar.zst", ".tar.xz"):
            for include_filename in (False, True):
                with self.subTest(suffix=suffix, include_filename=include_filename):
                    for artifact in artifact_cache.LLVM_ARTIFACTS:
                        archive = artifact + suffix
                        value = self.checksums[artifact]
                        if include_filename:
                            value += "  " + archive
                        (self.metadata / f"{archive}.sha256sum").write_text(
                            value + "\n"
                        )
                    plan = self.lookup_artifacts_by_fingerprint(
                        self.record_artifacts_from_ci_run(suffix=suffix)
                    )
                    self.assertEqual(plan.status, "hit")
                    for request in plan.copy_requests:
                        self.assertEqual(
                            request.source.archive, request.artifact + suffix
                        )
                        self.assertEqual(
                            request.source.sha256, self.checksums[request.artifact]
                        )

    def test_cli_hit_and_miss_both_succeed_with_explicit_plan_actions(self) -> None:
        metadata_before = {
            path.name: path.read_bytes() for path in self.metadata.iterdir()
        }
        for index, status in ((self.index, "hit"), (self.empty, "miss")):
            with self.subTest(status=status):
                artifact_cache.write_json(self.index_path, index)
                index_before = self.index_path.read_bytes()
                result = self.lookup_cli()
                self.assertEqual(result.returncode, 0, result.stderr)
                plan_json = json.loads(self.plan_path.read_text())
                self.assertEqual(plan_json["status"], status)
                self.assertEqual(
                    plan_json["action"], "populate" if status == "hit" else "build"
                )
                for match in plan_json["artifacts"]:
                    self.assertEqual("producer" in match, status == "hit")
                    self.assertEqual("source" in match, status == "hit")
                self.assertIn(status.upper(), result.stdout)
                self.assertEqual(self.index_path.read_bytes(), index_before)
                self.assertEqual(
                    metadata_before,
                    {path.name: path.read_bytes() for path in self.metadata.iterdir()},
                )

    def test_cli_input_errors_remove_a_previous_hit_plan(self) -> None:
        for invalid in (
            None,
            "broken JSON",
            '{"schema_version":2,"entries":[]}',
            "[]",
            "{}",
            '{"schema_version":1,"entries":{}}',
            '{"schema_version":1,"entries":[null]}',
            '{"schema_version":1,"entries":[{}]}',
        ):
            with self.subTest(invalid=invalid):
                self.plan_path.write_text('{"status":"hit","action":"populate"}')
                if invalid is None:
                    self.index_path.unlink(missing_ok=True)
                else:
                    self.index_path.write_text(invalid)
                result = self.lookup_cli()
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.plan_path.exists())

        # A directory triggers a real read error, even when running as root.
        with self.subTest(invalid="fingerprint is a directory"):
            artifact_cache.write_json(self.index_path, self.index)
            self.plan_path.write_text('{"status":"hit","action":"populate"}')
            path = self.metadata / f"{artifact_cache.LLVM_ARTIFACTS[0]}.fprint"
            path.unlink()
            path.mkdir()
            with self.assertRaises(OSError):
                self.lookup_artifacts_by_fingerprint()
            result = self.lookup_cli()
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(repr(str(path)), result.stderr)
            self.assertFalse(self.plan_path.exists())

    def test_cli_plan_output_cannot_overwrite_the_index(self) -> None:
        artifact_cache.write_json(self.index_path, self.index)
        before = self.index_path.read_bytes()
        self.plan_path = self.index_path
        result = self.lookup_cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.index_path.read_bytes(), before)

    def test_cli_record_failure_preserves_existing_index(self) -> None:
        artifact_cache.write_json(self.index_path, self.empty)
        before = self.index_path.read_bytes()
        cases = (
            ("amd-llvm_run_generic.fprint", None),
            ("amd-llvm_doc_generic.tar.zst.sha256sum", None),
            (
                "amd-llvm_lib_generic.tar.zst.sha256sum",
                "a" * 64 + "  wrong-file.tar.zst\n",
            ),
        )
        for name, contents in cases:
            with self.subTest(file=name, contents=contents):
                path = self.metadata / name
                original = path.read_bytes()
                try:
                    if contents is None:
                        path.unlink()
                    else:
                        path.write_text(contents)
                    result = self.record_cli()
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(self.index_path.read_bytes(), before)
                finally:
                    path.write_bytes(original)

    def test_cli_record_then_lookup(self) -> None:
        artifact_cache.write_json(self.index_path, self.empty)
        result = self.record_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(artifact_cache.load_index(self.index_path), self.index)
        self.assertEqual(self.lookup_cli().returncode, 0)
        self.assertEqual(json.loads(self.plan_path.read_text())["action"], "populate")

    def test_extra_index_fields_are_ignored_by_lookup_and_record(self) -> None:
        legacy_index = artifact_cache.to_json(self.index)
        legacy_index["note"] = "Unused index field"
        for entry in legacy_index["entries"]:
            entry["note"] = "Unused entry field"
            entry["source"]["note"] = "Unused source field"
            entry["producer"].update(
                job_id="67890",
                head_sha="c" * 40,
                url="https://github.com/ROCm/TheRock/actions/runs/12345",
            )
        self.index_path.write_text(json.dumps(legacy_index))
        self.assertEqual(artifact_cache.load_index(self.index_path), self.index)
        original_json = self.index_path.read_bytes()
        self.assertEqual(self.record_cli().returncode, 0)
        self.assertEqual(self.index_path.read_bytes(), original_json)

        result = self.lookup_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(self.plan_path.read_text())
        self.assertEqual(plan["status"], "hit")
        for entry in plan["artifacts"] + plan["copy_requests"]:
            self.assertEqual(set(entry["producer"]), {"repository", "run_id"})
            self.assertNotIn("note", entry)
            self.assertNotIn("note", entry["source"])

        # Adding a new fingerprint writes old and new entries in the minimal form.
        for artifact in artifact_cache.LLVM_ARTIFACTS:
            (self.metadata / f"{artifact}.fprint").write_text("b" * 64)
        result = self.record_cli(run_id="67890")
        self.assertEqual(result.returncode, 0, result.stderr)
        updated = json.loads(self.index_path.read_text())
        self.assertNotIn("note", updated)
        self.assertEqual(
            len(updated["entries"]),
            len(self.index.entries) + len(artifact_cache.LLVM_ARTIFACTS),
        )
        for entry in updated["entries"]:
            self.assertEqual(set(entry["producer"]), {"repository", "run_id"})
            self.assertNotIn("note", entry)
            self.assertNotIn("note", entry["source"])

    def test_cli_record_existing_keys_succeeds_without_rewriting_index(self) -> None:
        artifact_cache.write_json(self.index_path, self.index)
        before = self.index_path.read_bytes()
        modified = self.index_path.stat().st_mtime_ns
        for run_id in ("12345", "67890"):
            with self.subTest(run_id=run_id):
                if run_id == "67890":
                    # A matching key retains the original archive and checksum,
                    # even if another producer's compressed bytes differ.
                    for artifact in artifact_cache.LLVM_ARTIFACTS:
                        (self.metadata / f"{artifact}.tar.zst.sha256sum").write_text(
                            "f" * 64
                        )
                result = self.record_cli(run_id=run_id)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Already indexed:", result.stdout)
                self.assertIn("Index unchanged.", result.stdout)
                self.assertEqual(self.index_path.read_bytes(), before)
                self.assertEqual(self.index_path.stat().st_mtime_ns, modified)

        # Skipping an existing key must not bypass input validation.
        (self.metadata / f"{artifact_cache.LLVM_ARTIFACTS[-1]}.fprint").unlink()
        result = self.record_cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.index_path.read_bytes(), before)

    def test_cli_record_adds_only_missing_keys_and_preserves_existing_producers(
        self,
    ) -> None:
        partial = copy.deepcopy(self.index)
        partial.entries.pop()
        artifact_cache.write_json(self.index_path, partial)
        result = self.record_cli(run_id="67890")
        self.assertEqual(result.returncode, 0, result.stderr)
        updated = artifact_cache.load_index(self.index_path)
        self.assertEqual(len(updated.entries), len(artifact_cache.LLVM_ARTIFACTS))
        self.assertEqual(updated.entries[:-1], partial.entries)
        self.assertEqual(updated.entries[-1].producer.run_id, "67890")
        self.assertIn("Recorded 1 LLVM", result.stdout)
        self.assertIn(f"Already indexed: {len(partial.entries)}", result.stdout)
        self.assertEqual(self.lookup_artifacts_by_fingerprint(updated).status, "hit")

    def test_record_downloads_metadata_and_preserves_producer_location(self) -> None:
        for platform, repository, root in (
            ("linux", "ROCm/TheRock", None),
            ("windows", "ROCm/TheRock", None),
            ("linux", "example/TheRock", "https://example.invalid/custom/prefix"),
            ("linux", "example/TheRock", "https://example.invalid/cache%20runs"),
            ("linux", "ROCm/TheRock", "s3://test-bucket/custom/prefix"),
        ):
            with self.subTest(platform=platform, root=root):
                artifact_cache.write_json(self.index_path, self.empty)
                args = [
                    "record",
                    "--index",
                    str(self.index_path),
                    "--platform",
                    platform,
                    "--build-variant",
                    "ci-x86_64-release",
                    "--run-id",
                    "12345",
                ]
                if root is not None:
                    args.extend(["--source-root", root, "--repository", repository])
                expected_root = (
                    root
                    or f"https://therock-ci-artifacts.s3.amazonaws.com/12345-{platform}"
                )
                download_root = (
                    "https://test-bucket.s3.amazonaws.com/custom/prefix"
                    if root and root.startswith("s3://")
                    else expected_root
                )
                expected_urls = [
                    f"{download_root}/{artifact}{suffix}"
                    for artifact in artifact_cache.LLVM_ARTIFACTS
                    for suffix in (".fprint", ".tar.zst.sha256sum")
                ]

                def response(url: str, *, timeout: int) -> io.BytesIO:
                    self.assertIn(url, expected_urls)
                    return io.BytesIO(
                        (self.metadata / url.rsplit("/", 1)[1]).read_bytes()
                    )

                with patch.object(
                    artifact_cache, "urlopen", side_effect=response
                ) as fetch:
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(artifact_cache.main(args), 0)
                self.assertEqual(
                    [call.args[0] for call in fetch.call_args_list], expected_urls
                )
                index = artifact_cache.load_index(self.index_path)
                self.assertEqual(
                    self.lookup_artifacts_by_fingerprint(
                        index, platform=platform
                    ).status,
                    "hit",
                )
                for entry in index.entries:
                    self.assertEqual(entry.source.root, expected_root)
                    self.assertEqual(
                        entry.source.sha256, self.checksums[entry.artifact]
                    )
                    self.assertEqual(
                        entry.producer,
                        artifact_cache.Producer(repository=repository, run_id="12345"),
                    )

    def test_failed_download_or_invalid_metadata_preserves_index(self) -> None:
        for failure in (
            "missing sidecar",
            "timeout",
            "invalid checksum",
            "inconsistent fingerprint",
        ):
            with self.subTest(failure=failure):
                artifact_cache.write_json(self.index_path, self.empty)
                before = self.index_path.read_bytes()

                def response(url: str, *, timeout: int) -> io.BytesIO:
                    name = url.rsplit("/", 1)[1]
                    if name == artifact_cache.LLVM_ARTIFACTS[-1] + ".tar.zst.sha256sum":
                        if failure == "missing sidecar":
                            raise HTTPError(url, 404, "Not Found", None, None)
                        if failure == "timeout":
                            raise TimeoutError("Download timed out")
                        if failure == "invalid checksum":
                            return io.BytesIO(b"not a checksum")
                    if (
                        failure == "inconsistent fingerprint"
                        and name == artifact_cache.LLVM_ARTIFACTS[-1] + ".fprint"
                    ):
                        return io.BytesIO(b"b" * 64)
                    return io.BytesIO((self.metadata / name).read_bytes())

                with patch.object(artifact_cache, "urlopen", side_effect=response):
                    with redirect_stdout(io.StringIO()), redirect_stderr(
                        io.StringIO()
                    ) as errors:
                        result = artifact_cache.main(
                            [
                                "record",
                                "--index",
                                str(self.index_path),
                                "--platform",
                                "linux",
                                "--build-variant",
                                "release",
                                "--run-id",
                                "12345",
                            ]
                        )
                self.assertEqual(result, 2)
                self.assertIn("ERROR:", errors.getvalue())
                self.assertEqual(self.index_path.read_bytes(), before)

    def test_record_rejects_ambiguous_or_invalid_download_locations(self) -> None:
        artifact_cache.write_json(self.index_path, self.empty)
        before = self.index_path.read_bytes()
        for overrides in (
            ["--repository", "example/TheRock"],
            ["--platform", "unknown"],
            ["--run-id", "../invalid"],
            ["--source-root", "https://example.invalid/path?query=value"],
        ):
            with self.subTest(overrides=overrides):
                with patch.object(artifact_cache, "urlopen") as fetch:
                    with redirect_stderr(io.StringIO()):
                        result = artifact_cache.main(
                            [
                                "record",
                                "--index",
                                str(self.index_path),
                                "--platform",
                                "linux",
                                "--build-variant",
                                "release",
                                "--run-id",
                                "12345",
                                *overrides,
                            ]
                        )
                self.assertEqual(result, 2)
                fetch.assert_not_called()
                self.assertEqual(self.index_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
