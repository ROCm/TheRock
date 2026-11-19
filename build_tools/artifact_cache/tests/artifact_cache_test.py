#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Artifact cache tests; no build, network, or AWS credentials required.

CI runs are simulated with the local directory backend, which reads the same
layout as S3: {staging}/{run_id}-{platform}/{artifact files}.
"""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent.parent))

import artifact_cache
from artifact_cache import (
    CacheError,
    CacheIndex,
    LocalDirectoryBackend,
    S3Backend,
    lookup_artifacts_by_fingerprint,
    main,
)
from _therock_utils.workflow_outputs import WorkflowOutputRoot

LLVM = [f"amd-llvm_{c}_generic" for c in ("dbg", "dev", "doc", "lib", "run")]
BASE = [f"base_{c}_generic" for c in ("dev", "lib")]
RUN_ID = "43936062480"


def sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def make_run(
    staging: Path,
    run_id: str = RUN_ID,
    platform: str = "linux",
    fingerprints: dict[str, str | None] | None = None,
    suffix: str = ".tar.zst",
) -> Path:
    """Create a fake CI run. A None fingerprint means no .fprint file."""
    if fingerprints is None:
        fingerprints = {stem: sha("fingerprint-" + stem[:3]) for stem in LLVM}
    run_dir = staging / f"{run_id}-{platform}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for stem, fingerprint in fingerprints.items():
        archive = stem + suffix
        (run_dir / archive).write_bytes(b"archive")
        (run_dir / (archive + ".sha256sum")).write_text(f"{sha(archive)}  {archive}\n")
        if fingerprint is not None:
            (run_dir / (stem + ".fprint")).write_text(fingerprint + "\n")
    return run_dir


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.staging = self.root / "staging"
        self.index_path = self.root / "index.json"
        self.index_path.write_text('{"schema_version": 1, "entries": []}\n')
        env = mock.patch.dict(
            os.environ, {"THEROCK_LOCAL_STAGING_DIR": str(self.staging)}
        )
        env.start()
        self.addCleanup(env.stop)

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def record(self, *extra: str, run_id: str = RUN_ID, platform: str = "linux"):
        return self.run_cli(
            "record", "--index", self.index_path, "--run-id", run_id,
            "--platform", platform, *extra,
        )  # fmt: skip

    def lookup(self, fingerprints_dir: Path, *extra: str, platform: str = "linux"):
        output = self.root / "plan.json"
        code, out, err = self.run_cli(
            "lookup", "--index", self.index_path, "--fingerprints-dir", fingerprints_dir,
            "--platform", platform, "--output", output, *extra,
        )  # fmt: skip
        plan = json.loads(output.read_text()) if output.exists() else None
        return code, plan

    def index(self) -> dict:
        return json.loads(self.index_path.read_text())

    def local_fingerprints(self, fingerprints: dict[str, str | None]) -> Path:
        directory = self.root / f"local-{len(list(self.root.glob('local-*')))}"
        directory.mkdir()
        for stem, fingerprint in fingerprints.items():
            if fingerprint is not None:
                (directory / (stem + ".fprint")).write_text(fingerprint + "\n")
        return directory

    def statuses(self, plan: dict) -> dict[str, tuple[str, str]]:
        return {
            f"{a['name']}_{a['component']}_{a['target']}": (a["status"], a["reason"])
            for a in plan["artifacts"]
        }


class RecordTest(CacheTestCase):
    def test_default_records_only_amd_llvm(self):
        make_run(
            self.staging,
            fingerprints={
                **dict.fromkeys(LLVM, sha("a")),
                **dict.fromkeys(BASE, sha("b")),
            },
        )
        code, out, _ = self.record()
        self.assertEqual(code, 0)
        self.assertIn("5 added", out)
        entries = self.index()["entries"]
        self.assertEqual({e["name"] for e in entries}, {"amd-llvm"})
        self.assertEqual(
            {e["component"] for e in entries}, {"dbg", "dev", "doc", "lib", "run"}
        )
        entry = next(e for e in entries if e["component"] == "lib")
        archive = "amd-llvm_lib_generic.tar.zst"
        self.assertEqual(entry["platform"], "linux")
        self.assertEqual(entry["target"], "generic")
        self.assertEqual(entry["fingerprint"], sha("a"))
        self.assertEqual(
            entry["provenance"], {"repository": "ROCm/TheRock", "run_id": RUN_ID}
        )
        self.assertEqual(entry["outputs"][0]["key"], archive)
        self.assertEqual(entry["outputs"][0]["digest"], sha(archive))
        self.assertTrue(entry["outputs"][0]["uri"].startswith("file://"))
        self.assertTrue(entry["outputs"][0]["uri"].endswith("/" + archive))
        self.assertEqual(self.index()["status"], "prototype")

    def test_include_and_exclude_patterns_select_other_artifacts(self):
        make_run(
            self.staging,
            fingerprints={
                **dict.fromkeys(LLVM, sha("a")),
                **dict.fromkeys(BASE, sha("b")),
            },
        )
        code, _, _ = self.record("base")
        self.assertEqual(code, 0)
        self.assertEqual({e["name"] for e in self.index()["entries"]}, {"base"})
        code, _, _ = self.record(".", "--exclude", "_dbg_", "_doc_")
        self.assertEqual(code, 0)
        recorded = {(e["name"], e["component"]) for e in self.index()["entries"]}
        self.assertIn(("amd-llvm", "lib"), recorded)
        self.assertIn(("base", "dev"), recorded)
        self.assertNotIn(("amd-llvm", "dbg"), recorded)
        self.assertNotIn(("amd-llvm", "doc"), recorded)

    def test_repeating_a_record_leaves_the_index_unchanged(self):
        make_run(self.staging)
        self.record()
        before = self.index_path.read_bytes()
        code, out, _ = self.record(run_id=RUN_ID)
        self.assertEqual(code, 0)
        self.assertIn("5 already indexed; index unchanged", out)
        self.assertEqual(self.index_path.read_bytes(), before)

    def test_matching_fingerprints_keep_the_original_run(self):
        make_run(self.staging)
        self.record()
        make_run(self.staging, run_id="43936062999")
        code, out, _ = self.record(run_id="43936062999")
        self.assertEqual(code, 0)
        self.assertEqual(
            {e["provenance"]["run_id"] for e in self.index()["entries"]}, {RUN_ID}
        )

    def test_changed_fingerprint_replaces_only_that_artifact_and_platform(self):
        make_run(
            self.staging,
            fingerprints={
                **dict.fromkeys(LLVM, sha("a")),
                **dict.fromkeys(BASE, sha("b")),
            },
        )
        make_run(
            self.staging, platform="windows", fingerprints=dict.fromkeys(LLVM, sha("w"))
        )
        self.record(".")
        self.record(platform="windows")
        make_run(
            self.staging,
            run_id="43936062999",
            fingerprints=dict.fromkeys(LLVM, sha("new")),
        )
        code, out, _ = self.record(run_id="43936062999")
        self.assertEqual(code, 0)
        self.assertIn("5 replaced", out)
        by_key = {
            (e["platform"], e["name"], e["component"]): e
            for e in self.index()["entries"]
        }
        self.assertEqual(len(by_key), 12)
        for component in ("dbg", "dev", "doc", "lib", "run"):
            replaced = by_key["linux", "amd-llvm", component]
            self.assertEqual(replaced["fingerprint"], sha("new"))
            self.assertEqual(replaced["provenance"]["run_id"], "43936062999")
            self.assertEqual(
                by_key["windows", "amd-llvm", component]["fingerprint"], sha("w")
            )
        self.assertEqual(by_key["linux", "base", "lib"]["fingerprint"], sha("b"))
        self.assertEqual(by_key["linux", "base", "lib"]["provenance"]["run_id"], RUN_ID)

    def test_epoch_is_recorded_when_given(self):
        make_run(self.staging)
        self.record("--epoch", "7")
        self.assertEqual({e["epoch"] for e in self.index()["entries"]}, {"7"})

    def test_artifacts_without_fingerprints_are_skipped_and_reported(self):
        make_run(
            self.staging,
            fingerprints={**dict.fromkeys(LLVM, sha("a")), **dict.fromkeys(BASE, None)},
        )
        code, out, _ = self.record(".")
        self.assertEqual(code, 0)
        self.assertIn("Skipped base_dev_generic: no_fingerprint", out)
        self.assertEqual({e["name"] for e in self.index()["entries"]}, {"amd-llvm"})

    def test_fingerprint_without_a_checksum_is_an_error(self):
        run_dir = make_run(self.staging)
        (run_dir / "amd-llvm_dbg_generic.tar.zst.sha256sum").unlink()
        before = self.index_path.read_bytes()
        code, _, err = self.record()
        self.assertEqual(code, 2)
        self.assertIn("amd-llvm_dbg_generic has a fingerprint but no checksum", err)
        self.assertEqual(self.index_path.read_bytes(), before)

    def test_run_without_any_fingerprints_is_an_error(self):
        make_run(self.staging, fingerprints=dict.fromkeys(LLVM, None))
        code, _, err = self.record()
        self.assertEqual(code, 2)
        self.assertIn("No artifacts with fingerprints", err)

    def test_components_must_share_a_fingerprint_and_the_index_is_kept(self):
        fingerprints = dict.fromkeys(LLVM, sha("a"))
        fingerprints["amd-llvm_lib_generic"] = sha("different")
        make_run(self.staging, fingerprints=fingerprints)
        before = self.index_path.read_bytes()
        code, _, err = self.record()
        self.assertEqual(code, 2)
        self.assertIn("record all components together", err)
        self.assertEqual(self.index_path.read_bytes(), before)

    def test_updating_part_of_a_group_is_rejected(self):
        make_run(self.staging)
        self.record()
        make_run(
            self.staging,
            run_id="43936062999",
            fingerprints=dict.fromkeys(LLVM, sha("new")),
        )
        before = self.index_path.read_bytes()
        code, _, err = self.record("amd-llvm_lib", run_id="43936062999")
        self.assertEqual(code, 2)
        self.assertIn("record all components together", err)
        self.assertEqual(self.index_path.read_bytes(), before)

    def test_invalid_fingerprint_is_an_error(self):
        make_run(
            self.staging,
            fingerprints={
                **dict.fromkeys(LLVM, sha("a")),
                "amd-llvm_lib_generic": "not-a-hash",
            },
        )
        code, _, err = self.record()
        self.assertEqual(code, 2)
        self.assertIn("lowercase SHA-256", err)

    def read_checksum(self, text: str) -> str:
        archive = "amd-llvm_lib_generic.tar.zst"
        path = self.root / (archive + ".sha256sum")
        path.write_text(text)
        return artifact_cache.read_checksum(path, archive)

    def test_checksum_file_may_hold_only_the_digest(self):
        self.assertEqual(self.read_checksum(sha("x") + "\n"), sha("x"))

    def test_checksum_file_may_name_the_archive(self):
        archive = "amd-llvm_lib_generic.tar.zst"
        self.assertEqual(self.read_checksum(f"{sha('x')} *{archive}\n"), sha("x"))
        self.assertEqual(self.read_checksum(f"{sha('x')}  {archive}\n"), sha("x"))

    def test_checksum_file_for_another_archive_is_rejected(self):
        with self.assertRaisesRegex(CacheError, "Checksum filename mismatch"):
            self.read_checksum(f"{sha('x')}  other.tar.zst\n")

    def test_malformed_checksum_files_are_rejected(self):
        with self.assertRaisesRegex(CacheError, "Invalid checksum file"):
            self.read_checksum("a b c\n")
        with self.assertRaisesRegex(CacheError, "Empty checksum file"):
            self.read_checksum("\n")
        with self.assertRaisesRegex(CacheError, "lowercase SHA-256"):
            self.read_checksum("XYZ\n")

    def test_xz_archives_are_supported(self):
        make_run(self.staging, suffix=".tar.xz")
        self.record()
        self.assertEqual(
            {e["outputs"][0]["key"] for e in self.index()["entries"]},
            {stem + ".tar.xz" for stem in LLVM},
        )

    def test_a_missing_index_is_created(self):
        make_run(self.staging)
        self.index_path.unlink()
        code, _, _ = self.record()
        self.assertEqual(code, 0)
        self.assertEqual(len(self.index()["entries"]), 5)

    def test_run_id_must_be_numeric(self):
        code, _, err = self.record(run_id="abc")
        self.assertEqual(code, 2)
        self.assertIn("run-id must be numeric", err)

    def test_repository_comes_from_the_flag_then_the_environment(self):
        make_run(self.staging)
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "env-owner/env-repo"}):
            self.record()
            self.assertEqual(
                {e["provenance"]["repository"] for e in self.index()["entries"]},
                {"env-owner/env-repo"},
            )
            self.index_path.write_text('{"schema_version": 1, "entries": []}\n')
            self.record("--run-github-repo", "flag-owner/flag-repo")
            self.assertEqual(
                {e["provenance"]["repository"] for e in self.index()["entries"]},
                {"flag-owner/flag-repo"},
            )

    def test_invalid_repository_is_an_error(self):
        make_run(self.staging)
        code, _, err = self.record("--run-github-repo", "not-a-repository")
        self.assertEqual(code, 2)
        self.assertIn("owner/repository", err)

    def test_local_staging_dir_selects_the_local_backend(self):
        backend = artifact_cache.create_source_backend(RUN_ID, "linux", "o/r")
        self.assertIsInstance(backend, LocalDirectoryBackend)

    def test_s3_backend_looks_up_the_workflow_run(self):
        root = WorkflowOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id=RUN_ID,
            platform="linux",
        )
        with mock.patch.dict(os.environ):
            del os.environ["THEROCK_LOCAL_STAGING_DIR"]
            with mock.patch.object(
                WorkflowOutputRoot, "from_workflow_run", return_value=root
            ) as from_workflow_run:
                backend = artifact_cache.create_source_backend(RUN_ID, "linux", "o/r")
        self.assertIsInstance(backend, S3Backend)
        from_workflow_run.assert_called_once_with(
            run_id=RUN_ID,
            platform="linux",
            github_repository="o/r",
            lookup_workflow_run=True,
        )

    def test_s3_locations_are_public_https_urls(self):
        root = WorkflowOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id=RUN_ID,
            platform="linux",
        )
        self.assertEqual(
            artifact_cache.archive_uri(S3Backend(root), "amd-llvm_lib_generic.tar.zst"),
            "https://therock-ci-artifacts.s3.amazonaws.com/43936062480-linux/amd-llvm_lib_generic.tar.zst",
        )


class LookupTest(CacheTestCase):
    def setUp(self):
        super().setUp()
        make_run(
            self.staging,
            fingerprints={
                **dict.fromkeys(LLVM, sha("a")),
                **dict.fromkeys(BASE, sha("b")),
            },
        )
        self.record(".")

    def test_hit_requires_every_component(self):
        code, plan = self.lookup(self.local_fingerprints(dict.fromkeys(LLVM, sha("a"))))
        self.assertEqual(code, 0)
        self.assertEqual((plan["status"], plan["action"]), ("hit", "populate"))
        self.assertEqual(plan["plan_version"], 1)
        self.assertEqual(len(plan["artifacts"]), 5)
        for artifact in plan["artifacts"]:
            self.assertEqual(artifact["status"], "hit")
            self.assertEqual(artifact["provenance"]["run_id"], RUN_ID)
            self.assertEqual(len(artifact["outputs"]), 1)

    def test_empty_fingerprints_directory_is_a_miss(self):
        code, plan = self.lookup(self.local_fingerprints({}))
        self.assertEqual(code, 0)
        self.assertEqual((plan["status"], plan["action"]), ("miss", "build"))
        self.assertEqual(plan["reason"], "no_local_fingerprints")
        self.assertEqual(plan["artifacts"], [])

    def test_empty_fingerprints_directory_is_named_in_the_output(self):
        directory = self.local_fingerprints({})
        output = self.root / "plan.json"
        _, out, _ = self.run_cli(
            "lookup", "--index", self.index_path, "--fingerprints-dir", directory,
            "--platform", "linux", "--output", output,
        )  # fmt: skip
        self.assertIn(f"No selected fingerprints in {directory}", out)

    def test_indexed_components_missing_locally_are_misses(self):
        fingerprints = dict.fromkeys(LLVM[:3], sha("a"))
        _, plan = self.lookup(self.local_fingerprints(fingerprints))
        statuses = self.statuses(plan)
        self.assertEqual(len(statuses), 5)
        self.assertEqual(
            statuses["amd-llvm_run_generic"], ("miss", "missing_expected_fingerprint")
        )
        self.assertEqual(statuses["amd-llvm_dbg_generic"], ("miss", "incomplete_group"))

    def test_artifacts_the_build_does_not_produce_are_not_checked(self):
        directory = self.local_fingerprints(dict.fromkeys(LLVM, sha("a")))
        _, plan = self.lookup(directory, ".")  # The index also holds base.
        self.assertEqual({a["name"] for a in plan["artifacts"]}, {"amd-llvm"})
        self.assertEqual((plan["status"], plan["action"]), ("hit", "populate"))

    def test_incomplete_group_is_still_detected_with_a_broad_pattern(self):
        fingerprints = dict.fromkeys(LLVM, sha("a"))
        fingerprints["amd-llvm_dbg_generic"] = None
        _, plan = self.lookup(self.local_fingerprints(fingerprints), ".")
        self.assertEqual({a["name"] for a in plan["artifacts"]}, {"amd-llvm"})
        self.assertEqual(plan["status"], "miss")
        self.assertEqual(
            self.statuses(plan)["amd-llvm_dbg_generic"],
            ("miss", "missing_expected_fingerprint"),
        )

    def test_a_group_found_only_through_excluded_files_is_not_checked(self):
        directory = self.local_fingerprints({"amd-llvm_dbg_generic": sha("a")})
        _, plan = self.lookup(directory, "--exclude", "_dbg_")
        self.assertEqual(plan["artifacts"], [])
        self.assertEqual(plan["reason"], "no_artifacts_selected")

    def test_changed_fingerprints_miss(self):
        _, plan = self.lookup(
            self.local_fingerprints(dict.fromkeys(LLVM, sha("other")))
        )
        self.assertEqual(plan["status"], "miss")
        self.assertEqual(
            set(self.statuses(plan).values()), {("miss", "no_matching_index_entry")}
        )
        self.assertNotIn("outputs", plan["artifacts"][0])

    def test_incomplete_group_never_allows_partial_reuse(self):
        fingerprints = dict.fromkeys(LLVM, sha("a"))
        fingerprints["amd-llvm_dbg_generic"] = None
        _, plan = self.lookup(self.local_fingerprints(fingerprints))
        statuses = self.statuses(plan)
        self.assertEqual(plan["status"], "miss")
        self.assertEqual(
            statuses["amd-llvm_dbg_generic"], ("miss", "missing_expected_fingerprint")
        )
        for stem in LLVM[1:]:
            self.assertEqual(statuses[stem], ("miss", "incomplete_group"))
            self.assertNotIn(
                "outputs",
                next(
                    a for a in plan["artifacts"] if a["component"] == stem.split("_")[1]
                ),
            )

    def test_extra_local_component_that_is_not_indexed_misses(self):
        fingerprints = {
            **dict.fromkeys(LLVM, sha("a")),
            "amd-llvm_sys_generic": sha("a"),
        }
        _, plan = self.lookup(self.local_fingerprints(fingerprints))
        self.assertEqual(plan["status"], "miss")
        self.assertEqual(
            self.statuses(plan)["amd-llvm_lib_generic"], ("miss", "incomplete_group")
        )

    def test_inconsistent_fingerprints_in_one_group_miss(self):
        fingerprints = dict.fromkeys(LLVM, sha("a"))
        fingerprints["amd-llvm_run_generic"] = sha("stale")
        _, plan = self.lookup(self.local_fingerprints(fingerprints))
        statuses = self.statuses(plan)
        self.assertEqual(plan["status"], "miss")
        self.assertEqual(
            statuses["amd-llvm_lib_generic"],
            ("miss", "inconsistent_group_fingerprints"),
        )
        self.assertEqual(
            statuses["amd-llvm_run_generic"], ("miss", "no_matching_index_entry")
        )

    def test_invalid_fingerprint_file_is_a_miss(self):
        directory = self.local_fingerprints(dict.fromkeys(LLVM, sha("a")))
        (directory / "amd-llvm_lib_generic.fprint").write_text("garbage\n")
        _, plan = self.lookup(directory)
        self.assertEqual(
            self.statuses(plan)["amd-llvm_lib_generic"],
            ("miss", "invalid_expected_fingerprint"),
        )
        self.assertEqual(plan["status"], "miss")

    def test_other_artifacts_need_include_patterns(self):
        directory = self.local_fingerprints(
            {**dict.fromkeys(LLVM, sha("a")), **dict.fromkeys(BASE, sha("b"))}
        )
        _, plan = self.lookup(directory)
        self.assertEqual({a["name"] for a in plan["artifacts"]}, {"amd-llvm"})
        self.assertEqual(plan["includes"], ["amd-llvm"])
        _, plan = self.lookup(directory, "base")
        self.assertEqual({a["name"] for a in plan["artifacts"]}, {"base"})
        self.assertEqual(plan["status"], "hit")
        _, plan = self.lookup(directory, ".")
        self.assertEqual(len(plan["artifacts"]), 7)
        self.assertEqual(plan["status"], "hit")

    def test_exclude_removes_artifacts_from_the_plan(self):
        directory = self.local_fingerprints(
            {s: sha("a") for s in LLVM if "_dbg_" not in s}
        )
        _, plan = self.lookup(directory)
        self.assertEqual(plan["status"], "miss")  # The index still has dbg.
        _, plan = self.lookup(directory, "--exclude", "_dbg_")
        self.assertEqual(plan["status"], "hit")
        self.assertEqual(plan["excludes"], ["_dbg_"])
        self.assertEqual(len(plan["artifacts"]), 4)

    def test_groups_are_independent(self):
        directory = self.local_fingerprints(
            {**dict.fromkeys(LLVM, sha("a")), **dict.fromkeys(BASE, sha("changed"))}
        )
        _, plan = self.lookup(directory, ".")
        statuses = self.statuses(plan)
        self.assertEqual(plan["status"], "miss")
        self.assertEqual(
            statuses["amd-llvm_lib_generic"], ("hit", "matching_index_entry")
        )
        self.assertEqual(
            statuses["base_lib_generic"], ("miss", "no_matching_index_entry")
        )

    def test_entries_for_other_platforms_do_not_match(self):
        make_run(
            self.staging, platform="windows", fingerprints=dict.fromkeys(LLVM, sha("w"))
        )
        self.record(platform="windows")
        directory = self.local_fingerprints(dict.fromkeys(LLVM, sha("w")))
        _, plan = self.lookup(directory, platform="linux")
        self.assertEqual(plan["status"], "miss")
        _, plan = self.lookup(directory, platform="windows")
        self.assertEqual(plan["status"], "hit")

    def test_no_selected_artifacts_is_a_miss(self):
        directory = self.local_fingerprints(dict.fromkeys(LLVM, sha("a")))
        _, plan = self.lookup(directory, "nothing-matches-this")
        self.assertEqual(
            (plan["status"], plan["reason"]), ("miss", "no_artifacts_selected")
        )
        self.assertEqual(plan["artifacts"], [])

    def test_epoch_is_checked_only_when_both_sides_have_one(self):
        directory = self.local_fingerprints(dict.fromkeys(LLVM, sha("a")))
        _, plan = self.lookup(directory, "--epoch", "3")  # Index has no epoch.
        self.assertEqual(plan["status"], "hit")
        make_run(
            self.staging,
            run_id="43936062999",
            fingerprints=dict.fromkeys(LLVM, sha("e")),
        )
        self.record("--epoch", "2", run_id="43936062999")
        directory = self.local_fingerprints(dict.fromkeys(LLVM, sha("e")))
        _, plan = self.lookup(directory, "--epoch", "2")
        self.assertEqual(plan["status"], "hit")
        _, plan = self.lookup(directory, "--epoch", "3")
        self.assertEqual(
            set(self.statuses(plan).values()), {("miss", "epoch_mismatch")}
        )
        _, plan = self.lookup(directory)
        self.assertEqual(plan["status"], "hit")

    def test_unsuccessful_lookup_removes_an_old_plan(self):
        directory = self.local_fingerprints(dict.fromkeys(LLVM, sha("a")))
        _, plan = self.lookup(directory)
        self.assertEqual(plan["status"], "hit")
        self.index_path.write_text("not json")
        code, plan = self.lookup(directory)
        self.assertEqual(code, 2)
        self.assertIsNone(plan)

    def test_plan_may_not_overwrite_the_index(self):
        code, _, err = self.run_cli(
            "lookup", "--index", self.index_path, "--fingerprints-dir", self.root,
            "--platform", "linux", "--output", self.index_path,
        )  # fmt: skip
        self.assertEqual(code, 2)
        self.assertIn("must not overwrite the index", err)

    def test_lookup_function_leaves_the_index_unchanged(self):
        index = artifact_cache.load_index(self.index_path)
        before = json.dumps(artifact_cache.to_json(index))
        lookup_artifacts_by_fingerprint(
            index, self.local_fingerprints({}), "linux", ["."], []
        )
        self.assertEqual(json.dumps(artifact_cache.to_json(index)), before)


class IndexTest(CacheTestCase):
    def entry(self, **overrides) -> dict:
        archive = "amd-llvm_lib_generic.tar.zst"
        entry = {
            "platform": "linux",
            "name": "amd-llvm",
            "component": "lib",
            "target": "generic",
            "fingerprint": sha("a"),
            "provenance": {"repository": "ROCm/TheRock", "run_id": RUN_ID},
            "outputs": [
                {
                    "key": archive,
                    "uri": "https://example.com/" + RUN_ID + "-linux/" + archive,
                    "digest": sha("d"),
                }
            ],
        }
        entry.update(overrides)
        return entry

    def load(self, *entries: dict, schema_version: int = 1) -> CacheIndex:
        self.index_path.write_text(
            json.dumps({"schema_version": schema_version, "entries": list(entries)})
        )
        return artifact_cache.load_index(self.index_path)

    def test_valid_entries_load(self):
        index = self.load(self.entry(), self.entry(component="dev", outputs=[{
            "key": "amd-llvm_dev_generic.tar.xz",
            "uri": "file:///cache/amd-llvm_dev_generic.tar.xz",
            "digest": sha("d"),
        }]))  # fmt: skip
        self.assertEqual(len(index.entries), 2)
        self.assertEqual(index.status, "prototype")

    def test_invalid_entries_are_rejected(self):
        archive = "amd-llvm_lib_generic.tar.zst"
        output = {
            "key": archive,
            "uri": "https://example.com/" + archive,
            "digest": sha("d"),
        }

        def with_output(**changes: str) -> list[dict]:
            return [self.entry(outputs=[{**output, **changes}])]

        missing_provenance = self.entry()
        del missing_provenance["provenance"]
        cases = {
            "fingerprint": [self.entry(fingerprint="ABC")],
            "digest": with_output(digest="abc"),
            "scheme": with_output(uri="ftp://example.com/" + archive),
            "credentials": with_output(uri="https://user:pw@example.com/" + archive),
            "query": with_output(uri="https://example.com/" + archive + "?x=1"),
            "relative file": with_output(uri="file://relative/" + archive),
            "uri/key mismatch": with_output(uri="https://example.com/other.tar.zst"),
            "key/artifact mismatch": with_output(key="base_lib_generic.tar.zst"),
            "no outputs": [self.entry(outputs=[])],
            "underscore in name": [self.entry(name="amd_llvm")],
            "missing field": [missing_provenance],
            "duplicate": [self.entry(), self.entry()],
        }
        for label, entries in cases.items():
            with self.subTest(label):
                with self.assertRaises(CacheError):
                    self.load(*entries)

    def test_unsupported_schema_version_is_rejected(self):
        with self.assertRaisesRegex(CacheError, "Unsupported index schema_version"):
            self.load(self.entry(), schema_version=2)

    def test_invalid_index_gives_exit_code_2(self):
        self.index_path.write_text(
            json.dumps(
                {"schema_version": 1, "entries": [self.entry(fingerprint="bad")]}
            )
        )
        code, plan = self.lookup(self.root)
        self.assertEqual(code, 2)
        self.assertIsNone(plan)

    def test_json_output_omits_unset_fields_and_leaves_no_temporary_files(self):
        entry = artifact_cache.CacheEntry(
            platform="linux",
            name="amd-llvm",
            component="lib",
            target="generic",
            fingerprint=sha("a"),
            provenance=artifact_cache.Provenance(
                repository="ROCm/TheRock", run_id=RUN_ID
            ),
            outputs=[
                artifact_cache.Output(
                    key="amd-llvm_lib_generic.tar.zst",
                    uri="s3://b/amd-llvm_lib_generic.tar.zst",
                    digest=sha("d"),
                )
            ],
        )
        index = CacheIndex(schema_version=1, status="prototype", entries=[entry])
        path = self.root / "out" / "index.json"
        artifact_cache.write_json(path, index)
        self.assertNotIn("epoch", json.loads(path.read_text())["entries"][0])
        self.assertEqual([p.name for p in path.parent.iterdir()], ["index.json"])

    def test_checked_in_index_is_valid(self):
        index = artifact_cache.load_index(artifact_cache.DEFAULT_INDEX)
        self.assertEqual(index.status, "prototype")
        for platform in ("linux", "windows"):
            with self.subTest(platform):
                entries = [e for e in index.entries if e.platform == platform]
                self.assertEqual(
                    {e.component for e in entries}, {"dbg", "dev", "doc", "lib", "run"}
                )
                self.assertEqual({e.name for e in entries}, {"amd-llvm"})
                self.assertEqual(len({e.fingerprint for e in entries}), 1)


if __name__ == "__main__":
    unittest.main()
