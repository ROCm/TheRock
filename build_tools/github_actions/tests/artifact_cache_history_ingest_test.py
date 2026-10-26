# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Filename parsing and history evidence; ingestion never publishes payloads."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_db import CacheDB
from build_tools.github_actions.artifact_cache.manifest import OutputObject, TrustLevel, ValidationState
from build_tools.github_actions.artifact_cache.history_ingest import (
    ingest_runs, parse_artifact_filename, trust_for_run, validation_for_run,
)
from build_tools.github_actions.tests import _artifact_cache_fixtures as fixtures


class ParseFilenameTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(parse_artifact_filename("blas_lib_gfx94X-dcgpu.tar.zst"),
                         ("blas", "lib", "gfx94X-dcgpu"))

    def test_generic_family_and_xz(self):
        self.assertEqual(parse_artifact_filename("base_run_generic.tar.xz"),
                         ("base", "run", "generic"))

    def test_name_with_underscore(self):
        self.assertEqual(parse_artifact_filename("rocm_smi_lib_generic.tar.zst"),
                         ("rocm_smi", "lib", "generic"))

    def test_rejects_non_archive(self):
        self.assertIsNone(parse_artifact_filename("blas_lib_gfx94X.sha256sum"))
        self.assertIsNone(parse_artifact_filename("notes.txt"))

    def test_rejects_unknown_component(self):
        self.assertIsNone(parse_artifact_filename("blas_zzz_generic.tar.zst"))

    def test_rejects_archive_paths(self):
        for name in ("../blas_lib_generic.tar.zst", "folder/blas_lib_generic.tar.zst",
                     "folder\\blas_lib_generic.tar.zst"):
            with self.subTest(name=name):
                self.assertIsNone(parse_artifact_filename(name))


class ClassifyRunTest(unittest.TestCase):
    def test_trust_push_main_is_trusted(self):
        self.assertEqual(trust_for_run(fixtures.history_run(1)), TrustLevel.TRUSTED)

    def test_trust_schedule_main_is_trusted(self):
        self.assertEqual(trust_for_run(fixtures.history_run(1, event="schedule")), TrustLevel.TRUSTED)

    def test_trust_release_branch_is_untrusted(self):
        for event in ("push", "schedule"):
            with self.subTest(event=event):
                run = fixtures.history_run(1, branch="release/therock-10.1", event=event)
                self.assertEqual(trust_for_run(run), TrustLevel.UNTRUSTED)

    def test_trust_pull_request_is_untrusted(self):
        self.assertEqual(trust_for_run(fixtures.history_run(1, event="pull_request")), TrustLevel.UNTRUSTED)

    def test_trust_push_feature_branch_is_untrusted(self):
        self.assertEqual(trust_for_run(fixtures.history_run(1, branch="feature")), TrustLevel.UNTRUSTED)

    def test_manual_main_run_is_untrusted(self):
        self.assertEqual(trust_for_run(fixtures.history_run(1, event="workflow_dispatch")), TrustLevel.UNTRUSTED)

    def test_validation_from_conclusion(self):
        self.assertEqual(validation_for_run({"conclusion": "success"}), ValidationState.BUILD_PASSED)
        self.assertEqual(validation_for_run({"conclusion": "failure"}), ValidationState.UNVALIDATED)
        self.assertEqual(validation_for_run({"conclusion": "success", "status": "in_progress"}),
                         ValidationState.UNVALIDATED)


class IngestRunsTest(fixtures.RelationalFixture):
    def test_components_remain_evidence_without_inventing_a_complete_bundle(self):
        objects = [OutputObject("blas_lib_gfx94X-dcgpu.tar.zst", "s3://b/1"),
                   OutputObject("blas_dev_gfx94X-dcgpu.tar.zst", "s3://b/2"),
                   OutputObject("base_run_generic.tar.zst", "s3://b/3")]
        with CacheDB(self.database) as db:
            stats = ingest_runs(db=db, runs=[fixtures.history_run(100)], artifact_lister=lambda run: objects)
            self.assertEqual(stats.units_indexed, 3)
            self.assertEqual(stats.objects_indexed, 3)
            self.assertEqual(stats.runs_indexed, 1)
            self.assertEqual({u["artifact_identity"] for u in db.history()[0]["units"]},
                             {"blas_lib_gfx94X-dcgpu", "blas_dev_gfx94X-dcgpu", "base_run_generic"})
            self.assertEqual(db.stats()["artifact_keys"], 0)
            self.assertEqual(db.count(), 0)
            self.assertEqual(stats.to_dict()["cache_entries_published"], 0)

    def test_require_trusted_skips_pr_runs_before_artifact_listing(self):
        def unexpected_listing(run):
            self.fail("untrusted run must be rejected before artifact listing")
        with CacheDB(self.database) as db:
            stats = ingest_runs(db=db, runs=[fixtures.history_run(200, event="pull_request")],
                                artifact_lister=unexpected_listing, require_trusted=True)
            self.assertEqual(stats.runs_skipped, 1)
            self.assertEqual(db.history(), [])

    def test_failed_run_is_unvalidated_evidence_not_a_ready_entry(self):
        with CacheDB(self.database) as db:
            stats = ingest_runs(db=db, runs=[fixtures.history_run(300, conclusion="failure")],
                artifact_lister=lambda run: [OutputObject("blas_lib_generic.tar.zst", "s3://b/1")])
            self.assertEqual(stats.runs_indexed, 1)
            self.assertEqual(db.history()[0]["validation"], "unvalidated")
            self.assertEqual(db.stats()["cache_entries"], 0)
            self.assertEqual(db.stats()["artifact_files"], 0)


if __name__ == "__main__":
    unittest.main()
