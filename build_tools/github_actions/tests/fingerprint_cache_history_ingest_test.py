# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for fingerprint_cache.history_ingest (filename parsing + ingest)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprint_cache.cache_db import CacheDB
from fingerprint_cache.manifest import OutputObject, TrustLevel, ValidationState
from fingerprint_cache.history_ingest import (
    ingest_runs,
    parse_artifact_filename,
    trust_for_run,
    validation_for_run,
)


class ParseFilenameTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(
            parse_artifact_filename("blas_lib_gfx94X-dcgpu.tar.zst"),
            ("blas", "lib", "gfx94X-dcgpu"),
        )

    def test_generic_family_and_xz(self):
        self.assertEqual(
            parse_artifact_filename("base_run_generic.tar.xz"),
            ("base", "run", "generic"),
        )

    def test_name_with_underscore(self):
        self.assertEqual(
            parse_artifact_filename("rocm_smi_lib_generic.tar.zst"),
            ("rocm_smi", "lib", "generic"),
        )

    def test_rejects_non_archive(self):
        self.assertIsNone(parse_artifact_filename("blas_lib_gfx94X.sha256sum"))
        self.assertIsNone(parse_artifact_filename("notes.txt"))

    def test_rejects_unknown_component(self):
        # "zzz" is not a known component token.
        self.assertIsNone(parse_artifact_filename("blas_zzz.tar.zst"))


class ClassifyRunTest(unittest.TestCase):
    def test_trust_push_main_is_trusted(self):
        run = {"event": "push", "head_branch": "main"}
        self.assertEqual(trust_for_run(run), TrustLevel.TRUSTED)

    def test_trust_release_branch_is_trusted(self):
        run = {"event": "push", "head_branch": "release/therock-10.1"}
        self.assertEqual(trust_for_run(run), TrustLevel.TRUSTED)

    def test_trust_pull_request_is_untrusted(self):
        run = {"event": "pull_request", "head_branch": "feature"}
        self.assertEqual(trust_for_run(run), TrustLevel.UNTRUSTED)

    def test_trust_push_feature_branch_is_untrusted(self):
        run = {"event": "push", "head_branch": "my-feature"}
        self.assertEqual(trust_for_run(run), TrustLevel.UNTRUSTED)

    def test_validation_from_conclusion(self):
        self.assertEqual(
            validation_for_run({"conclusion": "success"}),
            ValidationState.BUILD_PASSED,
        )
        self.assertEqual(
            validation_for_run({"conclusion": "failure"}),
            ValidationState.UNVALIDATED,
        )


def _run(run_id, commit, event="push", branch="main", conclusion="success"):
    return {
        "id": run_id,
        "head_sha": commit,
        "head_branch": branch,
        "event": event,
        "conclusion": conclusion,
        "name": "multi_arch_ci.yml",
        "created_at": "2026-09-28T00:00:00Z",
        "html_url": f"https://github.com/ROCm/TheRock/actions/runs/{run_id}",
    }


def _fake_lister(objects_by_run):
    def lister(run):
        return objects_by_run.get(str(run["id"]), [])

    return lister


class IngestRunsTest(unittest.TestCase):
    def test_groups_components_into_bundle_units(self):
        # One run producing two components of blas + one of base.
        objects = {
            "100": [
                OutputObject("blas_lib_gfx94X-dcgpu.tar.zst", "s3://b/1"),
                OutputObject("blas_dev_gfx94X-dcgpu.tar.zst", "s3://b/2"),
                OutputObject("base_run_generic.tar.zst", "s3://b/3"),
            ]
        }
        with CacheDB() as db:
            stats = ingest_runs(
                db=db,
                runs=[_run("100", "aaaa")],
                artifact_lister=_fake_lister(objects),
            )
            # Two units: blas/gfx94X-dcgpu (2 objects) and base/generic (1 object).
            self.assertEqual(stats.units_published, 2)
            self.assertEqual(stats.objects_indexed, 3)
            self.assertEqual(stats.runs_indexed, 1)
            self.assertEqual(db.count(), 2)

    def test_require_trusted_skips_pr_runs(self):
        objects = {"200": [OutputObject("blas_lib_generic.tar.zst", "s3://b/1")]}
        with CacheDB() as db:
            stats = ingest_runs(
                db=db,
                runs=[_run("200", "bbbb", event="pull_request", branch="pr")],
                artifact_lister=_fake_lister(objects),
                require_trusted=True,
            )
            self.assertEqual(stats.units_published, 0)
            self.assertEqual(db.count(), 0)

    def test_failed_run_not_published_by_policy(self):
        # Unvalidated (failed) run: default producer policy refuses to publish.
        objects = {"300": [OutputObject("blas_lib_generic.tar.zst", "s3://b/1")]}
        with CacheDB() as db:
            stats = ingest_runs(
                db=db,
                runs=[_run("300", "cccc", conclusion="failure")],
                artifact_lister=_fake_lister(objects),
            )
        self.assertEqual(stats.units_published, 0)
        self.assertEqual(stats.units_skipped_policy, 1)


if __name__ == "__main__":
    unittest.main()
