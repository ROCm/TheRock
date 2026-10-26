# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replay imported history evidence without publishing or claiming real reuse."""

import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_db import CacheDB, ReadPolicy
from build_tools.github_actions.artifact_cache.history_ingest import ingest_runs
from build_tools.github_actions.artifact_cache.manifest import OutputObject, TrustLevel
from build_tools.github_actions.artifact_cache.replay import replay
from build_tools.github_actions.tests import _artifact_cache_fixtures as fixtures

UNITS = ("amd-llvm_lib_generic", "blas_lib_gfx94X-dcgpu", "fft_lib_gfx94X-dcgpu")


def objects(run):
    return [OutputObject(name + ".tar.zst", f"s3://history/{run['id']}/{name}.tar.zst") for name in UNITS]


class ReplayTest(fixtures.RelationalFixture):
    def report(self, runs, policy=None):
        with CacheDB(self.database) as db:
            ingest_runs(db=db, runs=runs, artifact_lister=objects)
            report = replay(db.history(), read_policy=policy)
            self.assertEqual(db.stats()["cache_entries"], 0)
            self.assertEqual(db.stats()["artifact_files"], 0)
        return report

    def pair(self, **later_options):
        return [fixtures.history_run(1), fixtures.history_run(
            2, start="2026-09-01T01:00:00Z", end="2026-09-01T01:10:00Z", **later_options)]

    def test_first_run_all_miss_then_same_commit_all_potential_hits(self):
        report = self.report(self.pair())
        first, second = report.results
        self.assertEqual(first["potential_hits"], 0)
        self.assertEqual(second["potential_hits"], 3)
        self.assertEqual(second["same_commit_hits"], 3)
        self.assertFalse(report.to_dict()["payloads_verified"])

    def test_cross_commit_reuse_beats_scoped_same_commit_baseline(self):
        before = {name: hashlib.sha256(name.encode()).hexdigest() for name in UNITS}
        after = {**before, UNITS[2]: hashlib.sha256(b"changed fft").hexdigest()}
        runs = self.pair(commit="c2", fingerprints=after)
        runs[0]["source_fingerprints"] = before
        report = self.report(runs)
        second = report.results[1]
        self.assertEqual(second["same_commit_hits"], 0)
        self.assertEqual(second["potential_hits"], 2)
        self.assertEqual(second["units"], 3)
        data = report.to_dict()
        self.assertGreater(data["potential_hit_rate"], data["same_commit_hit_rate"])

    def test_report_serializes_as_evidence_estimate(self):
        report = self.report([fixtures.history_run(1)])
        data = json.loads(report.render())
        self.assertEqual(data, report.to_dict())
        self.assertEqual(data["scope"], "history_model")
        self.assertEqual(len(data["per_run"]), 1)
        self.assertFalse(data["payloads_verified"])

    def test_trusted_replay_cannot_reuse_release_branch_evidence(self):
        runs = self.pair()
        runs[0]["head_branch"] = "release/therock-10.1"
        report = self.report(runs, ReadPolicy(min_trust=TrustLevel.TRUSTED))
        self.assertEqual(report.results[1]["potential_hits"], 0)
        self.assertEqual(report.results[1]["same_commit_hits"], 0)


if __name__ == "__main__":
    unittest.main()
