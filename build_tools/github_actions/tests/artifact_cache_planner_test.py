# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Planning uses ArtifactKey candidates and never asserts unverified stage skips."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_db import CacheDB, ReadPolicy
from build_tools.github_actions.artifact_cache.inventory import TopologyAdapter
from build_tools.github_actions.artifact_cache.planner import plan_cache
from build_tools.github_actions.artifact_cache.manifest import TrustLevel
from build_tools.github_actions.tests import _artifact_cache_fixtures as fixtures

OUTPUTS = ("amd-llvm_lib_generic", "core-hip_lib_generic",
           "blas_lib_gfx94X-dcgpu", "fft_lib_gfx94X-dcgpu")


class Topology:
    def get_build_order(self):
        return ["compiler", "core", "math"]

    def get_produced_artifacts(self, stage):
        return {"compiler": ["amd-llvm"], "core": ["core-hip"], "math": ["blas", "fft"]}[stage]

    def get_artifacts_for_path(self, path):
        return {"math-libs/BLAS/rocBLAS/x.cpp": ["blas"],
                "compiler/llvm/y.cpp": ["amd-llvm"]}.get(path, [])


class PlannerTest(fixtures.RelationalFixture):
    def plan(self, *, changed_paths=(), policy=None):
        records = [({"artifact_identity": name}, self.key(name)) for name in OUTPUTS]
        with CacheDB(self.database) as db:
            return plan_cache(db=db, records=records, topology=TopologyAdapter(Topology()),
                              changed_paths=changed_paths, read_policy=policy)

    def seed_full(self, trust=TrustLevel.TRUSTED):
        for name in OUTPUTS:
            self.seed(name, trust=trust)

    def test_empty_cache_all_miss(self):
        plan = self.plan()
        self.assertEqual(len(plan.entries), 4)
        self.assertEqual(plan.to_dict()["summary"], {"candidate": 0, "miss": 4, "uncacheable": 0})
        self.assertEqual(plan.to_dict()["skippable_stages"], [])

    def test_full_cache_all_candidates_but_no_stage_is_skippable(self):
        self.seed_full()
        plan = self.plan()
        self.assertEqual(plan.to_dict()["summary"]["candidate"], 4)
        self.assertTrue(all(stage.all_requested_candidates for stage in plan.stage_plans))
        self.assertFalse(any(stage.can_skip for stage in plan.stage_plans))
        self.assertFalse(any(entry.is_hit for entry in plan.entries))
        self.assertFalse(plan.to_dict()["payloads_verified"])

    def test_changed_path_forces_only_matching_unit_miss(self):
        self.seed_full()
        plan = self.plan(changed_paths=["math-libs/BLAS/rocBLAS/x.cpp"])
        decisions = {entry.unit.artifact_identity: entry.status for entry in plan.entries}
        self.assertEqual(decisions, {name: "miss" if name.startswith("blas_") else "candidate"
                                     for name in OUTPUTS})
        self.assertFalse(any(stage.can_skip for stage in plan.stage_plans))

    def test_unmapped_global_path_forces_full_rebuild(self):
        self.seed_full()
        plan = self.plan(changed_paths=["build_tools/github_actions/foo.py"])
        self.assertEqual(plan.to_dict()["summary"]["miss"], 4)
        self.assertEqual(plan.to_dict()["skippable_stages"], [])

    def test_explicit_trusted_policy_rejects_untrusted_candidates(self):
        self.seed_full(trust=TrustLevel.UNTRUSTED)
        strict = self.plan(policy=ReadPolicy(min_trust=TrustLevel.TRUSTED))
        loose = self.plan(policy=ReadPolicy(min_trust=TrustLevel.UNTRUSTED))
        self.assertEqual(strict.to_dict()["summary"]["candidate"], 0)
        self.assertEqual(loose.to_dict()["summary"]["candidate"], 4)


if __name__ == "__main__":
    unittest.main()
