# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for fingerprint_cache.planner (flowchart 1) using a dict topology."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprint_cache.cache_key import derive_fallback_fingerprint
from fingerprint_cache.cache_db import CacheDB
from fingerprint_cache.manifest import (
    OutputObject,
    Provenance,
    TrustLevel,
    ValidationState,
)
from fingerprint_cache.planner import plan_cache
from fingerprint_cache.publish import publish_build_outputs
from fingerprint_cache.topology_view import DictTopologyView

TARGETS = ["gfx94X-dcgpu"]
COMMIT = "c0ffee0000"


def _topology() -> DictTopologyView:
    return DictTopologyView(
        stages=[
            ("compiler", [("amd-llvm", False)]),
            ("core", [("core-hip", False)]),
            ("math", [("blas", True), ("fft", True)]),
        ],
        path_to_artifacts={
            "math-libs/BLAS/rocBLAS/x.cpp": ["blas"],
            "compiler/llvm/y.cpp": ["amd-llvm"],
        },
        global_paths=["build_tools/", ".github/"],
    )


def _seed_full_cache(db: CacheDB, topology: DictTopologyView, commit: str) -> None:
    """Publish a trusted, build-passed entry for every unit at ``commit``."""
    for stage in topology.build_order_stages():
        for unit in topology.units_for_stage(stage, TARGETS):
            fp = derive_fallback_fingerprint(source_commit=commit, artifact=unit)
            publish_build_outputs(
                db=db,
                artifact=unit,
                outputs=[
                    OutputObject(
                        f"{unit.name}_lib_{unit.target_family}.tar.zst", "s3://b/x"
                    )
                ],
                provenance=Provenance(
                    repository="ROCm/TheRock",
                    run_id="seed",
                    branch="main",
                    commit=commit,
                    workflow="multi_arch_ci.yml",
                    created_at="2026-09-01T00:00:00Z",
                ),
                source_fingerprint=fp,
                trust=TrustLevel.TRUSTED,
                validation=ValidationState.BUILD_PASSED,
            )


class PlannerMissTest(unittest.TestCase):
    def test_empty_cache_all_miss(self):
        topo = _topology()
        with CacheDB() as db:
            plan = plan_cache(
                topology=topo,
                changed_paths=[],
                target_families=TARGETS,
                source_commit=COMMIT,
                db=db,
            )
        self.assertEqual(plan.total_hits, 0)
        self.assertEqual(plan.total_misses, plan.total_units)
        self.assertEqual(plan.skippable_stages, [])
        # 2 generic units + 2 target-specific units (one family) = 4
        self.assertEqual(plan.total_units, 4)


class PlannerHitTest(unittest.TestCase):
    def test_full_cache_all_hit_all_skippable(self):
        topo = _topology()
        with CacheDB() as db:
            _seed_full_cache(db, topo, COMMIT)
            plan = plan_cache(
                topology=topo,
                changed_paths=[],
                target_families=TARGETS,
                source_commit=COMMIT,
                db=db,
            )
        self.assertEqual(plan.total_hits, plan.total_units)
        self.assertEqual(set(plan.skippable_stages), {"compiler", "core", "math"})

    def test_changed_path_forces_unit_miss(self):
        topo = _topology()
        with CacheDB() as db:
            _seed_full_cache(db, topo, COMMIT)
            plan = plan_cache(
                topology=topo,
                changed_paths=["math-libs/BLAS/rocBLAS/x.cpp"],
                target_families=TARGETS,
                source_commit=COMMIT,
                db=db,
            )
        # blas unit must rebuild; math stage cannot be skipped; fft still hit.
        self.assertNotIn("math", plan.skippable_stages)
        self.assertIn("compiler", plan.skippable_stages)
        self.assertIn("core", plan.skippable_stages)
        math_stage = next(sp for sp in plan.stage_plans if sp.stage == "math")
        built = [a.name for a in math_stage.units_to_build]
        self.assertEqual(built, ["blas"])

    def test_global_path_forces_full_rebuild(self):
        topo = _topology()
        with CacheDB() as db:
            _seed_full_cache(db, topo, COMMIT)
            plan = plan_cache(
                topology=topo,
                changed_paths=["build_tools/github_actions/foo.py"],
                target_families=TARGETS,
                source_commit=COMMIT,
                db=db,
            )
        self.assertEqual(plan.total_hits, 0)
        self.assertEqual(plan.skippable_stages, [])


class PlannerTrustPolicyTest(unittest.TestCase):
    def test_untrusted_entries_not_reused_by_default(self):
        topo = _topology()
        with CacheDB() as db:
            # Seed as UNTRUSTED (PR builds).
            for stage in topo.build_order_stages():
                for unit in topo.units_for_stage(stage, TARGETS):
                    fp = derive_fallback_fingerprint(
                        source_commit=COMMIT, artifact=unit
                    )
                    publish_build_outputs(
                        db=db,
                        artifact=unit,
                        outputs=[OutputObject("x_lib_generic.tar.zst", "s3://b/x")],
                        provenance=Provenance(
                            "ROCm/TheRock",
                            "pr",
                            "pr-branch",
                            COMMIT,
                            "ci",
                            "2026-09-01T00:00:00Z",
                        ),
                        source_fingerprint=fp,
                        trust=TrustLevel.UNTRUSTED,
                        validation=ValidationState.BUILD_PASSED,
                    )
            plan = plan_cache(
                topology=topo,
                changed_paths=[],
                target_families=TARGETS,
                source_commit=COMMIT,
                db=db,
            )
        # Default read policy requires TRUSTED, so untrusted entries are ignored.
        self.assertEqual(plan.total_hits, 0)


if __name__ == "__main__":
    unittest.main()
