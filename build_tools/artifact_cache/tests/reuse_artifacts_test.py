# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for reuse_artifacts.plan_reuse (artifact-level RFC0017 reuse)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reuse_artifacts


class FakeTopology:
    """Minimal BuildTopology stand-in: stage name -> produced artifact names."""

    def __init__(self, stages: dict[str, list[str]]):
        self._stages = stages

    def get_build_order(self):
        return list(self._stages.keys())

    def get_produced_artifacts(self, stage):
        return set(self._stages.get(stage, []))


# compiler stage produces amd-llvm + friends; math-libs produces blas/fft.
TOPOLOGY = FakeTopology(
    {
        "compiler": ["amd-llvm", "base", "hipify"],
        "math-libs": ["blas", "fft"],
    }
)


def _hit(name, run_id="42"):
    return {
        "name": name,
        "status": "hit",
        "provenance": {"repository": "ROCm/TheRock", "run_id": run_id},
    }


def _miss(name):
    return {"name": name, "status": "miss"}


class PlanReuseTest(unittest.TestCase):
    def test_build_plan_yields_no_reuse(self):
        plan = {"action": "build", "artifacts": [_hit("amd-llvm")]}
        self.assertEqual(reuse_artifacts.plan_reuse(plan, TOPOLOGY), ([], [], None))

    def test_no_hits_yields_no_reuse(self):
        plan = {"action": "populate", "artifacts": [_miss("amd-llvm")]}
        self.assertEqual(reuse_artifacts.plan_reuse(plan, TOPOLOGY), ([], [], None))

    def test_single_hit_reused_rest_rebuilt(self):
        # RFC0017 milestone 2: reuse amd-llvm, rebuild everything else.
        plan = {"action": "populate", "artifacts": [_hit("amd-llvm")]}
        reusable, rebuild, run_id = reuse_artifacts.plan_reuse(plan, TOPOLOGY)
        self.assertEqual(reusable, ["amd-llvm"])
        self.assertEqual(rebuild, ["base", "blas", "fft", "hipify"])
        self.assertEqual(run_id, "42")
        # a reused artifact is never also rebuilt
        self.assertNotIn("amd-llvm", rebuild)

    def test_multiple_hits_same_run(self):
        plan = {
            "action": "populate",
            "artifacts": [_hit("amd-llvm"), _hit("base"), _miss("blas")],
        }
        reusable, rebuild, run_id = reuse_artifacts.plan_reuse(plan, TOPOLOGY)
        self.assertEqual(reusable, ["amd-llvm", "base"])
        self.assertEqual(rebuild, ["blas", "fft", "hipify"])
        self.assertEqual(run_id, "42")

    def test_hits_across_multiple_runs_refused(self):
        plan = {
            "action": "populate",
            "artifacts": [_hit("amd-llvm", "42"), _hit("base", "99")],
        }
        self.assertEqual(reuse_artifacts.plan_reuse(plan, TOPOLOGY), ([], [], None))

    def test_rebuild_is_complement_over_whole_graph(self):
        plan = {"action": "populate", "artifacts": [_hit("blas")]}
        reusable, rebuild, _ = reuse_artifacts.plan_reuse(plan, TOPOLOGY)
        self.assertEqual(reusable, ["blas"])
        # blas lives in math-libs, but the complement spans every stage.
        self.assertEqual(rebuild, ["amd-llvm", "base", "fft", "hipify"])


if __name__ == "__main__":
    unittest.main()
