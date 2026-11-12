# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import argparse
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csan_repair
from workflow_utils import load_workflow, WORKFLOWS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _therock_utils.build_topology import BuildTopology


class MathParallelTest(unittest.TestCase):
    def test_predecessor_device_archives_require_family_expansion(self):
        from artifact_manager import parse_target_families, find_available_artifacts

        archive = "rand_lib_gfx942.tar.zst"
        for expand in (False, True):
            targets = parse_target_families(
                argparse.Namespace(
                    generic_only=False,
                    amdgpu_families="gfx94X-dcgpu",
                    expand_family_to_targets=expand,
                    amdgpu_targets=None,
                )
            )
            selected = find_available_artifacts({"rand"}, targets, {archive})
            self.assertEqual(selected, [archive] if expand else [])

    def test_every_math_artifact_has_one_owner_and_dependencies_publish_first(self):
        topology = BuildTopology(str(WORKFLOWS_DIR.parents[1] / "BUILD_TOPOLOGY.toml"))
        jobs = load_workflow(WORKFLOWS_DIR / "ci_csan_math.yml")["jobs"]
        owners = {}
        for name, job in jobs.items():
            for artifact in job["with"]["rebuild_artifacts"].split(","):
                self.assertNotIn(artifact, owners)
                owners[artifact] = name
            self.assertTrue(job["with"]["reuse_stage_artifacts"])
            self.assertEqual(job["with"]["artifact_suffix"], name)
            self.assertLessEqual(
                set(job["with"]["rebuild_artifacts"].split(",")),
                set(job["with"]["configure_artifacts"].split(",")),
            )
        self.assertEqual(set(owners), topology.get_produced_artifacts("math-libs"))

        def ancestors(name):
            result = set(jobs[name].get("needs", []))
            for dependency in list(result):
                result.update(ancestors(dependency))
            return result

        for artifact, owner in owners.items():
            for dependency in topology.artifacts[artifact].artifact_deps:
                if dependency in owners:
                    self.assertIn(owners[dependency], ancestors(owner) | {owner})
        # Source-level dependencies omitted from the artifact topology.
        self.assertIn(owners["rand"], ancestors(owners["composable-kernel"]))
        self.assertEqual(owners["support"], owners["blas"])
        self.assertEqual(owners["blas"], owners["solver"])

    def test_math_only_does_not_rebuild_nonmath_or_run_normal_pipeline(self):
        workflow = load_workflow(WORKFLOWS_DIR / "ci_csan.yml")
        self.assertIn("!inputs.math_parallel", workflow["jobs"]["linux_build"]["if"])
        self.assertIn("parallel_math", workflow["jobs"]["summary"]["needs"])
        self.assertIn(
            "copy_prebuilt_stages", workflow["jobs"]["parallel_math"]["needs"]
        )
        with patch.dict(
            "os.environ", {"MATH_PARALLEL": "true", "COMPONENT_REPAIR": "true"}
        ), patch.object(sys, "argv", ["csan_repair.py", "inputs"]):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                csan_repair.main()


if __name__ == "__main__":
    unittest.main()
