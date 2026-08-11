# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from unittest import mock
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_jax_test_matrix as matrix_script

# gfx94x is the outer key used to construct workflow pipelines, while
# gfx94X-dcgpu is the inner key, which we use for package names. Workflows pass
# the inner key, but a manually dispatched run may pass either.
GFX94X_SINGLE_GPU = "linux-gfx942-1gpu-ccs-csp-ossci-rocm"
GFX94X_MULTI_GPU = "linux-gfx942-8gpu-ossci-rocm"


def subsets(matrix: dict) -> list[str]:
    return [job["test_subset"] for job in matrix["include"]]


def runner_for(matrix: dict, subset: str) -> str:
    for job in matrix["include"]:
        if job["test_subset"] == subset:
            return job["test_runs_on"]
    raise AssertionError(f"no {subset} job in {matrix}")


class ResolveScopeTest(unittest.TestCase):
    def test_release_types_that_test_everything(self):
        for release_type in ["nightly", "prerelease"]:
            with self.subTest(release_type=release_type):
                self.assertEqual(
                    matrix_script.resolve_scope("auto", release_type), "full"
                )

    def test_release_types_that_run_per_change(self):
        # An 8-GPU queue slot per job is too much for these.
        for release_type in ["ci", "dev"]:
            with self.subTest(release_type=release_type):
                self.assertEqual(
                    matrix_script.resolve_scope("auto", release_type), "short"
                )

    def test_an_explicit_scope_wins(self):
        # So a workflow can put the full suite on its own cadence.
        self.assertEqual(matrix_script.resolve_scope("short", "prerelease"), "short")
        self.assertEqual(matrix_script.resolve_scope("full", "ci"), "full")


class BuildTestMatrixTest(unittest.TestCase):
    def matrix(self, target="gfx94X-dcgpu", platform="linux", scope="short") -> dict:
        return matrix_script.build_test_matrix(
            target=target, platform=platform, scope=scope
        )

    def test_short_scope_runs_only_the_single_gpu_job(self):
        # On one GPU the suite is the single-accelerator tests.
        matrix = self.matrix(scope="short")

        self.assertEqual(subsets(matrix), ["all"])
        self.assertEqual(runner_for(matrix, "all"), GFX94X_SINGLE_GPU)

    def test_full_scope_adds_the_multi_accelerator_subset(self):
        matrix = self.matrix(scope="full")

        self.assertEqual(subsets(matrix), ["all", "multi"])
        self.assertEqual(runner_for(matrix, "all"), GFX94X_SINGLE_GPU)
        self.assertEqual(runner_for(matrix, "multi"), GFX94X_MULTI_GPU)

    def test_the_outer_family_key_resolves_too(self):
        matrix = self.matrix(target="gfx94x", scope="full")

        self.assertEqual(runner_for(matrix, "all"), GFX94X_SINGLE_GPU)
        self.assertEqual(runner_for(matrix, "multi"), GFX94X_MULTI_GPU)

    def test_a_family_without_a_multi_gpu_runner_skips_that_subset(self):
        # Those tests need several GPUs, and a 1-GPU runner would skip every one
        # of them while reporting a pass.
        matrix = self.matrix(target="gfx1151", platform="windows", scope="full")

        self.assertEqual(subsets(matrix), ["all"])
        self.assertEqual(runner_for(matrix, "all"), "windows-gfx1151-gpu-rocm")

    def test_an_unknown_family_is_an_error(self):
        with self.assertRaises(ValueError):
            self.matrix(target="gfx-not-a-family")


class OutputsTest(unittest.TestCase):
    def outputs(self, argv: list[str]) -> dict[str, str]:
        with open(os.environ["GITHUB_OUTPUT"], "w") as f:
            f.truncate()
        matrix_script.main(argv)
        written = Path(os.environ["GITHUB_OUTPUT"]).read_text()
        return dict(line.split("=", 1) for line in written.splitlines() if "=" in line)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.dict(
            os.environ, {"GITHUB_OUTPUT": os.path.join(self.tmp.name, "out")}
        )
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_matrix_a_workflow_consumes(self):
        outputs = self.outputs(
            ["--target", "gfx94X-dcgpu", "--release-type", "prerelease"]
        )

        self.assertEqual(outputs["enabled"], "true")
        self.assertEqual(
            json.loads(outputs["matrix"]),
            {
                "include": [
                    {"test_subset": "all", "test_runs_on": GFX94X_SINGLE_GPU},
                    {"test_subset": "multi", "test_runs_on": GFX94X_MULTI_GPU},
                ]
            },
        )

    def test_a_pull_request_gets_one_job_on_the_single_gpu_runner(self):
        outputs = self.outputs(["--target", "gfx94X-dcgpu", "--release-type", "ci"])

        self.assertEqual(
            json.loads(outputs["matrix"]),
            {"include": [{"test_subset": "all", "test_runs_on": GFX94X_SINGLE_GPU}]},
        )

    def test_an_unknown_release_type_is_rejected(self):
        # Defaulting it would quietly drop the multi-accelerator job.
        with self.assertRaises(SystemExit):
            self.outputs(["--target", "gfx94X-dcgpu", "--release-type", "nighlty"])


if __name__ == "__main__":
    unittest.main()
