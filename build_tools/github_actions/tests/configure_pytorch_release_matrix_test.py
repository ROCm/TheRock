# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_pytorch_release_matrix as m
from workflow_utils import (
    WORKFLOWS_DIR,
    get_matrix_references,
    get_workflow_job,
    load_workflow,
)


class ConfigurePytorchReleaseMatrixTest(unittest.TestCase):
    def test_ci_linux_uses_reduced_matrix(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="ci",
            amdgpu_families="gfx94X-dcgpu",
            platform="linux",
        )

        # Compared to releases:
        #   * limited to python 3.12
        #   * not including "nightly" pytorch_git_ref
        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.10",
                    "amdgpu_families": "gfx94X-dcgpu",
                },
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.11",
                    "amdgpu_families": "gfx94X-dcgpu",
                },
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.12",
                    "amdgpu_families": "gfx94X-dcgpu",
                },
            ],
        )

    def test_ci_windows_uses_reduced_matrix(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="ci",
            amdgpu_families="gfx110X-all",
            platform="windows",
        )

        # Compared to releases:
        #   * limited to python 3.12
        # Compared to Linux:
        #   * limited to only a single pytorch_git_ref
        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "release/2.10",
                    "amdgpu_families": "gfx110X-all",
                },
            ],
        )

    def test_explicit_versions_and_refs_narrow_matrix(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="nightly",
            python_versions=["3.13"],
            pytorch_git_refs=["nightly"],
            amdgpu_families="gfx94X-dcgpu",
            platform="linux",
        )

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.13",
                    "pytorch_git_ref": "nightly",
                    "amdgpu_families": "gfx94X-dcgpu",
                }
            ],
        )

    def test_filters_exact_unsupported_family(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev",
            python_versions=["3.12"],
            pytorch_git_refs=["release/2.10"],
            amdgpu_families="gfx94X-dcgpu;gfx125X-dcgpu",
            platform="linux",
        )

        # gfx125X-dcgpu not supported on the release/2.10 ref, should filter
        self.assertEqual(matrix[0]["amdgpu_families"], "gfx94X-dcgpu")
        matrix_families = ";".join(row["amdgpu_families"] for row in matrix)
        self.assertNotIn("gfx125X", matrix_families)

    def test_unknown_explicit_ref_keeps_families(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev",
            python_versions=["3.12"],
            pytorch_git_refs=["users/alice/gfx125x-bringup"],
            amdgpu_families="gfx125X-dcgpu",
            platform="linux",
        )

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.12",
                    "pytorch_git_ref": "users/alice/gfx125x-bringup",
                    "amdgpu_families": "gfx125X-dcgpu",
                }
            ],
        )

    def test_generated_rows_cover_workflow_matrix_inputs(self):
        # The generate_pytorch_matrix_for_release_type script produces matrix
        # JSON for use in workflow files like:
        #
        #   matrix:
        #     include: ${{ fromJSON(needs.setup_matrix.outputs.pytorch_matrix) }}
        #
        # Then it passes values to the build workflow using expressions like:
        #   with:
        #     amdgpu_families: ${{ matrix.amdgpu_families }}
        #     python_version: ${{ matrix.python_version }}
        #     pytorch_git_ref: ${{ matrix.pytorch_git_ref }}
        #
        # This test checks that all `${{ matrix. }}` values are produced for
        # every row in the generated matrix. It intentionally does not check
        # that every generated key is consumed by each workflow; if we want to
        # enforce exact schemas, do that with generator-local tests.
        test_cases = [
            (
                "multi_arch_release_linux_pytorch_wheels.yml",
                "build_pytorch_wheels",
                "linux",
                "dev",
            ),
            (
                "multi_arch_release_windows_pytorch_wheels.yml",
                "build_pytorch_wheels",
                "windows",
                "dev",
            ),
            ("multi_arch_ci_linux.yml", "build_pytorch_wheel_fat", "linux", "ci"),
            ("multi_arch_ci_windows.yml", "build_pytorch_wheel_fat", "windows", "ci"),
        ]

        for workflow_filename, job_name, platform, release_type in test_cases:
            with self.subTest(workflow_filename=workflow_filename):
                workflow = load_workflow(WORKFLOWS_DIR / workflow_filename)
                job = get_workflow_job(workflow, job_name)
                matrix_references = get_matrix_references(job["with"])

                matrix = m.generate_pytorch_matrix_for_release_type(
                    release_type=release_type,
                    python_versions=["3.12"],
                    pytorch_git_refs=["release/2.10"],
                    amdgpu_families="gfx94X-dcgpu",
                    platform=platform,
                )

                self.assertGreater(len(matrix), 0)
                for row in matrix:
                    # This checks the row schema, not whether values are
                    # truthy. Empty values are allowed when a workflow handles
                    # them explicitly. Undefined values are not: if the
                    # workflow reads `matrix.unknown`, this test fails until
                    # the generator emits that key for every row.
                    self.assertEqual(matrix_references - set(row), set())


if __name__ == "__main__":
    unittest.main()
