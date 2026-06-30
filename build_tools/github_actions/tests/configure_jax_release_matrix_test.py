# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_jax_release_matrix as m
from workflow_utils import (
    WORKFLOWS_DIR,
    get_matrix_references,
    get_workflow_job,
    load_workflow,
)


class ConfigureJaxReleaseMatrixTest(unittest.TestCase):
    def test_default_matrix_includes_multiple_python_versions_and_refs(self):
        matrix = m.generate_jax_matrix(None)

        python_versions = {row["python_version"] for row in matrix}
        jax_refs = {row["jax_ref"] for row in matrix}

        self.assertGreater(len(matrix), 1)
        self.assertGreater(len(python_versions), 1)
        self.assertGreater(len(jax_refs), 1)
        self.assertEqual(
            set(matrix[0]),
            {"python_version", "jax_ref", "jax_repository", "build_mode", "gfx_arch"},
        )

    def test_explicit_python_version_narrows_matrix(self):
        matrix = m.generate_jax_matrix(["3.12"])

        self.assertGreater(len(matrix), 1)
        self.assertEqual(
            {row["python_version"] for row in matrix},
            {"3.12"},
        )

    def test_generated_rows_cover_workflow_matrix_inputs(self):
        # workflow file like:
        #
        #   matrix:
        #     include: ${{ fromJSON(needs.setup_matrix.outputs.jax_matrix) }}
        #
        # Then it passes values to the build workflow using expressions like:
        #   with:
        #     test_amdgpu_family: ${{ inputs.test_amdgpu_family }}
        #     python_version: ${{ matrix.python_version }}
        #     jax_ref: ${{ matrix.jax_ref }}
        #
        # This test checks that all `${{ matrix. }}` values are produced for
        # every row in the generated matrix. It intentionally does not check
        # that every generated key is consumed by each workflow; if we want to
        # enforce exact schemas, do that with generator-local tests.

        workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_release_linux_jax_wheels.yml"
        )
        job = get_workflow_job(workflow, "build_jax_wheels")
        matrix_references = get_matrix_references(job["with"])

        matrix = m.generate_jax_matrix(["3.12"])

        self.assertGreater(len(matrix), 0)
        for row in matrix:
            # This checks the row schema, not whether values are truthy. Empty
            # values are allowed, such as gfx_arch="" for native JAX builds.
            # Undefined values are not: if the workflow reads `matrix.unknown`,
            # this test fails until the generator emits that key for every row.
            self.assertEqual(matrix_references - set(row), set())


if __name__ == "__main__":
    unittest.main()
