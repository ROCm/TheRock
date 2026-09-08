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
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="dev",
            platform="linux",
        )

        python_versions = {row["python_version"] for row in matrix}
        jax_refs = {row["jax_ref"] for row in matrix}

        self.assertGreater(len(matrix), 1)
        self.assertGreater(len(python_versions), 1)
        self.assertGreater(len(jax_refs), 1)
        self.assertEqual(
            set(matrix[0]),
            {
                "python_version",
                "jax_ref",
                "jax_label",
                "jax_repository",
                "rocm_jax_ref",
                "gfx_arch",
            },
        )

    def test_explicit_python_version_narrows_matrix(self):
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="dev",
            platform="linux",
            python_versions=["3.12"],
        )

        self.assertGreater(len(matrix), 1)
        self.assertEqual(
            {row["python_version"] for row in matrix},
            {"3.12"},
        )

    def test_generate_jax_matrix_uses_requested_refs_only(self):
        matrix = m.generate_jax_matrix(
            jax_refs=["rocm-jaxlib-v0.10.1"],
            python_versions=["3.12"],
        )

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["python_version"], "3.12")
        self.assertEqual(matrix[0]["jax_ref"], "rocm-jaxlib-v0.10.1")
        self.assertEqual(matrix[0]["jax_label"], "0.10.1")
        self.assertEqual(matrix[0]["jax_repository"], "ROCm/jax")
        self.assertEqual(matrix[0]["rocm_jax_ref"], "rocm-jaxlib-v0.10.1")
        self.assertEqual(matrix[0]["gfx_arch"], "device-all")

    def test_release_refs_check_out_rocm_jax_at_the_jax_ref(self):
        # A release tag exists under the same name in ROCm/jax and
        # ROCm/rocm-jax, so a ref config that names no rocm_jax_ref keeps the
        # two checkouts on one ref, as the workflows did before the key existed.
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="dev",
            platform="linux",
        )

        self.assertGreater(len(matrix), 0)
        for row in matrix:
            self.assertEqual(row["rocm_jax_ref"], row["jax_ref"])

    def test_jax_main_ref_builds_upstream_with_rocm_jax_default_branch(self):
        # Upstream main has no rocm-jax ref of the same name, so the tip config
        # pins the Dockerfile checkout separately and names its own label.
        matrix = m.generate_jax_matrix(
            jax_refs=["jax-main"],
            python_versions=["3.11", "3.12"],
        )

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["python_version"], "3.12")
        self.assertEqual(matrix[0]["jax_ref"], "main")
        self.assertEqual(matrix[0]["jax_label"], "tip")
        self.assertEqual(matrix[0]["jax_repository"], "jax-ml/jax")
        self.assertEqual(matrix[0]["rocm_jax_ref"], "rocm-jax-infra")

    def test_jax_main_is_not_a_default_ref(self):
        # The tip build is opt-in: no release type builds upstream main unless
        # a caller asks for it by name.
        for release_type in m.RELEASE_TYPES:
            matrix = m.generate_jax_matrix_for_release_type(
                release_type=release_type,
                platform="linux",
            )
            self.assertNotIn("main", {row["jax_ref"] for row in matrix})

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
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="dev",
            platform="linux",
            python_versions=["3.12"],
        )
        self.assertGreater(len(matrix), 0)

        # Both matrix consumers: the release entry point and the CI entry point.
        for workflow_name in [
            "multi_arch_release_linux_jax_wheels.yml",
            "multi_arch_ci_linux.yml",
        ]:
            with self.subTest(workflow=workflow_name):
                workflow = load_workflow(WORKFLOWS_DIR / workflow_name)
                job = get_workflow_job(workflow, "build_jax_wheels")
                matrix_references = get_matrix_references(job["with"])
                self.assertGreater(len(matrix_references), 0)

                for row in matrix:
                    # This checks the row schema, not whether values are
                    # truthy. Empty values are allowed, such as gfx_arch=""
                    # for native JAX builds. Undefined values are not: if the
                    # workflow reads `matrix.unknown`, this test fails until
                    # the generator emits that key for every row.
                    self.assertEqual(matrix_references - set(row), set())

    def test_ref_excluded_python_versions_are_filtered(self):
        # exclude_python_versions in JAX_REF_CONFIGS drops those Python versions
        # for that ref only (e.g. JAX 0.11.0 excludes 3.11).
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="dev",
            platform="linux",
        )
        combos = {(row["python_version"], row["jax_ref"]) for row in matrix}

        self.assertNotIn(("3.11", "rocm-jaxlib-v0.11.0"), combos)
        self.assertNotIn(("3.11", "rocm-jaxlib-v0.11.1"), combos)
        # 3.11 is still valid for refs that don't exclude it.
        self.assertIn(("3.11", "rocm-jaxlib-v0.10.2"), combos)
        # Non-excluded Python versions are still paired with the 0.11 refs.
        self.assertIn(("3.12", "rocm-jaxlib-v0.11.0"), combos)
        self.assertIn(("3.12", "rocm-jaxlib-v0.11.1"), combos)

    def test_ci_builds_single_jax_ref(self):
        # CI keeps the runner queues short by building one ref, while still
        # fanning out over Python versions. Comparing the set of distinct refs
        # (rather than the row count) is what pins CI to exactly one ref.
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="ci",
            platform="linux",
        )
        self.assertEqual({row["jax_ref"] for row in matrix}, {"rocm-jaxlib-v0.11.1"})

    def test_unknown_release_type_raises(self):
        with self.assertRaises(ValueError):
            m.generate_jax_matrix_for_release_type(
                release_type="unknown",
                platform="linux",
            )

    def test_unknown_jax_ref_raises(self):
        with self.assertRaises(ValueError):
            m.generate_jax_matrix_for_release_type(
                release_type="ci",
                platform="linux",
                jax_refs=["unknown-jax-ref"],
            )


if __name__ == "__main__":
    unittest.main()
