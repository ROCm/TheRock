# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_jax_release_matrix as m
from workflow_utils import (
    WORKFLOWS_DIR,
    get_matrix_references,
    get_workflow_call_inputs,
    get_workflow_job,
    load_workflow,
)


class ConfigureJaxReleaseMatrixTest(unittest.TestCase):
    def setUp(self):
        # Keep matrix behavior independent of the current supported releases.
        self.enterContext(
            mock.patch.object(
                m,
                "JAX_REF_CONFIGS",
                {
                    "rocm-jaxlib-vFAKE-old": {
                        "jax_ref": "rocm-jaxlib-vFAKE-old",
                        "jax_repository": "example/fake-jax",
                        "gfx_arch": "fake-device-all",
                    },
                    "rocm-jaxlib-vFAKE-new": {
                        "jax_ref": "rocm-jaxlib-vFAKE-new",
                        "jax_repository": "example/fake-jax",
                        "gfx_arch": "fake-device-all",
                        "exclude_python_versions": ["3.11"],
                    },
                    "fake-tip": {
                        "jax_ref": "fake-upstream-main",
                        "jax_repository": "example/fake-upstream-jax",
                        "rocm_jax_ref": "fake-infra",
                        "gfx_arch": "fake-device-all",
                        "jax_label": "fake-tip-label",
                        "wheel_type": "nightly",
                        "publish": False,
                        "exclude_python_versions": ["3.11", "3.12"],
                    },
                },
            )
        )
        for name, refs in (
            ("CI_JAX_REFS", ["rocm-jaxlib-vFAKE-new"]),
            (
                "RELEASE_STABLE_JAX_REFS",
                ["rocm-jaxlib-vFAKE-old", "rocm-jaxlib-vFAKE-new"],
            ),
            (
                "RELEASE_JAX_REFS",
                ["rocm-jaxlib-vFAKE-old", "rocm-jaxlib-vFAKE-new", "fake-tip"],
            ),
        ):
            self.enterContext(mock.patch.object(m, name, {"linux": refs}))
        self.enterContext(
            mock.patch.object(m, "RELEASE_PYTHON_VERSIONS", ["3.11", "3.12", "3.13"])
        )
        self.enterContext(
            mock.patch.object(m, "CI_PYTHON_VERSIONS", {"linux": ["3.12"]})
        )

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
                "wheel_type",
                "publish",
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
            jax_refs=["rocm-jaxlib-vFAKE-old"],
            python_versions=["3.12"],
        )

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["python_version"], "3.12")
        self.assertEqual(matrix[0]["jax_ref"], "rocm-jaxlib-vFAKE-old")
        self.assertEqual(matrix[0]["jax_label"], "FAKE-old")
        self.assertEqual(matrix[0]["jax_repository"], "example/fake-jax")
        self.assertEqual(matrix[0]["rocm_jax_ref"], "rocm-jaxlib-vFAKE-old")
        self.assertEqual(matrix[0]["gfx_arch"], "fake-device-all")
        self.assertEqual(matrix[0]["wheel_type"], "release")

    def test_release_refs_check_out_rocm_jax_at_the_jax_ref(self):
        # A config with no rocm_jax_ref keeps both checkouts on the same ref.
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="prerelease",
            platform="linux",
        )

        self.assertGreater(len(matrix), 0)
        for row in matrix:
            self.assertEqual(row["rocm_jax_ref"], row["jax_ref"])

    def test_jax_main_ref_builds_upstream_with_rocm_jax_default_branch(self):
        # Upstream main has no rocm-jax ref of the same name, so the tip config
        # pins the Dockerfile checkout separately and names its own label.
        matrix = m.generate_jax_matrix(
            jax_refs=["fake-tip"],
            python_versions=["3.11", "3.13"],
        )

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["python_version"], "3.13")
        self.assertEqual(matrix[0]["jax_ref"], "fake-upstream-main")
        self.assertEqual(matrix[0]["jax_label"], "fake-tip-label")
        self.assertEqual(matrix[0]["jax_repository"], "example/fake-upstream-jax")
        self.assertEqual(matrix[0]["rocm_jax_ref"], "fake-infra")
        self.assertEqual(matrix[0]["wheel_type"], "nightly")

    def test_tip_builds_without_publishing(self):
        # whl-next does not route jax/jaxlib yet and the tests install from
        # there, so tip is a build-only signal while every pinned ref publishes.
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="nightly", platform="linux"
        )
        published = {row["jax_ref"]: row["publish"] for row in matrix}
        self.assertIs(published.pop("fake-upstream-main"), False)
        self.assertTrue(published)
        self.assertTrue(all(published.values()))

    def test_release_refs_build_release_wheels(self):
        # Configs without wheel_type default to release wheels; the fake tip
        # explicitly opts into nightly wheels.
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="prerelease",
            platform="linux",
        )

        self.assertGreater(len(matrix), 0)
        for row in matrix:
            self.assertEqual(row["wheel_type"], "release")

    def test_prerelease_builds_stable_refs_only(self):
        # A prerelease is a candidate for something we ship, so nothing in it
        # may change between two runs. Upstream tip moves, so it is left out.
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="prerelease",
            platform="linux",
        )

        self.assertGreater(len(matrix), 0)
        self.assertNotIn("fake-upstream-main", {row["jax_ref"] for row in matrix})

    def test_dev_and_nightly_build_tip(self):
        # Tip against the ROCm of the same run is a row of the matrix, the way
        # PyTorch's "nightly" ref is, so a break shows up as a failed cell.
        for release_type in ("dev", "nightly"):
            with self.subTest(release_type=release_type):
                matrix = m.generate_jax_matrix_for_release_type(
                    release_type=release_type,
                    platform="linux",
                )
                self.assertIn("fake-upstream-main", {row["jax_ref"] for row in matrix})

    def test_bkc_releases_have_empty_matrices(self):
        # BKC releases skip JAX builds, including when inputs select a version.
        for release_type in ("dev-bkc", "nightly-bkc"):
            with self.subTest(release_type=release_type):
                self.assertEqual(
                    m.generate_jax_matrix_for_release_type(
                        release_type=release_type,
                        platform="linux",
                    ),
                    [],
                )
                self.assertEqual(
                    m.generate_jax_matrix_for_release_type(
                        release_type=release_type,
                        platform="linux",
                        python_versions=["3.12"],
                        jax_refs=["rocm-jaxlib-vFAKE-old"],
                    ),
                    [],
                )

    def test_ci_never_builds_a_moving_ref(self):
        # A pull request is judged against refs whose content is fixed;
        # upstream moving under it would fail CI outside the author's control.
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="ci",
            platform="linux",
        )

        self.assertGreater(len(matrix), 0)
        self.assertNotIn("fake-upstream-main", {row["jax_ref"] for row in matrix})

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

    def test_build_workflows_accept_generated_row_keys(self):
        # Both build workflows must accept any row the generator can emit, so
        # a ref config can be moved between the CI and release defaults (or
        # dispatched to either build workflow) without a workflow change. Every
        # row key except jax_label, which only names the calling job, is an
        # input on each build workflow; the callers forward what they need.
        matrix = m.generate_jax_matrix(
            jax_refs=["rocm-jaxlib-vFAKE-new", "fake-tip"],
            python_versions=["3.13"],
        )
        self.assertEqual(len(matrix), 2)
        row_keys = set().union(*(set(row) for row in matrix)) - {"jax_label"}
        self.assertIn("wheel_type", row_keys)

        for workflow_name in [
            "multi_arch_build_linux_jax_wheels.yml",
            "multi_arch_build_linux_jax_wheels_ci.yml",
        ]:
            with self.subTest(workflow=workflow_name):
                workflow = load_workflow(WORKFLOWS_DIR / workflow_name)
                self.assertEqual(row_keys - get_workflow_call_inputs(workflow), set())

    def test_ref_excluded_python_versions_are_filtered(self):
        # The old ref has no exclusions, the new ref excludes one version, and
        # tip excludes two. Filtering must preserve all other combinations.
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="dev",
            platform="linux",
        )
        combos = {(row["python_version"], row["jax_ref"]) for row in matrix}

        self.assertEqual(
            combos,
            {
                ("3.11", "rocm-jaxlib-vFAKE-old"),
                ("3.12", "rocm-jaxlib-vFAKE-old"),
                ("3.13", "rocm-jaxlib-vFAKE-old"),
                ("3.12", "rocm-jaxlib-vFAKE-new"),
                ("3.13", "rocm-jaxlib-vFAKE-new"),
                ("3.13", "fake-upstream-main"),
            },
        )

    def test_ci_builds_single_jax_ref(self):
        # CI keeps the runner queues short by building one ref, while still
        # fanning out over Python versions. Comparing the set of distinct refs
        # (rather than the row count) is what pins CI to exactly one ref.
        matrix = m.generate_jax_matrix_for_release_type(
            release_type="ci",
            platform="linux",
        )
        self.assertEqual({row["jax_ref"] for row in matrix}, {"rocm-jaxlib-vFAKE-new"})

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
