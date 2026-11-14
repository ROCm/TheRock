# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test matrix behavior with fake policies, then validate the real configuration.

Synthetic release refs, GPU families, targets, and Python versions exercise
generation, filtering, and release-type selection without tying expectations to
the changing support tables. Separate unmocked tests check that the real default
configuration generates nonempty matrices without duplicate rows.
"""

import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import configure_pytorch_release_matrix as m
from workflow_utils import (
    WORKFLOWS_DIR,
    get_matrix_references,
    get_workflow_job,
    load_workflow,
)


# A fictional support history: exclusions shrink as releases add support.
FAKE_EXCLUDED_IMPLICIT_TARGETS = {
    "linux": {
        "release/fake-old": {"fake-target-z", "fake-target-a"},
        "release/fake-new": {"fake-target-z"},
        "release/fake-latest": set(),
    },
    "windows": {
        "release/fake-old": {"fake-target-windows"},
    },
}

FAKE_UNSUPPORTED_AMDGPU_FAMILIES = {
    "linux": {
        "release/fake-old": {"fake-family-new"},
        "release/fake-new": set(),
    },
    "windows": {},
}


FAKE_CI_PYTORCH_REFS = {
    "linux": ["release/fake-new"],
    "windows": ["release/fake-latest"],
}
FAKE_STABLE_PYTORCH_REFS = {
    "linux": ["release/fake-old", "release/fake-new"],
    "windows": ["release/fake-latest"],
}
FAKE_RELEASE_PYTORCH_REFS = {
    "linux": ["release/fake-old", "release/fake-new", "nightly"],
    "windows": ["release/fake-latest", "nightly"],
}
FAKE_PRIMARY_TEST_PYTHON_VERSIONS = {
    "release/fake-old": "3.11",
    "release/fake-new": "3.12",
    "release/fake-latest": "3.13",
    "nightly": "3.11",
}


@mock.patch.multiple(
    m,
    CI_PYTORCH_REFS=FAKE_CI_PYTORCH_REFS,
    RELEASE_STABLE_PYTORCH_REFS=FAKE_STABLE_PYTORCH_REFS,
    RELEASE_PYTORCH_REFS=FAKE_RELEASE_PYTORCH_REFS,
    CI_PYTHON_VERSIONS={"linux": ["3.12"], "windows": ["3.13"]},
    RELEASE_PYTHON_VERSIONS=["3.11", "3.12", "3.13"],
    PYTORCH_PRIMARY_TEST_PYTHON_VERSIONS=FAKE_PRIMARY_TEST_PYTHON_VERSIONS,
    EXCLUDED_IMPLICIT_TARGETS=FAKE_EXCLUDED_IMPLICIT_TARGETS,
    UNSUPPORTED_AMDGPU_FAMILIES=FAKE_UNSUPPORTED_AMDGPU_FAMILIES,
)
class ConfigurePytorchReleaseMatrixTest(unittest.TestCase):
    def test_ci_uses_platform_defaults_and_standard_tests(self):
        cases = (
            ("linux", "3.12", "release/fake-new", "fake-target-z"),
            ("windows", "3.13", "release/fake-latest", ""),
        )
        for platform, python_version, ref, exclusions in cases:
            with self.subTest(platform=platform):
                matrix = m.generate_pytorch_matrix_for_release_type(
                    release_type="ci",
                    amdgpu_families="fake-family-stable",
                    platform=platform,
                )
                self.assertEqual(
                    matrix,
                    [
                        {
                            "python_version": python_version,
                            "pytorch_git_ref": ref,
                            "amdgpu_families": "fake-family-stable",
                            "exclude_implicit_targets": exclusions,
                            "test_level": "standard",
                            "test_amdgpu_families": "auto",
                        }
                    ],
                )

    def test_release_types_select_default_refs_and_all_python_versions(self):
        # These expectations describe release-type behavior, not the live refs.
        cases = {
            "prerelease": ["release/fake-old", "release/fake-new"],
            "dev": ["release/fake-old", "release/fake-new", "nightly"],
            "dev-bkc": ["release/fake-old", "release/fake-new", "nightly"],
            "nightly": ["release/fake-old", "release/fake-new", "nightly"],
            "nightly-bkc": ["release/fake-old", "release/fake-new", "nightly"],
        }
        for release_type, refs in cases.items():
            with self.subTest(release_type=release_type):
                matrix = m.generate_pytorch_matrix_for_release_type(
                    release_type=release_type,
                    amdgpu_families="fake-family-stable",
                    platform="linux",
                )
                self.assertEqual(
                    [(row["python_version"], row["pytorch_git_ref"]) for row in matrix],
                    list(product(["3.11", "3.12", "3.13"], refs)),
                )

    def test_release_defaults_are_platform_specific(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="prerelease",
            amdgpu_families="fake-family-stable",
            platform="windows",
        )
        self.assertEqual(
            [(row["python_version"], row["pytorch_git_ref"]) for row in matrix],
            [
                ("3.11", "release/fake-latest"),
                ("3.12", "release/fake-latest"),
                ("3.13", "release/fake-latest"),
            ],
        )

    def test_explicit_versions_and_refs_narrow_matrix(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="nightly",
            python_versions=["3.13"],
            pytorch_git_refs=["nightly"],
            amdgpu_families="fake-family-stable",
            platform="linux",
        )

        self.assertEqual(
            matrix,
            [
                {
                    "python_version": "3.13",
                    "pytorch_git_ref": "nightly",
                    "amdgpu_families": "fake-family-stable",
                    "exclude_implicit_targets": "",
                    "test_level": "none",
                    "test_amdgpu_families": "auto",
                }
            ],
        )

    def test_filters_exact_unsupported_family(self):
        # Unsupported families are filtered while supported ones are kept.
        cases = (
            ("release/fake-old", "fake-family-stable", "fake-target-a;fake-target-z"),
            ("release/fake-new", "fake-family-stable;fake-family-new", "fake-target-z"),
            ("release/fake-latest", "fake-family-stable;fake-family-new", ""),
        )
        for pytorch_git_ref, families, excluded_targets in cases:
            with self.subTest(pytorch_git_ref=pytorch_git_ref):
                matrix = m.generate_pytorch_matrix_for_release_type(
                    release_type="dev",
                    python_versions=["3.12"],
                    pytorch_git_refs=[pytorch_git_ref],
                    amdgpu_families="fake-family-stable;fake-family-new",
                    platform="linux",
                )

                self.assertEqual(len(matrix), 1)
                self.assertEqual(matrix[0]["amdgpu_families"], families)
                self.assertEqual(
                    matrix[0]["exclude_implicit_targets"], excluded_targets
                )

    def test_unknown_explicit_ref_keeps_families(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev",
            python_versions=["3.12"],
            pytorch_git_refs=["users/alice/fake-bringup"],
            amdgpu_families="fake-family-stable;fake-family-new",
            platform="linux",
        )

        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["pytorch_git_ref"], "users/alice/fake-bringup")
        self.assertEqual(
            matrix[0]["amdgpu_families"], "fake-family-stable;fake-family-new"
        )
        self.assertEqual(matrix[0]["exclude_implicit_targets"], "")

    def test_fully_unsupported_family_omits_only_affected_ref(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev",
            python_versions=["3.12"],
            pytorch_git_refs=["release/fake-old", "release/fake-new"],
            amdgpu_families="fake-family-new",
            platform="linux",
        )
        self.assertEqual(
            [(row["pytorch_git_ref"], row["amdgpu_families"]) for row in matrix],
            [("release/fake-new", "fake-family-new")],
        )

    def test_family_filtering_matches_exact_names_case_insensitively(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev",
            python_versions=["3.12"],
            pytorch_git_refs=["release/fake-old"],
            amdgpu_families="FAKE-FAMILY-NEW;fake-family-new-extra;fake-family-stable",
            platform="linux",
        )
        self.assertEqual(len(matrix), 1)
        self.assertEqual(
            matrix[0]["amdgpu_families"],
            "fake-family-new-extra;fake-family-stable",
        )

    def test_resolve_excluded_implicit_targets_cli(self):
        cases = (
            ("linux", "release/fake-old", "fake-target-a;fake-target-z"),
            ("linux", "release/fake-new", "fake-target-z"),
            ("linux", "release/fake-latest", ""),
            ("linux", "users/alice/fake-bringup", ""),
            ("windows", "release/fake-old", "fake-target-windows"),
            ("windows", "release/fake-new", ""),
            ("windows", "users/alice/fake-bringup", ""),
        )
        for platform, pytorch_git_ref, expected in cases:
            with self.subTest(platform=platform, pytorch_git_ref=pytorch_git_ref):
                output = StringIO()
                with redirect_stdout(output):
                    rc = m.main(
                        [
                            "--platform",
                            platform,
                            "--resolve-excluded-implicit-targets",
                            pytorch_git_ref,
                        ]
                    )
                self.assertEqual(rc, 0)
                self.assertEqual(output.getvalue().strip(), expected)

    def test_release_uses_primary_python_version_for_standard_tests(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="nightly",
            pytorch_git_refs=["release/fake-old"],
            amdgpu_families="fake-family-stable",
            platform="linux",
        )

        self.assertEqual(
            [(row["python_version"], row["test_level"]) for row in matrix],
            [
                ("3.11", "standard"),
                ("3.12", "none"),
                ("3.13", "none"),
            ],
        )

    def test_each_ref_uses_its_own_primary_test_python_version(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="nightly",
            pytorch_git_refs=["release/fake-old", "release/fake-new"],
            amdgpu_families="fake-family-stable",
            platform="linux",
        )
        self.assertEqual(
            [
                (row["pytorch_git_ref"], row["python_version"])
                for row in matrix
                if row["test_level"] == "standard"
            ],
            [("release/fake-old", "3.11"), ("release/fake-new", "3.12")],
        )

    def test_unknown_ref_uses_nightly_primary_test_python_version(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev",
            pytorch_git_refs=["users/alice/fake-bringup"],
            amdgpu_families="fake-family-stable",
            platform="linux",
        )
        self.assertEqual(
            [(row["python_version"], row["test_level"]) for row in matrix],
            [("3.11", "standard"), ("3.12", "none"), ("3.13", "none")],
        )

    def test_ci_keeps_standard_tests_for_non_primary_python_version(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="ci",
            python_versions=["3.13"],
            pytorch_git_refs=["release/fake-old"],
            amdgpu_families="fake-family-stable",
            platform="linux",
        )
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["python_version"], "3.13")
        self.assertEqual(matrix[0]["pytorch_git_ref"], "release/fake-old")
        self.assertEqual(matrix[0]["test_level"], "standard")

    def test_empty_families_produce_no_rows(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev", amdgpu_families=" ; ; ", platform="linux"
        )
        self.assertEqual(matrix, [])

    def test_unknown_release_type_and_platform_raise(self):
        for release_type, platform in (("unknown", "linux"), ("dev", "unknown")):
            with self.subTest(release_type=release_type, platform=platform):
                with self.assertRaises(ValueError):
                    m.generate_pytorch_matrix_for_release_type(
                        release_type=release_type,
                        platform=platform,
                        amdgpu_families="fake-family-stable",
                    )

    def test_summary_explains_an_all_none_matrix(self):
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="dev",
            python_versions=["3.12"],
            pytorch_git_refs=["release/fake-old", "nightly"],
            amdgpu_families="fake-family-stable",
            platform="linux",
        )

        summary = m.format_matrix_summary(
            release_type="dev",
            platform="linux",
            python_versions=["3.12"],
            pytorch_git_refs=["release/fake-old", "nightly"],
            amdgpu_families="fake-family-stable",
            matrix=matrix,
        )

        self.assertIn("| Generated rows | 2 (`none`: 2) |", summary)
        self.assertIn("`release/fake-old`: `3.11`", summary)
        self.assertIn("`nightly`: `3.11`", summary)
        self.assertIn("All generated rows use `none`", summary)
        self.assertIn(
            "| `3.12` | `release/fake-old` | `fake-family-stable` | `none` |",
            summary,
        )

    def test_resolve_cli_pytorch_git_refs_all_and_empty_use_defaults(self):
        for raw in ("", "all", "ALL", "release/fake-new all"):
            with self.subTest(raw=raw):
                self.assertIsNone(
                    m._resolve_cli_pytorch_git_refs(
                        raw, release_type="nightly", platform="linux"
                    )
                )

    def test_resolve_cli_pytorch_git_refs_subset_passthrough(self):
        self.assertEqual(
            m._resolve_cli_pytorch_git_refs(
                "release/fake-old, release/fake-new",
                release_type="nightly",
                platform="linux",
            ),
            ["release/fake-old", "release/fake-new"],
        )

    def test_single_ref_covers_all_release_python_versions(self):
        # Selecting one PyTorch ref builds it against every release Python
        # version (1 ref x all Python).
        matrix = m.generate_pytorch_matrix_for_release_type(
            release_type="nightly",
            pytorch_git_refs=["release/fake-new"],
            amdgpu_families="fake-family-stable",
            platform="linux",
        )

        self.assertEqual(
            {row["pytorch_git_ref"] for row in matrix}, {"release/fake-new"}
        )
        self.assertEqual(
            [row["python_version"] for row in matrix],
            ["3.11", "3.12", "3.13"],
        )

    def test_resolve_cli_pytorch_git_refs_rejects_unknown_ref_for_non_dev(self):
        # Non-dev release types reject refs outside the available set.
        with self.assertRaises(ValueError):
            m._resolve_cli_pytorch_git_refs(
                "release/9.99", release_type="nightly", platform="linux"
            )

    def test_resolve_cli_pytorch_git_refs_dev_passes_through_custom_branch(self):
        # dev is the developer bring-up path: arbitrary refs flow through so a
        # custom branch/tag/SHA can opt into new GPU families early.
        self.assertEqual(
            m._resolve_cli_pytorch_git_refs(
                "users/alice/fake-bringup",
                release_type="dev",
                platform="linux",
            ),
            ["users/alice/fake-bringup"],
        )
        self.assertEqual(
            m._resolve_cli_pytorch_git_refs(
                "release/fake-new, users/alice/fake-bringup",
                release_type="dev",
                platform="linux",
            ),
            ["release/fake-new", "users/alice/fake-bringup"],
        )

    def test_resolve_cli_pytorch_git_refs_non_dev_rejects_custom_branch(self):
        # Outside dev, a custom branch is not a known release ref and is
        # rejected so production releases only build supported refs.
        with self.assertRaises(ValueError):
            m._resolve_cli_pytorch_git_refs(
                "users/alice/fake-bringup",
                release_type="nightly",
                platform="linux",
            )

    def test_resolve_cli_pytorch_git_refs_is_release_type_aware(self):
        # "nightly" is a valid ref for nightly releases but not for prerelease.
        self.assertEqual(
            m._resolve_cli_pytorch_git_refs(
                "nightly", release_type="nightly", platform="linux"
            ),
            ["nightly"],
        )
        with self.assertRaises(ValueError):
            m._resolve_cli_pytorch_git_refs(
                "nightly", release_type="prerelease", platform="linux"
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
                    pytorch_git_refs=["release/fake-old"],
                    amdgpu_families="fake-family-stable",
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


class ConfiguredPytorchReleaseMatrixTest(unittest.TestCase):
    def test_current_policy_generates_nonempty_matrices_without_duplicate_rows(self):
        # Exercise the real configuration without pinning its entries. Logic
        # and filtering expectations belong to the synthetic scenarios above.
        for platform in m.CI_PYTORCH_REFS:
            for release_type in m.RELEASE_TYPES:
                with self.subTest(platform=platform, release_type=release_type):
                    matrix = m.generate_pytorch_matrix_for_release_type(
                        release_type=release_type,
                        platform=platform,
                        amdgpu_families="all",
                    )
                    self.assertTrue(matrix)
                    combinations = {
                        (row["python_version"], row["pytorch_git_ref"])
                        for row in matrix
                    }
                    self.assertEqual(len(matrix), len(combinations))


if __name__ == "__main__":
    unittest.main()
