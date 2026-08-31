# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(THIS_DIR.parent))
sys.path.insert(0, os.fspath(THIS_DIR.parent.parent))

import configure_pytorch_test_matrix as m
from workflow_utils import (
    WORKFLOWS_DIR,
    get_choice_options,
    get_workflow_job,
    load_workflow,
)


FamilyMatrix = dict[str, dict[str, dict[str, object]]]


FAKE_FAMILY_MATRIX: FamilyMatrix = {
    "gfxalpha": {
        "linux": {
            "family": "gfxalpha-all",
            "test-runs-on": "linux-alpha",
        },
        "windows": {
            "family": "gfxalpha-all",
            "test-runs-on": "windows-alpha",
        },
    },
    "gfxnorunner": {
        "linux": {
            "family": "gfxnorunner",
            "test-runs-on": "",
        }
    },
}


def _fake_family_matrix(_trigger_types: list[str]) -> FamilyMatrix:
    return FAKE_FAMILY_MATRIX


class ConfigurePyTorchTestMatrixTest(unittest.TestCase):
    def test_direct_build_workflows_default_to_standard_tests(self) -> None:
        expected_options = {
            "multi_arch_build_portable_linux_pytorch_wheels.yml": [
                "sanity",
                "standard",
                "full",
            ],
            "multi_arch_build_windows_pytorch_wheels.yml": [
                "sanity",
                "standard",
            ],
        }

        for workflow_filename, options in expected_options.items():
            with self.subTest(workflow_filename=workflow_filename):
                workflow = load_workflow(WORKFLOWS_DIR / workflow_filename)
                on_block = workflow.get("on") or workflow[True]
                dispatch_input = on_block["workflow_dispatch"]["inputs"]["test_level"]
                call_input = on_block["workflow_call"]["inputs"]["test_level"]

                self.assertEqual(dispatch_input["default"], "standard")
                self.assertEqual(call_input["default"], "standard")
                self.assertEqual(get_choice_options(workflow, "test_level"), options)

    def test_full_dispatch_uses_resolved_test_level_only(self) -> None:
        workflow = load_workflow(
            WORKFLOWS_DIR / "multi_arch_build_portable_linux_pytorch_wheels.yml"
        )
        dispatch_job = get_workflow_job(workflow, "dispatch_pytorch_wheels_full_test")

        self.assertEqual(
            dispatch_job["if"],
            "${{ needs.configure_pytorch_tests.outputs.test_level == 'full' }}",
        )
        self.assertIn("configure_pytorch_tests", dispatch_job["needs"])

    def test_empty_family_list_returns_empty_matrix(self) -> None:
        matrix = m.build_test_matrix(
            amdgpu_families=[],
            platform="linux",
            test_level="standard",
        )
        self.assertEqual(matrix, {"include": []})

    def test_known_family_without_runner_is_skipped(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ):
            matrix = m.build_test_matrix(
                amdgpu_families=["gfxnorunner"],
                platform="linux",
                test_level="standard",
            )
        self.assertEqual(matrix, {"include": []})

    def test_family_match_is_platform_specific(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ):
            matrix = m.build_test_matrix(
                amdgpu_families=["gfxalpha-all"],
                platform="linux",
                test_level="standard",
            )
        # FAKE_FAMILY_MATRIX also has a windows-alpha runner. The Linux
        # request should only use the Linux platform entry and canonical family.
        self.assertEqual(
            matrix,
            {
                "include": [
                    {
                        "amdgpu_family": "gfxalpha-all",
                        "test_runs_on": "linux-alpha",
                    }
                ]
            },
        )

    def test_unknown_family_errors(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ), self.assertRaisesRegex(ValueError, "not-a-family"):
            m.build_test_matrix(
                amdgpu_families=["not-a-family"],
                platform="linux",
                test_level="standard",
            )

    def test_main_writes_outputs(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ), mock.patch.object(m, "gha_set_output") as gha_set_output:
            m.main(
                [
                    "--build-amdgpu-families",
                    "gfxalpha-all",
                    "--test-amdgpu-families",
                    "gfxalpha-all",
                    "--python-version",
                    "3.12",
                    "--platform",
                    "linux",
                ]
            )

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(outputs["enabled"], "true")
        self.assertEqual(outputs["test_level"], "standard")
        matrix = json.loads(outputs["matrix"])
        self.assertEqual(matrix["include"][0]["amdgpu_family"], "gfxalpha-all")

    def test_main_auto_uses_built_families(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ), mock.patch.object(m, "gha_set_output") as gha_set_output:
            m.main(
                [
                    "--build-amdgpu-families",
                    "gfxalpha-all;gfxalpha-all",
                    "--test-amdgpu-families",
                    "auto",
                    "--python-version",
                    "3.12",
                    "--platform",
                    "linux",
                ]
            )

        outputs = gha_set_output.call_args.args[0]
        matrix = json.loads(outputs["matrix"])
        self.assertEqual(matrix["include"][0]["amdgpu_family"], "gfxalpha-all")

    def test_main_rejects_mixed_control_and_explicit_families(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            m.main(
                [
                    "--build-amdgpu-families",
                    "gfxalpha-all",
                    "--test-amdgpu-families",
                    "auto;gfxalpha-all",
                    "--python-version",
                    "3.12",
                    "--platform",
                    "linux",
                ]
            )

    def test_main_none_skips_tests(self) -> None:
        with mock.patch.object(m, "gha_set_output") as gha_set_output:
            m.main(
                [
                    "--build-amdgpu-families",
                    "gfxalpha-all",
                    "--test-amdgpu-families",
                    "none",
                    "--python-version",
                    "3.12",
                    "--platform",
                    "linux",
                ]
            )

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(outputs["enabled"], "false")
        self.assertEqual(json.loads(outputs["matrix"]), {"include": []})

    def test_sanity_level_skips_gpu_tests(self) -> None:
        with mock.patch.object(m, "gha_set_output") as gha_set_output:
            m.main(
                [
                    "--build-amdgpu-families",
                    "not-a-family",
                    "--test-level",
                    "sanity",
                    "--python-version",
                    "3.13",
                    "--platform",
                    "linux",
                ]
            )

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(outputs["enabled"], "false")
        self.assertEqual(outputs["test_level"], "sanity")
        self.assertEqual(json.loads(outputs["matrix"]), {"include": []})

    def test_full_level_requires_existing_full_test_configuration(self) -> None:
        self.assertEqual(
            m.resolve_requested_test_level(
                requested_test_level="full",
                python_version="3.12",
                platform="linux",
                amdgpu_families=["gfx94X-dcgpu"],
            ),
            "full",
        )

        for python_version, platform, families in (
            ("3.13", "linux", ["gfx94X-dcgpu"]),
            ("3.12", "windows", ["gfx94X-dcgpu"]),
            ("3.12", "linux", ["gfx110X-all"]),
        ):
            with self.subTest(
                python_version=python_version,
                platform=platform,
                families=families,
            ):
                self.assertEqual(
                    m.resolve_requested_test_level(
                        requested_test_level="full",
                        python_version=python_version,
                        platform=platform,
                        amdgpu_families=families,
                    ),
                    "standard",
                )

    def test_full_level_includes_standard_gpu_matrix(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ):
            matrix = m.build_test_matrix(
                amdgpu_families=["gfxalpha-all"],
                platform="linux",
                test_level="full",
            )

        self.assertEqual(len(matrix["include"]), 1)

    def test_real_family_matrix_finds_gfx950_runner(self) -> None:
        matrix = m.build_test_matrix(
            amdgpu_families=["gfx950-dcgpu"],
            platform="linux",
            test_level="standard",
        )
        include = matrix["include"]
        self.assertEqual(len(include), 1)
        self.assertEqual(include[0]["amdgpu_family"], "gfx950-dcgpu")
        self.assertTrue(include[0]["test_runs_on"])


if __name__ == "__main__":
    unittest.main()
