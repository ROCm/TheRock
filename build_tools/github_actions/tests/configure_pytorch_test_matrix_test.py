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
    def test_empty_family_list_returns_empty_matrix(self) -> None:
        matrix = m.build_test_matrix(
            amdgpu_families=[],
            platform="linux",
        )
        self.assertEqual(matrix, {"include": []})

    def test_known_family_without_runner_is_skipped(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ):
            matrix = m.build_test_matrix(
                amdgpu_families=["gfxnorunner"],
                platform="linux",
            )
        self.assertEqual(matrix, {"include": []})

    def test_family_match_is_platform_specific(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ):
            matrix = m.build_test_matrix(
                amdgpu_families=["gfxalpha-all"],
                platform="linux",
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
                    "--platform",
                    "linux",
                ]
            )

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(outputs["enabled"], "true")
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
                    "--platform",
                    "linux",
                ]
            )

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(outputs["enabled"], "false")
        self.assertEqual(json.loads(outputs["matrix"]), {"include": []})

    def test_real_family_matrix_finds_gfx950_runner(self) -> None:
        matrix = m.build_test_matrix(
            amdgpu_families=["gfx950-dcgpu"],
            platform="linux",
        )
        include = matrix["include"]
        self.assertEqual(len(include), 1)
        self.assertEqual(include[0]["amdgpu_family"], "gfx950-dcgpu")
        self.assertTrue(include[0]["test_runs_on"])


if __name__ == "__main__":
    unittest.main()
