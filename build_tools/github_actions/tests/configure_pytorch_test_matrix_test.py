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
            "fetch-gfx-targets": ["gfxalpha0"],
            "test-runs-on": "linux-alpha",
        }
    },
    "gfxbeta": {
        "windows": {
            "family": "gfxbeta-all",
            "fetch-gfx-targets": ["gfxbeta0"],
            "test-runs-on": "windows-beta",
        }
    },
    "gfxnorunner": {
        "linux": {
            "family": "gfxnorunner",
            "fetch-gfx-targets": ["gfxnorunner0"],
            "test-runs-on": "",
        }
    },
}


def _fake_family_matrix(_trigger_types: list[str]) -> FamilyMatrix:
    return FAKE_FAMILY_MATRIX


class ConfigurePyTorchTestMatrixTest(unittest.TestCase):
    def test_empty_family_list_disables_matrix(self) -> None:
        matrix = m.build_test_matrix(
            amdgpu_families=[],
            platform="linux",
        )
        self.assertEqual(matrix, {"include": []})

    def test_auto_uses_built_families(self) -> None:
        test_families = m.resolve_requested_test_families(
            build_amdgpu_families="gfx950",
            test_amdgpu_families="auto",
        )
        self.assertEqual(test_families, ["gfx950"])

    def test_empty_test_families_uses_built_families(self) -> None:
        test_families = m.resolve_requested_test_families(
            build_amdgpu_families="gfx950",
            test_amdgpu_families="",
        )
        self.assertEqual(test_families, ["gfx950"])

    def test_none_skips_tests(self) -> None:
        test_families = m.resolve_requested_test_families(
            build_amdgpu_families="gfx950",
            test_amdgpu_families="none",
        )
        self.assertEqual(test_families, [])

    def test_gfx_target_builds_linux_test_matrix(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ):
            matrix = m.build_test_matrix(
                amdgpu_families=["gfxalpha0"],
                platform="linux",
            )
        self.assertEqual(
            matrix,
            {
                "include": [
                    {
                        "amdgpu_family": "gfxalpha0",
                        "test_runs_on": "linux-alpha",
                    }
                ]
            },
        )

    def test_windows_target_builds_windows_test_matrix(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ):
            matrix = m.build_test_matrix(
                amdgpu_families=["gfxbeta0"],
                platform="windows",
            )
        self.assertEqual(
            matrix,
            {
                "include": [
                    {
                        "amdgpu_family": "gfxbeta0",
                        "test_runs_on": "windows-beta",
                    }
                ]
            },
        )

    def test_known_family_without_runner_is_skipped(self) -> None:
        with mock.patch.object(
            m, "get_all_families_for_trigger_types", side_effect=_fake_family_matrix
        ):
            matrix = m.build_test_matrix(
                amdgpu_families=["gfxnorunner"],
                platform="linux",
            )
        self.assertEqual(matrix, {"include": []})

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
                    "gfxalpha0",
                    "--test-amdgpu-families",
                    "gfxalpha0",
                    "--platform",
                    "linux",
                ]
            )

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(outputs["enabled"], "true")
        matrix = json.loads(outputs["matrix"])
        self.assertEqual(matrix["include"][0]["amdgpu_family"], "gfxalpha0")

    def test_real_family_matrix_resolves_gfx950(self) -> None:
        matrix = m.build_test_matrix(
            amdgpu_families=["gfx950"],
            platform="linux",
        )
        include = matrix["include"]
        self.assertEqual(len(include), 1)
        self.assertEqual(include[0]["amdgpu_family"], "gfx950-dcgpu")
        self.assertTrue(include[0]["test_runs_on"])


if __name__ == "__main__":
    unittest.main()
