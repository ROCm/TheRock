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


class ConfigurePyTorchTestMatrixTest(unittest.TestCase):
    def test_empty_family_list_disables_matrix(self) -> None:
        matrix = m.build_test_matrix(
            amdgpu_families=[],
            platform="linux",
            package_index_url="https://example.com/whl/",
        )
        self.assertEqual(matrix, {"include": []})

    def test_auto_uses_built_families(self) -> None:
        build_families, test_families = m.resolve_requested_test_families(
            build_amdgpu_families="gfx950",
            test_amdgpu_families="auto",
        )
        self.assertEqual(build_families, ["gfx950"])
        self.assertEqual(test_families, ["gfx950"])

    def test_empty_test_families_uses_built_families(self) -> None:
        _build_families, test_families = m.resolve_requested_test_families(
            build_amdgpu_families="gfx950",
            test_amdgpu_families="",
        )
        self.assertEqual(test_families, ["gfx950"])

    def test_none_skips_quick_tests(self) -> None:
        _build_families, test_families = m.resolve_requested_test_families(
            build_amdgpu_families="gfx950",
            test_amdgpu_families="none",
        )
        self.assertEqual(test_families, [])

    def test_gfx950_target_builds_quick_test_matrix(self) -> None:
        matrix = m.build_test_matrix(
            amdgpu_families=["gfx950"],
            platform="linux",
            package_index_url="https://example.com/whl/",
        )
        self.assertEqual(
            matrix,
            {
                "include": [
                    {
                        "amdgpu_family": "gfx950-dcgpu",
                        "test_runs_on": "linux-gfx950-1gpu-ccs-ossci-rocm",
                        "package_index_url": "https://example.com/whl/",
                    }
                ]
            },
        )

    def test_unknown_family_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "not-a-family"):
            m.build_test_matrix(
                amdgpu_families=["not-a-family"],
                platform="linux",
                package_index_url="https://example.com/whl/",
            )

    def test_missing_package_index_url_errors_when_tests_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "--package-index-url"):
            m.build_test_matrix(
                amdgpu_families=["gfx950"],
                platform="linux",
                package_index_url="",
            )

    def test_main_writes_outputs(self) -> None:
        with mock.patch.object(
            m, "gha_set_output"
        ) as gha_set_output, mock.patch.object(m, "gha_append_step_summary"):
            m.main(
                [
                    "--build-amdgpu-families",
                    "gfx950",
                    "--test-amdgpu-families",
                    "gfx950",
                    "--package-index-url",
                    "https://example.com/whl/",
                ]
            )

        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(outputs["enabled"], "true")
        matrix = json.loads(outputs["matrix"])
        self.assertEqual(matrix["include"][0]["amdgpu_family"], "gfx950-dcgpu")


if __name__ == "__main__":
    unittest.main()
