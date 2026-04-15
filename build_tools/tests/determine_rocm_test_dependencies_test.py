# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from determine_rocm_test_dependencies import SubprojectDependencyAnalyzer


class SubprojectDependencyAnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.build_dir = Path(self.temp_dir) / "build"
        self.build_dir.mkdir()

    def tearDown(self):
        import shutil

        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def write_manifest(self, manifest_data: dict):
        manifest_file = self.build_dir / "subproject_test_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest_data, f)

    def test_test_subprojects_overrides_reverse_deps(self):
        """test_subprojects limits which tests run."""
        manifest = {
            "subprojects": {
                "rocBLAS": {
                    "runtime_deps": [],
                    "test_subprojects": ["hipBLAS", "rocSOLVER"],
                },
                "hipBLAS": {"runtime_deps": ["rocBLAS"]},
                "rocSOLVER": {"runtime_deps": ["rocBLAS"]},
                "MIOpen": {"runtime_deps": ["rocBLAS"]},
            }
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        result = analyzer.get_subprojects_to_test(["rocBLAS"])
        self.assertEqual(result, {"rocBLAS", "hipBLAS", "rocSOLVER"})

    def test_without_test_subprojects_uses_all_dependents(self):
        """Without test_subprojects, all direct dependents are tested."""
        manifest = {
            "subprojects": {
                "rocRAND": {"runtime_deps": []},
                "MIOpen": {"runtime_deps": ["rocRAND"]},
                "other": {"runtime_deps": ["rocRAND"]},
            }
        }
        self.write_manifest(manifest)
        analyzer = SubprojectDependencyAnalyzer(
            self.build_dir / "subproject_test_manifest.json"
        )

        result = analyzer.get_subprojects_to_test(["rocRAND"])
        self.assertEqual(result, {"rocRAND", "MIOpen", "other"})


if __name__ == "__main__":
    unittest.main()
