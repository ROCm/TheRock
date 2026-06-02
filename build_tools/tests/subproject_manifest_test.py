#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "artifact_subprojects.json"

CMAKE_ARGS = [
    "-GNinja",
    "-DTHEROCK_AMDGPU_FAMILIES=gfx1100",  # Required by cmake, but doesn't affect manifest
    "-DTHEROCK_ENABLE_ALL=ON",
    "-DTHEROCK_BUNDLE_SYSDEPS=OFF",
    "-DTHEROCK_ENABLE_LIBHIPCXX=OFF",
]


class SubprojectManifestTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            subprocess.run(["cmake", "--version"], capture_output=True, check=True)
            subprocess.run(["ninja", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise unittest.SkipTest("cmake or ninja not available")

        cls.temp_dir = tempfile.mkdtemp(prefix="therock-manifest-test-")
        result = subprocess.run(
            ["cmake", "-B", cls.temp_dir, "-S", str(REPO_ROOT)] + CMAKE_ARGS,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
            raise unittest.SkipTest(f"CMake configure failed: {result.stderr}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_manifest_is_fresh(self):
        with MANIFEST_PATH.open() as f:
            committed = json.load(f)
        with (Path(self.temp_dir) / "artifact_subprojects.json").open() as f:
            generated = json.load(f)

        committed_norm = {k: sorted(v) for k, v in sorted(committed.items())}
        generated_norm = {k: sorted(v) for k, v in sorted(generated.items())}

        self.assertEqual(
            committed_norm,
            generated_norm,
            "Run 'python build_tools/generate_subproject_manifest.py' to update",
        )


if __name__ == "__main__":
    unittest.main()
