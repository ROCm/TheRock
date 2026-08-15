#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(
    0,
    os.fspath(Path(__file__).parents[2] / "rocm-systems" / "shared" / "kpack" / "python"),
)

import pair_kpack_test_binaries


class PairKpackTestBinariesTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.artifacts_dir = self.tmp_dir / "artifacts"
        self.artifacts_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def write_manifest(self, artifact_dir: Path, entries: list[str]):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "artifact_manifest.txt").write_text(
            "".join(f"{entry}\n" for entry in entries)
        )

    def test_kpacked_host_binaries_move_to_arch_artifacts(self):
        generic_dir = self.artifacts_dir / "fft_test_generic"
        self.write_manifest(generic_dir, ["bin", "share"])
        (generic_dir / "bin").mkdir()
        (generic_dir / "bin" / "fft_test").write_text("kpacked host binary")
        (generic_dir / "bin" / "host_only").write_text("regular host binary")
        (generic_dir / "share").mkdir()
        (generic_dir / "share" / "data.txt").write_text("test data")

        gfx1100_dir = self.artifacts_dir / "fft_test_gfx1100"
        gfx1201_dir = self.artifacts_dir / "fft_test_gfx1201"
        self.write_manifest(gfx1100_dir, ["lib"])
        self.write_manifest(gfx1201_dir, ["lib"])

        def read_marker(path: Path):
            if path.name == "fft_test":
                return {"kpack_search_paths": ["lib"], "kernel_name": "fft_test"}
            return None

        with mock.patch.object(
            pair_kpack_test_binaries,
            "read_kpack_ref_marker",
            side_effect=read_marker,
        ):
            pair_kpack_test_binaries.run(
                argparse.Namespace(
                    artifacts_dir=self.artifacts_dir,
                    artifact_prefix="fft_test",
                )
            )

        self.assertFalse((generic_dir / "bin" / "fft_test").exists())
        self.assertTrue((generic_dir / "bin" / "host_only").exists())
        self.assertTrue((generic_dir / "share" / "data.txt").exists())

        for arch_dir in [gfx1100_dir, gfx1201_dir]:
            copied_binary = arch_dir / "bin" / "fft_test"
            self.assertEqual(copied_binary.read_text(), "kpacked host binary")
            manifest_entries = (arch_dir / "artifact_manifest.txt").read_text().splitlines()
            self.assertIn("bin", manifest_entries)
            self.assertIn("lib", manifest_entries)


if __name__ == "__main__":
    unittest.main()
