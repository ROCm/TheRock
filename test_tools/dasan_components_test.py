# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression checks for CI artifact reuse and runner GPU isolation."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build_tools"))
import dasan_components as build

spec = importlib.util.spec_from_file_location(
    "component_validation", ROOT / "test_tools/dasan_components/run_validation.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class ComponentReuseTest(unittest.TestCase):
    def test_ci_visibility_is_preserved_including_empty_masks(self):
        masks = {
            "ROCR_VISIBLE_DEVICES": "3,5",
            "HIP_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
            "GPU_DEVICE_ORDINAL": "0,1",
        }
        with patch.dict(os.environ, masks):
            env = validation.environment(ROOT, None)
        self.assertEqual({key: env[key] for key in masks}, masks)

    def test_wrong_baseline_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "wrong.tar.xz"
            archive.write_bytes(b"not the requested CI artifact")
            with self.assertRaisesRegex(RuntimeError, "differs from successful CI"):
                build.import_installation(archive, root)
            self.assertFalse((root / "baseline").exists())

    def test_reuse_verification_detects_payload_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "build/dist/rocm/lib/compiler-rt.a"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"downloaded CI runtime")
            build.save(
                root / "baseline-files.json",
                {"lib/compiler-rt.a": build.file_record(binary)},
            )
            build.verify(root)
            binary.write_bytes(b"unexpected rebuild")
            with self.assertRaisesRegex(RuntimeError, "reuse audit failed"):
                build.verify(root)
            result = json.loads((root / "reuse-verification.json").read_text())
            self.assertEqual(result["changed_or_missing"], ["lib/compiler-rt.a"])

    def test_reuse_verification_rejects_compiler_build_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build.save(root / "baseline-files.json", {})
            ninja = root / "build/compiler/amd-llvm/build/build.ninja"
            ninja.parent.mkdir(parents=True)
            ninja.touch()
            with self.assertRaisesRegex(RuntimeError, "reuse audit failed"):
                build.verify(root)


if __name__ == "__main__":
    unittest.main()
