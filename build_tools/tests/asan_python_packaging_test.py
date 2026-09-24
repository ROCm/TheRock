# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifacts import ArtifactCatalog
from build_python_packages import find_asan_runtime_rpath, resolve_package_version


class AsanVersionResolutionTest(unittest.TestCase):
    def test_version_defaults_to_release_asan(self):
        args = argparse.Namespace(asan=True, version="")
        manifest = {"rocm_version": "10.1.0"}

        self.assertEqual(
            resolve_package_version(args, manifest),
            "10.1.0+asan",
        )

    def test_explicit_channel_asan_version_must_match_artifact_base(self):
        args = argparse.Namespace(
            asan=True,
            version="7.15.0a20260807+asan",
        )
        with self.assertRaisesRegex(ValueError, "artifact base 10.1.0"):
            resolve_package_version(args, {"rocm_version": "10.1.0"})

    def test_explicit_channel_asan_version_is_accepted(self):
        args = argparse.Namespace(asan=True, version="10.1.0.dev0+abcdef.asan")
        self.assertEqual(
            resolve_package_version(args, {"rocm_version": "10.1.0"}),
            "10.1.0.dev0+abcdef.asan",
        )


class AsanRuntimeDiscoveryTest(unittest.TestCase):
    def _write_runtime(self, artifact_dir: Path, *runtime_parts: str) -> None:
        artifact = artifact_dir / "amd-llvm_lib_generic"
        stage = artifact / "compiler" / "amd-llvm" / "stage"
        runtime = stage.joinpath(*runtime_parts)
        runtime.parent.mkdir(parents=True)
        runtime.touch()
        (artifact / "artifact_manifest.txt").write_text("compiler/amd-llvm/stage\n")

    def test_finds_clang_resource_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            self._write_runtime(
                artifact_dir,
                "lib",
                "llvm",
                "lib",
                "clang",
                "23",
                "lib",
                "linux",
                "libclang_rt.asan-x86_64.so",
            )

            self.assertEqual(
                find_asan_runtime_rpath(ArtifactCatalog(artifact_dir)),
                "lib/llvm/lib/clang/23/lib/linux",
            )

    def test_finds_per_target_runtime_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            self._write_runtime(
                artifact_dir,
                "lib",
                "llvm",
                "lib",
                "clang",
                "20",
                "lib",
                "x86_64-unknown-linux-gnu",
                "libclang_rt.asan.so",
            )

            self.assertEqual(
                find_asan_runtime_rpath(ArtifactCatalog(artifact_dir)),
                "lib/llvm/lib/clang/20/lib/x86_64-unknown-linux-gnu",
            )

    def test_prefers_per_target_directory_when_both_layouts_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            artifact = artifact_dir / "amd-llvm_lib_generic"
            stage = artifact / "compiler" / "amd-llvm" / "stage"
            clang_lib = stage / "lib" / "llvm" / "lib" / "clang" / "20" / "lib"
            linux = clang_lib / "linux" / "libclang_rt.asan-x86_64.so"
            per_target = clang_lib / "x86_64-unknown-linux-gnu" / "libclang_rt.asan.so"
            linux.parent.mkdir(parents=True)
            per_target.parent.mkdir(parents=True)
            linux.touch()
            per_target.touch()
            (artifact / "artifact_manifest.txt").write_text("compiler/amd-llvm/stage\n")

            self.assertEqual(
                find_asan_runtime_rpath(ArtifactCatalog(artifact_dir)),
                "lib/llvm/lib/clang/20/lib/x86_64-unknown-linux-gnu",
            )

    def test_missing_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "no shared Clang ASAN"):
                find_asan_runtime_rpath(ArtifactCatalog(Path(temp_dir)))


if __name__ == "__main__":
    unittest.main()
