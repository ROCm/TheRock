# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifacts import ArtifactCatalog
from build_python_packages import (
    _rpath_resolves_directory,
    find_asan_runtime_rpath,
    resolve_package_version,
    validate_asan_runtime_resolution,
)


class AsanVersionResolutionTest(unittest.TestCase):
    def test_version_defaults_to_artifact_nightly_date(self):
        args = argparse.Namespace(asan=True, asan_build_id=None, version="")
        manifest = {
            "rocm_version": "10.1.0",
            "rocm_package_version": "10.1.0a20260807",
        }

        self.assertEqual(
            resolve_package_version(args, manifest),
            "10.1.0+asan.20260807",
        )

    def test_explicit_asan_version_must_match_artifact_base(self):
        args = argparse.Namespace(
            asan=True,
            asan_build_id=None,
            version="7.15.0+asan.20260807",
        )
        with self.assertRaisesRegex(ValueError, r"10\.1\.0\+asan"):
            resolve_package_version(args, {"rocm_version": "10.1.0"})

    def test_build_id_requires_asan_mode(self):
        args = argparse.Namespace(
            asan=False,
            asan_build_id="20260807",
            version="10.1.0",
        )
        with self.assertRaisesRegex(ValueError, "requires --asan"):
            resolve_package_version(args, {"rocm_version": "10.1.0"})


class AsanRuntimeDiscoveryTest(unittest.TestCase):
    def test_finds_clang_resource_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            artifact = artifact_dir / "base_lib_generic"
            stage = artifact / "base" / "aux-overlay" / "stage"
            runtime = (
                stage
                / "lib"
                / "llvm"
                / "lib"
                / "clang"
                / "23"
                / "lib"
                / "linux"
                / "libclang_rt.asan-x86_64.so"
            )
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            (artifact / "artifact_manifest.txt").write_text("base/aux-overlay/stage\n")

            self.assertEqual(
                find_asan_runtime_rpath(ArtifactCatalog(artifact_dir)),
                "lib/llvm/lib/clang/23/lib/linux",
            )

    def test_missing_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "no shared Clang ASAN"):
                find_asan_runtime_rpath(ArtifactCatalog(Path(temp_dir)))


class AsanRpathValidationTest(unittest.TestCase):
    def test_origin_rpath_resolves_cross_wheel_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary = root / "_rocm_sdk_libraries" / "lib" / "libfoo.so"
            runtime_dir = (
                root
                / "_rocm_sdk_core"
                / "lib"
                / "llvm"
                / "lib"
                / "clang"
                / "23"
                / "lib"
                / "linux"
            )

            self.assertTrue(
                _rpath_resolves_directory(
                    binary_path=binary,
                    rpaths=[
                        "$ORIGIN/../../_rocm_sdk_core/"
                        "lib/llvm/lib/clang/23/lib/linux"
                    ],
                    expected_dir=runtime_dir,
                )
            )

    def test_origin_rpath_resolves_separate_wheel_staging_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profiler_platform = root / "rocm-profiler" / "platform"
            core_platform = root / "rocm-sdk-core" / "platform"
            binary = profiler_platform / "_rocm_profiler" / "bin" / "rocprof"
            runtime_dir = (
                core_platform
                / "_rocm_sdk_core"
                / "lib"
                / "llvm"
                / "lib"
                / "clang"
                / "23"
                / "lib"
                / "linux"
            )

            self.assertTrue(
                _rpath_resolves_directory(
                    binary_path=binary,
                    rpaths=[
                        "$ORIGIN/../../_rocm_sdk_core/"
                        "lib/llvm/lib/clang/23/lib/linux"
                    ],
                    expected_dir=runtime_dir,
                    binary_platform_root=profiler_platform,
                    expected_platform_root=core_platform,
                )
            )

    def test_validator_projects_separate_wheels_into_site_packages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            core_platform = root / "rocm-sdk-core" / "platform"
            core_dir = core_platform / "_rocm_sdk_core"
            profiler_platform = root / "rocm-profiler" / "platform"
            profiler_dir = profiler_platform / "_rocm_profiler"
            runtime_rpath = "lib/llvm/lib/clang/23/lib/linux"
            runtime_dir = core_dir / runtime_rpath
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "libclang_rt.asan-x86_64.so").touch()

            binary = profiler_dir / "bin" / "rocprof"
            binary.parent.mkdir(parents=True)
            binary.touch()
            package = types.SimpleNamespace(
                platform_dir=profiler_dir,
                files=types.SimpleNamespace(
                    materialized_relpaths={"bin/rocprof": (None, binary)}
                ),
            )
            core = types.SimpleNamespace(platform_dir=core_dir)

            with mock.patch(
                "build_python_packages._elf_dynamic_info",
                return_value=(
                    ["libclang_rt.asan-x86_64.so"],
                    [
                        "$ORIGIN/../../_rocm_sdk_core/"
                        "lib/llvm/lib/clang/23/lib/linux"
                    ],
                ),
            ):
                validate_asan_runtime_resolution(
                    core=core,
                    packages=[package],
                    runtime_rpath=runtime_rpath,
                    require_instrumented=True,
                )

    def test_validator_accepts_resolvable_instrumented_elf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            core_dir = root / "_rocm_sdk_core"
            runtime_rpath = "lib/llvm/lib/clang/23/lib/linux"
            runtime_dir = core_dir / runtime_rpath
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "libclang_rt.asan-x86_64.so").touch()

            binary = root / "_rocm_sdk_libraries" / "lib" / "libfoo.so"
            binary.parent.mkdir(parents=True)
            binary.touch()
            package = types.SimpleNamespace(
                platform_dir=root / "_rocm_sdk_libraries",
                files=types.SimpleNamespace(
                    materialized_relpaths={"lib/libfoo.so": (None, binary)}
                )
            )
            core = types.SimpleNamespace(platform_dir=core_dir)
            dynamic_info = (
                ["libclang_rt.asan-x86_64.so"],
                ["$ORIGIN/../../_rocm_sdk_core/" "lib/llvm/lib/clang/23/lib/linux"],
            )

            with mock.patch(
                "build_python_packages._elf_dynamic_info",
                return_value=dynamic_info,
            ):
                validate_asan_runtime_resolution(
                    core=core,
                    packages=[package],
                    runtime_rpath=runtime_rpath,
                    require_instrumented=True,
                )

    def test_validator_rejects_unresolved_instrumented_elf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            core_dir = root / "_rocm_sdk_core"
            runtime_rpath = "lib/llvm/lib/clang/23/lib/linux"
            runtime_dir = core_dir / runtime_rpath
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "libclang_rt.asan-x86_64.so").touch()

            binary = root / "_rocm_sdk_libraries" / "lib" / "libfoo.so"
            binary.parent.mkdir(parents=True)
            binary.touch()
            package = types.SimpleNamespace(
                platform_dir=root / "_rocm_sdk_libraries",
                files=types.SimpleNamespace(
                    materialized_relpaths={"lib/libfoo.so": (None, binary)}
                )
            )
            core = types.SimpleNamespace(platform_dir=core_dir)

            with mock.patch(
                "build_python_packages._elf_dynamic_info",
                return_value=(
                    ["libclang_rt.asan-x86_64.so"],
                    ["$ORIGIN/llvm/lib/clang/23/lib/linux"],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot resolve"):
                    validate_asan_runtime_resolution(
                        core=core,
                        packages=[package],
                        runtime_rpath=runtime_rpath,
                        require_instrumented=True,
                    )


if __name__ == "__main__":
    unittest.main()
