#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for downstream_cmake_smoke.py. No ROCm prefix or cmake required."""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

LINUX_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LINUX_DIR))

import downstream_cmake_smoke as sdk  # noqa: E402

PROJECT_CMAKE = sdk.REPO_ROOT / "examples" / "cpp-sdk-user" / "CMakeLists.txt"
DRIVER_CMAKE = sdk.REPO_ROOT / "examples" / "clean_configure_test_project.cmake"
GROUPS = sdk.parse_enable_groups(PROJECT_CMAKE.read_text())

SAMPLE_PROJECT = """
if(THEROCK_ENABLE_HIP)
    find_package(hip CONFIG REQUIRED)
endif()
if(THEROCK_ENABLE_BLAS)
    find_package(hipblaslt CONFIG REQUIRED)
    find_package(hipblas CONFIG REQUIRED)
    find_package(rocblas CONFIG REQUIRED)
endif()
if(THEROCK_ENABLE_HIPTENSOR)
    find_package(hiptensor CONFIG REQUIRED)
endif()
"""


def _write_config(prefix: Path, package: str, filename: str | None = None) -> None:
    """Write a one-line CMake config file under ``prefix/lib/cmake/<package>``."""
    folder = prefix / "lib" / "cmake" / package
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (filename or f"{package}-config.cmake")).write_text("# test\n")


def _all_packages(prefix: Path) -> None:
    """Write a config file for every package the real cpp-sdk-user project finds."""
    for _flag, group in GROUPS:
        for package in group:
            if package == "rocWMMA":
                _write_config(prefix, package, "rocWMMAConfig.cmake")
            else:
                _write_config(prefix, package)


class EnableFlagTests(unittest.TestCase):
    def test_full_prefix_enables_every_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            _all_packages(prefix)
            flags = {
                flag.name: flag.value
                for flag in sdk.enable_flags(
                    sdk.installed_cmake_packages(prefix, GROUPS), GROUPS
                )
            }
        self.assertEqual(set(flags), {name for name, _group in GROUPS})
        self.assertTrue(all(value == "ON" for value in flags.values()))

    def test_absent_hiptensor_is_off_and_hip_stays_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            _all_packages(prefix)
            hiptensor = prefix / "lib" / "cmake" / "hiptensor"
            for path in hiptensor.iterdir():
                path.unlink()
            hiptensor.rmdir()
            flags = {
                flag.name: flag
                for flag in sdk.enable_flags(
                    sdk.installed_cmake_packages(prefix, GROUPS), GROUPS
                )
            }
        self.assertEqual(flags["THEROCK_ENABLE_HIPTENSOR"].value, "OFF")
        self.assertEqual(flags["THEROCK_ENABLE_HIPTENSOR"].missing, ("hiptensor",))
        self.assertEqual(flags["THEROCK_ENABLE_HIP"].value, "ON")
        self.assertEqual(flags["THEROCK_ENABLE_BLAS"].value, "ON")

    def test_absent_group_is_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            _all_packages(prefix)
            for package in ("hipfft", "rocfft"):
                folder = prefix / "lib" / "cmake" / package
                for path in folder.iterdir():
                    path.unlink()
                folder.rmdir()
            flags = {
                flag.name: flag
                for flag in sdk.enable_flags(
                    sdk.installed_cmake_packages(prefix, GROUPS), GROUPS
                )
            }
        self.assertEqual(flags["THEROCK_ENABLE_FFT"].value, "OFF")
        self.assertEqual(flags["THEROCK_ENABLE_FFT"].missing, ("hipfft", "rocfft"))

    def test_missing_hip_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required CMake packages: hip"):
            sdk.enable_flags({"hipblaslt"}, GROUPS)

    def test_missing_hipblaslt_fails(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "missing required CMake packages: hipblaslt"
        ):
            sdk.enable_flags({"hip"}, GROUPS)

    def test_partial_group_fails(self) -> None:
        packages = {"hip", "hipblaslt", "hipblas", "rocblas", "rocprim"}
        with self.assertRaisesRegex(
            ValueError, "THEROCK_ENABLE_PRIM is only partly installed"
        ):
            sdk.enable_flags(packages, GROUPS)

    def test_similar_config_names_do_not_satisfy_required_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            _write_config(prefix, "hip-lang", "hip-lang-config.cmake")
            _write_config(prefix, "hipblaslt", "hipblaslt-package-config.cmake")
            names = sdk.installed_cmake_packages(prefix, GROUPS)
        self.assertEqual(names, set())
        with self.assertRaisesRegex(ValueError, "hip"):
            sdk.enable_flags(names, GROUPS)

    def test_rocwmma_config_cmake_name(self) -> None:
        known = {"rocWMMA"}
        self.assertEqual(
            sdk.package_name_from_config(Path("rocWMMAConfig.cmake"), known),
            "rocWMMA",
        )
        self.assertEqual(
            sdk.package_name_from_config(Path("rocwmma-config.cmake"), known),
            "rocWMMA",
        )
        self.assertIsNone(sdk.package_name_from_config(Path("CMakeLists.txt"), known))

    def test_lowercase_rocwmma_file_enables_the_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            for package in ("hip", "hipblaslt", "hipblas", "rocblas"):
                _write_config(prefix, package)
            _write_config(prefix, "rocwmma", "rocwmma-config.cmake")
            flags = {
                flag.name: flag.value
                for flag in sdk.enable_flags(
                    sdk.installed_cmake_packages(prefix, GROUPS), GROUPS
                )
            }
        self.assertEqual(flags["THEROCK_ENABLE_ROCWMMA"], "ON")

    def test_nested_win32_package_stays_in_its_group(self) -> None:
        groups = dict(
            sdk.parse_enable_groups(
                """
                if(THEROCK_ENABLE_SPARSE)
                    if(NOT WIN32)
                        find_package(hipsparselt CONFIG REQUIRED)
                    endif()
                    find_package(hipsparse CONFIG REQUIRED)
                endif()
                """
            )
        )
        self.assertEqual(groups["THEROCK_ENABLE_SPARSE"], ("hipsparselt", "hipsparse"))

    def test_real_project_packages_come_from_the_cmake_file(self) -> None:
        text = PROJECT_CMAKE.read_text()
        found = re.findall(r"find_package\s*\(\s*([A-Za-z0-9_]+)", text)
        parsed = [package for _flag, group in GROUPS for package in group]
        self.assertCountEqual(parsed, found)
        by_flag = dict(GROUPS)
        self.assertIn("hipblaslt", by_flag["THEROCK_ENABLE_BLAS"])
        self.assertIn("hipsparselt", by_flag["THEROCK_ENABLE_SPARSE"])
        self.assertEqual(by_flag["THEROCK_ENABLE_ROCWMMA"], ("rocWMMA",))
        propagate_block = re.search(
            r"set\(\s*propagate_vars(.*?)\)",
            DRIVER_CMAKE.read_text(),
            re.S,
        )
        self.assertIsNotNone(propagate_block)
        propagated = set(
            re.findall(r"\bTHEROCK_ENABLE_[A-Za-z0-9_]+\b", propagate_block.group(1))
        )
        self.assertLessEqual({name for name, _group in GROUPS}, propagated)


class CommandTests(unittest.TestCase):
    def test_command_uses_driver_and_prefix_env(self) -> None:
        flags = [
            sdk.EnableFlag("THEROCK_ENABLE_HIP", "ON"),
            sdk.EnableFlag("THEROCK_ENABLE_HIPTENSOR", "OFF", ("hiptensor",)),
        ]
        command = sdk.build_command(
            cmake="cmake",
            source_dir=Path("/src/examples/cpp-sdk-user"),
            binary_dir=Path("/tmp/cpp-sdk-user"),
            driver=Path("/src/examples/clean_configure_test_project.cmake"),
            prefix=Path("/opt/rocm/core"),
            generator="Unix Makefiles",
            flags=flags,
        )
        self.assertEqual(
            command[:5],
            ["cmake", "-E", "env", "CMAKE_PREFIX_PATH=/opt/rocm/core", "--"],
        )
        self.assertIn("-DSOURCE_DIR=/src/examples/cpp-sdk-user", command)
        self.assertIn("-DBINARY_DIR=/tmp/cpp-sdk-user", command)
        self.assertIn("-DTHEROCK_ENABLE_HIP=ON", command)
        self.assertIn("-DTHEROCK_ENABLE_HIPTENSOR=OFF", command)
        self.assertNotIn("-DENABLE_DEVICE_TEST=ON", command)
        self.assertEqual(
            command[-2:], ["-P", "/src/examples/clean_configure_test_project.cmake"]
        )
        self.assertNotIn("-DCMAKE_HIP_ARCHITECTURES=gfx942", command)

    def test_prefix_clang_is_the_host_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            clangxx = prefix / "lib" / "llvm" / "bin" / "clang++"
            clangxx.parent.mkdir(parents=True)
            clangxx.write_text("")
            self.assertEqual(sdk.host_cxx_compiler(prefix), clangxx.as_posix())
            self.assertEqual(
                sdk.host_cxx_compiler(prefix, "/usr/bin/g++"), "/usr/bin/g++"
            )

    def test_missing_prefix_clang_leaves_the_compiler_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(sdk.host_cxx_compiler(Path(tmp)))

    def test_command_pins_one_hip_architecture(self) -> None:
        command = sdk.build_command(
            cmake="cmake",
            source_dir=Path("/src/examples/cpp-sdk-user"),
            binary_dir=Path("/tmp/cpp-sdk-user"),
            driver=Path("/src/examples/clean_configure_test_project.cmake"),
            prefix=Path("/opt/rocm/core"),
            generator="Unix Makefiles",
            flags=[sdk.EnableFlag("THEROCK_ENABLE_HIP", "ON")],
            hip_arch="gfx942",
        )
        self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx942", command)

    def test_family_group_pins_its_first_target(self) -> None:
        self.assertEqual(sdk.first_hip_architecture("gfx94X-dcgpu"), "gfx942")
        self.assertEqual(sdk.first_hip_architecture("gfx942"), "gfx942")
        self.assertEqual(
            sdk.first_hip_architecture("gfx94X-dcgpu;gfx950-dcgpu"), "gfx942"
        )

    def test_lowered_gfx_arch_token_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sdk.first_hip_architecture("gfx94x")

    def test_empty_hip_arch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            sdk.first_hip_architecture("  ")

    def test_build_dir_inside_prefix_fails(self) -> None:
        with tempfile.TemporaryDirectory() as prefix_dir, tempfile.TemporaryDirectory() as other:
            prefix = Path(prefix_dir)
            other_path = Path(other)
            source = other_path / "cpp-sdk-user"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(SAMPLE_PROJECT)
            driver = other_path / "clean_configure_test_project.cmake"
            driver.write_text("# driver\n")
            with self.assertRaisesRegex(ValueError, "inside the install prefix"):
                sdk.validate_layout(prefix, prefix / "build", source, driver)

    def test_main_invokes_cmake_with_hiptensor_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "prefix"
            _all_packages(prefix)
            hiptensor = prefix / "lib" / "cmake" / "hiptensor"
            for path in hiptensor.iterdir():
                path.unlink()
            hiptensor.rmdir()
            source = root / "cpp-sdk-user"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(SAMPLE_PROJECT)
            driver = root / "clean_configure_test_project.cmake"
            driver.write_text("# driver\n")
            cmake = root / "cmake"
            cmake.write_text("#!/bin/sh\n")
            binary = root / "build"
            with patch.object(sdk.subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
                code = sdk.main(
                    [
                        "--prefix",
                        str(prefix),
                        "--source-dir",
                        str(source),
                        "--driver",
                        str(driver),
                        "--cmake",
                        str(cmake),
                        "--binary-dir",
                        str(binary),
                    ]
                )
        self.assertEqual(code, 0)
        command = run.call_args.args[0]
        self.assertIn("-DTHEROCK_ENABLE_HIPTENSOR=OFF", command)
        self.assertIn("-DTHEROCK_ENABLE_HIP=ON", command)
        self.assertIn("-DTHEROCK_ENABLE_BLAS=ON", command)
        self.assertTrue(command[-1].endswith("clean_configure_test_project.cmake"))

    def test_main_returns_error_when_hipblaslt_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "prefix"
            _write_config(prefix, "hip")
            source = root / "cpp-sdk-user"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(SAMPLE_PROJECT)
            driver = root / "clean_configure_test_project.cmake"
            driver.write_text("# driver\n")
            cmake = root / "cmake"
            cmake.write_text("#!/bin/sh\n")
            with patch.object(sdk.subprocess, "run") as run:
                code = sdk.main(
                    [
                        "--prefix",
                        str(prefix),
                        "--source-dir",
                        str(source),
                        "--driver",
                        str(driver),
                        "--cmake",
                        str(cmake),
                        "--binary-dir",
                        str(root / "build"),
                    ]
                )
        self.assertEqual(code, 1)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
