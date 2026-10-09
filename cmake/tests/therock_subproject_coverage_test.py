#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests the link flags of the coverage CMAKE_PROJECT_INCLUDE file that
therock_subproject.cmake generates for each instrumented subproject."""

import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

THEROCK_ROOT = Path(__file__).resolve().parents[2]

# The subset of TheRock's CMake modules that therock_subproject.cmake needs to
# be usable outside the superproject.
HARNESS_INCLUDES = (
    "therock_globals",
    "therock_sanitizers",
    "therock_flag_utils",
    "therock_default_targets",
    "therock_subproject",
)

# Every entry point into the GPU half of the profile runtime.
STUB_ENTRY_POINTS = (
    "__llvm_profile_offload_register_shadow_variable",
    "__llvm_profile_offload_register_section_shadow_variable",
    "__llvm_profile_offload_register_dynamic_module",
    "__llvm_profile_offload_unregister_dynamic_module",
    "__llvm_profile_hip_collect_device_data",
)


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def write_harness(source_dir: Path, toolchain_root: Path) -> None:
    """Declares one subproject instrumented for host coverage only and one for
    device coverage too, both from the same sources. Self-instrumented, so
    neither needs an upstream coverage option."""
    includes = "\n".join(f"include({name})" for name in HARNESS_INCLUDES)
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        # C, for the subprojects' toolchain files to hand on.
        project(therock_subproject_coverage_harness LANGUAGES C)

        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
        list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)

        set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/rocm_build_flags_state.cmake")
        file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")

        {includes}

        set(THEROCK_COVERAGE_SELF_INSTRUMENTED_PROJECTS host_project device_project)
        set(HOST_PROJECT_ENABLE_COVERAGE ON)
        set(DEVICE_PROJECT_ENABLE_COVERAGE ON)
        set(DEVICE_PROJECT_ENABLE_DEVICE_COVERAGE ON)
        foreach(_name IN ITEMS host_project device_project)
          therock_cmake_subproject_declare(${{_name}}
            EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/project"
            BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/${{_name}}"
            CMAKE_ARGS "-DTHEROCK_TOOLCHAIN_ROOT={toolchain_root.as_posix()}"
          )
          therock_cmake_subproject_activate(${{_name}})
        endforeach()
        """,
    )
    # The project() in the subdirectory runs the include a second time.
    write_file(
        source_dir / "project" / "CMakeLists.txt",
        """
        cmake_minimum_required(VERSION 3.25)
        project(coverage_project LANGUAGES C)
        add_subdirectory(sub)
        """,
    )
    write_file(
        source_dir / "project" / "sub" / "CMakeLists.txt",
        """
        project(coverage_sub LANGUAGES C)
        foreach(_kind IN ITEMS EXE SHARED MODULE)
          file(WRITE "${CMAKE_BINARY_DIR}/${_kind}_linker_flags.txt"
            "${CMAKE_${_kind}_LINKER_FLAGS}")
        endforeach()
        """,
    )


def run(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise AssertionError(" ".join(args) + "\n" + result.stdout)
    return result


@unittest.skipIf(sys.platform == "win32", "coverage is not instrumented on Windows")
@unittest.skipUnless(
    shutil.which("cc") and shutil.which("nm") and shutil.which("ninja"),
    "needs a host C compiler, nm and ninja",
)
class CoverageIncludeLinkFlagsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.source_dir = root / "source"
        self.build_dir = root / "build"
        self.hip_runtime = root / "toolchain" / "lib" / "libamdhip64.so"
        write_file(self.hip_runtime, "")
        write_harness(self.source_dir, self.hip_runtime.parents[1])
        run("cmake", "-S", str(self.source_dir), "-B", str(self.build_dir), "-GNinja")

    def tearDown(self):
        self._tmp.cleanup()

    def test_host_only_project_links_the_stub_and_the_hip_runtime_first(self):
        run(
            "cmake",
            "--build",
            str(self.build_dir),
            "--target",
            "host_project+configure",
        )
        project_build = self.build_dir / "host_project" / "build"
        stub = project_build / "therock_coverage_profile_stub.o"
        hip_first = (
            f"-Wl,--push-state,--as-needed {self.hip_runtime.as_posix()} "
            "-Wl,--pop-state"
        )
        for kind in ("EXE", "SHARED", "MODULE"):
            with self.subTest(kind=kind):
                flags = (project_build / f"{kind}_linker_flags.txt").read_text()
                self.assertEqual(flags.count(stub.as_posix()), 1, flags)
                self.assertEqual(flags.count(hip_first), 1, flags)
                self.assertLess(flags.index(stub.as_posix()), flags.index(hip_first))

        defined = run("nm", "--defined-only", str(stub)).stdout.split()
        for entry_point in STUB_ENTRY_POINTS:
            self.assertIn(entry_point, defined)

    def test_device_coverage_project_keeps_the_real_gpu_half(self):
        include = (self.build_dir / "device_project" / "_coverage.cmake").read_text()
        self.assertNotIn("therock_coverage_profile_stub", include)
        self.assertIn("libamdhip64.so", include)


if __name__ == "__main__":
    unittest.main()
