#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that a prebuilt subproject stage respects RUNTIME_DEPS ordering.

A subproject whose stage directory is imported from an artifact (marked by a
`<stage_dir>.prebuilt` file) still copies its RUNTIME_DEPS stage directories
into its own dist directory. Those deps may be built from source in the same
build -- that is the normal shape of a staged CI build, where
`artifact_manager.py fetch --bootstrap` marks the inbound artifacts prebuilt
while the stage's own subprojects build normally.

See https://github.com/ROCm/TheRock/issues/7690.

These tests drive `cmake/therock_subproject.cmake` through a miniature
superproject rather than TheRock itself, so they need no submodules, no
compiler, and no bundled sysdeps.
"""

import subprocess
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


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def write_harness(source_dir: Path) -> None:
    """Writes a two-subproject superproject into `source_dir`.

    `main_project` declares `RUNTIME_DEPS dep_project`, so its dist directory
    must contain `lib/dep.txt` (installed by `dep_project`) in addition to
    whatever is in its own stage directory.
    """
    includes = "\n".join(f"include({name})" for name in HARNESS_INCLUDES)
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        project(therock_subproject_prebuilt_harness NONE)

        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
        list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)

        # therock_subproject.cmake fingerprints this file. The harness declares
        # no flags, so an empty state file is enough.
        set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/rocm_build_flags_state.cmake")
        file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")

        {includes}

        therock_cmake_subproject_declare(dep_project
          EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/dep"
          BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/dep"
        )
        therock_cmake_subproject_activate(dep_project)

        therock_cmake_subproject_declare(main_project
          EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/main"
          BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/main"
          RUNTIME_DEPS dep_project
        )
        therock_cmake_subproject_activate(main_project)
        """,
    )
    write_file(
        source_dir / "dep" / "CMakeLists.txt",
        """
        cmake_minimum_required(VERSION 3.25)
        project(dep_project NONE)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dep.txt" "dep\\n")
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dep.txt" DESTINATION lib)
        """,
    )
    write_file(
        source_dir / "main" / "CMakeLists.txt",
        """
        cmake_minimum_required(VERSION 3.25)
        project(main_project NONE)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main.txt" "main\\n")
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/main.txt" DESTINATION lib)
        """,
    )


def mark_prebuilt(build_dir: Path) -> None:
    """Imports `main_project`'s stage from an "artifact".

    This mirrors what `ArtifactPopulator` in `build_tools/artifact_manager.py`
    does under `--bootstrap`: populate the stage directory and drop a
    `<stage_dir>.prebuilt` marker beside it, before CMake configures.
    """
    stage_dir = build_dir / "main" / "stage"
    write_file(stage_dir / "lib" / "main.txt", "main\n")
    (build_dir / "main" / "stage.prebuilt").touch()


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


class PrebuiltRuntimeDepsTest(unittest.TestCase):
    def _stage_main_project(self, *, prebuilt: bool) -> list[str]:
        """Builds `main_project+stage` and returns its dist contents."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            build_dir = temp_dir / "build"
            write_harness(source_dir)
            if prebuilt:
                mark_prebuilt(build_dir)

            run("cmake", "-S", str(source_dir), "-B", str(build_dir), "-GNinja")
            run("cmake", "--build", str(build_dir), "--target", "main_project+stage")

            dist_dir = build_dir / "main" / "dist"
            return sorted(
                p.relative_to(dist_dir).as_posix()
                for p in dist_dir.rglob("*")
                if p.is_file()
            )

    def test_built_subproject_dist_contains_runtime_deps(self):
        # Control: when main_project is built normally, its dist picks up the
        # runtime dep. This is the behavior the prebuilt path must match.
        self.assertEqual(
            self._stage_main_project(prebuilt=False),
            ["lib/dep.txt", "lib/main.txt"],
        )

    def test_prebuilt_subproject_dist_contains_runtime_deps(self):
        # Regression: the prebuilt stage rule copies from dep_project's stage
        # directory, so it must also depend on dep_project having been staged.
        # Without that dependency the copy runs first and silently produces a
        # dist directory holding only the prebuilt artifact's own files.
        self.assertEqual(
            self._stage_main_project(prebuilt=True),
            ["lib/dep.txt", "lib/main.txt"],
        )


class ToolchainFlagsTest(unittest.TestCase):
    def test_sanitizer_flags_and_launchers_reach_subprojects(self):
        for sanitizer in ("TSAN", "HOST_ASAN", "ASAN", "OFF"):
            with self.subTest(
                sanitizer=sanitizer
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                includes = "\n".join(f"include({name})" for name in HARNESS_INCLUDES)
                write_file(
                    root / "parent/CMakeLists.txt",
                    f"""
                    cmake_minimum_required(VERSION 3.25)
                    project(toolchain_flags NONE)
                    set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
                    list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
                    {includes}
                    add_custom_target(hip-clr)
                    set_target_properties(hip-clr PROPERTIES
                      THEROCK_SUBPROJECT cmake THEROCK_DIST_DIR "${{CMAKE_BINARY_DIR}}/llvm")
                    file(MAKE_DIRECTORY "${{CMAKE_BINARY_DIR}}/llvm/lib/llvm/lib/clang/24")
                    add_custom_target(child)
                    set_target_properties(child PROPERTIES THEROCK_AMDGPU_TARGETS "gfx942;gfx1100")
                    foreach(language C CXX HIP)
                      set(CMAKE_${{language}}_FLAGS "-DKEEP_${{language}}")
                      set(CMAKE_${{language}}_COMPILER_LAUNCHER "ccache;--verbose")
                    endforeach()
                    set(THEROCK_SANITIZER {sanitizer})
                    _therock_cmake_subproject_setup_toolchain(child amd-hip
                      "${{CMAKE_BINARY_DIR}}/toolchain.cmake")
                    """,
                )
                write_file(
                    root / "child/CMakeLists.txt",
                    """
                    cmake_minimum_required(VERSION 3.25)
                    project(observe_flags NONE)
                    include("${TOOLCHAIN}")
                    file(WRITE "${CMAKE_BINARY_DIR}/observed.txt" "${GPU_TARGETS}\n")
                    foreach(language C CXX HIP)
                      file(APPEND "${CMAKE_BINARY_DIR}/observed.txt"
                        "${CMAKE_${language}_FLAGS_INIT}\n${CMAKE_${language}_COMPILER_LAUNCHER}\n")
                    endforeach()
                    """,
                )
                run(
                    "cmake",
                    "-S",
                    str(root / "parent"),
                    "-B",
                    str(root / "parent-build"),
                    "-GNinja",
                )
                run(
                    "cmake",
                    "-S",
                    str(root / "child"),
                    "-B",
                    str(root / "child-build"),
                    "-GNinja",
                    f"-DTOOLCHAIN={root / 'parent-build/toolchain.cmake'}",
                )
                targets, *observed = (
                    (root / "child-build/observed.txt").read_text().splitlines()
                )
                self.assertEqual(
                    targets,
                    (
                        "gfx942:xnack+;gfx1100"
                        if sanitizer == "ASAN"
                        else "gfx942;gfx1100"
                    ),
                )
                for index, language in enumerate(("C", "CXX", "HIP")):
                    flags, launcher = observed[index * 2 : index * 2 + 2]
                    self.assertIn(f"-DKEEP_{language}", flags)
                    self.assertTrue(launcher.endswith("ccache;--verbose"))
                    host_only = sanitizer in ("TSAN", "HOST_ASAN")
                    self.assertEqual("CCACHE_DISABLE=1" in launcher, host_only)
                    if sanitizer == "OFF" or (
                        language == "HIP" and sanitizer == "ASAN"
                    ):
                        self.assertNotIn("-fsanitize=", flags)
                    else:
                        kind = "thread" if sanitizer == "TSAN" else "address"
                        prefix = "-Xarch_host " if host_only else ""
                        self.assertIn(f"{prefix}-fsanitize={kind}", flags)
                        self.assertIn(f"{prefix}-fno-omit-frame-pointer", flags)


if __name__ == "__main__":
    unittest.main()
