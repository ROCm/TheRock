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

import platform
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


BASELINE_SUPPORTED = platform.system() == "Linux" and platform.machine().lower() in (
    "x86_64",
    "amd64",
)


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def write_harness(
    source_dir: Path,
    *,
    main_project: str = "main_project",
    deferred_builds: tuple[str, ...] = (),
) -> None:
    """Writes a two-subproject superproject into `source_dir`.

    `main_project` declares `RUNTIME_DEPS dep_project`, so its dist directory
    must contain `lib/dep.txt` (installed by `dep_project`) in addition to
    whatever is in its own stage directory.
    """
    includes = "\n".join(f"include({name})" for name in HARNESS_INCLUDES)
    deferred_args = " ".join(f'"{name}"' for name in deferred_builds)
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

        therock_cmake_subproject_declare({main_project}
          EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/main"
          BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/main"
          RUNTIME_DEPS dep_project
          HARDENING_DEFERRED {deferred_args}
        )
        therock_cmake_subproject_activate({main_project})

        if(COVERAGE_TEST_WRAPPERS)
          # Avoid downloads and packaging while exercising the real wrapper
          # declarations and parent-level coverage diagnostics.
          function(therock_subproject_fetch target)
            file(WRITE "${{CMAKE_CURRENT_BINARY_DIR}}/source/CMakeLists.txt"
              "cmake_minimum_required(VERSION 3.25)\\nproject(download_fixture NONE)\\n")
            file(TOUCH "${{CMAKE_CURRENT_BINARY_DIR}}/source/download.stamp")
          endfunction()
          function(therock_provide_artifact)
          endfunction()
          add_custom_target(therock-third-party)
          add_subdirectory("${{THEROCK_SOURCE_DIR}}/third-party/fftw3" fftw)
          add_subdirectory("${{THEROCK_SOURCE_DIR}}/third-party/openmpi" openmpi)
        endif()
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
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.source_dir = Path(temp_dir.name) / "source"
        self.build_dir = Path(temp_dir.name) / "build"

    def _configure_main_project(
        self,
        *,
        prebuilt: bool,
        hardening_policy: str = "OFF",
        main_project: str = "main_project",
        deferred_builds: tuple[str, ...] = (),
        cmake_args: tuple[str, ...] = (),
    ) -> str:
        """Configures the harness and returns its diagnostics without building."""
        write_harness(
            self.source_dir,
            main_project=main_project,
            deferred_builds=deferred_builds,
        )
        if prebuilt:
            mark_prebuilt(self.build_dir)
        return run(
            "cmake",
            "-S",
            str(self.source_dir),
            "-B",
            str(self.build_dir),
            "-GNinja",
            f"-DTHEROCK_HARDENING_POLICY={hardening_policy}",
            *cmake_args,
        ).stdout

    def _stage_main_project(self, *, prebuilt: bool) -> list[str]:
        """Builds `main_project+stage` and returns the resulting dist contents."""
        self._configure_main_project(prebuilt=prebuilt)
        run("cmake", "--build", str(self.build_dir), "--target", "main_project+stage")
        dist_dir = self.build_dir / "main" / "dist"
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

    @unittest.skipUnless(BASELINE_SUPPORTED, "BASELINE requires Linux x86-64")
    def test_prebuilt_hardening_coverage_is_reported(self) -> None:
        # Artifact provenance takes precedence over source-build declarations.
        configure_output = self._configure_main_project(
            prebuilt=True,
            hardening_policy="BASELINE",
            deferred_builds=("inner build",),
        )
        self.assertIn(
            "Hardening coverage for prebuilt subproject main_project: unverified",
            configure_output,
        )
        self.assertNotIn("no TheRock policy injection", configure_output)

    @unittest.skipUnless(BASELINE_SUPPORTED, "BASELINE requires Linux x86-64")
    def test_deferred_builds_are_reported_during_top_level_configure(self) -> None:
        """Keeps every declared gap visible without running child configure.

        Child-only reporting can disappear into suppressed logs or be skipped
        in retained builds. Repeating BASELINE must still report the gaps;
        OFF and projects without declared gaps must remain quiet.
        """
        deferred_builds = ("first (Meson)", "second (autotools)")
        for policy in ("OFF", "BASELINE", "BASELINE", "OFF"):
            with self.subTest(policy=policy):
                output = self._configure_main_project(
                    prebuilt=False,
                    hardening_policy=policy,
                    deferred_builds=deferred_builds,
                )
                for name in deferred_builds:
                    diagnostic = (
                        f"Hardening coverage for subproject main_project ({name}): "
                        "deferred (no TheRock policy injection)"
                    )
                    if policy == "BASELINE":
                        self.assertIn(diagnostic, output)
                    else:
                        self.assertNotIn("Hardening coverage", output)
                self.assertNotIn("subproject dep_project:", output)
                self.assertFalse(
                    (self.build_dir / "main" / "build" / "CMakeCache.txt").exists()
                )

    @unittest.skipUnless(BASELINE_SUPPORTED, "BASELINE requires Linux x86-64")
    def test_wrapper_coverage_follows_the_selected_fftw_build_route(self) -> None:
        """Reports actual independent builds without downloading their sources.

        FFTW's native CMake branch receives target flags; only its autotools
        branch needs deferral. OpenMPI always uses an independent build.
        """
        for autotools in ("OFF", "ON"):
            for policy in ("OFF", "BASELINE"):
                with self.subTest(autotools=autotools, policy=policy):
                    output = self._configure_main_project(
                        prebuilt=False,
                        hardening_policy=policy,
                        cmake_args=(
                            "-DCOVERAGE_TEST_WRAPPERS=ON",
                            f"-DTHEROCK_FLAG_FFTW3_AUTOTOOLS_BUILD={autotools}",
                        ),
                    )
                    for project in ("therock-fftw3", "therock-fftw3f"):
                        self.assertEqual(
                            f"Hardening coverage for subproject {project} " in output,
                            policy == "BASELINE" and autotools == "ON",
                        )
                    self.assertEqual(
                        "Hardening coverage for subproject therock-openmpi " in output,
                        policy == "BASELINE",
                    )

    @unittest.skipUnless(BASELINE_SUPPORTED, "BASELINE requires Linux x86-64")
    def test_compiler_exclusion_and_prebuilt_provenance_are_reported(self) -> None:
        """Keeps deferred compiler scope separate from artifact provenance.

        Compiler exclusion cannot establish how an imported artifact was
        built, so its hardening status must still be reported as unverified.
        """
        for prebuilt in (False, True):
            with self.subTest(prebuilt=prebuilt):
                output = self._configure_main_project(
                    prebuilt=prebuilt,
                    hardening_policy="BASELINE",
                    main_project="amd-llvm",
                    deferred_builds=("inner build",),
                )
                self.assertIn("deferred compiler-group coverage", output)
                self.assertNotIn("no TheRock policy injection", output)
                self.assertEqual(
                    "Hardening coverage for prebuilt subproject amd-llvm: unverified"
                    in output,
                    prebuilt,
                )


if __name__ == "__main__":
    unittest.main()
