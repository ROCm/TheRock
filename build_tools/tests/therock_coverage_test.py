# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for `cmake/therock_coverage.cmake`.

These drive a miniature superproject through `therock_cmake_subproject_activate`
and inspect the `project_init.cmake` it generates for the sub-project, which is
where the coverage link options land.
"""

import re
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
    "therock_coverage",
    "therock_subproject",
)


def write_harness(source_dir: Path) -> None:
    """Writes a one-subproject superproject into `source_dir`.

    `main_project` is given an EXTERNAL_SOURCE_DIR under a stand-in
    rocm-libraries tree so that the monorepo group flags resolve for it the way
    they would for a real component.
    """
    includes = "\n".join(f"include({name})" for name in HARNESS_INCLUDES)
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        # C is enabled because FindThreads, which decides whether the profile
        # runtime needs a separate thread library, is a hard error without it.
        project(therock_coverage_harness C)

        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
        list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)

        # therock_subproject.cmake fingerprints this file. The harness declares
        # no flags, so an empty state file is enough.
        set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/rocm_build_flags_state.cmake")
        file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")

        {includes}

        set(THEROCK_ROCM_LIBRARIES_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/rocm-libraries")
        therock_coverage_init()

        therock_cmake_subproject_declare(main_project
          EXTERNAL_SOURCE_DIR "${{THEROCK_ROCM_LIBRARIES_SOURCE_DIR}}/projects/main"
          BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/main"
        )
        therock_cmake_subproject_activate(main_project)
        """,
    )
    write_file(
        source_dir / "rocm-libraries" / "projects" / "main" / "CMakeLists.txt",
        """
        cmake_minimum_required(VERSION 3.25)
        project(main_project NONE)
        """,
    )


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


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


def probe_profile_runtime_deps() -> list[str]:
    """Returns the libraries the coverage stanza is expected to link here.

    The answer is platform-dependent -- libdl and libpthread are folded into
    libc from glibc 2.34, and neither applies on Windows -- so it is taken from
    CMake rather than assumed, and the same values the stanza is built from are
    the ones asserted against.
    """
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        source_dir = temp_dir / "source"
        write_file(
            source_dir / "CMakeLists.txt",
            """
            cmake_minimum_required(VERSION 3.25)
            project(therock_coverage_probe C)
            find_package(Threads QUIET)
            message(STATUS "PROBE=[${CMAKE_DL_LIBS}][${CMAKE_THREAD_LIBS_INIT}]")
            """,
        )
        output = run(
            "cmake", "-S", str(source_dir), "-B", str(temp_dir / "build"), "-GNinja"
        ).stdout

    match = re.search(r"PROBE=\[(.*)\]\[(.*)\]", output)
    if not match:
        raise AssertionError(f"probe did not report its libraries:\n{output}")
    return [lib for lib in match.groups() if lib]


class CoverageInitStanzaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected_link_libraries = (
            f"link_libraries({' '.join(probe_profile_runtime_deps())})"
        )

    def _project_init_contents(self, *cmake_args: str) -> str:
        """Configures the harness and returns main_project's project_init file."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            build_dir = temp_dir / "build"
            write_harness(source_dir)

            run(
                "cmake",
                "-S",
                str(source_dir),
                "-B",
                str(build_dir),
                "-GNinja",
                *cmake_args,
            )

            # The file is named from the sub-project's build dir, so find it
            # rather than restating the naming scheme here.
            init_files = list((build_dir / "main").glob("*_init.cmake"))
            self.assertEqual(
                len(init_files),
                1,
                f"expected one project_init file, got {init_files}",
            )
            return init_files[0].read_text(encoding="utf-8")

    def test_no_coverage_does_not_link_profile_runtime_deps(self):
        # Control: an uninstrumented sub-project must not pick up link options it
        # has no use for.
        self.assertNotIn("link_libraries", self._project_init_contents())

    def test_per_project_option_links_profile_runtime_deps(self):
        # Regression: the profile runtime that -fprofile-instr-generate links
        # calls into libdl (dlsym, dladdr) and libpthread (pthread_once,
        # pthread_getattr_np), which are separate libraries before glibc 2.34. A
        # shared library missing them still links, and the first executable to
        # link against that library then fails --no-allow-shlib-undefined.
        self.assertIn(
            self.expected_link_libraries,
            self._project_init_contents("-DMAIN_PROJECT_ENABLE_COVERAGE=ON"),
        )

    def test_project_list_links_profile_runtime_deps(self):
        # The path CI actually takes: coverage_nightly.yml passes a project list
        # rather than the per-project option.
        self.assertIn(
            self.expected_link_libraries,
            self._project_init_contents("-DTHEROCK_COVERAGE_PROJECTS=main_project"),
        )

    def test_monorepo_group_flag_links_profile_runtime_deps(self):
        self.assertIn(
            self.expected_link_libraries,
            self._project_init_contents("-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON"),
        )

    def test_explicit_opt_out_beats_group_flag(self):
        # A nightly run instruments a whole monorepo except for one component,
        # so an explicit OFF has to win over the group flag.
        self.assertNotIn(
            "link_libraries",
            self._project_init_contents(
                "-DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON",
                "-DMAIN_PROJECT_ENABLE_COVERAGE=OFF",
            ),
        )


if __name__ == "__main__":
    unittest.main()
