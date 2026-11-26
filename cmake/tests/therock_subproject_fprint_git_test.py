#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests how git state affects subproject source fingerprints.

A subproject's fingerprint includes the identity of its source directory. For a
git source that is its commit, and the fingerprint is dropped when the source is
dirty. Sources that were downloaded into an ignored build directory carry a
stable `.<name>.smrev` fingerprint instead. They are not tracked by git, but
they sit inside TheRock's own work tree, so an enclosing repository is always
found above them.

The fixtures cover a downloaded source, a tracked source, and an unrelated
tracked file. The tests check that editing the unrelated file does not
invalidate the fingerprint of the downloaded source, and that editing a tracked
source still does.

They drive `cmake/therock_subproject.cmake` through a miniature superproject
inside a throwaway git repository rather than TheRock itself, so they need no
submodules, no compiler, and no bundled sysdeps.
"""

import shutil
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

# Stable fingerprint of the simulated downloaded source.
SMREV = "0123456789abcdef"
FINGERPRINT_PROJECTS = ("downloaded_project", "tracked_project")


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def write_source_project(source_dir: Path, project: str) -> None:
    """Writes a minimal project that a subproject can use as its source."""
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        project({project} NONE)
        """,
    )


def write_fixture(repo_dir: Path) -> None:
    """Writes a repository with a downloaded source and a tracked source.

    The layout mirrors TheRock, whose `build/` directory is ignored:
      CMakeLists.txt, unrelated.txt   tracked
      tracked_src/                    tracked source directory
      build/downloaded/source         ignored, downloaded source
      build/downloaded/.source.smrev  its stable fingerprint
    """
    includes = "\n".join(f"include({name})" for name in HARNESS_INCLUDES)
    write_file(
        repo_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        project(therock_subproject_fprint_git_harness NONE)

        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
        list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)

        # therock_subproject.cmake fingerprints this file. The harness
        # declares no flags, so an empty state file is enough.
        set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/rocm_build_flags_state.cmake")
        file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")

        {includes}

        therock_cmake_subproject_declare(downloaded_project
          EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/build/downloaded/source"
          BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/downloaded"
        )
        therock_cmake_subproject_activate(downloaded_project)

        therock_cmake_subproject_declare(tracked_project
          EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/tracked_src"
          BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/tracked"
        )
        therock_cmake_subproject_activate(tracked_project)

        # Record each fingerprint ("" means it could not be computed).
        foreach(_project downloaded_project tracked_project)
          get_target_property(_fprint "${{_project}}" THEROCK_FPRINT)
          if(NOT _fprint)
            set(_fprint "")
          endif()
          file(WRITE "${{CMAKE_BINARY_DIR}}/${{_project}}.fprint" "${{_fprint}}")
        endforeach()
        """,
    )
    write_file(repo_dir / ".gitignore", "build/\n")
    write_file(repo_dir / "unrelated.txt", "unrelated\n")
    write_file(repo_dir / "tracked_src" / "lib.txt", "source\n")
    write_source_project(repo_dir / "tracked_src", "tracked_project")
    write_source_project(
        repo_dir / "build" / "downloaded" / "source", "downloaded_project"
    )
    (repo_dir / "build" / "downloaded" / ".source.smrev").write_text(SMREV)


def run_command(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(args)}\n{result.stdout}")
    return result


def git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Runs git with a fixed identity, independent of the user's git config."""
    return run_command(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        *args,
        cwd=repo_dir,
    )


def configure(source_dir: Path, build_dir: Path) -> None:
    run_command(
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-GNinja",
        cwd=source_dir,
    )


def read_fingerprints(build_dir: Path) -> dict[str, str]:
    """Returns each project's fingerprint; "" means none could be computed."""
    return {
        project: (build_dir / f"{project}.fprint").read_text(encoding="utf-8")
        for project in FINGERPRINT_PROJECTS
    }


@unittest.skipIf(shutil.which("git") is None, "git is not available")
class SubprojectFprintGitTest(unittest.TestCase):
    """Exercises fingerprints of downloaded and tracked sources."""

    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.source_dir = Path(temp_dir.name)
        self.build_dir = self.source_dir / "build" / "out"
        write_fixture(self.source_dir)
        git(self.source_dir, "init", "-q")
        git(self.source_dir, "add", "-A")
        git(self.source_dir, "commit", "-q", "--no-verify", "-m", "initial")

    def configure_and_read(self) -> dict[str, str]:
        configure(self.source_dir, self.build_dir)
        return read_fingerprints(self.build_dir)

    def test_clean_tree_has_valid_fingerprints(self) -> None:
        """Both a downloaded and a tracked source get a fingerprint."""
        fingerprints = self.configure_and_read()

        self.assertTrue(fingerprints["downloaded_project"])
        self.assertTrue(fingerprints["tracked_project"])

    def test_unrelated_edit_keeps_downloaded_source_fingerprint(self) -> None:
        """Editing a tracked file that no source uses must not drop the fingerprint."""
        clean = self.configure_and_read()

        (self.source_dir / "unrelated.txt").write_text("edited\n")
        dirty = self.configure_and_read()

        self.assertTrue(dirty["downloaded_project"])
        self.assertEqual(dirty["downloaded_project"], clean["downloaded_project"])

    def test_edit_inside_tracked_source_invalidates_its_fingerprint(self) -> None:
        """A real change to a tracked source must still be detected as dirty."""
        self.assertTrue(self.configure_and_read()["tracked_project"])

        (self.source_dir / "tracked_src" / "lib.txt").write_text("edited\n")
        fingerprints = self.configure_and_read()

        self.assertFalse(fingerprints["tracked_project"])


if __name__ == "__main__":
    unittest.main()
