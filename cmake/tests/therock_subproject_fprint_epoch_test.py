#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that THEROCK_FPRINT_EPOCH participates in subproject fingerprints.

The fingerprint epoch is a global cache-invalidation lever: it is folded into
every subproject's fingerprint content before that content is hashed into the
``THEROCK_FPRINT`` target property. Bumping ``THEROCK_FPRINT_EPOCH`` must change
every fingerprint (invalidating every cached artifact), while leaving it
unchanged must reproduce the same fingerprint for the same sources.

These tests drive ``cmake/therock_subproject.cmake`` through a miniature
superproject rather than TheRock itself, so they need no submodules, no
compiler, and no bundled sysdeps. The harness subproject is declared with
``FPRINT_SOURCE_HASH`` so its source fingerprint comes from file content rather
than git, letting the test run against a throwaway source tree.
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
    "therock_subproject",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def run_cmake(
    source_dir: Path,
    build_dir: Path,
    *args: str,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["cmake", "-S", str(source_dir), "-B", str(build_dir), "-GNinja", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(result.stdout)
    if not expect_success and result.returncode == 0:
        raise AssertionError("CMake unexpectedly succeeded:\n" + result.stdout)
    return result


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def write_harness(source_dir: Path) -> None:
    """Writes a single-subproject superproject that records its fingerprint.

    After activating ``main_project`` the top-level list file reads back the
    ``THEROCK_FPRINT`` target property and writes it to ``fprint.txt`` in the
    binary dir, which the test reads.
    """
    includes = "\n".join(f"include({name})" for name in HARNESS_INCLUDES)
    write_file(
        source_dir / "CMakeLists.txt",
        f"""
        cmake_minimum_required(VERSION 3.25)
        project(therock_subproject_fprint_epoch_harness NONE)

        set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
        set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
        list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)

        # therock_subproject.cmake fingerprints this file. The harness declares
        # no flags, so an empty state file is enough.
        set(ROCM_BUILD_FLAGS_STATE_FILE "${{CMAKE_BINARY_DIR}}/rocm_build_flags_state.cmake")
        file(WRITE "${{ROCM_BUILD_FLAGS_STATE_FILE}}" "# no flags\\n")

        {includes}

        therock_cmake_subproject_declare(main_project
          EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/main"
          BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/main"
          FPRINT_SOURCE_HASH
        )
        therock_cmake_subproject_activate(main_project)

        get_target_property(_fprint main_project THEROCK_FPRINT)
        file(WRITE "${{CMAKE_BINARY_DIR}}/fprint.txt" "${{_fprint}}")
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


def fingerprint(source_dir: Path, build_dir: Path, *, epoch: str | None = None) -> str:
    """Configures the harness and returns main_project's fingerprint."""
    args = [f"-DTHEROCK_FPRINT_EPOCH={epoch}"] if epoch is not None else []
    run_cmake(source_dir, build_dir, *args)
    return (build_dir / "fprint.txt").read_text(encoding="utf-8").strip()


class FingerprintEpochTest(unittest.TestCase):
    def test_fingerprint_is_a_sha256_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            write_harness(source_dir)
            self.assertRegex(
                fingerprint(source_dir, temp_dir / "build", epoch="1"), SHA256_RE
            )

    def test_same_epoch_produces_same_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            write_harness(source_dir)
            self.assertEqual(
                fingerprint(source_dir, temp_dir / "build_a", epoch="1"),
                fingerprint(source_dir, temp_dir / "build_b", epoch="1"),
            )

    def test_bumping_epoch_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            write_harness(source_dir)
            self.assertNotEqual(
                fingerprint(source_dir, temp_dir / "build_e1", epoch="1"),
                fingerprint(source_dir, temp_dir / "build_e2", epoch="2"),
            )

    def test_default_epoch_matches_explicit_one(self):
        # Configuring without -D must behave like the documented default ("1").
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            write_harness(source_dir)
            self.assertEqual(
                fingerprint(source_dir, temp_dir / "build_default"),
                fingerprint(source_dir, temp_dir / "build_one", epoch="1"),
            )


if __name__ == "__main__":
    unittest.main()
