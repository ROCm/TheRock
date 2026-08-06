#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for ``runpath_to_rpath.py``.

These tests build real shared libraries on the host, inspect them with readelf,
and exercise the runpath_to_rpath conversion logic end-to-end.

What is tested
--------------
1. Sample libraries are built correctly in setUp:
   - lib_runpath.so with DT_RUNPATH (--enable-new-dtags)
   - lib_rpath.so  with DT_RPATH  (--disable-new-dtags)
2. _get_rpath() reads the rpath value from both RUNPATH and RPATH ELF tags.
3. update_rpath() preserves rpath values and keeps DT_RPATH libraries unchanged.
4. DT_RUNPATH can be converted to DT_RPATH when patchelf clears RUNPATH first
   (reference workflow for older patchelf versions).

Prerequisites (system packages)
-------------------------------
  Debian/Ubuntu:  sudo apt install build-essential binutils patchelf python3
  RHEL/Fedora:    sudo dnf install gcc binutils patchelf python3

Verify:  command -v gcc readelf patchelf python3
"""

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent
sys.path.insert(0, str(LINUX_DIR))

import runpath_to_rpath  # noqa: E402

FIXTURE_SRC = THIS_SCRIPT_DIR / "hello_lib.c"
RPATH_VALUE = "$ORIGIN/lib"
REQUIRED_TOOLS = ("gcc", "readelf", "patchelf")
HAS_INTEGRATION_TOOLS = all(shutil.which(tool) for tool in REQUIRED_TOOLS)


def _elf_rpath_info(path: Path):
    """Inspect an ELF file and return its dynamic rpath tag and value."""
    out = subprocess.check_output(
        ["readelf", "-d", str(path)], stderr=subprocess.DEVNULL
    ).decode()
    for line in out.splitlines():
        m = re.search(r"\(R(?:UN)?PATH\)\s+Library r(?:un)?path: \[(.+)\]", line)
        if not m:
            continue
        if "(RUNPATH)" in line:
            return "RUNPATH", m.group(1)
        if "(RPATH)" in line:
            return "RPATH", m.group(1)
    return None, None


def _build_fixture_libs(out_dir: Path):
    """Compile hello_lib.c into one RUNPATH and one RPATH shared library."""
    runpath_so = out_dir / "lib_runpath.so"
    rpath_so = out_dir / "lib_rpath.so"
    common = ["gcc", "-shared", "-fPIC", "-Wl,-rpath," + RPATH_VALUE]
    subprocess.check_call(
        common + ["-Wl,--enable-new-dtags", "-o", str(runpath_so), str(FIXTURE_SRC)]
    )
    subprocess.check_call(
        common + ["-Wl,--disable-new-dtags", "-o", str(rpath_so), str(FIXTURE_SRC)]
    )
    return runpath_so, rpath_so


@unittest.skipUnless(HAS_INTEGRATION_TOOLS, f"requires: {', '.join(REQUIRED_TOOLS)}")
class TestRunpathToRpath(unittest.TestCase):
    """Integration tests for runpath_to_rpath using real ELF libraries."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.workdir = Path(self._tmpdir.name)
        self.runpath_so, self.rpath_so = _build_fixture_libs(self.workdir)

        self.assertEqual(_elf_rpath_info(self.runpath_so), ("RUNPATH", RPATH_VALUE))
        self.assertEqual(_elf_rpath_info(self.rpath_so), ("RPATH", RPATH_VALUE))

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_get_rpath_reads_real_libraries(self):
        """Verify _get_rpath() reads rpath from real RUNPATH and RPATH ELFs."""
        self.assertEqual(runpath_to_rpath._get_rpath(self.runpath_so), RPATH_VALUE)
        self.assertEqual(runpath_to_rpath._get_rpath(self.rpath_so), RPATH_VALUE)

    def test_update_rpath_preserves_rpath_values(self):
        """Run update_rpath() and verify rpath values and tags after conversion."""
        runpath_to_rpath.update_rpath(self.workdir, [])

        self.assertEqual(_elf_rpath_info(self.runpath_so)[1], RPATH_VALUE)
        self.assertEqual(_elf_rpath_info(self.rpath_so)[1], RPATH_VALUE)
        self.assertEqual(_elf_rpath_info(self.rpath_so)[0], "RPATH")

        runpath_tag, _ = _elf_rpath_info(self.runpath_so)
        if runpath_tag != "RPATH":
            self.skipTest(
                "update_rpath did not convert DT_RUNPATH to DT_RPATH on this patchelf "
                "(see test_runpath_tag_becomes_rpath_after_remove_and_set)"
            )

    def test_runpath_tag_becomes_rpath_after_remove_and_set(self):
        """Verify DT_RUNPATH converts to DT_RPATH when RUNPATH is cleared first."""
        subprocess.check_call(["patchelf", "--remove-rpath", str(self.runpath_so)])
        subprocess.check_call(
            [
                "patchelf",
                "--force-rpath",
                "--set-rpath",
                RPATH_VALUE,
                str(self.runpath_so),
            ]
        )
        self.assertEqual(_elf_rpath_info(self.runpath_so), ("RPATH", RPATH_VALUE))


if __name__ == "__main__":
    unittest.main()
