#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for setup_repo_build_deps.sh's --os-profile handling.

The package managers and sudo are replaced by stubs that only record how they
were called, so nothing is installed.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "setup_repo_build_deps.sh"
STUBS = ("apt-get", "zypper", "dnf", "tdnf")


@unittest.skipIf(sys.platform == "win32", "needs bash")
class SetupRepoBuildDepsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.calls = self.tmp / "calls.log"
        self.calls.touch()

    def _run(self, os_profile: str, stubs=STUBS) -> subprocess.CompletedProcess:
        """Run the script with the given stubs ahead of the real PATH."""
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir(exist_ok=True)
        for name in stubs:
            stub = bin_dir / name
            stub.write_text(
                f'#!/bin/bash\necho "{name} $*" >> "{self.calls}"\n',
                encoding="utf-8",
            )
            stub.chmod(0o755)
        # Real sudo resets PATH, which would bypass the stubs on a non-root
        # machine, so it only runs its arguments here.
        sudo = bin_dir / "sudo"
        sudo.write_text('#!/bin/bash\nexec "$@"\n', encoding="utf-8")
        sudo.chmod(0o755)
        env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
        return subprocess.run(
            ["bash", str(SCRIPT), "--os-profile", os_profile],
            env=env,
            capture_output=True,
            text=True,
        )

    def _managers_called(self) -> set[str]:
        return {line.split()[0] for line in self.calls.read_text().splitlines()}

    def test_unrecognised_profile_is_rejected_before_any_package_manager(self):
        result = self._run("typo-profile")

        self.assertEqual(result.returncode, 1)
        self.assertIn("unrecognised --os-profile 'typo-profile'", result.stderr)
        self.assertEqual(self._managers_called(), set())

    def test_each_family_uses_its_package_manager(self):
        cases = {
            "ubuntu2404": "apt-get",
            "debian12": "apt-get",
            "sles15": "zypper",
            "rhel9": "dnf",
            "azl3": "dnf",
        }
        for os_profile, manager in cases.items():
            with self.subTest(os_profile=os_profile):
                self.calls.write_text("")
                result = self._run(os_profile)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self._managers_called(), {manager})

    @unittest.skipIf(shutil.which("dnf"), "dnf is installed on this machine")
    def test_azl_uses_tdnf_where_dnf_is_missing(self):
        result = self._run("azl3", stubs=("tdnf",))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._managers_called(), {"tdnf"})


if __name__ == "__main__":
    unittest.main()
