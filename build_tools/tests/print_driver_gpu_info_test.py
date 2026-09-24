#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for build_tools/print_driver_gpu_info.py."""

import sys
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ImportTest(unittest.TestCase):
    def test_module_imports_on_any_platform(self):
        """The module must be importable without error on any OS.

        fcntl is a Unix-only stdlib module used inside print_driver_gpu_info.
        Importing it at the top level causes a ModuleNotFoundError on Windows.
        This test catches that class of mistakes by running on CPU-only runners
        before any GPU job has a chance to fail.
        """
        import print_driver_gpu_info  # noqa: F401


class IsWslTest(unittest.TestCase):
    """WSL detection gates whether amd-smi and the KFD ioctl are attempted.

    Getting this wrong is silent in opposite directions: a false negative makes
    every WSL GPU job fail on amd-smi (exit 255), and a false positive skips the
    real GPU checks on a bare-metal runner.
    """

    def test_wsl_with_gpu_detected(self):
        import print_driver_gpu_info as m

        with patch("os.path.exists", return_value=True), patch(
            "builtins.open",
            mock_open(read_data="Linux version 6.6 microsoft-standard-WSL2"),
        ):
            self.assertTrue(m._is_wsl())

    def test_bare_metal_linux_is_not_wsl(self):
        """/dev/dxg absent -> ordinary Linux, keep the amd-smi + KFD path."""
        import print_driver_gpu_info as m

        with patch("os.path.exists", return_value=False):
            self.assertFalse(m._is_wsl())

    def test_dxg_present_but_not_microsoft_kernel(self):
        """Both signals are required; a stray /dev/dxg alone is not WSL."""
        import print_driver_gpu_info as m

        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data="Linux version 6.8 generic")
        ):
            self.assertFalse(m._is_wsl())

    def test_unreadable_proc_version_is_not_wsl(self):
        import print_driver_gpu_info as m

        with patch("os.path.exists", return_value=True), patch(
            "builtins.open", side_effect=OSError("boom")
        ):
            self.assertFalse(m._is_wsl())


class RunSanityTest(unittest.TestCase):
    def test_wsl_runs_amd_smi_and_rocminfo_but_skips_kfd(self):
        """On WSL the driver tools work; only the KFD ioctl cannot.

        Verified empirically (actions/runs/36042414950): with the wsl-rocdxg
        artifact installed, amd-smi static and rocminfo both succeed, and
        /dev/kfd is the only thing genuinely missing. Skipping more than that
        would throw away real GPU validation.
        """
        import print_driver_gpu_info as m

        ran = []
        with patch.object(m, "_is_wsl", return_value=True), patch.object(
            m,
            "run_command_with_search",
            side_effect=lambda **kw: ran.append(kw["command"]),
        ), patch.object(
            m, "_get_kfd_version", side_effect=AssertionError("KFD must not be queried")
        ):
            rc = m.run_sanity("Linux")

        self.assertEqual(rc, 0)
        self.assertIn("amd-smi", ran)
        self.assertIn("rocminfo", ran)

    def test_bare_metal_linux_missing_kfd_still_fails(self):
        """Off WSL, a missing /dev/kfd must remain a hard failure."""
        import print_driver_gpu_info as m

        with patch.object(m, "_is_wsl", return_value=False), patch.object(
            m, "run_command_with_search"
        ), patch("os.path.exists", return_value=False):
            rc = m.run_sanity("Linux")

        self.assertEqual(rc, 1)

    def test_bare_metal_linux_still_runs_amd_smi(self):
        """The existing Linux path must be unchanged."""
        import print_driver_gpu_info as m

        ran = []
        with patch.object(m, "_is_wsl", return_value=False), patch.object(
            m,
            "run_command_with_search",
            side_effect=lambda **kw: ran.append(kw["command"]),
        ), patch("os.path.exists", return_value=True), patch.object(
            m, "_get_kfd_version", return_value=(1, 15)
        ):
            rc = m.run_sanity("Linux")

        self.assertEqual(rc, 0)
        self.assertIn("amd-smi", ran)
        self.assertIn("rocminfo", ran)

    def test_windows_path_unchanged(self):
        import print_driver_gpu_info as m

        ran = []
        with patch.object(
            m,
            "run_command_with_search",
            side_effect=lambda **kw: ran.append(kw["command"]),
        ):
            rc = m.run_sanity("Windows")

        self.assertEqual(rc, 0)
        self.assertEqual(ran, ["hipInfo.exe"])


if __name__ == "__main__":
    unittest.main()
