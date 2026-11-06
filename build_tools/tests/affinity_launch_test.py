#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for affinity_launch.py (Windows single-CPU linker launcher)."""

import os
import random
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import affinity_launch

LAUNCHER = Path(__file__).parent.parent / "affinity_launch.py"

# Prints this process's affinity mask; with "child", also starts itself again as
# a grandchild and prints that process's mask too.
AFFINITY_PROBE = textwrap.dedent(
    """\
    import ctypes, ctypes.wintypes as wt, subprocess, sys
    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = wt.HANDLE
    k.GetProcessAffinityMask.argtypes = [
        wt.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    k.GetProcessAffinityMask(
        k.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask))
    if sys.argv[1:] == ["child"]:
        grandchild = subprocess.run(
            [sys.executable, __file__], capture_output=True, text=True).stdout.strip()
        print(process_mask.value, grandchild)
    else:
        print(process_mask.value)
    """
)


def popcount(mask: int) -> int:
    return bin(mask).count("1")


class ChooseAffinityMaskTest(unittest.TestCase):
    def test_single_cpu_within_available(self):
        available = 0b1111_0000_1111
        for seed in range(50):
            mask = affinity_launch.choose_affinity_mask(
                available, 1, random.Random(seed)
            )
            self.assertEqual(popcount(mask), 1)
            self.assertEqual(mask & ~available, 0)

    def test_spreads_across_cpus(self):
        available = (1 << 40) - 1
        rng = random.Random(1234)
        chosen = {
            affinity_launch.choose_affinity_mask(available, 1, rng) for _ in range(400)
        }
        self.assertGreater(len(chosen), 30)

    def test_multiple_cpus(self):
        available = 0b1011_0110
        mask = affinity_launch.choose_affinity_mask(available, 3, random.Random(7))
        self.assertEqual(popcount(mask), 3)
        self.assertEqual(mask & ~available, 0)

    def test_request_more_than_available(self):
        available = 0b101
        mask = affinity_launch.choose_affinity_mask(available, 8, random.Random(0))
        self.assertEqual(mask, available)

    def test_empty_mask(self):
        self.assertEqual(
            affinity_launch.choose_affinity_mask(0, 1, random.Random(0)), 0
        )


class LauncherProcessTest(unittest.TestCase):
    def run_launcher(self, *cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-S", "-E", str(LAUNCHER), *cmd],
            capture_output=True,
            text=True,
        )

    def test_exit_code_passthrough(self):
        for code in (0, 1, 7):
            result = self.run_launcher(
                sys.executable, "-c", f"import sys; sys.exit({code})"
            )
            self.assertEqual(result.returncode, code)

    def test_arguments_preserved(self):
        args = ["a b", 'q"uote', ""]
        result = self.run_launcher(
            sys.executable, "-c", "import sys; print(sys.argv[1:])", *args
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(args))

    def test_missing_command_is_an_error(self):
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)

    @unittest.skipUnless(sys.platform == "win32", "CPU affinity is only set on Windows")
    def test_child_and_grandchild_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "affinity_probe.py"
            probe.write_text(AFFINITY_PROBE)
            result = self.run_launcher(sys.executable, str(probe), "child")
        self.assertEqual(result.returncode, 0, result.stderr)
        child, grandchild = (int(x) for x in result.stdout.split())
        self.assertEqual(popcount(child), 1)
        self.assertEqual(child, grandchild)


if __name__ == "__main__":
    unittest.main()
