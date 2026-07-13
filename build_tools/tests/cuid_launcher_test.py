# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for build_tools/cuid_launcher.py."""

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import cuid_launcher
from cuid_launcher import (
    build_command,
    has_explicit_cuid,
    is_hip_compile,
    output_path,
    stable_cuid,
)


class OutputPathTest(unittest.TestCase):
    def test_space_separated(self):
        self.assertEqual(output_path(["-c", "k.hip", "-o", "a.o"]), "a.o")

    def test_joined(self):
        self.assertEqual(output_path(["-c", "k.hip", "-oa.o"]), "a.o")

    def test_long_form_equals(self):
        self.assertEqual(output_path(["-c", "k.hip", "--output=a.o"]), "a.o")

    def test_long_form_space(self):
        self.assertEqual(output_path(["-c", "k.hip", "--output", "a.o"]), "a.o")

    def test_missing(self):
        self.assertIsNone(output_path(["-c", "k.hip"]))


class IsHipCompileTest(unittest.TestCase):
    def test_offload_arch(self):
        self.assertTrue(is_hip_compile(["--offload-arch=gfx942", "-c", "k.cpp"]))

    def test_x_hip(self):
        self.assertTrue(is_hip_compile(["-x", "hip", "-c", "k.cpp"]))

    def test_hip_suffix(self):
        self.assertTrue(is_hip_compile(["-c", "k.hip"]))

    def test_cu_suffix(self):
        self.assertTrue(is_hip_compile(["-c", "k.cu"]))

    def test_requires_compile_flag(self):
        # Offload markers without -c (e.g. a link step) are not compiles.
        self.assertFalse(is_hip_compile(["--offload-arch=gfx942", "-o", "libx.so"]))

    def test_host_cpp_is_not_hip(self):
        self.assertFalse(is_hip_compile(["-c", "host.cpp", "-o", "host.o"]))


class HasExplicitCuidTest(unittest.TestCase):
    def test_joined(self):
        self.assertTrue(has_explicit_cuid(["-cuid=abc"]))

    def test_bare(self):
        self.assertTrue(has_explicit_cuid(["-cuid", "abc"]))

    def test_absent(self):
        self.assertFalse(has_explicit_cuid(["-c", "k.hip"]))


class StableCuidTest(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(stable_cuid("d/a.o"), stable_cuid("d/a.o"))

    def test_distinct_per_output(self):
        self.assertNotEqual(stable_cuid("d/a.o"), stable_cuid("d/b.o"))

    def test_shape(self):
        cuid = stable_cuid("d/a.o")
        self.assertEqual(len(cuid), 16)
        int(cuid, 16)  # raises if not hex


class BuildCommandTest(unittest.TestCase):
    HIP = ["--offload-arch=gfx942", "-c", "k.hip"]

    def _cuids(self, args, sccache=None):
        with mock.patch.object(cuid_launcher, "find_sccache", return_value=sccache):
            cmd = build_command(["cuid_launcher.py", "clang++", *args])
        return cmd, [c for c in cmd if c.startswith("-cuid=")]

    def test_injects_for_hip(self):
        _, cuids = self._cuids([*self.HIP, "-o", "d/a.o"])
        self.assertEqual(len(cuids), 1)

    def test_distinct_per_output(self):
        _, a = self._cuids([*self.HIP, "-o", "d/a.o"])
        _, b = self._cuids([*self.HIP, "-o", "d/b.o"])
        self.assertNotEqual(a, b)

    def test_deterministic(self):
        _, a1 = self._cuids([*self.HIP, "-o", "d/a.o"])
        _, a2 = self._cuids([*self.HIP, "-o", "d/a.o"])
        self.assertEqual(a1, a2)

    def test_no_inject_for_host_compile(self):
        _, cuids = self._cuids(["-c", "host.cpp", "-o", "host.o"])
        self.assertEqual(cuids, [])

    def test_respects_existing_cuid(self):
        _, cuids = self._cuids([*self.HIP, "-o", "d/a.o", "-cuid=PRESET"])
        self.assertEqual(cuids, ["-cuid=PRESET"])

    def test_no_inject_without_output(self):
        _, cuids = self._cuids(self.HIP)
        self.assertEqual(cuids, [])

    def test_sccache_prefixed_when_found(self):
        cmd, _ = self._cuids([*self.HIP, "-o", "d/a.o"], sccache="/x/sccache")
        self.assertEqual(cmd[0], "/x/sccache")
        self.assertEqual(cmd[1], "clang++")

    def test_preserves_original_args(self):
        cmd, _ = self._cuids([*self.HIP, "-o", "d/a.o"])
        for a in [*self.HIP, "-o", "d/a.o"]:
            self.assertIn(a, cmd)


if __name__ == "__main__":
    unittest.main()
