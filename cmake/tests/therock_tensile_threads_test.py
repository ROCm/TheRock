#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the real hipBLASLt hook's nested-worker budget with CMake."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "math-libs/BLAS/pre_hook_hipBLASLt.cmake"


class TensileThreadsTest(unittest.TestCase):
    def run_hook(self, sanitizer, budget, previous=None):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "hook.cmake"
            result = Path(directory) / "result.txt"
            script.write_text(
                f'set(THEROCK_SANITIZER "{sanitizer}")\n'
                f'set(THEROCK_TOOLCHAIN_ROOT "{Path(directory).as_posix()}")\n'
                + (
                    f'set(TENSILELITE_BUILD_PARALLEL_LEVEL "{previous}" CACHE STRING "")\n'
                    if previous is not None
                    else ""
                )
                + f'include("{HOOK.as_posix()}")\n'
                + f'file(WRITE "{result.as_posix()}" "${{TENSILELITE_BUILD_PARALLEL_LEVEL}}")\n'
            )
            env = os.environ.copy()
            env.pop("CMAKE_BUILD_PARALLEL_LEVEL", None)
            if budget is not None:
                env["CMAKE_BUILD_PARALLEL_LEVEL"] = budget
            process = subprocess.run(
                ["cmake", "-P", str(script)], env=env, text=True, capture_output=True
            )
            return process, result.read_text() if result.exists() else None

    def test_budget_caps_default_and_larger_limit_but_preserves_smaller(self):
        for previous, expected in ((None, "22"), ("192", "22"), ("4", "4")):
            with self.subTest(previous=previous):
                process, value = self.run_hook("CSAN", "22", previous)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(value, expected)

    def test_other_modes_and_unset_budget_keep_existing_behavior(self):
        for sanitizer, budget, previous in (
            ("OFF", "22", None),
            ("ASAN", "22", "192"),
            ("CSAN", None, "192"),
        ):
            with self.subTest(sanitizer=sanitizer, budget=budget):
                process, value = self.run_hook(sanitizer, budget, previous)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(value, previous or "")

    def test_invalid_budget_fails_instead_of_falling_back_to_all_cpus(self):
        for budget in ("", "0", "-1", "22;192"):
            with self.subTest(budget=budget):
                process, _ = self.run_hook("CSAN", budget)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("positive CMAKE_BUILD_PARALLEL_LEVEL", process.stderr)


if __name__ == "__main__":
    unittest.main()
