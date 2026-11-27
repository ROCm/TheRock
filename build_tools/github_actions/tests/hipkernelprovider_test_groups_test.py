# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "test_executable_scripts"))

from hipkernelprovider_test_groups import ENGINE_GROUPS, add_group_filter

BASE = ["ctest", "-L", "^standard$", "--test-dir", "build/bin/hip_kernel_provider"]
ALL_ENGINES = "|".join(ENGINE_GROUPS.values())


class AddGroupFilterTest(unittest.TestCase):
    def test_engine_group_narrows_with_its_label(self):
        cmd = add_group_filter(BASE, "asm-sdpa")
        self.assertEqual(cmd, BASE + ["-L", "^ASM_SDPA_ENGINE$"])

    def test_unit_group_excludes_every_engine(self):
        self.assertEqual(add_group_filter(BASE, "unit"), BASE + ["-LE", ALL_ENGINES])

    def test_unit_group_joins_an_existing_exclusion(self):
        # ctest ANDs repeated -LE, so a second -LE would exclude only tests
        # matching both patterns; the engine labels must join the same regex.
        cmd = add_group_filter(BASE + ["-LE", "ex_gpu"], "unit")
        self.assertEqual(cmd.count("-LE"), 1)
        self.assertEqual(cmd[cmd.index("-LE") + 1], f"ex_gpu|{ALL_ENGINES}")

    def test_unknown_group_rejected(self):
        with self.assertRaises(ValueError):
            add_group_filter(BASE, "nope")


if __name__ == "__main__":
    unittest.main()
