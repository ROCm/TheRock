#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for build_tools/print_driver_gpu_info.py."""

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
