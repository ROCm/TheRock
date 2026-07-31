# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import runner_capability


class RunnerCapabilityTest(unittest.TestCase):
    def test_parse_dot_version(self):
        self.assertEqual(
            runner_capability.parse_dot_version("6.19.14.31400000"),
            (6, 19, 14, 31400000),
        )

    def test_amdgpu_driver_meets_min(self):
        with mock.patch.object(
            runner_capability,
            "read_amdgpu_driver_version",
            return_value="6.19.14.31400000",
        ):
            self.assertTrue(
                runner_capability.amdgpu_driver_meets_min("6.19.14.31400000")
            )
            self.assertTrue(runner_capability.amdgpu_driver_meets_min("6.19.14.0"))
            self.assertFalse(runner_capability.amdgpu_driver_meets_min("6.19.15.0"))

    def test_check_runner_requirements_passes(self):
        with mock.patch.object(
            runner_capability,
            "read_amdgpu_driver_version",
            return_value="6.19.14.31400000",
        ):
            runner_capability.check_runner_requirements(
                {"amdgpu_driver_min": "6.19.14.31400000"}
            )

    def test_check_runner_requirements_fails(self):
        with mock.patch.object(
            runner_capability,
            "read_amdgpu_driver_version",
            return_value="6.18.0.0",
        ):
            with self.assertRaises(SystemExit):
                runner_capability.check_runner_requirements(
                    {"amdgpu_driver_min": "6.19.14.31400000"}
                )


if __name__ == "__main__":
    unittest.main()
