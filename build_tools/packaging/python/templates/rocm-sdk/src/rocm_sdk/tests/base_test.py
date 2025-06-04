"""Installation package tests for the base installation."""

import sys
import unittest

from .. import _dist_info as di

from . import utils


class ROCmBaseTest(unittest.TestCase):
    def testCLI(self):
        output = utils.exec(
            [sys.executable, "-P", "-m", "rocm_sdk", "--help"], capture=True
        ).decode()
        self.assertIn("usage:", output)

    def testVersion(self):
        output = (
            utils.exec(
                [sys.executable, "-P", "-m", "rocm_sdk", "version"], capture=True
            )
            .decode()
            .strip()
        )
        self.assertTrue(output)
        self.assertIn(".", output)

    def testTargets(self):
        output = (
            utils.exec(
                [sys.executable, "-P", "-m", "rocm_sdk", "targets"], capture=True
            )
            .decode()
            .strip()
        )
        self.assertTrue(output)
        self.assertIn("gfx", output)
