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
