"""Installation package tests for the base installation."""

import re
import os
import sys
import unittest

import rocm_sdk
from .. import _dist_info as di
from . import utils


class ROCmBaseTest(unittest.TestCase):
    def setUp(self):
        self.orig_ROCM_SDK_PRELOAD_LIBRARIES = os.getenv("ROCM_SDK_PRELOAD_LIBRARIES")
        os.environ.pop("ROCM_SDK_PRELOAD_LIBRARIES", None)
        rocm_sdk._ALL_CDLLS.clear()

    def tearDown(self):
        orig_env = self.orig_ROCM_SDK_PRELOAD_LIBRARIES
        if orig_env:
            os.putenv("ROCM_SDK_PRELOAD_LIBRARIES", orig_env)
        else:
            os.environ.pop("ROCM_SDK_PRELOAD_LIBRARIES", None)

    def testCLI(self):
        # -P flag is only available in Python 3.11+
        cmd = [sys.executable]
        if sys.version_info >= (3, 11):
            cmd.append("-P")
        cmd.extend(["-m", "rocm_sdk", "--help"])
        output = utils.exec(cmd, capture=True).decode()
        self.assertIn("usage:", output)

    def testVersion(self):
        # -P flag is only available in Python 3.11+
        cmd = [sys.executable]
        if sys.version_info >= (3, 11):
            cmd.append("-P")
        cmd.extend(["-m", "rocm_sdk", "version"])
        output = (
            utils.exec(cmd, capture=True)
            .decode()
            .strip()
        )
        self.assertTrue(output)
        self.assertIn(".", output)

    def testTargets(self):
        # -P flag is only available in Python 3.11+
        cmd = [sys.executable]
        if sys.version_info >= (3, 11):
            cmd.append("-P")
        cmd.extend(["-m", "rocm_sdk", "targets"])
        output = (
            utils.exec(cmd, capture=True)
            .decode()
            .strip()
        )
        self.assertTrue(output)
        self.assertIn("gfx", output)

    def test_initialize_process_preload_libraries(self):
        rocm_sdk.initialize_process(preload_shortnames=["amdhip64"])
        self.assertIn("amdhip64", rocm_sdk._ALL_CDLLS)

    def test_initialize_process_env_preload_1(self):
        os.environ["ROCM_SDK_PRELOAD_LIBRARIES"] = "amdhip64"
        rocm_sdk.initialize_process()
        self.assertIn("amdhip64", rocm_sdk._ALL_CDLLS)

    def test_initialize_process_env_preload_2_comma(self):
        os.environ["ROCM_SDK_PRELOAD_LIBRARIES"] = " ,amdhip64, ,hiprtc,"
        rocm_sdk.initialize_process()
        self.assertIn("amdhip64", rocm_sdk._ALL_CDLLS)
        self.assertIn("hiprtc", rocm_sdk._ALL_CDLLS)

    def test_initialize_process_env_preload_2_semi(self):
        os.environ["ROCM_SDK_PRELOAD_LIBRARIES"] = " ;amdhip64; ;hiprtc;"
        rocm_sdk.initialize_process()
        self.assertIn("amdhip64", rocm_sdk._ALL_CDLLS)
        self.assertIn("hiprtc", rocm_sdk._ALL_CDLLS)

    def test_initialize_process_check_version(self):
        rocm_sdk.initialize_process(
            check_version=rocm_sdk.__version__, fail_on_version_mismatch=True
        )

    def test_initialize_process_check_version_asterisk(self):
        rocm_sdk.initialize_process(check_version="*", fail_on_version_mismatch=True)

    def test_initialize_process_check_version_pattern(self):
        rocm_sdk.initialize_process(
            check_version=re.compile(".+"), fail_on_version_mismatch=True
        )

    def test_initialize_process_check_version_mismatch(self):
        with self.assertRaisesRegex(
            RuntimeError, "The program was compiled against a ROCm version matching"
        ):
            rocm_sdk.initialize_process(
                check_version="badversion", fail_on_version_mismatch=True
            )

    def test_initialize_process_check_version_mismatch_warning(self):
        with self.assertWarnsRegex(
            UserWarning, "The program was compiled against a ROCm version matching"
        ):
            rocm_sdk.initialize_process(check_version="badversion")
