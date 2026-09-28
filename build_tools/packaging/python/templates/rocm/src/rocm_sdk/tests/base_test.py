# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installation package tests for the base installation."""

import os
import re
import subprocess
import sys
import unittest

import rocm_sdk
from .. import _dist_info as di
from . import utils

_RTLD_CONTRACT_SCRIPT = r"""
import ctypes
import sys
import rocm_sdk

rocm_sdk.initialize_process(
    preload_shortnames=["amd_comgr", "amdhip64", "hiprtc"],
    rtld_global=True,
)
rtld = ctypes.CDLL(None)

def addr(name: str) -> int | None:
    fn = getattr(rtld, name, None)
    if fn is None:
        return None
    return ctypes.cast(fn, ctypes.c_void_p).value

required = ["hipGetDeviceCount", "amd_comgr_create_data"]
forbidden = ["LLVMInitializeAMDGPUTarget"]
missing = [n for n in required if not addr(n)]
present = [n for n in forbidden if addr(n)]
if missing or present:
    sys.stderr.write(
        f"RTLD_DEFAULT contract failed: missing={missing} forbidden_present={present}\n"
    )
    sys.exit(1)
"""

# No torch here, so a cold interpreter start is the only real cost.
_CHILD_TIMEOUT_SECONDS = 300


def _run_child(script: str, timeout: int = _CHILD_TIMEOUT_SECONDS) -> None:
    """Run `script` in a fresh interpreter, failing loudly on crash or hang.

    The failure under test is a SIGSEGV (returncode -11), but a loader bug can
    hang instead; without a timeout that stalls CI until the job limit and the
    regression signal is lost.
    """
    completed = subprocess.run(
        [sys.executable, "-c", script],
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        signal_note = (
            f" (killed by signal {-completed.returncode})"
            if completed.returncode < 0
            else ""
        )
        raise AssertionError(
            f"child exited {completed.returncode}{signal_note}\n"
            f"--- child stdout ---\n{completed.stdout}\n"
            f"--- child stderr ---\n{completed.stderr}"
        )


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
        cmd = [sys.executable, "-m", "rocm_sdk", "--help"]
        output = utils.run_command(cmd, capture=True).decode()
        self.assertIn("usage:", output)

    def testVersion(self):
        cmd = [sys.executable, "-m", "rocm_sdk", "version"]
        output = utils.run_command(cmd, capture=True).decode().strip()
        self.assertTrue(output)
        self.assertIn(".", output)

    def testTargets(self):
        cmd = [sys.executable, "-m", "rocm_sdk", "targets"]
        output = utils.run_command(cmd, capture=True).decode().strip()
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

    @unittest.skipIf(sys.platform != "linux", "RTLD_DEFAULT contract is Linux/glibc")
    def test_initialize_process_hip_global_llvm_not_global(self):
        _run_child(_RTLD_CONTRACT_SCRIPT)

    def test_run_child_times_out_instead_of_hanging(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            _run_child("import time; time.sleep(30)", timeout=1)

    def test_run_child_reports_child_stderr(self):
        with self.assertRaisesRegex(AssertionError, "distinctive-child-marker"):
            _run_child(
                "import sys; sys.stderr.write('distinctive-child-marker'); sys.exit(3)"
            )
