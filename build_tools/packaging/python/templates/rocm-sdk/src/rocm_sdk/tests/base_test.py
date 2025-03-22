"""Installation package tests for the base installation."""

from pathlib import Path
import shlex
import subprocess
import sys
import unittest

from .. import _dist_info as di


def exec(args: list[str | Path], cwd: Path | None = None, capture: bool = False):
    args = [str(arg) for arg in args]
    if cwd is None:
        cwd = Path.cwd()
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    sys.stdout.flush()
    if capture:
        return subprocess.check_output(args, cwd=str(cwd), stdin=subprocess.DEVNULL)
    else:
        subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


class ROCmBaseTest(unittest.TestCase):
    def testCLI(self):
        output = exec(
            [sys.executable, "-P", "-m", "rocm_sdk", "--help"], capture=True
        ).decode()
        self.assertIn("usage:", output)
