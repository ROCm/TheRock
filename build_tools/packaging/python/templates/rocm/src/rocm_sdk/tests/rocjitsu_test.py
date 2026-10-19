# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise Rocjitsu from the installed development SDK without a physical GPU."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rocm_sdk import _dist_info as di


class RocjitsuTest(unittest.TestCase):
    def test_installed_worker(self):
        import rocm_sdk_devel

        prefix = (
            Path(rocm_sdk_devel.__file__).parent.parent
            / di.ALL_PACKAGES["devel"].get_py_package_name()
        )
        # The child must locate and expand the installed SDK on first use,
        # without relying on a source tree or the caller's search paths.
        env = os.environ.copy()
        env["PATH"] = os.defpath
        for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
            env.pop(name, None)
        script = """
import os
from pathlib import Path
import sys
import rocjitsu

prefix = Path(sys.argv[1])
assert Path(rocjitsu.diagnostics()["executable"]) == prefix / "bin/rocjitsu"
presets = prefix / "share/rocjitsu/configs"
assert rocjitsu.config.names() == tuple(sorted(p.stem for p in presets.glob("*.json")))
assert rocjitsu.config.gfx1250_mi455x == presets / "gfx1250_mi455x.json"
assert os.access(prefix / "libexec/rocjitsu/rocjitsu-memory", os.X_OK)
rocjitsu.cli(["--version"])
with rocjitsu.Session(rocjitsu.config.gfx1250_mi455x) as session:
    value, worker_pid = session.run(lambda x: (x + 1, os.getpid()), 41, timeout=30)
    assert value == 42 and worker_pid != os.getpid()
    assert session.run(os.getpid, timeout=30) == worker_pid
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-c", script, str(prefix)],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
