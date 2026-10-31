# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "profile_build.py"
spec = importlib.util.spec_from_file_location("profile_build", SCRIPT)
profile_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile_build)


class ProfileBuildTest(unittest.TestCase):
    def test_visible_hierarchy_capacity_uses_tightest_bound(self):
        self.assertEqual(
            profile_build.capacity(
                list(range(96)),
                {
                    "child": {
                        "cpu.max": "800000 100000",
                        "cpuset.cpus.effective": "0-31",
                    },
                    "parent": {"cpu.max": "400000 100000"},
                },
            ),
            4,
        )
        self.assertIsNone(profile_build.capacity(None, {}))

    def test_nested_cpu_work_and_exit_code_are_preserved(self):
        child = (
            "import time; t=time.process_time();\nwhile time.process_time()-t<1.4: pass"
        )
        command = (
            "import subprocess,sys,time; "
            f"subprocess.run([sys.executable,'-c',{child!r}],check=True);"
            "time.sleep(.3);sys.exit(7)"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--interval",
                    ".1",
                    "--output",
                    str(output),
                    "--",
                    sys.executable,
                    "-c",
                    command,
                ],
                check=False,
            )
            self.assertEqual(result.returncode, 7)
            summary = json.loads((output / "summary.json").read_text())
            rows = [
                json.loads(line)
                for line in (output / "samples.jsonl").read_text().splitlines()
            ]
            self.assertGreaterEqual(summary["observed_process_identities"], 2)
            self.assertGreater(summary["sampled_process_cpu_lower_bound_s"], 0.5)
            self.assertFalse([row for row in rows if row["event"] == "sampling_error"])
            if sys.platform != "win32":
                self.assertGreater(sum(summary["posix_waited_child_cpu"].values()), 0.5)
            else:
                self.assertIsNone(summary["posix_waited_child_cpu"])

    def test_immediate_exit_and_launch_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            for command, expected in (
                ([sys.executable, "-c", "pass"], 0),
                (["therock-missing-executable-123456"], 127),
            ):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--output",
                        temporary,
                        "--",
                        *command,
                    ],
                    check=False,
                )
                self.assertEqual(result.returncode, expected)


if __name__ == "__main__":
    unittest.main()
