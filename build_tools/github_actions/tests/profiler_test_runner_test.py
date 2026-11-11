# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check profiler subprocess budgets and complete CTest/GTest coverage."""

import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


RUNNER = (
    Path(__file__).resolve().parent.parent / "test_executable_scripts/test_runner.py"
)


class ProfilerTestRunnerTest(unittest.TestCase):
    def load(self, component, tier="comprehensive", csan=True, timeout=None):
        env = {
            "TEST_COMPONENT": component,
            "TEST_TYPE": tier,
            "THEROCK_BIN_DIR": "/tmp/profiler-sdk/bin",
            "SHARD_INDEX": "5",
            "TOTAL_SHARDS": "8",
        }
        if csan:
            env["CSAN_RUNTIME_PATH"] = "/sdk/libclang_rt.csan.so"
        if timeout is not None:
            env["ROCPROFSYS_CI_TIMEOUT"] = timeout
        with patch.dict(os.environ, env, clear=True):
            return runpy.run_path(str(RUNNER))["environ_vars"]

    def test_systems_budget_and_explicit_override(self):
        for tier in ("comprehensive", "full"):
            self.assertEqual(
                self.load("rocprofiler-systems", tier)["ROCPROFSYS_CI_TIMEOUT"], "1800"
            )
            self.assertEqual(
                self.load("rocprofiler-systems", tier, timeout="2400")[
                    "ROCPROFSYS_CI_TIMEOUT"
                ],
                "2400",
            )

    def test_other_budgets_unchanged(self):
        for component, tier, csan in (
            ("rocprofiler-compute", "full", True),
            ("rocprofiler-systems", "quick", True),
            ("rocprofiler-systems", "standard", True),
            ("rocprofiler-systems", "full", False),
            ("rocblas", "full", True),
        ):
            self.assertNotIn("ROCPROFSYS_CI_TIMEOUT", self.load(component, tier, csan))
        env = self.load("rocblas")
        self.assertEqual(
            (env["GTEST_SHARD_INDEX"], env["GTEST_TOTAL_SHARDS"]), ("4", "8")
        )

    @unittest.skipUnless(shutil.which("ctest"), "requires actual CTest")
    def test_ctest_stride_preserves_every_inner_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "cases.jsonl"
            child = root / "gtest_like.py"
            child.write_text(
                "import json,os,sys\n"
                "index=int(os.environ['GTEST_SHARD_INDEX'])\n"
                "total=int(os.environ['GTEST_TOTAL_SHARDS'])\n"
                "with open(sys.argv[1],'a') as f:\n"
                " for case in range(6):\n"
                "  if case % total == index: f.write(json.dumps([int(sys.argv[2]),case])+'\\n')\n"
            )
            (root / "CTestTestfile.cmake").write_text(
                "\n".join(
                    f'add_test(entry{i} "{Path(sys.executable).as_posix()}" "{child.as_posix()}" "{record.as_posix()}" "{i}")'
                    for i in range(8)
                )
            )
            for component in ("rocprofiler-compute", "rocprofiler-systems"):
                record.unlink(missing_ok=True)
                env = {**os.environ, **self.load(component)}
                for shard in range(1, 9):
                    subprocess.run(
                        [
                            shutil.which("ctest"),
                            "--test-dir",
                            str(root),
                            "-I",
                            f"{shard},,8",
                            "--output-on-failure",
                        ],
                        env=env,
                        check=True,
                        capture_output=True,
                    )
                cases = [
                    tuple(json.loads(line)) for line in record.read_text().splitlines()
                ]
                self.assertEqual(len(cases), 48)
                self.assertEqual(
                    set(cases),
                    {(entry, case) for entry in range(8) for case in range(6)},
                )


if __name__ == "__main__":
    unittest.main()
