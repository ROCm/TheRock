# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise RCCL test commands with real subprocesses and fake GPU executables."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RCCL_SCRIPT = (
    Path(__file__).resolve().parent.parent / "test_executable_scripts" / "test_rccl.py"
)
CORRECTNESS_EXECUTABLES = {
    "all_gather_perf",
    "alltoallv_perf",
    "broadcast_perf",
    "alltoall_perf",
    "all_reduce_perf",
    "reduce_perf",
    "hypercube_perf",
    "gather_perf",
    "scatter_perf",
    "sendrecv_perf",
    "reduce_scatter_perf",
}
VISIBILITY_VARIABLES = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)
FAKE_EXECUTABLE = r"""
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
if name == "rocminfo":
    for index in range(int(os.environ["FAKE_GPU_COUNT"])):
        print(f"Agent {index}:\n  Name: gfx942")
    sys.exit(0)

visibility = {
    key: os.environ.get(key)
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
}
record = {
    "name": name,
    "args": sys.argv[1:],
    "visibility": visibility,
    "unit_range": [os.environ.get(key) for key in
                   ("UT_MIN_GPUS", "UT_MAX_GPUS", "UT_POW2_GPUS")],
}
with open(os.environ["FAKE_RCCL_RECORD"], "a") as output:
    output.write(json.dumps(record) + "\n")

assert visibility == json.loads(os.environ["FAKE_EXPECTED_VISIBILITY"]), record
if name == "rccl-UnitTests":
    assert record["unit_range"] == ["2", os.environ["FAKE_EXPECTED_UNIT_MAX"], "1"], record
    assert sys.argv[1:] == [], record
else:
    count = os.environ["FAKE_EXPECTED_PERF_GPUS"]
    assert sys.argv[1:] == (["-g", count] if count else []), record
"""


@unittest.skipUnless(os.name == "posix", "RCCL fake executables use POSIX shebangs")
class RCCLTestRunnerTest(unittest.TestCase):
    def run_rccl(
        self,
        test_type: str,
        multi_gpu: str,
        gpu_count: int,
        visibility: dict[str, str] | None = None,
        expected_perf_gpus: int | None = None,
        expected_unit_max: int = 2,
    ) -> tuple[subprocess.CompletedProcess, list[dict]]:
        with tempfile.TemporaryDirectory() as temporary_dir:
            binary_dir = Path(temporary_dir)
            for executable in CORRECTNESS_EXECUTABLES | {"rocminfo", "rccl-UnitTests"}:
                path = binary_dir / executable
                path.write_text(f"#!{sys.executable}\n" + FAKE_EXECUTABLE)
                path.chmod(0o755)
            record_path = binary_dir / "calls.jsonl"
            env = os.environ.copy()
            for key in VISIBILITY_VARIABLES:
                env.pop(key, None)
            env.update(visibility or {})
            env.update(
                {
                    "THEROCK_BIN_DIR": str(binary_dir),
                    "TEST_TYPE": test_type,
                    "MULTI_GPU_TEST": multi_gpu,
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "PYTEST_ADDOPTS": "",
                    "FAKE_GPU_COUNT": str(gpu_count),
                    "FAKE_RCCL_RECORD": str(record_path),
                    "FAKE_EXPECTED_VISIBILITY": json.dumps(
                        {key: env.get(key) for key in VISIBILITY_VARIABLES}
                    ),
                    "FAKE_EXPECTED_UNIT_MAX": str(expected_unit_max),
                    "FAKE_EXPECTED_PERF_GPUS": (
                        "" if expected_perf_gpus is None else str(expected_perf_gpus)
                    ),
                }
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    str(RCCL_SCRIPT),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            records = (
                [json.loads(line) for line in record_path.read_text().splitlines()]
                if record_path.exists()
                else []
            )
            return result, records

    def assert_all_executed(self, result, records):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            {record["name"] for record in records},
            CORRECTNESS_EXECUTABLES | {"rccl-UnitTests"},
        )
        self.assertEqual(len(records), len(CORRECTNESS_EXECUTABLES) + 1)

    def test_full_multi_gpu_tiers_use_all_eight_visible_gpus(self):
        for tier in ("comprehensive", "full"):
            with self.subTest(tier=tier):
                result, records = self.run_rccl(
                    tier, "1", 8, expected_perf_gpus=8, expected_unit_max=8
                )
                self.assert_all_executed(result, records)

    def test_multi_gpu_preserves_and_caps_caller_visibility(self):
        for hip_variable in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            with self.subTest(mask=hip_variable):
                result, records = self.run_rccl(
                    "full",
                    "1",
                    8,
                    visibility={hip_variable: "4,5", "ROCR_VISIBLE_DEVICES": "4,5"},
                    expected_perf_gpus=2,
                    expected_unit_max=2,
                )
                self.assert_all_executed(result, records)

    def test_multi_gpu_requires_two_visible_gpus(self):
        for gpu_count, visibility in (
            (0, {}),
            (1, {}),
            (8, {"HIP_VISIBLE_DEVICES": "4"}),
        ):
            with self.subTest(gpu_count=gpu_count, visibility=visibility):
                result, records = self.run_rccl("full", "1", gpu_count, visibility)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("require at least 2 visible GPUs", result.stdout)
                self.assertEqual(records, [], "No RCCL binary may run with <2 GPUs")

    def test_ordinary_tiers_retain_default_correctness_and_two_gpu_units(self):
        for tier, multi_gpu in (("quick", "1"), ("standard", "1"), ("full", "0")):
            with self.subTest(tier=tier, multi_gpu=multi_gpu):
                result, records = self.run_rccl(
                    tier, multi_gpu, 8, visibility={"HIP_VISIBLE_DEVICES": "4,5"}
                )
                self.assert_all_executed(result, records)

    def test_ordinary_single_gpu_skips_units_and_runs_correctness(self):
        result, records = self.run_rccl("standard", "0", 1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 skipped", result.stdout)
        self.assertEqual(
            {record["name"] for record in records}, CORRECTNESS_EXECUTABLES
        )


if __name__ == "__main__":
    unittest.main()
