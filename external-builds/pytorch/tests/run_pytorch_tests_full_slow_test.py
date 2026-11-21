# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the slow test config in run_pytorch_tests_full.py."""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

import run_pytorch_tests_full
from generate_test_sharding_matrix import build_matrix


class SetupEnvTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pytorch_dir = Path(self.temp_dir.name)
        self.saved_sys_path = list(sys.path)

    def tearDown(self):
        sys.path[:] = self.saved_sys_path
        self.temp_dir.cleanup()

    def _setup_env(self, test_config: str, amdgpu_family: str) -> dict[str, str]:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            run_pytorch_tests_full, "reconcile_agent_visibility_env"
        ):
            run_pytorch_tests_full.setup_env(
                pytorch_dir=self.pytorch_dir,
                test_config=test_config,
                amdgpu_family=amdgpu_family,
            )
            return dict(os.environ)

    def test_slow_enables_only_slow_tests(self):
        env = self._setup_env("slow", "gfx94X-dcgpu")
        self.assertEqual(env["TEST_CONFIG"], "slow")
        self.assertEqual(env["PYTORCH_TEST_WITH_SLOW"], "1")
        self.assertEqual(env["PYTORCH_TEST_SKIP_FAST"], "1")
        self.assertEqual(
            env["BUILD_ENVIRONMENT"], run_pytorch_tests_full.SLOW_BUILD_ENVIRONMENT
        )

    def test_default_leaves_slow_tests_disabled(self):
        env = self._setup_env("default", "gfx94X-dcgpu")
        self.assertNotIn("PYTORCH_TEST_WITH_SLOW", env)
        self.assertNotIn("PYTORCH_TEST_SKIP_FAST", env)
        self.assertEqual(env["BUILD_ENVIRONMENT"], "linux-noble-rocm-py3.12-mi300")


class BuildRunTestCmdTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pytorch_dir = Path(self.temp_dir.name)
        (self.pytorch_dir / "test").mkdir()
        (self.pytorch_dir / "test" / "run_test.py").write_text("")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _cmd(self, test_config: str) -> list[str]:
        args = argparse.Namespace(
            pytorch_dir=self.pytorch_dir,
            test_config=test_config,
            exclude_jit_executor=False,
            exclude_distributed=True,
            exclude_quantization=True,
            dry_run=False,
            shard=1,
            num_shards=4,
            include=None,
            exclude=None,
            cache=True,
        )
        return run_pytorch_tests_full.build_run_test_cmd(args, "", [])

    def _timeout(self, cmd: list[str]) -> int:
        return int(cmd[cmd.index("--timeout") + 1])

    def test_slow_uses_default_flags_and_longer_timeout(self):
        cmd = self._cmd("slow")
        self.assertIn("--exclude-distributed-tests", cmd)
        self.assertIn("--exclude-quantization-tests", cmd)
        self.assertNotIn("--distributed-tests", cmd)
        self.assertEqual(
            self._timeout(cmd), run_pytorch_tests_full.SLOW_PYTEST_TIMEOUT_SECONDS
        )

    def test_default_timeout_unchanged(self):
        self.assertEqual(
            self._timeout(self._cmd("default")),
            run_pytorch_tests_full.PYTEST_TIMEOUT_SECONDS,
        )


class ShardingMatrixTest(unittest.TestCase):
    def test_slow_runs_four_shards_on_single_gpu_runner(self):
        matrix = build_matrix(["slow"], "single-gpu", "multi-gpu")
        self.assertEqual(
            [(e["shard"], e["num_shards"], e["runs_on"]) for e in matrix["include"]],
            [(n, 4, "single-gpu") for n in range(1, 5)],
        )


if __name__ == "__main__":
    unittest.main()
