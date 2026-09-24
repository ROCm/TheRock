"""Unit tests for external-builds/pytorch/run_pytorch_test_sh.py."""

import argparse
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
PYTORCH_DIR = THIS_DIR.parents[2] / "external-builds" / "pytorch"
sys.path.insert(0, os.fspath(PYTORCH_DIR))

import run_pytorch_test_sh as runner


class ParseArgsTest(unittest.TestCase):
    def test_distributed_selects_all_gpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_sh = Path(tmp) / ".ci" / "pytorch" / "test.sh"
            test_sh.parent.mkdir(parents=True)
            test_sh.touch()

            args, pytest_args = runner.parse_args(
                [
                    "--pytorch-dir",
                    tmp,
                    "--test-config",
                    "distributed",
                    "--",
                    "--continue-on-collection-errors",
                ]
            )

        self.assertEqual(args.device_query, "all")
        self.assertEqual(args.gpu_policy, "all")
        self.assertEqual(pytest_args, ["--continue-on-collection-errors"])


class ConfigureEnvironmentTest(unittest.TestCase):
    def args(self, pytorch_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            amdgpu_family="gfx94X-dcgpu",
            pytorch_dir=pytorch_dir,
            test_config="default",
            shard=2,
            num_shards=6,
            include=["test_nn", "test_torch"],
            exclude=["test_bad"],
            cache=False,
        )

    def test_translates_therock_options_for_test_sh(self):
        with tempfile.TemporaryDirectory() as tmp:
            pytorch_dir = Path(tmp)
            existing_module = pytorch_dir / "test" / "nn" / "test_convolution.py"
            existing_module.parent.mkdir(parents=True)
            existing_module.touch()
            with mock.patch.dict(os.environ, {"PYTEST_ADDOPTS": "--reruns 2"}):
                env = runner.configure_environment(
                    self.args(pytorch_dir),
                    ["--continue-on-collection-errors"],
                    "not flaky_test",
                )

        self.assertEqual(env["TEST_CONFIG"], "default")
        self.assertEqual(env["SHARD_NUMBER"], "2")
        self.assertEqual(env["NUM_TEST_SHARDS"], "6")
        self.assertEqual(env["TESTS_TO_INCLUDE"], "test_nn test_torch")
        self.assertEqual(env["TESTS_TO_EXCLUDE"], "nn/test_convolution test_bad")
        self.assertEqual(
            shlex.split(env["PYTEST_ADDOPTS"]),
            [
                "--reruns",
                "2",
                "--continue-on-collection-errors",
                "-k",
                "not flaky_test",
                "-p",
                "no:cacheprovider",
                "--timeout",
                "900",
            ],
        )


if __name__ == "__main__":
    unittest.main()
