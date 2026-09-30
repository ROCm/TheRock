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
                    ["nn/test_convolution", "inductor/test_max_autotune"],
                )

        self.assertEqual(env["TEST_CONFIG"], "default")
        self.assertEqual(env["IN_WHEEL_TEST"], "1")
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


class ExcludedModulesTest(unittest.TestCase):
    def test_reads_exclusions_from_skip_tests(self):
        modules = runner.get_excluded_modules(
            amdgpu_family=["gfx942"], pytorch_version="2.15", platform="Linux"
        )
        self.assertIn("nn/test_convolution", modules)
        self.assertIn("inductor/test_max_autotune", modules)

    def test_exclusions_are_module_paths_not_test_cases(self):
        skips = runner.get_tests(
            amdgpu_family=["gfx942"], pytorch_version="2.15", platform="Linux"
        )
        for module in runner.get_excluded_modules(
            amdgpu_family=["gfx942"], pytorch_version="2.15", platform="Linux"
        ):
            self.assertNotIn(module, skips)


class MainTest(unittest.TestCase):
    def test_invokes_relative_test_sh_from_pytorch_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            pytorch_dir = Path(tmp)
            test_sh = pytorch_dir / ".ci" / "pytorch" / "test.sh"
            test_sh.parent.mkdir(parents=True)
            test_sh.touch()
            result = mock.Mock(returncode=0)
            with (
                mock.patch.object(runner, "check_pytorch_source_version"),
                mock.patch.object(runner, "reconcile_agent_visibility_env"),
                mock.patch.object(
                    runner, "configure_gpu_visibility", return_value=["gfx942"]
                ),
                mock.patch.object(
                    runner, "detect_pytorch_version", return_value="2.15"
                ),
                mock.patch.object(runner, "get_tests", return_value=""),
                mock.patch.object(runner, "get_excluded_modules", return_value=[]),
                mock.patch.object(
                    runner, "configure_environment", return_value={"CI": "1"}
                ),
                mock.patch.object(runner.subprocess, "run", return_value=result) as run,
            ):
                returncode = runner.main(["--pytorch-dir", tmp])

        self.assertEqual(returncode, 0)
        run.assert_called_once_with(
            ["bash", ".ci/pytorch/test.sh"],
            cwd=pytorch_dir,
            env={"CI": "1"},
        )


if __name__ == "__main__":
    unittest.main()
