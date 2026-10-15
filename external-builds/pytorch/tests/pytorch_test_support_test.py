# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
PYTORCH_TOOLS_DIR = THIS_DIR.parent
sys.path.insert(0, os.fspath(PYTORCH_TOOLS_DIR))

import pytorch_test_support as support


class PinnedDependencyTest(unittest.TestCase):
    def test_reads_current_and_legacy_pin_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            pytorch_dir = Path(tmp)
            current_pin = pytorch_dir / ".github/ci_commit_pins/torchcomms.txt"
            legacy_pin = pytorch_dir / ".ci/docker/ci_commit_pins/spmd_types.txt"
            current_pin.parent.mkdir(parents=True)
            legacy_pin.parent.mkdir(parents=True)
            current_pin.write_text("torchcomms-sha\n", encoding="utf-8")
            legacy_pin.write_text("spmd-sha\n", encoding="utf-8")

            self.assertEqual(
                support.read_pinned_commit(pytorch_dir, "torchcomms"),
                "torchcomms-sha",
            )
            self.assertEqual(
                support.read_pinned_commit(pytorch_dir, "spmd_types"),
                "spmd-sha",
            )

    def test_installs_pinned_distributed_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            pytorch_dir = Path(tmp)
            pin_dir = pytorch_dir / ".github/ci_commit_pins"
            pin_dir.mkdir(parents=True)
            for package in support.PINNED_PACKAGE_REPOSITORIES:
                (pin_dir / f"{package}.txt").write_text(
                    f"{package}-sha\n", encoding="utf-8"
                )

            def create_wheel(command, **_kwargs):
                if "wheel" in command:
                    wheel_dir = Path(command[command.index("--wheel-dir") + 1])
                    package = command[-1].split("/")[-1].split(".git")[0]
                    (wheel_dir / f"{package}-1.0-py3-none-any.whl").touch()

            with mock.patch.object(
                support.subprocess, "check_call", side_effect=create_wheel
            ) as check_call:
                support.install_distributed_test_dependencies(pytorch_dir)

        self.assertEqual(check_call.call_count, 4)
        torchcomms_build = next(
            call
            for call in check_call.call_args_list
            if "torchcomms.git" in call.args[0][-1]
        )
        self.assertEqual(torchcomms_build.kwargs["env"]["USE_GLOO"], "1")
        self.assertEqual(torchcomms_build.kwargs["env"]["USE_NCCL"], "0")

    def test_skips_dependencies_not_pinned_by_older_refs(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            support.subprocess, "check_call"
        ) as check_call:
            support.install_distributed_test_dependencies(Path(tmp))

        check_call.assert_not_called()


class CustomTestSelectionTest(unittest.TestCase):
    def test_full_default_shard_two_runs_custom_tests(self):
        self.assertTrue(
            support.should_run_custom_tests(
                test_config="default",
                shard=2,
                num_shards=6,
                included_tests=None,
            )
        )

    def test_other_suites_do_not_run_custom_tests(self):
        test_cases = [
            ("default", 1, 6, None),
            ("default", 2, 6, ["test_torch"]),
            ("distributed", 2, 3, None),
            ("inductor", 2, 2, None),
        ]
        for test_config, shard, num_shards, included_tests in test_cases:
            with self.subTest(test_config=test_config, shard=shard):
                self.assertFalse(
                    support.should_run_custom_tests(
                        test_config=test_config,
                        shard=shard,
                        num_shards=num_shards,
                        included_tests=included_tests,
                    )
                )

    def test_unsharded_full_default_runs_custom_tests(self):
        self.assertTrue(
            support.should_run_custom_tests(
                test_config="default",
                shard=0,
                num_shards=0,
                included_tests=None,
            )
        )

    def test_cmake_command_matches_upstream_build_inputs(self):
        pytorch_dir = Path("/pytorch")
        source_dir = pytorch_dir / "test/custom_operator"
        build_dir = Path("/tmp/custom-operator")
        with mock.patch.object(
            support.site, "getsitepackages", return_value=["/venv/site-packages"]
        ):
            command = support.cmake_configure_command(
                pytorch_dir, source_dir, build_dir
            )

        self.assertIn("-G", command)
        self.assertIn("Ninja", command)
        self.assertIn(
            "-DCMAKE_PREFIX_PATH=/venv/site-packages;/venv/site-packages/torch",
            command,
        )
        self.assertIn(
            "-DCMAKE_MODULE_PATH=/pytorch/cmake/public",
            command,
        )
        self.assertIn("-DUSE_ROCM=ON", command)


if __name__ == "__main__":
    unittest.main()
