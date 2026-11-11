# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class RppShardingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "share/rpp/test").mkdir(parents=True)
        script = Path(__file__).parent.parent / "test_executable_scripts/test_rpp.py"
        spec = importlib.util.spec_from_file_location(
            "test_rpp_sharding_fixture", script
        )
        self.runner = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"THEROCK_BIN_DIR": str(root / "bin")}):
            spec.loader.exec_module(self.runner)
        self.runner.THEROCK_TEST_DIR = root / "build"

    def test_csan_native_categories_receive_distinct_ctest_strides(self):
        for tier in ("comprehensive", "full"):
            self.runner.TEST_TYPE = tier
            for index in range(1, 9):
                with patch.dict(
                    os.environ,
                    {
                        "BUILD_VARIANT": "csan",
                        "TOTAL_SHARDS": "8",
                        "SHARD_INDEX": str(index),
                    },
                ):
                    self.assertEqual(
                        self.runner.shard_args(), ["--tests-information", f"{index},,8"]
                    )

    def test_other_variants_and_categories_keep_existing_commands(self):
        for variant, tier in (
            ("release", "comprehensive"),
            ("asan", "full"),
            ("csan", "standard"),
            ("csan", "quick"),
            ("csan", "ffm-comprehensive"),
        ):
            self.runner.TEST_TYPE = tier
            with patch.dict(
                os.environ, {"BUILD_VARIANT": variant, "TOTAL_SHARDS": "8"}
            ):
                self.assertEqual(self.runner.shard_args(), [])

    def test_list_and_execution_use_the_same_selected_case(self):
        self.runner.TEST_TYPE = "comprehensive"
        with patch.dict(
            os.environ,
            {"BUILD_VARIANT": "csan", "TOTAL_SHARDS": "8", "SHARD_INDEX": "5"},
        ), patch.object(
            self.runner.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0),
                subprocess.CompletedProcess([], 0, "Total Tests: 1\n", ""),
                subprocess.CompletedProcess([], 0),
            ],
        ) as run:
            self.runner.execute_tests({})
        listing = run.call_args_list[1].args[0]
        execution = run.call_args_list[2].args[0]
        self.assertIn("-N", listing)
        self.assertEqual(listing[-2:], ["--tests-information", "5,,8"])
        self.assertEqual(execution[-2:], listing[-2:])

    def test_invalid_shard_fails(self):
        self.runner.TEST_TYPE = "comprehensive"
        with patch.dict(
            os.environ,
            {"BUILD_VARIANT": "csan", "TOTAL_SHARDS": "8", "SHARD_INDEX": "9"},
        ):
            with self.assertRaises(ValueError):
                self.runner.shard_args()


if __name__ == "__main__":
    unittest.main()
