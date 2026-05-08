# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import test_utils


class TestCategoryTest(unittest.TestCase):
    def test_valid_categories_are_preserved(self):
        self.assertEqual(
            test_utils.VALID_TEST_CATEGORIES,
            {"quick", "standard", "comprehensive", "full"},
        )
        self.assertEqual(test_utils.normalize_test_category("quick"), "quick")
        self.assertEqual(test_utils.normalize_test_category(" STANDARD "), "standard")

    def test_invalid_categories_fall_back_to_quick(self):
        self.assertEqual(test_utils.normalize_test_category(None), "quick")
        self.assertEqual(test_utils.normalize_test_category(""), "quick")
        self.assertEqual(test_utils.normalize_test_category("smoke"), "quick")


class GpuArchTest(unittest.TestCase):
    def test_extract_gpu_arch(self):
        self.assertEqual(test_utils.extract_gpu_arch(None), "")
        self.assertEqual(test_utils.extract_gpu_arch(""), "")
        self.assertEqual(test_utils.extract_gpu_arch("gfx1151"), "gfx1151")
        self.assertEqual(test_utils.extract_gpu_arch("family=gfx942,gfx90a"), "gfx942")
        self.assertEqual(test_utils.extract_gpu_arch("GFX90A"), "gfx90a")
        self.assertEqual(test_utils.extract_gpu_arch("generic"), "")

    def test_find_matching_gpu_arch_exact_match(self):
        available = {"gfx1151", "gfx115X", "gfx11X"}
        self.assertEqual(
            test_utils.find_matching_gpu_arch("gfx1151", available), "gfx1151"
        )

    def test_find_matching_gpu_arch_uses_most_specific_wildcard(self):
        available = {"gfx115X", "gfx11X"}
        self.assertEqual(
            test_utils.find_matching_gpu_arch("gfx1151", available), "gfx115X"
        )

    def test_find_matching_gpu_arch_uses_less_specific_wildcard(self):
        available = {"gfx1150", "gfx94X", "gfx11X"}
        self.assertEqual(
            test_utils.find_matching_gpu_arch("gfx1151", available), "gfx11X"
        )

    def test_find_matching_gpu_arch_returns_none_without_match(self):
        self.assertIsNone(
            test_utils.find_matching_gpu_arch("gfx1151", {"gfx94X", "gfx90a"})
        )
        self.assertIsNone(test_utils.find_matching_gpu_arch("gfx1151", set()))

    def test_find_matching_gpu_arch_stops_before_too_broad_pattern(self):
        self.assertEqual(
            test_utils.find_matching_gpu_arch("gfx90a", {"gfx90X"}), "gfx90X"
        )
        self.assertIsNone(test_utils.find_matching_gpu_arch("gfx90a", {"gfx9X"}))


class ShardingTest(unittest.TestCase):
    def test_gtest_shard_env_converts_to_zero_based_index(self):
        self.assertEqual(
            test_utils.gtest_shard_env("2", "4"),
            {"GTEST_SHARD_INDEX": "1", "GTEST_TOTAL_SHARDS": "4"},
        )

    def test_ctest_shard_args_keep_one_based_index(self):
        self.assertEqual(
            test_utils.ctest_shard_args("2", "4"),
            ["--tests-information", "2,,4"],
        )

    def test_shard_values_must_be_positive(self):
        with self.assertRaises(ValueError):
            test_utils.gtest_shard_env("0", "4")
        with self.assertRaises(ValueError):
            test_utils.ctest_shard_args("1", "0")

    def test_shard_index_must_not_exceed_total_shards(self):
        with self.assertRaises(ValueError):
            test_utils.gtest_shard_env("5", "4")
        with self.assertRaises(ValueError):
            test_utils.ctest_shard_args("5", "4")


class CTestLabelArgsTest(unittest.TestCase):
    def test_category_label_is_included(self):
        self.assertEqual(
            test_utils.build_ctest_label_args("quick", "", set(), set()),
            ["-L", "quick", "-LE", "ex_gpu"],
        )

    def test_generic_gpu_excludes_gpu_specific_tests(self):
        self.assertEqual(
            test_utils.build_ctest_label_args("standard", "generic", set(), set()),
            ["-L", "standard", "-LE", "ex_gpu"],
        )

    def test_matching_gpu_adds_gpu_label(self):
        args = test_utils.build_ctest_label_args(
            "quick", "gfx1151", {"gfx115X", "gfx11X"}, set()
        )
        self.assertEqual(args, ["-L", "quick", "-L", "ex_gpu_gfx115X"])

    def test_no_matching_gpu_excludes_gpu_specific_tests(self):
        args = test_utils.build_ctest_label_args("quick", "gfx1151", {"gfx94X"}, set())
        self.assertEqual(args, ["-L", "quick", "-LE", "ex_gpu"])

    def test_category_exclude_is_combined_with_gpu_exclude(self):
        args = test_utils.build_ctest_label_args(
            "quick", "", set(), {"quick_exclude", "standard_exclude"}
        )
        self.assertEqual(args, ["-L", "quick", "-LE", "quick_exclude|ex_gpu"])

    def test_invalid_category_uses_quick_policy(self):
        args = test_utils.build_ctest_label_args("smoke", "", set(), {"quick_exclude"})
        self.assertEqual(args, ["-L", "quick", "-LE", "quick_exclude|ex_gpu"])


class CTestDiscoveryTest(unittest.TestCase):
    def test_count_ctest_tests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []

            def runner(cmd, **kwargs):
                calls.append((cmd, kwargs))
                return SimpleNamespace(
                    stdout="Test project\n  Test #1: alpha\n  Test #2: beta\n"
                )

            self.assertEqual(test_utils.count_ctest_tests(temp_dir, runner), 2)
            self.assertEqual(
                calls[0][0], ["ctest", "-N", "--test-dir", os.fspath(temp_dir)]
            )
            self.assertTrue(calls[0][1]["capture_output"])
            self.assertTrue(calls[0][1]["text"])
            self.assertTrue(calls[0][1]["check"])

    def test_read_ctest_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:

            def runner(cmd, **kwargs):
                return SimpleNamespace(
                    stdout="quick\nstandard\nex_gpu_gfx115X\nquick_exclude\n"
                )

            self.assertEqual(
                test_utils.read_ctest_labels(temp_dir, runner),
                {"quick", "standard", "ex_gpu_gfx115X", "quick_exclude"},
            )

    def test_missing_test_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            test_utils.count_ctest_tests(Path("does-not-exist"))

    def test_parse_ctest_labels(self):
        labels = test_utils.parse_ctest_labels(
            [
                "quick",
                " standard_exclude ",
                "ex_gpu_gfx115X",
                "ex_gpu_gfx942",
                "ex_gpu_notgfx",
            ]
        )
        self.assertEqual(labels.gpu_archs, {"gfx115X", "gfx942"})
        self.assertEqual(labels.exclude_labels, {"standard_exclude"})


class RocminfoTest(unittest.TestCase):
    ROCMINFO_OUTPUT = """
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Name:                    gfx942
  Name:                    cpu
  Name:                    GFX1100
"""

    def test_parse_rocminfo_gpu_archs_preserves_visible_gpu_records(self):
        self.assertEqual(
            test_utils.parse_rocminfo_gpu_archs(self.ROCMINFO_OUTPUT),
            ["gfx942", "gfx942", "gfx1100"],
        )

    def test_get_visible_gpu_count_uses_runner(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return SimpleNamespace(stdout=self.ROCMINFO_OUTPUT)

        self.assertEqual(
            test_utils.get_visible_gpu_count(
                env={"ROCR_VISIBLE_DEVICES": "0"}, runner=runner
            ),
            3,
        )
        self.assertEqual(calls[0][0], ["rocminfo"])
        self.assertEqual(calls[0][1]["env"], {"ROCR_VISIBLE_DEVICES": "0"})
        self.assertFalse(calls[0][1]["check"])

    def test_get_visible_gpu_count_prefers_rocm_bin_dir_rocminfo(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rocminfo_path = Path(temp_dir) / "rocminfo"
            rocminfo_path.write_text("")
            calls = []

            def runner(cmd, **kwargs):
                calls.append((cmd, kwargs))
                return SimpleNamespace(stdout=self.ROCMINFO_OUTPUT)

            self.assertEqual(
                test_utils.get_visible_gpu_count(rocm_bin_dir=temp_dir, runner=runner),
                3,
            )
            self.assertEqual(calls[0][0], [os.fspath(rocminfo_path)])

    def test_get_first_gpu_architecture_returns_first_visible_gpu(self):
        def runner(cmd, **kwargs):
            return SimpleNamespace(stdout=self.ROCMINFO_OUTPUT)

        self.assertEqual(test_utils.get_first_gpu_architecture(runner=runner), "gfx942")

    def test_get_first_gpu_architecture_raises_without_visible_gpu(self):
        def runner(cmd, **kwargs):
            return SimpleNamespace(stdout="Name: cpu\n")

        with self.assertRaises(RuntimeError):
            test_utils.get_first_gpu_architecture(runner=runner)


class ArtifactGroupTest(unittest.TestCase):
    def test_is_asan_artifact_group(self):
        self.assertFalse(test_utils.is_asan_artifact_group(None))
        self.assertFalse(test_utils.is_asan_artifact_group(""))
        self.assertFalse(test_utils.is_asan_artifact_group("core-runtime"))
        self.assertTrue(test_utils.is_asan_artifact_group("core-asan"))
        self.assertTrue(test_utils.is_asan_artifact_group("CORE-ASAN"))


if __name__ == "__main__":
    unittest.main()
