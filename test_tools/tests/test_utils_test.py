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


class TestRunSettingsTest(unittest.TestCase):
    def test_from_env_parses_common_ci_settings(self):
        settings = test_utils.TestRunSettings.from_env(
            test_dir="tests",
            rocm_path="install",
            env={
                "TEST_TYPE": "STANDARD",
                "AMDGPU_FAMILIES": "family=gfx942",
                "SHARD_INDEX": "2",
                "TOTAL_SHARDS": "4",
            },
        )

        self.assertEqual(settings.test_dir, Path("tests"))
        self.assertEqual(settings.rocm_path, Path("install"))
        self.assertEqual(settings.category, "standard")
        self.assertEqual(settings.gpu_arch, "gfx942")
        self.assertEqual(settings.shard_index, 2)
        self.assertEqual(settings.total_shards, 4)

    def test_invalid_env_category_falls_back_to_quick(self):
        settings = test_utils.TestRunSettings.from_env(
            test_dir="tests",
            env={
                "TEST_TYPE": "smoke",
                "SHARD_INDEX": "1",
                "TOTAL_SHARDS": "1",
            },
        )
        self.assertEqual(settings.category, "quick")

    def test_invalid_shards_raise(self):
        with self.assertRaises(ValueError):
            test_utils.TestRunSettings(
                test_dir="tests", shard_index="3", total_shards="2"
            )

    def test_with_ctest_returns_updated_settings(self):
        settings = test_utils.TestRunSettings(
            test_dir="tests",
            category="quick",
            gpu_arch="gfx1151",
        )

        updated = settings.with_ctest(
            available_gpu_archs={"gfx115X"},
            exclude_labels={"quick_exclude"},
            parallel="8",
            timeout_seconds="7200",
            output_on_failure=False,
            verbose=False,
            extra_args=["--tests-regex", "smoke"],
        )

        self.assertEqual(settings.available_gpu_archs, frozenset())
        self.assertEqual(updated.available_gpu_archs, frozenset({"gfx115X"}))
        self.assertEqual(updated.exclude_labels, frozenset({"quick_exclude"}))
        self.assertEqual(updated.ctest_parallel, 8)
        self.assertEqual(updated.ctest_timeout_seconds, 7200)
        self.assertFalse(updated.ctest_output_on_failure)
        self.assertFalse(updated.ctest_verbose)
        self.assertEqual(updated.extra_ctest_args, ("--tests-regex", "smoke"))

    def test_with_ctest_labels_applies_discovered_label_sets(self):
        settings = test_utils.TestRunSettings(test_dir="tests")
        updated = settings.with_ctest_labels(
            test_utils.CTestLabels(
                gpu_archs={"gfx942"},
                exclude_labels={"quick_exclude"},
            )
        )

        self.assertEqual(updated.available_gpu_archs, frozenset({"gfx942"}))
        self.assertEqual(updated.exclude_labels, frozenset({"quick_exclude"}))


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

    def test_discover_ctest_labels_requires_tests_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:

            def runner(cmd, **kwargs):
                if "-N" in cmd:
                    return SimpleNamespace(stdout="Test project\n")
                return SimpleNamespace(stdout="quick\n")

            with self.assertRaises(RuntimeError):
                test_utils.discover_ctest_labels(temp_dir, runner)

    def test_discover_ctest_labels_can_skip_test_count_check(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []

            def runner(cmd, **kwargs):
                calls.append(cmd)
                return SimpleNamespace(stdout="quick\nex_gpu_gfx115X\n")

            labels = test_utils.discover_ctest_labels(
                temp_dir, runner, require_tests=False
            )

            self.assertEqual(labels.gpu_archs, {"gfx115X"})
            self.assertEqual(labels.exclude_labels, set())
            self.assertEqual(
                calls,
                [["ctest", "--print-labels", "--test-dir", os.fspath(temp_dir)]],
            )


class CTestRunSettingsHelpersTest(unittest.TestCase):
    def test_build_ctest_command_from_settings(self):
        settings = test_utils.TestRunSettings(
            test_dir="tests",
            category="standard",
            gpu_arch="gfx1151",
            shard_index=2,
            total_shards=4,
        ).with_ctest(
            available_gpu_archs={"gfx115X"},
            exclude_labels={"standard_exclude"},
            parallel=8,
            timeout_seconds=7200,
            extra_args=["--tests-regex", "smoke"],
        )

        self.assertEqual(
            test_utils.build_ctest_command(settings),
            [
                "ctest",
                "-L",
                "standard",
                "-L",
                "ex_gpu_gfx115X",
                "-LE",
                "standard_exclude",
                "--output-on-failure",
                "--parallel",
                "8",
                "--timeout",
                "7200",
                "--test-dir",
                "tests",
                "-V",
                "--tests-information",
                "2,,4",
                "--tests-regex",
                "smoke",
            ],
        )

    def test_build_test_env_applies_common_and_project_settings(self):
        settings = test_utils.TestRunSettings(
            test_dir="tests",
            rocm_path="install",
            shard_index=2,
            total_shards=4,
        )

        env = test_utils.build_test_env(
            settings,
            base_env={"PATH": "base-bin", "LD_LIBRARY_PATH": "base-lib"},
            path_prepend={
                "PATH": [Path("install") / "bin"],
                "LD_LIBRARY_PATH": [Path("install") / "lib"],
            },
            extra_env={"HIP_VISIBLE_DEVICES": "0"},
        )

        self.assertEqual(env["ROCM_PATH"], "install")
        self.assertEqual(env["GTEST_SHARD_INDEX"], "1")
        self.assertEqual(env["GTEST_TOTAL_SHARDS"], "4")
        self.assertEqual(
            env["PATH"],
            os.pathsep.join([os.fspath(Path("install") / "bin"), "base-bin"]),
        )
        self.assertEqual(
            env["LD_LIBRARY_PATH"],
            os.pathsep.join([os.fspath(Path("install") / "lib"), "base-lib"]),
        )
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")

    def test_run_ctest_uses_settings_command_and_default_cwd(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return SimpleNamespace(returncode=0)

        settings = test_utils.TestRunSettings(
            test_dir="tests",
            rocm_path="install",
            category="quick",
            shard_index=1,
            total_shards=2,
        )
        env = {"ROCM_PATH": "install"}

        result = test_utils.run_ctest(settings, env=env, runner=runner, check=True)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls[0][0][0], "ctest")
        self.assertEqual(calls[0][1]["cwd"], Path("install"))
        self.assertEqual(calls[0][1]["env"], env)
        self.assertTrue(calls[0][1]["check"])

    def test_run_ctest_can_discover_labels_before_running(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []

            def runner(cmd, **kwargs):
                calls.append((cmd, kwargs))
                if "-N" in cmd:
                    return SimpleNamespace(stdout="  Test #1: alpha\n")
                if "--print-labels" in cmd:
                    return SimpleNamespace(stdout="quick_exclude\nex_gpu_gfx115X\n")
                return SimpleNamespace(returncode=0)

            settings = test_utils.TestRunSettings(
                test_dir=temp_dir,
                category="quick",
                gpu_arch="gfx1151",
            )

            result = test_utils.run_ctest(
                settings,
                runner=runner,
                discover_labels=True,
            )

            self.assertEqual(result.returncode, 0)
            final_cmd = calls[-1][0]
            self.assertIn("ex_gpu_gfx115X", final_cmd)
            self.assertIn("quick_exclude", final_cmd)
            self.assertEqual(len(calls), 3)


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
