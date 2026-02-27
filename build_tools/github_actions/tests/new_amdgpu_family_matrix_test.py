import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from new_amdgpu_family_matrix_data import (
    amdgpu_family_info_matrix_all,
    amdgpu_family_predefined_groups,
    all_build_variants,
)
from new_amdgpu_family_matrix_types import (
    BuildConfig,
    GpuRunners,
    MatrixEntry,
    PlatformConfig,
    ReleaseConfig,
    TestConfig,
)


class TestMatrixKeys(unittest.TestCase):
    """Verify the integrity of canonical keys in the matrix."""

    def test_all_keys_are_unique(self):
        keys = amdgpu_family_info_matrix_all.keys()
        self.assertEqual(len(keys), len(set(keys)), "Duplicate canonical keys found")

    def test_predefined_groups_reference_valid_keys(self):
        all_keys = set(amdgpu_family_info_matrix_all.keys())
        for group_name, group_keys in amdgpu_family_predefined_groups.items():
            for key in group_keys:
                self.assertIn(
                    key,
                    all_keys,
                    f"Group {group_name!r} references unknown key {key!r}",
                )

    def test_all_known_archs_present(self):
        keys = amdgpu_family_info_matrix_all.keys()
        for expected in [
            "gfx906",
            "gfx908",
            "gfx90a",
            "gfx94X-dcgpu",
            "gfx950-dcgpu",
            "gfx101X-dgpu",
            "gfx103X-dgpu",
            "gfx110X-all",
            "gfx1150",
            "gfx1151",
            "gfx1152",
            "gfx1153",
            "gfx120X-all",
        ]:
            self.assertIn(
                expected, keys, f"Expected key {expected!r} missing from matrix"
            )


class TestEntryLookup(unittest.TestCase):
    """Verify get_entry and get_default_for_family behavior."""

    def test_exact_key_lookup(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx94X-dcgpu")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfx94X-dcgpu")

    def test_family_name_resolves_to_default(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx950")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfx950-dcgpu")

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(
            amdgpu_family_info_matrix_all.get_entry("GFX950"),
            amdgpu_family_info_matrix_all.get_entry("gfx950"),
        )
        self.assertEqual(
            amdgpu_family_info_matrix_all.get_entry("GFX94X-DCGPU"),
            amdgpu_family_info_matrix_all.get_entry("gfx94X-dcgpu"),
        )

    def test_family_without_default_returns_none(self):
        # gfx115X has no family default — each chip is registered individually
        self.assertIsNone(
            amdgpu_family_info_matrix_all.get_default_for_family("gfx115X")
        )

    def test_unknown_key_returns_none(self):
        self.assertIsNone(amdgpu_family_info_matrix_all.get_entry("gfx9999"))

    def test_get_entries_for_groups(self):
        entries = amdgpu_family_info_matrix_all.get_entries_for_groups(
            amdgpu_family_predefined_groups["amdgpu_presubmit"]
        )
        self.assertGreater(len(entries), 0)
        for entry in entries:
            self.assertIsInstance(entry, MatrixEntry)


class TestDefaultConfig(unittest.TestCase):
    """Verify default field values for entries with minimal configuration."""

    def test_default_build_config(self):
        cfg = BuildConfig()
        self.assertEqual(cfg.build_variants, ["release"])
        self.assertFalse(cfg.expect_failure)

    def test_default_test_config_no_runner(self):
        cfg = TestConfig()
        self.assertFalse(cfg.run_tests)
        self.assertEqual(cfg.test_scope, "all")
        self.assertFalse(cfg.sanity_check_only_for_family)
        self.assertFalse(cfg.expect_pytorch_failure)
        self.assertFalse(cfg.runs_on)
        self.assertEqual(cfg.fetch_gfx_targets, [])

    def test_default_release_config(self):
        cfg = ReleaseConfig()
        self.assertFalse(cfg.bypass_tests_for_releases)

    def test_platform_config_initializes_all_three(self):
        cfg = PlatformConfig()
        self.assertIsInstance(cfg.build, BuildConfig)
        self.assertIsInstance(cfg.test, TestConfig)
        self.assertIsInstance(cfg.release, ReleaseConfig)


class TestRunTestsInference(unittest.TestCase):
    """Verify run_tests is correctly inferred from GpuRunners."""

    def test_run_tests_false_when_no_runner(self):
        cfg = TestConfig()
        self.assertFalse(cfg.run_tests)

    def test_run_tests_true_when_test_runner_set(self):
        cfg = TestConfig(runs_on=GpuRunners(test="linux-mi325-1gpu"))
        self.assertTrue(cfg.run_tests)

    def test_run_tests_true_when_only_benchmark_set(self):
        cfg = TestConfig(runs_on=GpuRunners(benchmark="linux-mi325-bench"))
        self.assertTrue(cfg.run_tests)

    def test_run_tests_true_when_only_extra_set(self):
        cfg = TestConfig(runs_on=GpuRunners(extra={"custom": "linux-mi325-custom"}))
        self.assertTrue(cfg.run_tests)

    def test_run_tests_explicit_false_overrides_runner(self):
        cfg = TestConfig(runs_on=GpuRunners(test="linux-gfx1153"), run_tests=False)
        self.assertFalse(cfg.run_tests)


class TestDedicatedConfig(unittest.TestCase):
    """Verify entries with non-default configuration."""

    def test_gfx94x_has_asan_build_variant(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx94X-dcgpu")
        self.assertIn("asan", entry.linux.build.build_variants)

    def test_gfx1151_has_bypass_tests_for_releases(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx1151")
        self.assertTrue(entry.linux.release.bypass_tests_for_releases)

    def test_gfx1151_windows_test_scope(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx1151")
        self.assertEqual(entry.windows.test.test_scope, "full")

    def test_gfx90x_has_linux_config(self):
        for key in ["gfx906", "gfx908", "gfx90a"]:
            entry = amdgpu_family_info_matrix_all.get_entry(key)
            self.assertIsNotNone(entry.linux, f"{key} should have linux config")

    def test_gfx94x_has_no_windows_config(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx94x")
        self.assertIsNone(entry.windows, f"{entry.key} should not have windows config")


class TestExtraRunners(unittest.TestCase):
    """Verify extra runners are included in to_dict output."""

    def test_gfx94x_sandbox_runner_in_to_dict(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx94X-dcgpu")
        runners_dict = entry.linux.test.runs_on.to_dict()
        self.assertIn("test-sandbox", runners_dict)
        self.assertEqual(
            runners_dict["test-sandbox"], "linux-mi325-8gpu-ossci-rocm-sandbox"
        )

    def test_gfx1151_oem_runner_in_to_dict(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx1151")
        runners_dict = entry.linux.test.runs_on.to_dict()
        self.assertIn("oem", runners_dict)

    def test_extra_runners_accessible_directly(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx94X-dcgpu")
        extra = entry.linux.test.runs_on.extra
        self.assertIn("test-sandbox", extra)


class TestBuildVariants(unittest.TestCase):
    """Verify all_build_variants structure and lookup."""

    def test_linux_release_variant_exists(self):
        bvi = all_build_variants.get("linux", "release")
        self.assertIsNotNone(bvi)
        self.assertEqual(bvi.label, "release")

    def test_linux_asan_variant_exists(self):
        bvi = all_build_variants.get("linux", "asan")
        self.assertIsNotNone(bvi)
        self.assertEqual(bvi.suffix, "asan")

    def test_windows_release_variant_exists(self):
        bvi = all_build_variants.get("windows", "release")
        self.assertIsNotNone(bvi)

    def test_unknown_platform_raises(self):
        with self.assertRaises(ValueError):
            all_build_variants.get("macos", "release")

    def test_unknown_variant_returns_none(self):
        self.assertIsNone(all_build_variants.get("linux", "nonexistent"))


class TestToDict(unittest.TestCase):
    """Verify to_dict always includes all fields including defaults."""

    def test_build_config_to_dict_includes_all_fields(self):
        d = BuildConfig().to_dict()
        self.assertIn("build_variants", d)
        self.assertIn("expect_failure", d)

    def test_test_config_to_dict_includes_all_fields(self):
        d = TestConfig().to_dict()
        self.assertIn("run_tests", d)
        self.assertIn("runs_on", d)
        self.assertIn("fetch-gfx-targets", d)
        self.assertIn("sanity_check_only_for_family", d)
        self.assertIn("test_scope", d)
        self.assertIn("expect_pytorch_failure", d)

    def test_platform_config_to_dict_includes_all_sections(self):
        d = PlatformConfig().to_dict()
        self.assertIn("build", d)
        self.assertIn("test", d)
        self.assertIn("release", d)

    def test_matrix_entry_to_dict_includes_amdgpu_family(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx950")
        d = entry.to_dict()
        self.assertEqual(d["amdgpu_family"], "gfx950-dcgpu")

    def test_matrix_entry_to_dict_with_platform(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx950")
        d = entry.to_dict("linux")
        self.assertEqual(d["amdgpu_family"], "gfx950-dcgpu")
        self.assertIn("build", d)
        self.assertNotIn("linux", d)


if __name__ == "__main__":
    unittest.main()
