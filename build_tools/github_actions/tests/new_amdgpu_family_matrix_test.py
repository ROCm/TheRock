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
    AmdGpuFamilyMatrix,
    BuildConfig,
    EntryLookupResult,
    GroupLookupResult,
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
        for group_name, group_keys in amdgpu_family_predefined_groups.items():
            for key in group_keys:
                self.assertIsNotNone(
                    amdgpu_family_info_matrix_all.get_entry(key),
                    f"Group {group_name!r} references unresolvable key {key!r}",
                )

    def test_all_known_archs_present(self):
        keys = amdgpu_family_info_matrix_all.keys()
        for expected in [
            "gfx906",
            "gfx908",
            "gfx90a",
            "gfx942",
            "gfx950",
            "gfx1010",
            "gfx1011",
            "gfx1012",
            "gfx1030",
            "gfx1031",
            "gfx1032",
            "gfx1034",
            "gfx1100",
            "gfx1101",
            "gfx1102",
            "gfx1103",
            "gfx1150",
            "gfx1151",
            "gfx1152",
            "gfx1153",
            "gfx1200",
            "gfx1201",
        ]:
            self.assertIn(
                expected, keys, f"Expected target {expected!r} missing from matrix"
            )


class TestEntryLookup(unittest.TestCase):
    """Verify get_entry and get_default_for_family behavior."""

    def test_exact_key_lookup(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx942")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfx942")

    def test_family_name_resolves_to_default(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx94X-dcgpu")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfx942")

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(
            amdgpu_family_info_matrix_all.get_entry("GFX942"),
            amdgpu_family_info_matrix_all.get_entry("gfx942"),
        )
        self.assertEqual(
            amdgpu_family_info_matrix_all.get_entry("GFX94X-DCGPU"),
            amdgpu_family_info_matrix_all.get_entry("gfx94X-dcgpu"),
        )

    def test_family_prefix_resolves_for_gfx94X(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx94X")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfx942")

    def test_family_prefix_resolves_for_gfx110X(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx110X")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfx1101")

    def test_family_without_default_returns_none(self):
        # gfx115X has no family default — each chip is registered individually
        self.assertIsNone(
            amdgpu_family_info_matrix_all.get_default_for_family("gfx115X-all")
        )

    def test_unknown_key_returns_none(self):
        self.assertIsNone(amdgpu_family_info_matrix_all.get_entry("gfx9999"))

    def test_get_entries_for_groups(self):
        result = amdgpu_family_info_matrix_all.get_entries_for_groups(
            amdgpu_family_predefined_groups["amdgpu_presubmit"]
        )
        self.assertIsInstance(result, GroupLookupResult)
        self.assertGreater(len(result.entries), 0)
        self.assertEqual(result.unmatched_keys, [])
        for entry in result.entries:
            self.assertIsInstance(entry, MatrixEntry)

    def test_get_entries_for_groups_reports_unmatched(self):
        result = amdgpu_family_info_matrix_all.get_entries_for_groups(
            ["gfx94X-dcgpu", "gfx9999-unknown", "gfx1151"]
        )
        self.assertEqual(len(result.entries), 2)
        self.assertEqual(result.unmatched_keys, ["gfx9999-unknown"])

    def test_family_default_resolves_for_gfx110X(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx110X-all")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfx1101")

    def test_family_default_resolves_for_gfx101X(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx101X-dgpu")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfx1010")


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
        entry = amdgpu_family_info_matrix_all.get_entry("gfx942")
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
        entry = amdgpu_family_info_matrix_all.get_entry("gfx942")
        self.assertIsNone(entry.windows, f"{entry.key} should not have windows config")


class TestExtraRunners(unittest.TestCase):
    """Verify extra runners are included in to_dict output."""

    def test_gfx94x_sandbox_runner_in_to_dict(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx942")
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
        entry = amdgpu_family_info_matrix_all.get_entry("gfx942")
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
        self.assertEqual(d["amdgpu_family"], "gfx950")

    def test_matrix_entry_to_dict_with_platform(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx950")
        d = entry.to_dict("linux")
        self.assertEqual(d["amdgpu_family"], "gfx950")
        self.assertIn("build", d)
        self.assertNotIn("linux", d)


class TestToNestedDict(unittest.TestCase):
    """Verify to_nested_dict structure."""

    def test_to_nested_dict_is_flat_by_target(self):
        d = amdgpu_family_info_matrix_all.to_nested_dict()
        # Top-level keys are target names, not family names
        self.assertIn("gfx942", d)
        self.assertIn("gfx1151", d)
        self.assertNotIn("gfx94X", d)

    def test_to_nested_dict_entry_has_amdgpu_family(self):
        d = amdgpu_family_info_matrix_all.to_nested_dict()
        self.assertEqual(d["gfx942"]["amdgpu_family"], "gfx942")

    def test_to_nested_dict_gfx1101_match_layout(self):
        d = amdgpu_family_info_matrix_all.to_nested_dict()
        subsection_gfx1101 = {
            "amdgpu_family": "gfx1101",
            "linux": {
                "build": {"build_variants": ["release"], "expect_failure": False},
                "release": {"bypass_tests_for_releases": True},
                "test": {
                    "expect_pytorch_failure": False,
                    "fetch-gfx-targets": ["gfx1101"],
                    "run_tests": False,
                    "runs_on": {
                        "benchmark": "",
                        "test": "linux-gfx110X-gpu-rocm",
                        "test-multi-gpu": "",
                    },
                    "sanity_check_only_for_family": True,
                    "test_scope": "all",
                },
            },
            "windows": {
                "build": {"build_variants": ["release"], "expect_failure": False},
                "release": {"bypass_tests_for_releases": True},
                "test": {
                    "expect_pytorch_failure": False,
                    "fetch-gfx-targets": ["gfx1101"],
                    "run_tests": True,
                    "runs_on": {
                        "benchmark": "",
                        "test": "windows-gfx110X-gpu-rocm",
                        "test-multi-gpu": "",
                    },
                    "sanity_check_only_for_family": True,
                    "test_scope": "all",
                },
            },
        }
        self.assertEqual(d["gfx1101"], subsection_gfx1101)


class TestGetEntriesForGroupsUnmatched(unittest.TestCase):
    """Verify get_entries_for_groups unmatched key reporting."""

    def test_all_unmatched_keys_reported(self):
        result = amdgpu_family_info_matrix_all.get_entries_for_groups(
            ["gfx9999", "gfxBAD"]
        )
        self.assertEqual(result.entries, [])
        self.assertEqual(result.unmatched_keys, ["gfx9999", "gfxBAD"])

    def test_unmatched_keys_preserve_order(self):
        result = amdgpu_family_info_matrix_all.get_entries_for_groups(
            ["gfx94X-dcgpu", "gfx9999", "gfx1151", "gfxBAD"]
        )
        self.assertEqual(len(result.entries), 2)
        self.assertEqual(result.unmatched_keys, ["gfx9999", "gfxBAD"])

    def test_empty_input_returns_empty_result(self):
        result = amdgpu_family_info_matrix_all.get_entries_for_groups([])
        self.assertEqual(result.entries, [])
        self.assertEqual(result.unmatched_keys, [])

    def test_deduplicate_removes_duplicate_entries(self):
        # gfx94X, gfx94X-dcgpu, gfx942 all resolve to the same entry
        result = amdgpu_family_info_matrix_all.get_entries_for_groups(
            ["gfx94X", "gfx94X-dcgpu", "gfx942"], deduplicate=True
        )
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].target, "gfx942")

    def test_deduplicate_false_allows_duplicates(self):
        result = amdgpu_family_info_matrix_all.get_entries_for_groups(
            ["gfx94X", "gfx94X-dcgpu", "gfx942"], deduplicate=False
        )
        self.assertEqual(len(result.entries), 3)


class TestLookup(unittest.TestCase):
    """Verify lookup() returns EntryLookupResult with correct resolved_via."""

    def test_lookup_by_target(self):
        result = amdgpu_family_info_matrix_all.lookup("gfx942")
        self.assertIsInstance(result, EntryLookupResult)
        self.assertEqual(result.entry.target, "gfx942")
        self.assertEqual(result.amdgpu_family, "gfx942")
        self.assertEqual(result.resolved_via, "target")

    def test_lookup_by_family(self):
        result = amdgpu_family_info_matrix_all.lookup("gfx94X-dcgpu")
        self.assertIsInstance(result, EntryLookupResult)
        self.assertEqual(result.entry.target, "gfx942")
        self.assertEqual(result.amdgpu_family, "gfx94X-dcgpu")
        self.assertEqual(result.resolved_via, "family")

    def test_lookup_by_family_prefix(self):
        result = amdgpu_family_info_matrix_all.lookup("gfx94X")
        self.assertIsInstance(result, EntryLookupResult)
        self.assertEqual(result.entry.target, "gfx942")
        self.assertEqual(result.amdgpu_family, "gfx94X")
        self.assertEqual(result.resolved_via, "family_prefix")

    def test_lookup_unknown_returns_none(self):
        self.assertIsNone(amdgpu_family_info_matrix_all.lookup("gfx9999"))

    def test_lookup_preserves_original_key_case(self):
        result = amdgpu_family_info_matrix_all.lookup("GFX94X-DCGPU")
        self.assertIsNotNone(result)
        self.assertEqual(result.amdgpu_family, "GFX94X-DCGPU")


class TestFamilyPopulation(unittest.TestCase):
    """Verify that entry.family is auto-populated from cmake data."""

    def test_gfx942_family_contains_gfx94X_dcgpu(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx942")
        self.assertIn("gfx94X-dcgpu", entry.family)

    def test_gfx1010_family_contains_gfx101X_dgpu(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx1010")
        self.assertIn("gfx101X-dgpu", entry.family)

    def test_gfx1151_family_contains_gfx115X_all(self):
        entry = amdgpu_family_info_matrix_all.get_entry("gfx1151")
        self.assertIn("gfx115X-all", entry.family)

    def test_get_targets_for_family_gfx120X_all(self):
        targets = amdgpu_family_info_matrix_all.get_targets_for_family("gfx120X-all")
        self.assertCountEqual(targets, ["gfx1200", "gfx1201"])

    def test_get_targets_for_family_gfx115X_all(self):
        targets = amdgpu_family_info_matrix_all.get_targets_for_family("gfx115X-all")
        self.assertCountEqual(targets, ["gfx1150", "gfx1151", "gfx1152", "gfx1153"])

    def test_get_targets_for_family_unknown_returns_empty(self):
        targets = amdgpu_family_info_matrix_all.get_targets_for_family("gfx999X-fake")
        self.assertEqual(targets, [])

    def test_is_family_default_validation_prevents_duplicates(self):
        with self.assertRaises(ValueError):
            AmdGpuFamilyMatrix(
                entries=[
                    MatrixEntry(target="gfx1010", is_family_default=True),
                    MatrixEntry(target="gfx1011", is_family_default=True),
                ],
                cmake_families={
                    "gfx1010": ["dgpu-all", "gfx101X-all", "gfx101X-dgpu"],
                    "gfx1011": ["dgpu-all", "gfx101X-all", "gfx101X-dgpu"],
                },
            )


if __name__ == "__main__":
    unittest.main()
