# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# test_runner has module-level code that reads env vars and calls sys.exit
# if TEST_COMPONENT is missing. Set required env vars before importing.
_tmpdir = tempfile.mkdtemp()
os.environ.setdefault("THEROCK_BIN_DIR", _tmpdir)
os.environ.setdefault("TEST_COMPONENT", "miopen")

sys.path.insert(0, os.fspath(Path(__file__).parent.parent / "test_executable_scripts"))

import test_runner


class FindMatchingGpuArchTest(unittest.TestCase):
    """Tests for find_matching_gpu_arch()."""

    def test_exact_match(self):
        available = {"gfx1151", "gfx115X", "gfx11X"}
        self.assertEqual(
            test_runner.find_matching_gpu_arch("gfx1151", available), "gfx1151"
        )

    def test_wildcard_one_char(self):
        available = {"gfx1150", "gfx115X", "gfx11X"}
        self.assertEqual(
            test_runner.find_matching_gpu_arch("gfx1151", available), "gfx115X"
        )

    def test_wildcard_two_chars(self):
        available = {"gfx1150", "gfx94X", "gfx11X"}
        self.assertEqual(
            test_runner.find_matching_gpu_arch("gfx1151", available), "gfx11X"
        )

    def test_most_specific_wildcard_wins(self):
        available = {"gfx115X", "gfx11X"}
        self.assertEqual(
            test_runner.find_matching_gpu_arch("gfx1151", available), "gfx115X"
        )

    def test_no_match_returns_none(self):
        available = {"gfx94X", "gfx90a"}
        self.assertIsNone(test_runner.find_matching_gpu_arch("gfx1151", available))

    def test_empty_available_returns_none(self):
        self.assertIsNone(test_runner.find_matching_gpu_arch("gfx1151", set()))

    def test_short_arch_exact_match(self):
        available = {"gfx950"}
        self.assertEqual(
            test_runner.find_matching_gpu_arch("gfx950", available), "gfx950"
        )

    def test_short_arch_wildcard_reaches_minimum(self):
        # "gfx90a" (6 chars): only "gfx90X" is tried (loop stops before 5-char patterns)
        available = {"gfx90X"}
        self.assertEqual(
            test_runner.find_matching_gpu_arch("gfx90a", available), "gfx90X"
        )

    def test_wildcard_too_short_not_tried(self):
        # "gfx9X" (5 chars) is never generated for "gfx90a" — loop doesn't go that far
        available = {"gfx9X"}
        self.assertIsNone(test_runner.find_matching_gpu_arch("gfx90a", available))


class BuildCtestCommandTest(unittest.TestCase):
    """Tests for build_ctest_command()."""

    def _build(self, category, gpu_arch, available_gpu_archs, exclude_labels=None):
        if exclude_labels is None:
            exclude_labels = set()
        return test_runner.build_ctest_command(
            category, gpu_arch, available_gpu_archs, exclude_labels
        )

    def test_category_is_first_label(self):
        cmd = self._build("quick", "", set())
        self.assertEqual(cmd[0], "ctest")
        idx = cmd.index("-L")
        # Include labels are anchored (^...$) for exact ctest -L matching.
        self.assertEqual(cmd[idx + 1], "^quick$")

    def test_generic_gpu_excludes_ex_gpu(self):
        cmd = self._build("quick", "generic", set())
        self.assertIn("-LE", cmd)
        le_idx = cmd.index("-LE")
        self.assertEqual(cmd[le_idx + 1], "ex_gpu")

    def test_empty_gpu_excludes_ex_gpu(self):
        cmd = self._build("standard", "", set())
        self.assertIn("-LE", cmd)
        le_idx = cmd.index("-LE")
        self.assertEqual(cmd[le_idx + 1], "ex_gpu")

    def test_matching_gpu_adds_gpu_label(self):
        available = {"gfx115X", "gfx11X"}
        cmd = self._build("quick", "gfx1151", available)
        label_indices = [i for i, v in enumerate(cmd) if v == "-L"]
        # Should have category label and GPU label, both anchored (^...$).
        labels = [cmd[i + 1] for i in label_indices]
        self.assertIn("^quick$", labels)
        self.assertIn("^ex_gpu_gfx115X$", labels)

    def test_no_matching_gpu_excludes_ex_gpu(self):
        available = {"gfx94X"}
        cmd = self._build("quick", "gfx1151", available)
        self.assertIn("-LE", cmd)
        le_idx = cmd.index("-LE")
        self.assertEqual(cmd[le_idx + 1], "ex_gpu")

    def test_common_params_present(self):
        cmd = self._build("quick", "", set())
        self.assertIn("--output-on-failure", cmd)
        self.assertIn("--parallel", cmd)
        self.assertIn("--timeout", cmd)
        self.assertIn("--test-dir", cmd)
        self.assertIn("-V", cmd)
        self.assertIn("--tests-information", cmd)

    def test_category_exclude_label_applied(self):
        exclude_labels = {"quick_exclude", "standard_exclude"}
        cmd = self._build("quick", "", set(), exclude_labels)
        # build_ctest_command emits a single -LE flag with all exclude
        # patterns joined by "|" (ctest regex OR). Split the value back
        # into its component patterns so the assertion matches logical
        # excludes rather than the emitted -LE arg string.
        le_indices = [i for i, v in enumerate(cmd) if v == "-LE"]
        le_patterns = [p for i in le_indices for p in cmd[i + 1].split("|")]
        self.assertIn("quick_exclude", le_patterns)

    def test_category_exclude_label_not_applied_when_absent(self):
        exclude_labels = {"standard_exclude"}
        cmd = self._build("quick", "generic", set(), exclude_labels)
        le_indices = [i for i, v in enumerate(cmd) if v == "-LE"]
        le_patterns = [p for i in le_indices for p in cmd[i + 1].split("|")]
        self.assertNotIn("quick_exclude", le_patterns)

    def test_verbose_flag_default(self):
        # Components without a ctest_verbose override keep -V.
        original = test_runner.test_component_job_name
        try:
            test_runner.test_component_job_name = "some-other-component"
            cmd = self._build("quick", "", set())
            self.assertIn("-V", cmd)
        finally:
            test_runner.test_component_job_name = original

    def test_verbose_flag_dropped_for_rocprofiler_systems(self):
        # rocprofiler-systems opts out of -V via COMPONENT_OVERRIDES.
        original = test_runner.test_component_job_name
        try:
            test_runner.test_component_job_name = "rocprofiler-systems"
            cmd = self._build("quick", "", set())
            self.assertNotIn("-V", cmd)
            # --output-on-failure must still be present so failures show output.
            self.assertIn("--output-on-failure", cmd)
        finally:
            test_runner.test_component_job_name = original

    def test_therock_ci_exclude_label_applied(self):
        exclude_labels = {"quick_therock_ci_exclude"}
        cmd = self._build("quick", "", set(), exclude_labels)
        le_indices = [i for i, v in enumerate(cmd) if v == "-LE"]
        le_patterns = [p for i in le_indices for p in cmd[i + 1].split("|")]
        self.assertIn("quick_therock_ci_exclude", le_patterns)

    def test_category_and_therock_ci_exclude_combined(self):
        exclude_labels = {"quick_exclude", "quick_therock_ci_exclude"}
        cmd = self._build("quick", "", set(), exclude_labels)
        le_indices = [i for i, v in enumerate(cmd) if v == "-LE"]
        le_patterns = [p for i in le_indices for p in cmd[i + 1].split("|")]
        self.assertIn("quick_exclude", le_patterns)
        self.assertIn("quick_therock_ci_exclude", le_patterns)

    def test_comprehensive_category(self):
        cmd = self._build("comprehensive", "", set())
        idx = cmd.index("-L")
        self.assertEqual(cmd[idx + 1], "^comprehensive$")

    def test_full_category(self):
        cmd = self._build("full", "", set())
        idx = cmd.index("-L")
        self.assertEqual(cmd[idx + 1], "^full$")

    def test_include_labels_are_anchored(self):
        # ctest -L is a partial regex match, so include labels must be anchored
        # (^...$) to avoid e.g. "full" matching "multigpu_full" or "ffm-full".
        cmd = self._build("full", "gfx1151", {"gfx115X"})
        label_values = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-L"]
        self.assertTrue(label_values)
        for value in label_values:
            self.assertTrue(
                value.startswith("^") and value.endswith("$"),
                f"include label not anchored: {value}",
            )


class ApplyComponentOverridesTest(unittest.TestCase):
    """Tests for apply_component_overrides()."""

    def _apply(self, job_name, default_parallel_count):
        return test_runner.apply_component_overrides(
            job_name,
            "quick",
            Path("/rocm"),
            Path("/therock"),
            "/rocm/bin/default",
            {},
            default_parallel_count,
        )

    def test_rocprofiler_compute_pins_serial_ctest(self):
        _, parallel = self._apply("rocprofiler-compute", 8)
        self.assertEqual(parallel, 1)

    def test_unknown_component_uses_default_parallel(self):
        test_dir, parallel = self._apply("some-unknown-component", 4)
        self.assertEqual(parallel, 4)
        self.assertEqual(test_dir, "/rocm/bin/default")

    def test_component_without_parallel_override_uses_default(self):
        _, parallel = self._apply("rocwmma", 4)
        self.assertEqual(parallel, 4)


class ValidTestCategoriesTest(unittest.TestCase):
    """Tests for VALID_TEST_CATEGORIES and category validation."""

    def test_all_expected_categories_present(self):
        self.assertEqual(
            test_runner.VALID_TEST_CATEGORIES,
            {
                "quick",
                "standard",
                "comprehensive",
                "full",
                "ffm-quick",
                "ffm-standard",
                "ffm-comprehensive",
                "ffm-full",
            },
        )

    def test_valid_category_accepted(self):
        for cat in ("quick", "standard", "comprehensive", "full"):
            self.assertIn(cat, test_runner.VALID_TEST_CATEGORIES)

    def test_invalid_category_not_accepted(self):
        self.assertNotIn("smoke", test_runner.VALID_TEST_CATEGORIES)
        self.assertNotIn("", test_runner.VALID_TEST_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
