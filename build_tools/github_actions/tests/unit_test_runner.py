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
        self.assertEqual(cmd[idx + 1], "quick")

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
        # Should have category label and GPU label
        labels = [cmd[i + 1] for i in label_indices]
        self.assertIn("quick", labels)
        self.assertIn("ex_gpu_gfx115X", labels)

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
        le_indices = [i for i, v in enumerate(cmd) if v == "-LE"]
        le_values = [cmd[i + 1] for i in le_indices]
        self.assertIn("quick_exclude", le_values)

    def test_category_exclude_label_not_applied_when_absent(self):
        exclude_labels = {"standard_exclude"}
        cmd = self._build("quick", "generic", set(), exclude_labels)
        le_indices = [i for i, v in enumerate(cmd) if v == "-LE"]
        le_values = [cmd[i + 1] for i in le_indices]
        self.assertNotIn("quick_exclude", le_values)

    def test_comprehensive_category(self):
        cmd = self._build("comprehensive", "", set())
        idx = cmd.index("-L")
        self.assertEqual(cmd[idx + 1], "comprehensive")

    def test_full_category(self):
        cmd = self._build("full", "", set())
        idx = cmd.index("-L")
        self.assertEqual(cmd[idx + 1], "full")


class ValidTestCategoriesTest(unittest.TestCase):
    """Tests for VALID_TEST_CATEGORIES and category validation."""

    def test_all_expected_categories_present(self):
        self.assertEqual(
            test_runner.VALID_TEST_CATEGORIES,
            {"quick", "standard", "comprehensive", "full"},
        )

    def test_valid_category_accepted(self):
        for cat in ("quick", "standard", "comprehensive", "full"):
            self.assertIn(cat, test_runner.VALID_TEST_CATEGORIES)

    def test_invalid_category_not_accepted(self):
        self.assertNotIn("smoke", test_runner.VALID_TEST_CATEGORIES)
        self.assertNotIn("", test_runner.VALID_TEST_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
