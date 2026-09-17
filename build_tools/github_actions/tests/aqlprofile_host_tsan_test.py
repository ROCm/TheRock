# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).parent.parent / "test_executable_scripts"
sys.path.insert(0, os.fspath(SCRIPT_DIR))

import test_aqlprofile_host_tsan


class AqlprofileHostTsanTest(unittest.TestCase):
    def test_packaged_manifest_matches_frozen_inventory(self):
        manifest = (
            Path(__file__).resolve().parents[3]
            / "profiler"
            / "aqlprofile_host_tsan_tests.json"
        )
        entries = test_aqlprofile_host_tsan._load_manifest(manifest)
        self.assertEqual(len(entries), 15)
        self.assertEqual(sum(len(entry["tests"]) for entry in entries), 80)
        logger_tests = next(
            entry["tests"] for entry in entries if entry["name"] == "logger-test"
        )
        self.assertIn("LoggerTest.ConcurrentLogging", logger_tests)
        self.assertNotIn(
            "logger-test:LoggerTest.ConcurrentLogging",
            {
                test
                for tests in test_aqlprofile_host_tsan.EXPECTED_EXCLUSIONS.values()
                for test in tests
            },
        )

    def test_exact_positive_binary_and_case_counts(self):
        counts = test_aqlprofile_host_tsan.EXPECTED_EXECUTABLE_COUNTS
        self.assertEqual(len(counts), 15)
        self.assertEqual(sum(counts.values()), 80)
        self.assertEqual(counts["utility_tests"], 0)
        self.assertEqual(
            test_aqlprofile_host_tsan.EXPECTED_INVENTORY_SHA256,
            "234c36c1302354dc934768f6e7705ff0fd33f1a71fcc0b7ed2160e96c26cc104",
        )

    def test_hardware_disabled_and_stale_exclusions_are_exact(self):
        exclusions = test_aqlprofile_host_tsan.EXPECTED_EXCLUSIONS
        self.assertEqual(
            {name: len(tests) for name, tests in exclusions.items()},
            {
                "requires_hsa_hardware": 7,
                "disabled": 1,
                "stale_registration": 1,
            },
        )
        self.assertTrue(
            all(
                test.startswith("utility_tests:")
                for test in exclusions["requires_hsa_hardware"]
            )
        )

    def test_missing_manifest_fails_closed(self):
        with mock.patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "manifest is missing"):
                test_aqlprofile_host_tsan._load_manifest(Path("missing.json"))

    def test_manifest_schema_and_executable_inventory_fail_closed(self):
        invalid_manifests = (
            {},
            {"schema_version": 1, "executables": []},
        )
        for manifest in invalid_manifests:
            with self.subTest(manifest=manifest), mock.patch.object(
                Path, "is_file", return_value=True
            ), mock.patch.object(
                Path, "read_text", return_value=json.dumps(manifest)
            ):
                with self.assertRaises(RuntimeError):
                    test_aqlprofile_host_tsan._load_manifest(Path("manifest.json"))

    def test_manifest_entries_require_exact_keys_and_allowed_kinds(self):
        manifest_path = (
            Path(__file__).resolve().parents[3]
            / "profiler"
            / "aqlprofile_host_tsan_tests.json"
        )
        base = json.loads(manifest_path.read_text())
        invalid_entries = []

        missing_kind = json.loads(json.dumps(base))
        del missing_kind["executables"][0]["kind"]
        invalid_entries.append(missing_kind)

        extra_key = json.loads(json.dumps(base))
        extra_key["executables"][0]["comment"] = "not in schema"
        invalid_entries.append(extra_key)

        unknown_kind = json.loads(json.dumps(base))
        unknown_kind["executables"][0]["kind"] = "benchmark"
        invalid_entries.append(unknown_kind)

        non_string_kind = json.loads(json.dumps(base))
        non_string_kind["executables"][0]["kind"] = ["gtest"]
        invalid_entries.append(non_string_kind)

        for manifest in invalid_entries:
            with self.subTest(entry=manifest["executables"][0]), mock.patch.object(
                Path, "is_file", return_value=True
            ), mock.patch.object(
                Path, "read_text", return_value=json.dumps(manifest)
            ):
                with self.assertRaises(RuntimeError):
                    test_aqlprofile_host_tsan._load_manifest(Path("manifest.json"))

    def test_inventory_normalization_is_order_independent(self):
        entries = [
            {"name": "second", "tests": ["Suite.z", "Suite.a"]},
            {"name": "first", "tests": ["Case.b"]},
        ]
        self.assertEqual(
            test_aqlprofile_host_tsan._normalized_inventory(entries),
            "first\tCase.b\nsecond\tSuite.a\nsecond\tSuite.z\n",
        )


if __name__ == "__main__":
    unittest.main()
