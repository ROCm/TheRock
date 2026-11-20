# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for _therock_utils.external_repo_config."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from _therock_utils.external_repo_config import (
    ExternalRepoInput,
    parse_and_validate_external_repo_config,
    parse_external_repo,
)


class ParseExternalRepoValidTest(unittest.TestCase):
    """Happy-path tests for parse_external_repo()."""

    def test_minimal_valid_payload(self):
        raw = json.dumps({"repository": "ROCm/rocm-libraries"})
        cfg = parse_external_repo(raw)
        self.assertEqual(cfg.repository, "ROCm/rocm-libraries")
        self.assertEqual(cfg.repo_name, "rocm-libraries")
        self.assertEqual(cfg.event_name, "")
        self.assertIsNone(cfg.base_sha)
        self.assertIsNone(cfg.pr_number)

    def test_full_payload(self):
        raw = json.dumps(
            {
                "repository": "ROCm/rocm-libraries",
                "ref": "refs/heads/develop",
                "event_name": "pull_request",
                "base_sha": "abc123",
                "head_sha": "def456",
                "pr_number": 42,
                "base_ref": "develop",
                "skip_ci_config": ".github/skip-ci.toml",
                "family_overrides": {"gfx942": {"test_labels": ["smoke"]}},
            }
        )
        cfg = parse_external_repo(raw)
        self.assertEqual(cfg.repository, "ROCm/rocm-libraries")
        self.assertEqual(cfg.ref, "refs/heads/develop")
        self.assertEqual(cfg.pr_number, 42)
        self.assertEqual(cfg.family_overrides, {"gfx942": {"test_labels": ["smoke"]}})

    def test_pr_number_as_string_is_coerced(self):
        raw = json.dumps({"repository": "ROCm/rocm-libraries", "pr_number": "123"})
        cfg = parse_external_repo(raw)
        self.assertEqual(cfg.pr_number, 123)

    def test_unknown_fields_are_ignored(self):
        raw = json.dumps({"repository": "ROCm/rocm-libraries", "future_field": "hello"})
        cfg = parse_external_repo(raw)
        self.assertEqual(cfg.repository, "ROCm/rocm-libraries")

    def test_repo_name_property(self):
        cfg = ExternalRepoInput(repository="ROCm/rocm-systems")
        self.assertEqual(cfg.repo_name, "rocm-systems")


class ParseExternalRepoInvalidTest(unittest.TestCase):
    """Error-path tests for parse_external_repo()."""

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo("not json")

    def test_non_object_json_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo('"just a string"')

    def test_array_json_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo("[1, 2, 3]")

    def test_missing_repository_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo(json.dumps({"event_name": "push"}))

    def test_empty_repository_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo(json.dumps({"repository": ""}))

    def test_repository_without_slash_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo(json.dumps({"repository": "rocm-libraries"}))

    def test_repository_non_string_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo(json.dumps({"repository": 123}))

    def test_pr_number_non_numeric_string_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo(
                json.dumps({"repository": "ROCm/rocm-libraries", "pr_number": "12a"})
            )

    def test_pr_number_float_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo(
                json.dumps({"repository": "ROCm/rocm-libraries", "pr_number": 12.0})
            )

    def test_pr_number_list_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo(
                json.dumps({"repository": "ROCm/rocm-libraries", "pr_number": []})
            )

    def test_base_sha_wrong_type_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo(
                json.dumps({"repository": "ROCm/rocm-libraries", "base_sha": 5})
            )

    def test_event_name_wrong_type_raises(self):
        with self.assertRaises(ValueError):
            parse_external_repo(
                json.dumps({"repository": "ROCm/rocm-libraries", "event_name": 42})
            )


class ExternalRepoInputPostInitTest(unittest.TestCase):
    """Direct construction validation tests."""

    def test_valid_construction(self):
        cfg = ExternalRepoInput(repository="ROCm/rocm-libraries")
        self.assertEqual(cfg.repository, "ROCm/rocm-libraries")

    def test_no_slash_in_repository_raises(self):
        with self.assertRaises(ValueError):
            ExternalRepoInput(repository="rocm-libraries")

    def test_wrong_type_field_raises(self):
        with self.assertRaises(ValueError):
            ExternalRepoInput(repository="ROCm/rocm-libraries", base_sha=5)


class ParseAndValidateExternalRepoConfigTest(unittest.TestCase):
    """Tests for parsing EXTERNAL_REPO_CONFIG."""

    def test_missing_config_returns_none(self):
        self.assertIsNone(parse_and_validate_external_repo_config(None))
        self.assertIsNone(parse_and_validate_external_repo_config(""))

    def test_valid_config(self):
        config = {
            "repository": "ROCm/rocm-systems",
            "extra_cmake_options": "-DFOO=ON",
            "submodule_path": "rocm-systems",
            "ref": "c1dba564e7e3a9158ce56bd0d314c7769ca0a073",
        }

        result = parse_and_validate_external_repo_config(json.dumps(config))

        self.assertEqual(result, config)

    def test_rejects_non_object_json(self):
        invalid_values = [
            None,
            [],
            "rocm-systems",
            123,
            True,
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "EXTERNAL_REPO_CONFIG must be a JSON object",
                ):
                    parse_and_validate_external_repo_config(json.dumps(value))

    def test_rejects_invalid_submodule_path(self):
        invalid_values = [
            None,
            "",
            "   ",
            123,
            True,
            [],
            {},
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                config = {
                    "submodule_path": value,
                    "ref": "c1dba564e7e3a9158ce56bd0d314c7769ca0a073",
                }

                with self.assertRaisesRegex(
                    ValueError,
                    "'submodule_path' must be a non-empty string",
                ):
                    parse_and_validate_external_repo_config(json.dumps(config))

    def test_rejects_invalid_ref(self):
        invalid_values = [
            None,
            "",
            "   ",
            123,
            True,
            [],
            {},
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                config = {
                    "submodule_path": "rocm-systems",
                    "ref": value,
                }

                with self.assertRaisesRegex(
                    ValueError,
                    "'ref' must be a non-empty string",
                ):
                    parse_and_validate_external_repo_config(json.dumps(config))


if __name__ == "__main__":
    unittest.main()
