"""Tests for generate_therock_manifest.py."""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from generate_therock_manifest import (
    parse_and_validate_external_repo_config,
)


class ParseAndValidateExternalRepoConfigTest(unittest.TestCase):
    """Tests for parsing EXTERNAL_REPO_CONFIG."""

    def test_missing_config_returns_none(self):
        self.assertIsNone(parse_and_validate_external_repo_config(None))
        self.assertIsNone(parse_and_validate_external_repo_config(""))

    def test_valid_config(self):
        config = {
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
