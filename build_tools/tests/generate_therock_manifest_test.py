"""Tests for generate_therock_manifest.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from generate_therock_manifest import build_manifest_schema


class ManifestSchemaTest(unittest.TestCase):
    def test_no_override_uses_committed_source_identity(self):
        repo_root = Path("unused")
        therock_commit = "a" * 40
        submodule_commit = "b" * 40
        entries = [
            {
                "name": "rocgdb",
                "path": "debug-tools/rocgdb/source",
                "url": "https://github.com/ROCm/rocgdb.git",
            }
        ]
        with (
            patch(
                "generate_therock_manifest.list_submodules_from_gitmodules_at_commit",
                return_value=entries,
            ),
            patch(
                "generate_therock_manifest.submodule_pin",
                return_value=submodule_commit,
            ) as submodule_pin,
            patch(
                "generate_therock_manifest.patches_for_submodule_by_name",
                return_value=["local.patch"],
            ),
        ):
            manifest = build_manifest_schema(repo_root, therock_commit)

        submodule_pin.assert_called_once_with(
            repo_root, therock_commit, "debug-tools/rocgdb/source"
        )
        self.assertEqual(
            manifest,
            {
                "the_rock_commit": therock_commit,
                "submodules": [
                    {
                        "submodule_name": "rocgdb",
                        "submodule_path": "debug-tools/rocgdb/source",
                        "submodule_url": "https://github.com/ROCm/rocgdb.git",
                        "pin_sha": submodule_commit,
                        "patches": ["local.patch"],
                    }
                ],
            },
        )

    def test_only_external_source_replaces_committed_identity(self):
        entries = [
            {"name": name, "path": name, "url": f"https://github.com/ROCm/{name}.git"}
            for name in ("rocm-libraries", "rocm-systems")
        ]
        default_pin = "a" * 40
        external_pin = "b" * 40
        # If we provide an external_repo_config, that data should be used in the
        # manifest instead of the original values.
        external_repo_config = {
            "repository": "my-fork/rocm-systems",
            "ref": external_pin,
            "submodule_path": "rocm-systems",
        }
        with (
            patch(
                "generate_therock_manifest.list_submodules_from_gitmodules_at_commit",
                return_value=entries,
            ),
            patch("generate_therock_manifest.submodule_pin", return_value=default_pin),
            patch(
                "generate_therock_manifest.patches_for_submodule_by_name",
                return_value=["local.patch"],
            ),
        ):
            manifest = build_manifest_schema(
                Path("unused"),
                "therock-commit",
                external_repo_config=external_repo_config,
            )

        rows = {row["submodule_name"]: row for row in manifest["submodules"]}
        self.assertEqual(rows["rocm-systems"]["pin_sha"], external_pin)
        self.assertEqual(
            rows["rocm-systems"]["submodule_url"],
            "https://github.com/my-fork/rocm-systems.git",
        )
        self.assertEqual(rows["rocm-systems"]["patches"], [])
        self.assertEqual(rows["rocm-libraries"]["pin_sha"], default_pin)
        self.assertEqual(
            rows["rocm-libraries"]["submodule_url"],
            "https://github.com/ROCm/rocm-libraries.git",
        )
        self.assertEqual(rows["rocm-libraries"]["patches"], ["local.patch"])


if __name__ == "__main__":
    unittest.main()
