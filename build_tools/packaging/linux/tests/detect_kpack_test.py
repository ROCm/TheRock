#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Tests for detect_kpack.py"""

import json
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import detect_kpack


class TestDetectKpack(unittest.TestCase):
    """Test KPACK detection from TheRock manifest files."""

    def setUp(self):
        """Create temporary directory for test artifacts."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifacts_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _create_manifest(self, subdir: str, kpack_enabled: bool):
        """Helper to create a therock_manifest.json file."""
        manifest_dir = self.artifacts_dir / subdir
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "therock_manifest.json"

        manifest_data = {
            "KPACK_SPLIT_ARTIFACTS": kpack_enabled,
            "other_field": "value",
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        return manifest_path

    def test_kpack_enabled(self):
        """Test detection when KPACK_SPLIT_ARTIFACTS is true."""
        self._create_manifest("comp1/stage/share/therock", kpack_enabled=True)

        enabled, manifest_path = detect_kpack.detect_kpack(self.artifacts_dir)

        self.assertTrue(enabled)
        self.assertIsNotNone(manifest_path)
        self.assertTrue(manifest_path.name == "therock_manifest.json")

    def test_kpack_disabled(self):
        """Test detection when KPACK_SPLIT_ARTIFACTS is false."""
        self._create_manifest("comp1/stage/share/therock", kpack_enabled=False)

        enabled, manifest_path = detect_kpack.detect_kpack(self.artifacts_dir)

        self.assertFalse(enabled)
        self.assertIsNone(manifest_path)

    def test_multiple_manifests_one_enabled(self):
        """Test with multiple manifests where one has KPACK enabled."""
        self._create_manifest("comp1/stage/share/therock", kpack_enabled=False)
        self._create_manifest("comp2/stage/share/therock", kpack_enabled=True)
        self._create_manifest("comp3/stage/share/therock", kpack_enabled=False)

        enabled, manifest_path = detect_kpack.detect_kpack(self.artifacts_dir)

        self.assertTrue(enabled)
        self.assertIsNotNone(manifest_path)

    def test_no_manifests(self):
        """Test when no therock_manifest.json files exist."""
        enabled, manifest_path = detect_kpack.detect_kpack(self.artifacts_dir)

        self.assertFalse(enabled)
        self.assertIsNone(manifest_path)

    def test_manifest_missing_kpack_field(self):
        """Test manifest file without KPACK_SPLIT_ARTIFACTS field."""
        manifest_dir = self.artifacts_dir / "comp1/stage/share/therock"
        manifest_dir.mkdir(parents=True)
        manifest_path = manifest_dir / "therock_manifest.json"

        # Create manifest without KPACK_SPLIT_ARTIFACTS field
        with open(manifest_path, "w") as f:
            json.dump({"other_field": "value"}, f)

        enabled, _ = detect_kpack.detect_kpack(self.artifacts_dir)

        self.assertFalse(enabled)


if __name__ == "__main__":
    unittest.main()
