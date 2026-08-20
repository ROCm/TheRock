# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for setup_ccache.py config generation.

The prerelease build is what actually ships, so these tests pin down which
release types read from the shared remote cache and which are restricted to the
local cache only.
"""

from pathlib import Path
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import setup_ccache


class GenConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def gen_config(self, *extra_args: str) -> dict[str, str]:
        """Runs the tool and returns the generated ccache.conf as a dict."""
        ccache_dir = self.tmp_dir / ".ccache"
        setup_ccache.main(
            [
                "--dir",
                os.fspath(ccache_dir),
                "--log-dir",
                os.fspath(self.tmp_dir / "logs"),
                "--no-reset-stats",
                "--init",
                *extra_args,
            ]
        )
        config = {}
        for line in (ccache_dir / "ccache.conf").read_text().splitlines():
            if not line.strip():
                continue
            key, _, value = line.partition(" = ")
            config[key] = value
        return config

    def test_prerelease_with_no_remote_cache_is_local_only(self):
        """Shipped builds must not be able to read from the remote cache."""
        config = self.gen_config("--release-type", "prerelease", "--no-remote-cache")
        self.assertNotIn("remote_storage", config)
        self.assertIn("cache_dir", config)

    def test_prerelease_without_flag_still_uses_remote(self):
        """The local-only behavior comes from the flag, not the release type."""
        config = self.gen_config("--release-type", "prerelease")
        self.assertEqual(config["remote_storage"], setup_ccache.CACHE_SRV_REL)
        self.assertIn("cache_dir", config)

    def test_nightly_uses_release_remote_cache(self):
        config = self.gen_config("--release-type", "nightly")
        self.assertEqual(config["remote_storage"], setup_ccache.CACHE_SRV_REL)

    def test_dev_uses_dev_remote_cache(self):
        config = self.gen_config("--release-type", "dev")
        self.assertEqual(config["remote_storage"], setup_ccache.CACHE_SRV_DEV)

    def test_local_path_is_honored_with_no_remote_cache(self):
        local_path = self.tmp_dir / "cache"
        config = self.gen_config(
            "--release-type",
            "prerelease",
            "--no-remote-cache",
            "--local-path",
            os.fspath(local_path),
        )
        self.assertEqual(config["cache_dir"], os.fspath(local_path))

    def test_no_remote_cache_rejects_remote(self):
        """--remote bypasses the preset, so the combination must not be silent."""
        with self.assertRaises(SystemExit):
            self.gen_config(
                "--remote",
                "--remote-storage",
                "http://example.invalid|layout=bazel",
                "--no-remote-cache",
            )

    def test_no_remote_cache_fails_if_remote_survives(self):
        """A preset change must not be able to quietly re-add remote_storage."""
        preset = dict(setup_ccache.CONFIG_PRESETS_MAP["github-oss-release"])
        # Whitespace ccache tolerates but a naive prefix check would miss.
        preset["remote_storage "] = "http://example.invalid"
        with patch.dict(
            setup_ccache.CONFIG_PRESETS_MAP, {"github-oss-release": preset}
        ):
            with self.assertRaises(ValueError):
                self.gen_config("--release-type", "prerelease", "--no-remote-cache")


class ConfigValueTest(unittest.TestCase):
    def test_parses_like_ccache(self):
        lines = ["remote_storage  =  http://host:8080|connect-timeout=50"]
        self.assertEqual(
            setup_ccache._config_value(lines, "remote_storage"),
            "http://host:8080|connect-timeout=50",
        )

    def test_missing_key_returns_none(self):
        self.assertIsNone(setup_ccache._config_value(["max_size = 10G"], "cache_dir"))


if __name__ == "__main__":
    unittest.main()
