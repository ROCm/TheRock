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

    def test_prerelease_is_local_only(self):
        """Shipped builds must not be able to read from the remote cache."""
        config = self.gen_config("--release-type", "prerelease")
        self.assertNotIn("remote_storage", config)
        self.assertIn("cache_dir", config)

    def test_nightly_bkc_is_local_only(self):
        config = self.gen_config("--release-type", "nightly-bkc")
        self.assertNotIn("remote_storage", config)
        self.assertIn("cache_dir", config)

    def test_nightly_uses_release_remote_cache(self):
        config = self.gen_config("--release-type", "nightly")
        self.assertEqual(config["remote_storage"], setup_ccache.CACHE_SRV_REL)

    def test_dev_uses_dev_remote_cache(self):
        config = self.gen_config("--release-type", "dev")
        self.assertEqual(config["remote_storage"], setup_ccache.CACHE_SRV_DEV)

    def test_dev_bkc_uses_dev_remote_cache(self):
        config = self.gen_config("--release-type", "dev-bkc")
        self.assertEqual(config["remote_storage"], setup_ccache.CACHE_SRV_DEV)

    def test_local_path_is_honored_for_prerelease(self):
        local_path = self.tmp_dir / "cache"
        config = self.gen_config(
            "--release-type",
            "prerelease",
            "--local-path",
            os.fspath(local_path),
        )
        self.assertEqual(config["cache_dir"], os.fspath(local_path))

    def test_explicit_local_preset_is_local_only(self):
        config = self.gen_config("--config-preset=local")
        self.assertNotIn("remote_storage", config)
        self.assertIn("cache_dir", config)


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
