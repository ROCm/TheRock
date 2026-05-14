#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ci_config_loader.py."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from ci_config_loader import (
    ConfigError,
    config_exists,
    get_build_runners,
    get_gpu_families,
    load_runner_config,
)

SAMPLE_CONFIG = {
    "version": "1.0.0",
    "build_runners": {
        "linux": {
            "default": [{"label": "azure-linux-scale-rocm", "weight": 1.0}],
        },
        "windows": {
            "default": [{"label": "azure-windows-scale-rocm", "weight": 1.0}],
        },
    },
    "gpu_families": {
        "presubmit": {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "linux-gfx942-1gpu-ossci-rocm",
                    "family": "gfx94X-dcgpu",
                    "fetch-gfx-targets": ["gfx942"],
                    "build_variants": ["release"],
                }
            },
        },
        "postsubmit": {
            "gfx950": {
                "linux": {
                    "test-runs-on": "linux-gfx950-1gpu-ccs-ossci-rocm",
                    "family": "gfx950-dcgpu",
                    "fetch-gfx-targets": ["gfx950"],
                    "build_variants": ["release"],
                }
            },
        },
    },
}


class TestLoadRunnerConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def _write_config(self, config: dict):
        config_file = self.config_path / "runner-config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

    def test_load_valid_config(self):
        self._write_config(SAMPLE_CONFIG)
        config = load_runner_config(self.config_path)
        self.assertEqual(config["version"], "1.0.0")

    def test_missing_config_raises_error(self):
        with self.assertRaises(ConfigError) as ctx:
            load_runner_config(self.config_path)
        self.assertIn("not found", str(ctx.exception))

    def test_invalid_json_raises_error(self):
        config_file = self.config_path / "runner-config.json"
        with open(config_file, "w") as f:
            f.write("{ invalid json }")
        with self.assertRaises(ConfigError) as ctx:
            load_runner_config(self.config_path)
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_missing_required_keys_raises_error(self):
        self._write_config({"version": "1.0.0"})
        with self.assertRaises(ConfigError) as ctx:
            load_runner_config(self.config_path)
        self.assertIn("missing required keys", str(ctx.exception))


class TestGetBuildRunners(unittest.TestCase):
    def test_returns_build_runners(self):
        runners = get_build_runners(SAMPLE_CONFIG)
        self.assertIn("linux", runners)
        self.assertIn("windows", runners)

    def test_empty_config_returns_empty(self):
        runners = get_build_runners({})
        self.assertEqual(runners, {})


class TestGetGpuFamilies(unittest.TestCase):
    def test_single_trigger_type(self):
        families = get_gpu_families(SAMPLE_CONFIG, ["presubmit"])
        self.assertIn("gfx94x", families)
        self.assertNotIn("gfx950", families)

    def test_multiple_trigger_types(self):
        families = get_gpu_families(SAMPLE_CONFIG, ["presubmit", "postsubmit"])
        self.assertIn("gfx94x", families)
        self.assertIn("gfx950", families)

    def test_unknown_trigger_type_ignored(self):
        families = get_gpu_families(SAMPLE_CONFIG, ["unknown"])
        self.assertEqual(families, {})


class TestConfigExists(unittest.TestCase):
    def test_exists_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "runner-config.json"
            config_file.write_text("{}")
            self.assertTrue(config_exists(Path(temp_dir)))

    def test_not_exists_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(config_exists(Path(temp_dir)))


if __name__ == "__main__":
    unittest.main()
