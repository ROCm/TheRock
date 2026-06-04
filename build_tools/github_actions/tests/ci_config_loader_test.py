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
    CURRENT_VERSION,
    ConfigError,
    ConfigV1,
    get_build_runners,
    get_gpu_families,
    load_config_v1,
    load_runner_config,
)

SAMPLE_CONFIG = {
    "version": "1",
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


def _write_config(path: Path, config: dict) -> None:
    (path / "runner-config.json").write_text(json.dumps(config))


class TestLoadConfigV1(unittest.TestCase):
    def test_loads_valid_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_config(Path(temp_dir), SAMPLE_CONFIG)
            config = load_config_v1(Path(temp_dir))
            self.assertIsInstance(config, ConfigV1)
            self.assertIn("linux", config.build_runners)

    def test_missing_config_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ConfigError) as ctx:
                load_config_v1(Path(temp_dir))
            self.assertIn("not found", str(ctx.exception))

    def test_invalid_json_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "runner-config.json").write_text("{invalid")
            with self.assertRaises(ConfigError) as ctx:
                load_config_v1(Path(temp_dir))
            self.assertIn("Invalid JSON", str(ctx.exception))

    def test_missing_required_keys_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_config(Path(temp_dir), {"version": "1"})
            with self.assertRaises(ConfigError) as ctx:
                load_config_v1(Path(temp_dir))
            self.assertIn("missing required", str(ctx.exception))

    def test_version_mismatch_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {**SAMPLE_CONFIG, "version": "99"}
            _write_config(Path(temp_dir), config)
            with self.assertRaises(ConfigError) as ctx:
                load_config_v1(Path(temp_dir))
            self.assertIn("version mismatch", str(ctx.exception))


class TestConfigV1(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        _write_config(Path(self.temp_dir), SAMPLE_CONFIG)
        self.config = load_config_v1(Path(self.temp_dir))

    def test_build_runners(self):
        self.assertIn("linux", self.config.build_runners)
        self.assertIn("windows", self.config.build_runners)
        self.assertEqual(
            self.config.build_runners["linux"]["default"][0]["label"],
            "azure-linux-scale-rocm",
        )

    def test_get_gpu_families_single_trigger(self):
        families = self.config.get_gpu_families(["presubmit"])
        self.assertIn("gfx94x", families)
        self.assertNotIn("gfx950", families)

    def test_get_gpu_families_multiple_triggers(self):
        families = self.config.get_gpu_families(["presubmit", "postsubmit"])
        self.assertIn("gfx94x", families)
        self.assertIn("gfx950", families)

    def test_family_structure(self):
        families = self.config.get_gpu_families(["presubmit"])
        self.assertEqual(families["gfx94x"]["linux"]["family"], "gfx94X-dcgpu")


class TestBackwardsCompatibility(unittest.TestCase):
    """Test legacy function aliases."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        _write_config(Path(self.temp_dir), SAMPLE_CONFIG)

    def test_load_runner_config_returns_dict(self):
        config = load_runner_config(Path(self.temp_dir))
        self.assertIsInstance(config, dict)
        self.assertIn("build_runners", config)

    def test_get_build_runners(self):
        config = load_runner_config(Path(self.temp_dir))
        runners = get_build_runners(config)
        self.assertIn("linux", runners)

    def test_get_gpu_families(self):
        config = load_runner_config(Path(self.temp_dir))
        families = get_gpu_families(config, ["presubmit"])
        self.assertIn("gfx94x", families)


if __name__ == "__main__":
    unittest.main()
