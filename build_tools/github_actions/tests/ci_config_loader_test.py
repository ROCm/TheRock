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
    get_build_runners,
    get_gpu_families,
    load_runner_config,
)

SAMPLE_CONFIG = {
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


class TestGetBuildRunners(unittest.TestCase):
    def test_returns_build_runners(self):
        runners = get_build_runners(SAMPLE_CONFIG)
        self.assertIn("linux", runners)
        self.assertIn("windows", runners)
        self.assertEqual(
            runners["linux"]["default"][0]["label"], "azure-linux-scale-rocm"
        )


class TestGetGpuFamilies(unittest.TestCase):
    def test_presubmit_families(self):
        families = get_gpu_families(SAMPLE_CONFIG, ["presubmit"])
        self.assertIn("gfx94x", families)
        self.assertNotIn("gfx950", families)

    def test_combined_trigger_types(self):
        families = get_gpu_families(SAMPLE_CONFIG, ["presubmit", "postsubmit"])
        self.assertIn("gfx94x", families)
        self.assertIn("gfx950", families)

    def test_family_structure(self):
        families = get_gpu_families(SAMPLE_CONFIG, ["presubmit"])
        self.assertEqual(families["gfx94x"]["linux"]["family"], "gfx94X-dcgpu")
        self.assertEqual(families["gfx94x"]["linux"]["build_variants"], ["release"])


class TestLoadRunnerConfig(unittest.TestCase):
    def test_missing_config_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ConfigError):
                load_runner_config(Path(temp_dir))

    def test_missing_required_keys_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "runner-config.json"
            config_file.write_text("{}")
            with self.assertRaises(ConfigError):
                load_runner_config(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
