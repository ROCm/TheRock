# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_executable_scripts"))
from miopen_rocjitsu_targets import load_targets
from prepare_miopen_rocjitsu import normalize_profile


class MiopenRocjitsuTargetsTest(unittest.TestCase):
    def test_invalid_registration_fails(self):
        for registration in (
            {},
            {"gfx950": {"family": "gfx950-dcgpu"}},
            {"gfx950": {"family": "gfx950-dcgpu", "config": "../profile.json"}},
            {"gfx950;echo": {"family": "gfx950-dcgpu", "config": "profile.json"}},
            {"gfx950": {"family": "gfx950;echo", "config": "profile.json"}},
        ):
            with self.subTest(
                registration=registration
            ), tempfile.TemporaryDirectory() as directory:
                registry = Path(directory) / "targets.json"
                registry.write_text(json.dumps(registration), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_targets(registry)

    def test_profile_normalization_preserves_sku_and_input(self):
        for arch, version, vm_arch in (
            ("gfx1250", 120500, "cdna5"),
            ("gfx950", 90500, "cdna4"),
            ("gfx90a", 90010, "cdna2"),
        ):
            with self.subTest(arch=arch):
                profile = {
                    "thread_allocations": [
                        {
                            "num_threads": 8,
                            "cpu_dispatch_threads": 17,
                            "async_helper_threads": 8,
                        }
                    ],
                    "vm": {
                        "arch": vm_arch,
                        "gpu": {
                            "num_gpus": 4,
                            "device": {
                                "gfx_target_version": version,
                                "simd_count": 1024,
                                "num_shader_engines": 4,
                                "num_cu_per_sh": 9,
                            },
                        },
                    },
                    "topology": {"root": {"name": "xcd[0:8]", "type": "xcd"}},
                }
                original = copy.deepcopy(profile)
                result = normalize_profile(profile, arch)
                self.assertEqual(result["vm"]["target"], arch)
                self.assertEqual(result["vm"]["gpu"]["num_gpus"], 1)
                allocation = {
                    "num_threads": 1,
                    "cpu_dispatch_threads": 1,
                    "async_helper_threads": 0,
                }
                self.assertEqual(result["thread_allocations"], [allocation])
                for key, value in allocation.items():
                    self.assertEqual(result[key], value)
                self.assertEqual(
                    result["vm"]["gpu"]["device"], profile["vm"]["gpu"]["device"]
                )
                self.assertEqual(result["topology"], profile["topology"])
                self.assertEqual(profile, original)

    def test_profile_architecture_mismatch_fails(self):
        for vm in (
            {"gpu": {"device": {"gfx_target_version": 120500}}},
            {"target": "gfx942", "gpu": {"device": {"gfx_target_version": 90500}}},
        ):
            with self.subTest(vm=vm), self.assertRaises(ValueError):
                normalize_profile({"vm": vm}, "gfx950")


if __name__ == "__main__":
    unittest.main()
