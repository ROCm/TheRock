#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ROCm alternatives JSON and shared maintainer-script templates."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


REPO_ROOT = Path(__file__).resolve().parents[4]

SCRIPTS_DIR = (
    REPO_ROOT
    / "build_tools"
    / "packaging"
    / "linux"
    / "template"
    / "scripts"
)

ALTERNATIVES_JSON = SCRIPTS_DIR / "amdrocm-alternatives.json"

SUPPORTED_SCRIPTS = {
    "postinst",
    "prerm",
}

EXPECTED_PACKAGES = {
    "amdrocm-base": {
        "binaries": {
            "rocm_agent_enumerator",
            "rocminfo",
        },
        "core_symlinks": "True",
        "llvm_symlinks": "False",
    },
    "amdrocm-llvm": {
        "binaries": {
            "hipcc",
            "hipconfig",
            "amdclang",
            "amdclang++",
            "amdclang-cl",
            "amdclang-cpp",
            "amdflang",
            "amdlld",
            "amdllvm",
            "offload-arch",
        },
        "core_symlinks": "False",
        "llvm_symlinks": "True",
    },
    "amdrocm-hipify": {
        "binaries": {
            "hipify-clang",
        },
        "core_symlinks": "False",
        "llvm_symlinks": "False",
    },
    "amdrocm-dnn-devel": {
        "binaries": {
            "MIOpenDriver",
        },
        "core_symlinks": "False",
        "llvm_symlinks": "False",
    },
    "amdrocm-debugger": {
        "binaries": {
            "rocgdb",
        },
        "core_symlinks": "False",
        "llvm_symlinks": "False",
    },
    "amdrocm-profiler-base": {
        "binaries": {
            "rocprofv3",
            "rocprofv3-avail",
            "rocprof-attach",
            "rocpd",
        },
        "core_symlinks": "False",
        "llvm_symlinks": "False",
    },
    "amdrocm-profiler": {
        "binaries": {
            "rocprof-compute",
            "rocprof-sys-attach",
            "rocprof-sys-avail",
            "rocprof-sys-causal",
            "rocprof-sys-instrument",
            "rocprof-sys-python",
            "rocprof-sys-run",
            "rocprof-sys-sample",
        },
        "core_symlinks": "False",
        "llvm_symlinks": "False",
    },
    "amdrocm-amdsmi": {
        "binaries": {
            "amd-smi",
        },
        "core_symlinks": "False",
        "llvm_symlinks": "False",
    },
}


class AlternativesScriptsTest(unittest.TestCase):
    """Test alternatives configuration and shared Jinja templates."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.alternatives = json.loads(
            ALTERNATIVES_JSON.read_text(encoding="utf-8")
        )

        cls.environment = Environment(
            loader=FileSystemLoader(str(REPO_ROOT)),
            keep_trailing_newline=True,
        )

    def render_script(self, package_name: str, script_name: str) -> str:
        """Render a shared maintainer script for one configured package."""

        package_config = self.alternatives[package_name]

        context = {
            "install_prefix": "/opt/rocm/core-10.2",
            "version_major": 10,
            "version_minor": 2,
            "version_patch": 0,
            "target": "deb",
            "package_name": package_name,
            "binaries": package_config["binaries"],
            "core_symlinks": "core_symlinks" in package_config,
            "llvm_symlinks": "llvm_symlinks" in package_config,
        }

        template = self.environment.get_template(
            f"build_tools/packaging/linux/template/scripts/amdrocm-{script_name}.j2"
        )

        return template.render(context)

    def test_alternatives_json_exists_and_is_valid(self) -> None:
        self.assertTrue(ALTERNATIVES_JSON.is_file(), f"Missing JSON file: {ALTERNATIVES_JSON}")
        self.assertIsInstance(self.alternatives, dict)
        self.assertTrue(self.alternatives)

    def test_expected_packages_are_configured(self) -> None:
        for package_name in EXPECTED_PACKAGES:
            with self.subTest(package=package_name):
                self.assertIn(package_name, self.alternatives)

    def test_package_configuration_has_valid_structure(self) -> None:
        for package_name, package_config in self.alternatives.items():
            with self.subTest(package=package_name):
                self.assertIsInstance(package_config, dict)
                self.assertIn("scripts", package_config)
                self.assertIn("binaries", package_config)
                self.assertIsInstance(package_config["scripts"], list)
                self.assertIsInstance(package_config["binaries"], list)
                self.assertTrue(package_config["scripts"])
                self.assertTrue(package_config["binaries"])
                self.assertTrue(set(package_config["scripts"]).issubset(SUPPORTED_SCRIPTS))
                self.assertTrue(all(isinstance(script, str) for script in package_config["scripts"]))
                self.assertTrue(all(isinstance(binary, str) for binary in package_config["binaries"]))

    def test_symlink_flags_use_true_strings(self) -> None:
        for package_name, package_config in self.alternatives.items():
            with self.subTest(package=package_name):
                if "core_symlinks" in package_config:
                    self.assertEqual(package_config["core_symlinks"], "True")
                if "llvm_symlinks" in package_config:
                    self.assertEqual(package_config["llvm_symlinks"], "True")

    def test_expected_binary_ownership(self) -> None:
        configured_binaries: dict[str, str] = {}

        for package_name, expected in EXPECTED_PACKAGES.items():
            with self.subTest(package=package_name):
                actual_binaries = set(self.alternatives[package_name]["binaries"])
                self.assertEqual(actual_binaries, expected["binaries"])

                for binary in actual_binaries:
                    with self.subTest(binary=binary):
                        self.assertNotIn(binary, configured_binaries, f"{binary!r} is owned by both {configured_binaries.get(binary)!r} and {package_name!r}")
                        configured_binaries[binary] = package_name

    def test_expected_symlink_ownership(self) -> None:
        self.assertEqual(self.alternatives["amdrocm-base"].get("core_symlinks"), "True")
        self.assertEqual(self.alternatives["amdrocm-llvm"].get("llvm_symlinks"), "True")

        for package_name, package_config in self.alternatives.items():
            with self.subTest(package=package_name):
                if package_name != "amdrocm-base":
                    self.assertNotEqual(package_config.get("core_symlinks", "False"), "True")
                if package_name != "amdrocm-llvm":
                    self.assertNotEqual(package_config.get("llvm_symlinks", "False"), "True")

    def test_shared_templates_exist(self) -> None:
        for script_name in SUPPORTED_SCRIPTS:
            template_path = SCRIPTS_DIR / f"amdrocm-{script_name}.j2"
            self.assertTrue(template_path.is_file(), f"Missing template: {template_path}")

    def test_configured_scripts_render(self) -> None:
        for package_name, package_config in self.alternatives.items():
            for script_name in package_config["scripts"]:
                with self.subTest(package=package_name, script=script_name):
                    rendered = self.render_script(package_name, script_name)
                    self.assertTrue(rendered.strip())
                    self.assertIn("update-alternatives", rendered)

    def test_base_postinst_renders_core_symlink_logic(self) -> None:
        rendered = self.render_script("amdrocm-base", "postinst")
        self.assertIn('"/opt/rocm/core"', rendered)
        self.assertIn('"/opt/rocm/core-10"', rendered)
        self.assertIn('"core"', rendered)
        self.assertIn('"core-10"', rendered)
        self.assertIn('"rocm-bin"', rendered)
        self.assertIn('"rocm-lib"', rendered)
        self.assertIn('"rocm-include"', rendered)
        self.assertIn('"rocm_agent_enumerator"', rendered)
        self.assertIn('"rocminfo"', rendered)
        self.assertNotIn('"rocm-llvm"', rendered)
        self.assertNotIn('"rocm-amdgcn"', rendered)

    def test_base_prerm_renders_core_removal_logic(self) -> None:
        rendered = self.render_script("amdrocm-base", "prerm")
        self.assertIn('"core"', rendered)
        self.assertIn('"core-10"', rendered)
        self.assertIn('"rocm-bin"', rendered)
        self.assertIn('"rocm-lib"', rendered)
        self.assertIn('"rocm-include"', rendered)
        self.assertIn('"rocm_agent_enumerator"', rendered)
        self.assertIn('"rocminfo"', rendered)
        self.assertNotIn('"rocm-llvm"', rendered)
        self.assertNotIn('"rocm-amdgcn"', rendered)

    def test_llvm_postinst_renders_llvm_symlink_logic(self) -> None:
        rendered = self.render_script("amdrocm-llvm", "postinst")
        self.assertIn('"/opt/rocm/llvm"', rendered)
        self.assertIn('"/opt/rocm/amdgcn"', rendered)
        self.assertIn('"/opt/rocm/libexec"', rendered)
        self.assertIn('"/opt/rocm/share"', rendered)
        self.assertIn('"rocm-llvm"', rendered)
        self.assertIn('"rocm-amdgcn"', rendered)
        self.assertIn('"rocm-libexec"', rendered)
        self.assertIn('"rocm-share"', rendered)
        self.assertIn('"hipcc"', rendered)
        self.assertIn('"hipconfig"', rendered)
        self.assertNotIn('"/opt/rocm/core"', rendered)
        self.assertNotIn('"rocm-bin"', rendered)

    def test_llvm_prerm_renders_llvm_removal_logic(self) -> None:
        rendered = self.render_script("amdrocm-llvm", "prerm")
        self.assertIn('"rocm-llvm"', rendered)
        self.assertIn('"rocm-amdgcn"', rendered)
        self.assertIn('"rocm-libexec"', rendered)
        self.assertIn('"rocm-share"', rendered)
        self.assertIn('"hipcc"', rendered)
        self.assertIn('"hipconfig"', rendered)
        self.assertNotIn('"core"', rendered)
        self.assertNotIn('"rocm-bin"', rendered)

    def test_false_symlink_flags_do_not_enable_logic(self) -> None:
        rendered = self.render_script("amdrocm-hipify", "postinst")
        self.assertNotIn('"/opt/rocm/core-10"', rendered)
        self.assertNotIn('"/opt/rocm/llvm"', rendered)
        self.assertIn('"hipify-clang"', rendered)

    def test_binary_entries_generate_commands(self) -> None:
        for package_name, package_config in self.alternatives.items():
            rendered = self.render_script(package_name, "postinst")

            for binary in package_config["binaries"]:
                with self.subTest(package=package_name, binary=binary):
                    self.assertIn(binary, rendered)
                    self.assertIn(f'"/usr/bin/{binary}"', rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
