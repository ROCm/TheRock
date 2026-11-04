#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for amdrocm-alternatives.json and the shared postinst/prerm templates.

Scripts come from the real deb and RPM generators. Packages and binaries are
read from the config files; only the template-defined symlink groups are listed.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

LINUX_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(LINUX_DIR.parents[1]), str(LINUX_DIR)]

import deb_package  # noqa: E402
import rpm_package  # noqa: E402
from packaging_utils import (  # noqa: E402
    PackageConfig,
    is_key_defined,
    is_postinstallscripts_available,
    load_alternatives_binaries,
    read_package_json_file,
)

MAJOR, MINOR = 10, 2
VERSION = f"{MAJOR}.{MINOR}.0"
PREFIX = f"/opt/rocm/core-{MAJOR}.{MINOR}"
RPM_SECTIONS = {"postinst": "%post", "prerm": "%preun"}
ACTIONS = {"postinst": "--install", "prerm": "--remove"}
PREFIX_LINES = {
    "deb": f"PKG_INSTALL_PREFIX={PREFIX}",
    "rpm": 'PKG_INSTALL_PREFIX="$RPM_INSTALL_PREFIX0"',
}
# Symlink flag -> (only package allowed to set it, alternatives it controls).
SYMLINK_GROUPS = {
    "core_symlinks": (
        "amdrocm-base",
        ["core", f"core-{MAJOR}", "rocm-bin", "rocm-lib", "rocm-include"]
        + ["rocm-libexec", "rocm-share"],
    ),
    "llvm_symlinks": ("amdrocm-llvm", ["rocm-llvm", "rocm-amdgcn"]),
}

# Packages whose scripts run against a real update-alternatives, the binaries
# that exist in the sandbox (the rest must be skipped), and the
# /opt/rocm/<link> -> <prefix>/<dir> links the two packages own.
SANDBOX_PACKAGES = ("amdrocm-base", "amdrocm-llvm")
SANDBOX_BINARIES = ("rocminfo", "hipcc", "amdclang")
SANDBOX_DIR_LINKS = {"core": "", f"core-{MAJOR}": ""} | {
    d: d for d in ("bin", "lib", "include", "libexec", "share", "llvm", "amdgcn")
}
UPDATE_ALTERNATIVES = shutil.which("update-alternatives")


@unittest.skipIf(
    sys.platform == "win32",
    "Linux maintainer scripts are only built and installed on Linux.",
)
class AlternativesScriptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.alternatives = load_alternatives_binaries()
        cls.packages = {pkg["Package"]: pkg for pkg in read_package_json_file()}
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.deb_root = Path(temp.name)

        cls.scripts = {}  # (package, target, script) -> rendered text
        for name in cls.alternatives.keys() & cls.packages.keys():
            for target in PREFIX_LINES:
                config = PackageConfig(
                    cls.deb_root, cls.deb_root, target, VERSION, "", PREFIX, ""
                )
                if target == "deb":
                    deb_dir = cls.deb_root / name
                    deb_package.generate_debian_postscripts(
                        cls.packages[name], deb_dir, config
                    )
                    for file in deb_dir.iterdir():
                        cls.scripts[(name, target, file.name)] = file.read_text(
                            encoding="utf-8"
                        )
                else:
                    sections = rpm_package.generate_rpm_postscripts(
                        cls.packages[name], config
                    )
                    for script, section in RPM_SECTIONS.items():
                        cls.scripts[(name, target, script)] = sections.get(section)

    def test_config_matches_postinstall_flags(self) -> None:
        flagged = {
            name
            for name, pkg in self.packages.items()
            if is_postinstallscripts_available(pkg)
        }
        self.assertEqual(set(self.alternatives), flagged)

    def test_each_binary_and_symlink_group_has_one_owner(self) -> None:
        counts = Counter(
            binary for cfg in self.alternatives.values() for binary in cfg["binaries"]
        )
        self.assertEqual([b for b, n in counts.items() if n > 1], [])
        for flag, (owner, _) in SYMLINK_GROUPS.items():
            owners = [
                n for n, cfg in self.alternatives.items() if is_key_defined(cfg, flag)
            ]
            self.assertEqual(owners, [owner], flag)

    def test_generated_scripts(self) -> None:
        for name, cfg in self.alternatives.items():
            for target in PREFIX_LINES:
                for script in ACTIONS:
                    with self.subTest(package=name, target=target, script=script):
                        text = self.scripts.get((name, target, script))
                        self.assertIsNotNone(text, "script was not generated")

                        bash = subprocess.run(["bash", "-n"], input=text, text=True)
                        self.assertEqual(bash.returncode, 0)
                        if target == "deb":
                            path = self.deb_root / name / script
                            self.assertTrue(os.access(path, os.X_OK))

                        # postinst only installs and prerm only removes.
                        for other, other_action in ACTIONS.items():
                            self.assertEqual(other_action in text, other == script)
                        self.assertIn(PREFIX_LINES[target], text)

                        for binary in cfg["binaries"]:
                            self.assertIn(f'"$PKG_INSTALL_PREFIX/bin/{binary}"', text)
                        for flag, (_, alternatives) in SYMLINK_GROUPS.items():
                            enabled = is_key_defined(cfg, flag)
                            for alternative in alternatives:
                                self.assertEqual(f'"{alternative}"' in text, enabled)

    def sandbox(self, name: str, target: str, opt_rocm: bool = True):
        """Render the scripts of SANDBOX_PACKAGES next to a sandboxed update-alternatives.

        Returns run(script, action, names, **env), links() and the expected links.
        """
        root = (self.deb_root / name).resolve()
        prefix = root / "prefix"
        for directory in ("usr/bin", "alt", "admin", "tools"):
            (root / directory).mkdir(parents=True)
        if opt_rocm:
            (root / "opt/rocm").mkdir(parents=True)
        for directory in SANDBOX_DIR_LINKS.values():
            (prefix / directory).mkdir(parents=True, exist_ok=True)
        for binary in SANDBOX_BINARIES:
            (prefix / "bin" / binary).touch(mode=0o755)

        # Same tools, but the /usr/bin and /opt/rocm paths they get live in root.
        rewrite = (
            "args=()\n"
            'for arg in "$@"; do\n'
            f'    case "$arg" in /usr/bin/*|/opt/rocm|/opt/rocm/*) arg="{root}$arg" ;; esac\n'
            '    args+=("$arg")\n'
            "done\n"
        )
        tools = {
            "update-alternatives": f"{UPDATE_ALTERNATIVES} --altdir {root}/alt "
            f"--admindir {root}/admin",
            "mkdir": shutil.which("mkdir"),
        }
        for tool, command in tools.items():
            wrapper = root / "tools" / tool
            wrapper.write_text(
                f'#!/bin/bash\n{rewrite}exec {command} "${{args[@]}}"\n',
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{root / 'tools'}{os.pathsep}{os.environ['PATH']}",
        }
        env.pop("RPM_INSTALL_PREFIX0", None)

        config = PackageConfig(root, root, target, VERSION, "", str(prefix), "")
        for package in SANDBOX_PACKAGES:
            if target == "deb":
                deb_package.generate_debian_postscripts(
                    self.packages[package], root / package, config
                )
                continue
            sections = rpm_package.generate_rpm_postscripts(
                self.packages[package], config
            )
            (root / package).mkdir()
            for script, section in RPM_SECTIONS.items():
                path = root / package / script
                path.write_text(sections[section], encoding="utf-8")
                path.chmod(0o755)

        def run(script: str, action: str, names=SANDBOX_PACKAGES, **extra) -> None:
            for package in names:
                subprocess.run(
                    [root / package / script, action], env=env | extra, check=True
                )

        def links() -> dict[str, Path]:
            return {
                str(link.relative_to(root)): Path(os.path.realpath(link))
                for directory in ("usr/bin", "opt/rocm")
                if (root / directory).is_dir()
                for link in (root / directory).iterdir()
            }

        expected = {f"usr/bin/{b}": prefix / "bin" / b for b in SANDBOX_BINARIES}
        expected |= {
            f"opt/rocm/{link}": prefix / directory
            for link, directory in SANDBOX_DIR_LINKS.items()
        }
        return run, links, expected, {"RPM_INSTALL_PREFIX0": str(prefix)}

    @unittest.skipUnless(UPDATE_ALTERNATIVES, "update-alternatives is not installed")
    def test_update_alternatives_install_and_remove(self) -> None:
        """Run the real deb postinst and prerm against a sandboxed update-alternatives."""
        run, links, expected, _ = self.sandbox("deb", "deb")

        run("postinst", "configure")
        self.assertEqual(links(), expected)

        run("prerm", "remove", reversed(SANDBOX_PACKAGES))
        self.assertEqual(links(), {})

    @unittest.skipUnless(UPDATE_ALTERNATIVES, "update-alternatives is not installed")
    def test_rpm_upgrade_keeps_alternatives(self) -> None:
        """A relocated RPM install creates /opt/rocm, and only an erase removes links."""
        run, links, expected, rpm_env = self.sandbox("rpm", "rpm", opt_rocm=False)

        run("postinst", "1", **rpm_env)
        self.assertEqual(links(), expected)

        # On upgrade RPM runs the new %post, then the old %preun with $1 = 1.
        run("postinst", "2", **rpm_env)
        run("prerm", "1", **rpm_env)
        self.assertEqual(links(), expected)

        run("prerm", "0", reversed(SANDBOX_PACKAGES), **rpm_env)
        self.assertEqual(links(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
