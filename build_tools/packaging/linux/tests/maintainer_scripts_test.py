#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for generated native package maintainer scripts."""

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LINUX_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINUX_DIR))

from deb_package import generate_debian_postscripts
from packaging_utils import (
    PackageConfig,
    get_package_info,
    is_postinstallscripts_available,
)
from rpm_package import generate_rpm_postscripts


class MaintainerScriptsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp_dir = Path(temporary.name)
        self.site_dir = self.temp_dir / "site-packages"
        self.query_dir = self.temp_dir / "python-query"
        self.query_dir.mkdir()
        # Redirect only sysconfig's installation directory. The generated shell,
        # system Python, hashing, atomic writes and cleanup all execute normally.
        (self.query_dir / "sysconfig.py").write_text(
            "def get_scheme_names(): return ('deb_system', 'posix_prefix')\n"
            f"def get_path(name, scheme): return {str(self.site_dir)!r}\n"
            f"def get_paths(): return {{'purelib': {str(self.site_dir)!r}}}\n"
        )
        self.bin_dir = self.temp_dir / "bin"
        self.bin_dir.mkdir()
        shadow_python = self.bin_dir / "python3"
        shadow_python.write_text("#!/bin/sh\nexit 77\n")
        shadow_python.chmod(0o755)

    def generate(
        self, package: str, target: str, prefix: str = "/srv/relocated/core-7.1"
    ) -> dict[str, str]:
        config = PackageConfig(
            artifacts_dir=self.temp_dir,
            dest_dir=self.temp_dir,
            pkg_type=target,
            rocm_version="7.1.0",
            version_suffix="1",
            install_prefix=prefix,
            gfx_arch="",
        )
        info = get_package_info(package)
        self.assertTrue(is_postinstallscripts_available(info))
        if target == "rpm":
            return generate_rpm_postscripts(info, config)
        output = self.temp_dir / package
        output.mkdir(exist_ok=True)
        generate_debian_postscripts(info, output, config)
        for script in output.iterdir():
            self.assertTrue(script.stat().st_mode & 0o111)
        return {script.name: script.read_text() for script in output.iterdir()}

    def execute(
        self, script: str, target: str, prefix: str, action: str, hooks: str = ""
    ) -> str:
        os_id = "debian" if target == "deb" else "rhel"
        result = subprocess.run(
            ["bash", "-e", "-s", "--", action],
            input=f"source() {{ ID={os_id}; ID_LIKE={os_id}; }}\n" + hooks + script,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "PATH": f"{self.bin_dir}:/usr/bin:/bin",
                "PYTHONPATH": str(self.query_dir),
                "PYTHONDONTWRITEBYTECODE": "1",
                "RPM_INSTALL_PREFIX0": prefix,
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def registration(self, prefix: str) -> Path:
        digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        return self.site_dir / f"amdrocm-amdsmi-{digest}.pth"

    def test_prefix_ownership_and_late_purge(self) -> None:
        for target, post, postrm, install_arg, remove_arg in (
            ("deb", "postinst", "postrm", "configure", "purge"),
            ("rpm", "%post", "%postun", "1", "0"),
        ):
            with self.subTest(target=target):
                prefixes = [
                    str(self.temp_dir / target / parent / "core-7.1")
                    for parent in ("old", "new")
                ]
                scripts = [self.generate("amdrocm-amdsmi", target, p) for p in prefixes]
                self.assertEqual(set(scripts[0]), {post, postrm})
                self.site_dir.mkdir(exist_ok=True)
                legacy = self.site_dir / "amdsmi.pth"
                legacy.write_text("wheel-owned-entry\n")
                registrations = [self.registration(p) for p in prefixes]
                for prefix, generated, pth in zip(prefixes, scripts, registrations):
                    self.execute(generated[post], target, prefix, install_arg)
                    self.assertEqual(pth.read_text(), "/opt/rocm/share/amd_smi\n")
                    self.assertEqual(pth.stat().st_mode & 0o777, 0o644)
                self.assertNotEqual(*registrations)

                # Atomic reinstall must replace a symlink, not overwrite its target.
                victim = self.temp_dir / "unrelated"
                victim.write_text("untouched")
                registrations[0].unlink()
                registrations[0].symlink_to(victim)
                self.execute(scripts[0][post], target, prefixes[0], install_arg)
                self.assertFalse(registrations[0].is_symlink())
                self.assertEqual(victim.read_text(), "untouched")
                self.assertEqual(
                    len(list(self.site_dir.glob("amdrocm-amdsmi-*.pth"))), 2
                )
                self.assertEqual(list(self.site_dir.glob("*.pth.*")), [])

                cache = Path(prefixes[0]) / "share/amd_smi/amdsmi/__pycache__"
                cache.mkdir(parents=True)
                (cache / "stale.pyc").write_bytes(b"cache")
                for _ in range(2):
                    self.execute(scripts[0][postrm], target, prefixes[0], remove_arg)
                    self.assertFalse(registrations[0].exists())
                    self.assertTrue(registrations[1].exists())
                self.assertFalse(cache.exists())
                self.execute(scripts[1][postrm], target, prefixes[1], remove_arg)
                self.assertEqual(list(self.site_dir.glob("amdrocm-amdsmi-*.pth")), [])
                self.assertEqual(legacy.read_text(), "wheel-owned-entry\n")

    def test_rpm_old_postun_preserves_replacement_payload(self) -> None:
        prefix = str(self.temp_dir / "core-7.1")
        scripts = self.generate("amdrocm-amdsmi", "rpm", prefix)
        self.execute(scripts["%post"], "rpm", prefix, "2")
        module = Path(prefix) / "share/amd_smi/amdsmi"
        module.mkdir(parents=True)
        initializer = module / "__init__.py"
        initializer.write_text("VERSION = 'replacement'\n")
        self.execute(scripts["%postun"], "rpm", prefix, "1")
        self.assertTrue(self.registration(prefix).exists())
        initializer.unlink()
        self.execute(scripts["%postun"], "rpm", prefix, "0")
        self.assertFalse(self.registration(prefix).exists())

    def test_registration_identity_uses_literal_prefix(self) -> None:
        prefix = str(self.temp_dir / "core-7.1")
        for literal in (prefix, prefix + "/"):
            scripts = self.generate("amdrocm-amdsmi", "rpm", literal)
            self.execute(scripts["%post"], "rpm", literal, "1")
            self.assertTrue(self.registration(literal).exists())
        self.assertEqual(len(list(self.site_dir.glob("amdrocm-amdsmi-*.pth"))), 2)

    def test_foreign_content_is_not_removed(self) -> None:
        prefix = str(self.temp_dir / "core-7.1")
        scripts = self.generate("amdrocm-amdsmi", "deb", prefix)
        self.execute(scripts["postinst"], "deb", prefix, "configure")
        pth = self.registration(prefix)
        content = "/opt/rocm/share/amd_smi\n/another/module\n"
        pth.write_text(content)
        self.execute(scripts["postrm"], "deb", prefix, "purge")
        self.assertEqual(pth.read_text(), content)

    def test_query_failure_is_nonfatal_under_errexit(self) -> None:
        (self.query_dir / "sysconfig.py").write_text(
            "raise RuntimeError('query failed')\n"
        )
        for target, post, postrm, install_arg, remove_arg in (
            ("deb", "postinst", "postrm", "configure", "remove"),
            ("rpm", "%post", "%postun", "1", "0"),
        ):
            for name, action in ((post, install_arg), (postrm, remove_arg)):
                with self.subTest(target=target, script=name):
                    prefix = str(self.temp_dir / target)
                    scripts = self.generate("amdrocm-amdsmi", target, prefix)
                    self.execute(scripts[name], target, prefix, action)
                    self.assertFalse(self.site_dir.exists())


if __name__ == "__main__":
    unittest.main()
