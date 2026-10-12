# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for install_pytorch_ci_packages.py."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

import install_pytorch_ci_packages as ipcp

SDK_ROOT = "/opt/rocm-sdk"
ROCM_VERSION = "10.2.0a20261012"
INDEX_URL = "https://example.com/whl-next/"


class InstallPytorchCiPackagesTest(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.pytorch_dir = Path(temp_dir.name)
        self.pin_dir = self.pytorch_dir / ".ci" / "docker" / "ci_commit_pins"
        self.pin_dir.mkdir(parents=True)

        self.calls = []
        self.installed_version = "0.0.2"
        self.origami_install_failures = 0
        self.devel_installed = False

        patchers = [
            mock.patch.object(ipcp.platform, "system", return_value="Linux"),
            mock.patch.object(ipcp.subprocess, "run", side_effect=self.fake_run),
            mock.patch.object(
                ipcp, "rocm_devel_installed", side_effect=lambda: self.devel_installed
            ),
            mock.patch.object(ipcp, "PIP_RETRY_WAIT_SECONDS", 0),
            mock.patch.dict(os.environ, {"CMAKE_PREFIX_PATH": "/existing"}),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("CC", None)
        os.environ.pop("CXX", None)

    def fake_run(self, cmd, env=None, check=True, text=True, stdout=None):
        self.calls.append((cmd, env))
        if cmd[1:3] == ["-m", "rocm_sdk"]:
            outputs = {
                "version": ROCM_VERSION,
                "init": "",
                "--cmake": f"{SDK_ROOT}/lib/cmake",
                "--root": SDK_ROOT,
            }
            return subprocess.CompletedProcess(cmd, 0, stdout=outputs[cmd[-1]])
        if cmd[1] == "-c":
            return subprocess.CompletedProcess(cmd, 0, stdout=self.installed_version)
        if cmd[1:4] == ["-m", "pip", "install"]:
            if cmd[-1].startswith("rocm-origami") and self.origami_install_failures:
                self.origami_install_failures -= 1
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=None)
        raise AssertionError(f"Unexpected command: {cmd}")

    def write_pin(self, text):
        (self.pin_dir / "rocm-origami.txt").write_text(text)

    def run_main(self, *extra_args):
        return ipcp.main(
            [
                "--pytorch-dir",
                str(self.pytorch_dir),
                "--index-url",
                INDEX_URL,
                *extra_args,
            ]
        )

    def pip_installs(self):
        return [
            (cmd[4:], env)
            for cmd, env in self.calls
            if cmd[1:4] == ["-m", "pip", "install"]
        ]

    def test_no_pin_file_installs_nothing(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.calls, [])

    def test_missing_pin_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            ipcp.main(["--pytorch-dir", str(self.pytorch_dir / "missing")])

    def test_non_linux_installs_nothing(self):
        self.write_pin("0.0.2\n")
        with mock.patch.object(ipcp.platform, "system", return_value="Windows"):
            self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.calls, [])

    def test_malformed_pin_raises(self):
        self.write_pin("not a version\n")
        with self.assertRaises(ValueError):
            self.run_main()

    def test_installs_rocm_devel_then_pinned_version(self):
        self.write_pin("0.0.2\n")
        self.assertEqual(self.run_main(), 0)

        devel_install, origami_install = self.pip_installs()
        self.assertEqual(
            devel_install[0],
            [f"--index-url={INDEX_URL}", f"rocm[devel]=={ROCM_VERSION}"],
        )
        args, env = origami_install
        self.assertEqual(args, ["rocm-origami==0.0.2"])
        self.assertEqual(
            env["CMAKE_PREFIX_PATH"],
            os.pathsep.join([f"{SDK_ROOT}/lib/cmake", "/existing"]),
        )
        llvm_bin = Path(SDK_ROOT) / "lib" / "llvm" / "bin"
        self.assertEqual(env["CC"], str(llvm_bin / "clang"))
        self.assertEqual(env["CXX"], str(llvm_bin / "clang++"))

    def test_find_links_is_passed_to_rocm_devel_install(self):
        self.write_pin("0.0.2\n")
        self.run_main("--find-links", "https://example.com/run/index.html")
        devel_install, _ = self.pip_installs()
        self.assertIn(
            "--find-links=https://example.com/run/index.html", devel_install[0]
        )

    def test_keeps_compilers_from_environment(self):
        self.write_pin("0.0.2\n")
        with mock.patch.dict(os.environ, {"CC": "my-cc", "CXX": "my-cxx"}):
            self.run_main()
        _, (_, env) = self.pip_installs()
        self.assertEqual((env["CC"], env["CXX"]), ("my-cc", "my-cxx"))

    def test_skips_rocm_devel_when_installed(self):
        self.write_pin("0.0.2\n")
        self.devel_installed = True
        self.run_main()
        self.assertEqual(
            [args for args, _ in self.pip_installs()], [["rocm-origami==0.0.2"]]
        )

    def test_retries_failed_install_once(self):
        self.write_pin("0.0.2\n")
        self.origami_install_failures = 1
        self.assertEqual(self.run_main(), 0)
        origami_installs = [
            args for args, _ in self.pip_installs() if args == ["rocm-origami==0.0.2"]
        ]
        self.assertEqual(len(origami_installs), 2)

    def test_gives_up_after_second_failure(self):
        self.write_pin("0.0.2\n")
        self.origami_install_failures = 2
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_main()

    def test_version_mismatch_raises(self):
        self.write_pin("0.0.2\n")
        self.installed_version = "0.0.3"
        with self.assertRaises(RuntimeError):
            self.run_main()

    def test_dry_run_installs_nothing(self):
        self.write_pin("0.0.2\n")
        self.assertEqual(self.run_main("--dry-run"), 0)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
