#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for native_windows_package_install_test.py.

All Windows-specific I/O (msiexec, downloads, winreg, filesystem) is mocked, so
these run on any platform including the Linux unit-test runner.
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
WINDOWS_DIR = THIS_DIR.parent
BUILD_TOOLS_DIR = WINDOWS_DIR.parent.parent

for path in (BUILD_TOOLS_DIR, WINDOWS_DIR):
    path_str = os.fspath(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_module(name: str, path: Path) -> types.ModuleType:
    """Load a module by path under a unique registered name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


install_mod = _load_module(
    "native_windows_package_install_test",
    WINDOWS_DIR / "native_windows_package_install_test.py",
)


class MajorMinorTest(unittest.TestCase):
    def test_parses_major_minor(self):
        self.assertEqual(install_mod.major_minor("10.2.0"), ("10", "2"))

    def test_rejects_malformed(self):
        with self.assertRaises(ValueError):
            install_mod.major_minor("10")


class ExpectedMsiFilenameTest(unittest.TestCase):
    def test_runtime_and_core(self):
        self.assertEqual(
            install_mod.expected_msi_filename("runtime"), "amdrocm-runtime.msi"
        )
        self.assertEqual(install_mod.expected_msi_filename("core"), "amdrocm-core.msi")

    def test_unknown_package_raises(self):
        with self.assertRaises(ValueError):
            install_mod.expected_msi_filename("bogus")


class InstallDirForTest(unittest.TestCase):
    def test_expands_subdir_from_version(self):
        root = Path("C:/pf")
        got = install_mod.install_dir_for("runtime", "10.2.0", program_files=root)
        self.assertEqual(got, root / "AMD" / "ROCm" / "core-10.2")


class ResolveMsiPrefixUrlTest(unittest.TestCase):
    def test_passes_no_api_lookup_and_returns_https_url(self):
        fake_loc = mock.MagicMock()
        fake_loc.https_url = "https://bucket.s3.amazonaws.com/99-windows/packages/msi"
        fake_root = mock.MagicMock()
        fake_root.native_windows_packages.return_value = fake_loc
        with mock.patch.object(
            install_mod.WorkflowOutputRoot,
            "from_workflow_run",
            return_value=fake_root,
        ) as from_run:
            url = install_mod.resolve_msi_prefix_url("99", "ROCm/TheRock", "nightly")
        self.assertEqual(url, fake_loc.https_url)
        _, kwargs = from_run.call_args
        self.assertFalse(kwargs["lookup_workflow_run"])
        self.assertEqual(kwargs["release_type"], "nightly")
        self.assertEqual(kwargs["github_repository"], "ROCm/TheRock")
        fake_root.native_windows_packages.assert_called_once_with("msi")


class DownloadMsisTest(unittest.TestCase):
    def test_builds_url_and_downloads_each(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            with mock.patch.object(
                install_mod.urllib.request, "urlretrieve"
            ) as urlretrieve:
                paths = install_mod.download_msis(
                    "https://b.s3.amazonaws.com/99-windows/packages/msi/",
                    ["runtime"],
                    dest,
                )
            self.assertEqual(paths, [dest / "amdrocm-runtime.msi"])
            url, _ = urlretrieve.call_args.args
            self.assertEqual(
                url,
                "https://b.s3.amazonaws.com/99-windows/packages/msi/amdrocm-runtime.msi",
            )

    def test_download_failure_raises_filenotfound(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                install_mod.urllib.request,
                "urlretrieve",
                side_effect=install_mod.urllib.error.URLError("boom"),
            ):
                with self.assertRaises(FileNotFoundError):
                    install_mod.download_msis("https://b/msi", ["runtime"], Path(tmp))


class InstallMsiTest(unittest.TestCase):
    def test_builds_silent_argv(self):
        with mock.patch.object(
            install_mod.subprocess, "run", return_value=mock.MagicMock(returncode=0)
        ) as run:
            install_mod.install_msi(Path("x.msi"), Path("log.txt"))
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], ["msiexec", "/i"])
        self.assertIn("/qn", argv)
        self.assertIn("/norestart", argv)
        self.assertIn("/l*v", argv)

    def test_nonzero_returncode_raises(self):
        with mock.patch.object(
            install_mod.subprocess, "run", return_value=mock.MagicMock(returncode=1603)
        ):
            with mock.patch.object(install_mod, "_print_log_tail"):
                with self.assertRaises(RuntimeError):
                    install_mod.install_msi(Path("x.msi"), Path("log.txt"))


class VerifyInstallDirTest(unittest.TestCase):
    def test_passes_on_populated_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "f.txt").write_text("x")
            install_mod.verify_install_dir(d)  # no raise

    def test_raises_on_missing(self):
        with self.assertRaises(RuntimeError):
            install_mod.verify_install_dir(Path("/no/such/dir"))

    def test_raises_on_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                install_mod.verify_install_dir(Path(tmp))


class VerifyPayloadDllsTest(unittest.TestCase):
    def _populate(self, root: Path):
        bindir = root / "lib"
        bindir.mkdir(parents=True)
        for name in [
            "amdhip64_7.dll",
            "hiprtc0717.dll",
            "amd_comgr.dll",
            "rocm_kpack.dll",
        ]:
            (bindir / name).write_bytes(b"x")

    def test_passes_when_all_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._populate(root)
            install_mod.verify_payload_dlls(root, "runtime")  # no raise

    def test_raises_listing_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lib").mkdir()
            (root / "lib" / "amdhip64_7.dll").write_bytes(b"x")  # only one present
            with self.assertRaises(RuntimeError):
                install_mod.verify_payload_dlls(root, "runtime")


class VerifySystem32DllsTest(unittest.TestCase):
    def test_runtime_passes_when_all_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys32 = Path(tmp)
            from generate_msi_wxs import PACKAGES

            for name in PACKAGES["runtime"].legacy_system32_dlls:
                (sys32 / name).write_bytes(b"x")
            install_mod.verify_system32_dlls("runtime", system32=sys32)  # no raise

    def test_runtime_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                install_mod.verify_system32_dlls("runtime", system32=Path(tmp))

    def test_core_is_noop(self):
        # 'core' declares no legacy System32 DLLs, so an empty dir passes.
        with tempfile.TemporaryDirectory() as tmp:
            install_mod.verify_system32_dlls("core", system32=Path(tmp))  # no raise


class VerifyRegistryKeyTest(unittest.TestCase):
    def _fake_winreg(
        self, install_dir="C:\\pf\\AMD\\ROCm\\core-10.2", raise_open=False
    ):
        fake = mock.MagicMock()
        fake.HKEY_LOCAL_MACHINE = 0
        fake.KEY_READ = 1
        fake.KEY_WOW64_64KEY = 2
        if raise_open:
            fake.OpenKey.side_effect = FileNotFoundError()
        else:
            ctx = mock.MagicMock()
            ctx.__enter__.return_value = "key"
            fake.OpenKey.return_value = ctx
            fake.QueryValueEx.return_value = (install_dir, 1)
        return fake

    def test_passes_when_key_and_value_present(self):
        fake = self._fake_winreg()
        with mock.patch.dict(sys.modules, {"winreg": fake}):
            install_mod.verify_registry_key("runtime", "10.2.0")
        subkey = fake.OpenKey.call_args.args[1]
        self.assertEqual(subkey, "Software\\AMD\\ROCm\\runtime\\10.2")
        # 64-bit view requested.
        flags = fake.OpenKey.call_args.args[3]
        self.assertEqual(flags, fake.KEY_READ | fake.KEY_WOW64_64KEY)

    def test_raises_when_key_absent(self):
        fake = self._fake_winreg(raise_open=True)
        with mock.patch.dict(sys.modules, {"winreg": fake}):
            with self.assertRaises(RuntimeError):
                install_mod.verify_registry_key("runtime", "10.2.0")


class ArgvFromCiEnvTest(unittest.TestCase):
    def test_none_when_required_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(install_mod._argv_from_ci_env())

    def test_builds_argv_with_optionals(self):
        env = {
            "ARTIFACT_RUN_ID": "99",
            "ROCM_VERSION": "10.2.0",
            "RELEASE_TYPE": "nightly",
            "PACKAGES": "runtime,core",
            "ARTIFACT_GITHUB_REPO": "me/fork",
            "MSI_DIR": "d:/msis",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            argv = install_mod._argv_from_ci_env()
        self.assertIn("--artifact-run-id", argv)
        self.assertIn("me/fork", argv)
        self.assertIn("--msi-dir", argv)

    def test_parse_rejects_unknown_package(self):
        argv = [
            "--artifact-run-id",
            "1",
            "--rocm-version",
            "10.2.0",
            "--packages",
            "bogus",
        ]
        args = install_mod.parse_cli_arguments(argv, raise_instead_of_exit=True)
        with self.assertRaises(ValueError):
            install_mod.WindowsPackageInstallTest(
                artifact_run_id=args.artifact_run_id,
                artifact_github_repo=args.artifact_github_repo,
                release_type=args.release_type,
                packages=args.packages,
                rocm_version=args.rocm_version,
            )


class NonWindowsSkipTest(unittest.TestCase):
    def test_run_returns_2_on_non_windows(self):
        runner = install_mod.WindowsPackageInstallTest(
            artifact_run_id="1",
            artifact_github_repo="ROCm/TheRock",
            release_type="nightly",
            packages=["runtime"],
            rocm_version="10.2.0",
        )
        with mock.patch.object(install_mod.sys, "platform", "linux"):
            self.assertEqual(runner.run(), 2)


if __name__ == "__main__":
    unittest.main()
