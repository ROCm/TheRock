#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for native_windows_package_uninstall_test.py.

All Windows-specific I/O (msiexec, winreg, filesystem) is mocked, so these run
on any platform including the Linux unit-test runner.
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
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the install module first (the uninstall module imports from it by name).
_load_module(
    "native_windows_package_install_test",
    WINDOWS_DIR / "native_windows_package_install_test.py",
)
uninstall_mod = _load_module(
    "native_windows_package_uninstall_test",
    WINDOWS_DIR / "native_windows_package_uninstall_test.py",
)


class FindProductCodeTest(unittest.TestCase):
    def _fake_winreg(self, entries: dict[str, str]):
        """Return a fake winreg whose Uninstall hive exposes ``entries``.

        entries maps {product_code: display_name}. EnumKey walks the codes;
        OpenKey on a code records it so QueryValueEx("DisplayName") returns the
        mapped name.
        """
        fake = mock.MagicMock()
        fake.HKEY_LOCAL_MACHINE = 0
        fake.KEY_READ = 1
        fake.KEY_WOW64_64KEY = 2

        codes = list(entries)
        hive_ctx = mock.MagicMock()
        hive_ctx.__enter__.return_value = mock.MagicMock()
        last_code = {"code": None}

        def enum_key(_hive, index):
            if index < len(codes):
                return codes[index]
            raise OSError("no more")

        def open_key(_root, subkey, *_a, **_k):
            if subkey == uninstall_mod.UNINSTALL_KEY:
                return hive_ctx
            last_code["code"] = subkey
            entry_ctx = mock.MagicMock()
            entry_ctx.__enter__.return_value = mock.MagicMock()
            entry_ctx.__exit__.return_value = False
            return entry_ctx

        fake.EnumKey.side_effect = enum_key
        fake.OpenKey.side_effect = open_key
        fake.QueryValueEx.side_effect = lambda _entry, _name: (
            entries[last_code["code"]],
            1,
        )
        return fake

    def test_finds_matching_display_name(self):
        entries = {
            "{AAA}": "Some Other Product",
            "{BBB}": "AMD ROCm Runtime",
        }
        fake = self._fake_winreg(entries)
        with mock.patch.dict(sys.modules, {"winreg": fake}):
            code = uninstall_mod.find_product_code("AMD ROCm Runtime")
        self.assertEqual(code, "{BBB}")

    def test_missing_display_name_raises(self):
        fake = self._fake_winreg({"{AAA}": "Nope"})
        with mock.patch.dict(sys.modules, {"winreg": fake}):
            with self.assertRaises(RuntimeError):
                uninstall_mod.find_product_code("AMD ROCm Runtime")


class UninstallMsiTest(unittest.TestCase):
    def test_builds_remove_argv(self):
        with mock.patch.object(
            uninstall_mod.subprocess, "run", return_value=mock.MagicMock(returncode=0)
        ) as run:
            uninstall_mod.uninstall_msi("{BBB}", Path("log.txt"))
        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], ["msiexec", "/x", "{BBB}"])
        self.assertIn("/qn", argv)

    def test_nonzero_raises(self):
        with mock.patch.object(
            uninstall_mod.subprocess,
            "run",
            return_value=mock.MagicMock(returncode=1612),
        ):
            with mock.patch.object(uninstall_mod, "_print_log_tail"):
                with self.assertRaises(RuntimeError):
                    uninstall_mod.uninstall_msi("{BBB}", Path("log.txt"))


class VerifyInstallDirRemovedTest(unittest.TestCase):
    def test_passes_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            # program_files/AMD/ROCm/core-10.2 does not exist.
            uninstall_mod.verify_install_dir_removed(
                "runtime", "10.2.0", program_files=Path(tmp)
            )

    def test_raises_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            pf = Path(tmp)
            install_dir = pf / "AMD" / "ROCm" / "core-10.2"
            install_dir.mkdir(parents=True)
            (install_dir / "f.txt").write_text("x")
            with self.assertRaises(RuntimeError):
                uninstall_mod.verify_install_dir_removed(
                    "runtime", "10.2.0", program_files=pf
                )


class VerifyRegistryKeyRemovedTest(unittest.TestCase):
    def test_passes_when_open_raises_filenotfound(self):
        fake = mock.MagicMock()
        fake.HKEY_LOCAL_MACHINE = 0
        fake.KEY_READ = 1
        fake.KEY_WOW64_64KEY = 2
        fake.OpenKey.side_effect = FileNotFoundError()
        with mock.patch.dict(sys.modules, {"winreg": fake}):
            uninstall_mod.verify_registry_key_removed("runtime", "10.2.0")  # no raise

    def test_raises_when_key_still_opens(self):
        fake = mock.MagicMock()
        fake.HKEY_LOCAL_MACHINE = 0
        fake.KEY_READ = 1
        fake.KEY_WOW64_64KEY = 2
        fake.OpenKey.return_value = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"winreg": fake}):
            with self.assertRaises(RuntimeError):
                uninstall_mod.verify_registry_key_removed("runtime", "10.2.0")


class NonWindowsSkipTest(unittest.TestCase):
    def test_run_returns_2_on_non_windows(self):
        runner = uninstall_mod.WindowsPackageUninstallTest(
            packages=["runtime"], rocm_version="10.2.0"
        )
        with mock.patch.object(uninstall_mod.sys, "platform", "linux"):
            self.assertEqual(runner.run(), 2)


if __name__ == "__main__":
    unittest.main()
