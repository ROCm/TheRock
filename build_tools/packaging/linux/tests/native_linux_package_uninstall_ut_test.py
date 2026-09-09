# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Unit test coverage for native_linux_package_uninstall_test.py:
#   Uninstall Step 4a/4b behaviour with mocked subprocess and run_streaming.
#   Integration-only (real apt/dnf/zypper, root): main() and pytest CI entry paths.
#   Requires Python 3.12+ (matches TheRock packaging test runtime).

import contextlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_this_file = Path(__file__).resolve()
_search_dirs = [_this_file.parent, _this_file.parent.parent]
_module_path = None
for _d in _search_dirs:
    _candidate = _d / "native_linux_package_uninstall_test.py"
    if _candidate.is_file():
        _module_path = _candidate
        break
if _module_path is None:
    _checked = ", ".join(str(d) for d in _search_dirs)
    raise FileNotFoundError(
        f"native_linux_package_uninstall_test.py not found in: {_checked}"
    )

_common_path = _module_path.parent / "native_linux_package_test_common.py"
_common_spec = importlib.util.spec_from_file_location(
    "native_linux_package_test_common", _common_path
)
native_linux_package_test_common = importlib.util.module_from_spec(_common_spec)
sys.modules["native_linux_package_test_common"] = native_linux_package_test_common
_common_spec.loader.exec_module(native_linux_package_test_common)

_pu_path = _module_path.parent / "packaging_utils.py"
_pu_spec = importlib.util.spec_from_file_location("packaging_utils", _pu_path)
packaging_utils = importlib.util.module_from_spec(_pu_spec)
sys.modules["packaging_utils"] = packaging_utils
_pu_spec.loader.exec_module(packaging_utils)

_spec = importlib.util.spec_from_file_location(
    "native_linux_package_uninstall_test",
    _module_path,
)
native_linux_package_uninstall_test = importlib.util.module_from_spec(_spec)
sys.modules["native_linux_package_uninstall_test"] = native_linux_package_uninstall_test
_spec.loader.exec_module(native_linux_package_uninstall_test)


@contextlib.contextmanager
def _suppress_script_output():
    import builtins

    orig = builtins.print
    try:
        builtins.print = lambda *args, **kwargs: None
        yield
    finally:
        builtins.print = orig


class ArgvFromCiEnvTest(unittest.TestCase):
    """Tests for :func:`_argv_from_ci_env` env-to-argv mapping."""

    def test_builds_argv_from_env(self):
        env = {
            "OS_PROFILE": "ubuntu2404",
            "INSTALL_PREFIX": "/opt/rocm/core",
            "GFX_ARCH": "gfx94x",
            "NATIVE_LINUX_INSTALL_ROCM_VERSION": "7.13",
            "BUILD_VARIANT": "asan",
        }
        with patch.dict(os.environ, env, clear=False):
            argv = native_linux_package_uninstall_test._argv_from_ci_env()
        self.assertIsNotNone(argv)
        self.assertEqual(argv[argv.index("--os-profile") + 1], "ubuntu2404")
        self.assertEqual(argv[argv.index("--install-prefix") + 1], "/opt/rocm/core")
        self.assertIn("--gfx-arch", argv)
        self.assertIn("--rocm-version", argv)
        self.assertIn("--build-variant", argv)

    def test_returns_none_when_required_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(native_linux_package_uninstall_test._argv_from_ci_env())


class ListInstalledRocmPackagesTest(unittest.TestCase):
    """Tests for ``list_installed_rocm_packages()``."""

    @patch("native_linux_package_uninstall_test.subprocess.run")
    def test_deb_parses_installed_package_names(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "Desired=Unknown/Install/Remove/Purge/Hold\n"
                "ii  amdrocm  1.0  amd64  ROCm metapackage\n"
                "ii  libc6  2.35  amd64  GNU C Library\n"
                "ii  rocm-dev  1.0  amd64  ROCm dev\n"
            ),
        )
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="ubuntu2404",
        )
        names = t.list_installed_rocm_packages()
        self.assertEqual(names, ["amdrocm", "rocm-dev"])

    @patch("native_linux_package_uninstall_test.subprocess.run")
    def test_rpm_parses_installed_package_names(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="amdrocm-7.13-1.x86_64\nkernel-5.14-1.x86_64\n",
        )
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="rhel8",
        )
        names = t.list_installed_rocm_packages()
        self.assertEqual(names, ["amdrocm-7.13-1.x86_64"])

    @patch("native_linux_package_uninstall_test.subprocess.run")
    def test_returns_empty_list_on_query_failure(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "dpkg")
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="ubuntu2404",
        )
        self.assertEqual(t.list_installed_rocm_packages(), [])


class UninstallDebPackagesTest(unittest.TestCase):
    """Tests for ``uninstall_packages()`` on deb (apt remove + autoremove)."""

    @patch("native_linux_package_uninstall_test.run_streaming")
    def test_remove_and_autoremove_in_reverse_order(self, mock_streaming):
        mock_streaming.return_value = 0
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="ubuntu2404",
            gfx_arch=["gfx94x", "gfx1100"],
            rocm_version="7.13",
        )
        with _suppress_script_output():
            self.assertTrue(t.uninstall_packages())
        self.assertEqual(mock_streaming.call_count, 2)
        remove_cmd = mock_streaming.call_args_list[0][0][0]
        autoremove_cmd = mock_streaming.call_args_list[1][0][0]
        self.assertEqual(remove_cmd[:4], ["sudo", "apt", "remove", "-y"])
        self.assertEqual(
            remove_cmd[4:],
            [
                "amdrocm-core-sdk7.13-gfx1100",
                "amdrocm7.13-gfx1100",
                "amdrocm-core-sdk7.13-gfx94x",
                "amdrocm7.13-gfx94x",
            ],
        )
        self.assertEqual(autoremove_cmd, ["sudo", "apt", "autoremove", "-y"])

    @patch("native_linux_package_uninstall_test.run_streaming")
    def test_returns_false_when_remove_fails(self, mock_streaming):
        mock_streaming.return_value = 1
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="ubuntu2404",
            gfx_arch="gfx94x",
        )
        with _suppress_script_output():
            self.assertFalse(t.uninstall_packages())


class UninstallRpmPackagesTest(unittest.TestCase):
    """Tests for ``uninstall_packages()`` on rpm (dnf / zypper --clean-deps)."""

    @patch("native_linux_package_uninstall_test.run_streaming")
    def test_dnf_remove_for_rhel(self, mock_streaming):
        mock_streaming.return_value = 0
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="rhel8",
            gfx_arch="gfx94x",
        )
        with _suppress_script_output():
            self.assertTrue(t.uninstall_packages())
        cmd = mock_streaming.call_args[0][0]
        self.assertEqual(cmd[:3], ["dnf", "remove", "-y"])
        self.assertIn("amdrocm-core-sdk", cmd)
        self.assertIn("amdrocm", cmd)

    @patch("native_linux_package_uninstall_test.run_streaming")
    def test_zypper_remove_for_sles(self, mock_streaming):
        """SLES must pass --clean-deps so dependency packages are removed."""
        mock_streaming.return_value = 0
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="sles16",
            gfx_arch="gfx94x",
        )
        with _suppress_script_output():
            self.assertTrue(t.uninstall_packages())
        cmd = mock_streaming.call_args[0][0]
        self.assertEqual(
            cmd[:5],
            ["zypper", "--non-interactive", "remove", "-y", "--clean-deps"],
        )
        self.assertEqual(mock_streaming.call_count, 1)


class RunUninstallVerificationTest(unittest.TestCase):
    """Tests for ``run_uninstall_verification()``."""

    @patch.object(
        native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest,
        "list_installed_rocm_packages",
        return_value=[],
    )
    def test_passes_when_no_packages_remain(self, mock_list):
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="ubuntu2404",
            install_prefix="/nonexistent/prefix",
        )
        with _suppress_script_output():
            self.assertTrue(t.run_uninstall_verification())

    @patch.object(
        native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest,
        "list_installed_rocm_packages",
        return_value=["amdrocm"],
    )
    def test_fails_when_packages_remain(self, mock_list):
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="ubuntu2404",
        )
        with _suppress_script_output():
            self.assertFalse(t.run_uninstall_verification())


class RunUninstallAndVerifyTest(unittest.TestCase):
    """Tests for ``run_uninstall_and_verify()`` orchestration."""

    @patch.object(
        native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest,
        "run_uninstall_verification",
        return_value=True,
    )
    @patch.object(
        native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest,
        "uninstall_packages",
        return_value=True,
    )
    @patch.object(
        native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest,
        "list_installed_rocm_packages",
        return_value=["amdrocm"],
    )
    def test_orchestrates_uninstall_and_verify(
        self, mock_list, mock_uninstall, mock_verify
    ):
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="ubuntu2404",
        )
        with _suppress_script_output():
            self.assertTrue(t.run_uninstall_and_verify())
        mock_list.assert_called()
        mock_uninstall.assert_called_once()
        mock_verify.assert_called_once()

    @patch.object(
        native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest,
        "run_uninstall_verification",
    )
    @patch.object(
        native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest,
        "uninstall_packages",
        return_value=False,
    )
    @patch.object(
        native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest,
        "list_installed_rocm_packages",
        return_value=[],
    )
    def test_returns_false_when_uninstall_fails(
        self, mock_list, mock_uninstall, mock_verify
    ):
        t = native_linux_package_uninstall_test.NativeLinuxPackageUninstallTest(
            os_profile="ubuntu2404",
        )
        with _suppress_script_output():
            self.assertFalse(t.run_uninstall_and_verify())
        mock_verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
