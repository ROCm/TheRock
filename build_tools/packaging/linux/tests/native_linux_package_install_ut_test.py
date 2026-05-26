# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Unit test coverage for native_linux_package_install_test.py:
#   All testable behaviour is covered with unit tests (pure logic or mocked I/O/subprocess).
#   Integration-only (real apt/rpm/zypper, network, root): main() execution path after validation.

import contextlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Load the module: look in same dir as this file, then parent (covers linux/ or linux/tests/ layout).
_this_file = Path(__file__).resolve()
_search_dirs = [_this_file.parent, _this_file.parent.parent]
_module_path = None
for _d in _search_dirs:
    _candidate = _d / "native_linux_package_install_test.py"
    if _candidate.is_file():
        _module_path = _candidate
        break
if _module_path is None:
    _checked = ", ".join(str(d) for d in _search_dirs)
    raise FileNotFoundError(
        f"native_linux_package_install_test.py not found in: {_checked}"
    )
_spec = importlib.util.spec_from_file_location(
    "native_linux_package_install_test",
    _module_path,
)
native_linux_package_install_test = importlib.util.module_from_spec(_spec)
sys.modules["native_linux_package_install_test"] = native_linux_package_install_test
_spec.loader.exec_module(native_linux_package_install_test)


def _noop_print(*args, **kwargs):
    """No-op replacement for print to suppress script output during tests."""


@contextlib.contextmanager
def _suppress_script_output():
    """Temporarily replace builtins.print with a no-op so script output does not appear.

    The script is loaded via importlib and may resolve print from builtins. Patching
    builtins.print ensures all print() calls (including from the script) are
    suppressed during the with block.
    """
    import builtins

    orig = builtins.print
    try:
        builtins.print = _noop_print
        yield
    finally:
        builtins.print = orig


class EnvHelperTest(unittest.TestCase):
    """Tests for _env()."""

    def test_env_returns_value_when_set(self):
        # Test that _env returns the environment variable value when it is set.
        with patch.dict(os.environ, {"ROCM_TEST_KEY": "custom"}, clear=False):
            self.assertEqual(
                native_linux_package_install_test._env("ROCM_TEST_KEY", "default"),
                "custom",
            )

    def test_env_returns_default_when_unset(self):
        # Test that _env returns the default when the environment variable is not set.
        with patch.dict(os.environ, {}, clear=False):
            if "ROCM_TEST_KEY" in os.environ:
                del os.environ["ROCM_TEST_KEY"]
            self.assertEqual(
                native_linux_package_install_test._env("ROCM_TEST_KEY", "rocm-default"),
                "rocm-default",
            )

    def test_env_returns_default_when_empty_string(self):
        # Test that _env returns the default when the variable is set to empty string.
        with patch.dict(os.environ, {"ROCM_TEST_KEY": ""}, clear=False):
            self.assertEqual(
                native_linux_package_install_test._env("ROCM_TEST_KEY", "default"),
                "default",
            )

    def test_env_strips_whitespace(self):
        # Test that _env strips leading and trailing whitespace from the value.
        with patch.dict(os.environ, {"ROCM_TEST_KEY": "  value  "}, clear=False):
            self.assertEqual(
                native_linux_package_install_test._env("ROCM_TEST_KEY", "default"),
                "value",
            )


class DerivePackageTypeTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest._derive_package_type()."""

    def test_ubuntu_returns_deb(self):
        # Test that Ubuntu OS profiles (e.g. ubuntu2404, Ubuntu2204) derive package type "deb".
        self.assertEqual(
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "ubuntu2404"
            ),
            "deb",
        )
        self.assertEqual(
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "Ubuntu2204"
            ),
            "deb",
        )

    def test_debian_returns_deb(self):
        # Test that Debian OS profiles derive package type "deb".
        self.assertEqual(
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "debian12"
            ),
            "deb",
        )

    def test_rhel_returns_rpm(self):
        # Test that RHEL OS profiles derive package type "rpm".
        self.assertEqual(
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "rhel8"
            ),
            "rpm",
        )

    def test_sles_returns_rpm(self):
        # Test that SLES OS profiles (sles15, sles16) derive package type "rpm".
        self.assertEqual(
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "sles16"
            ),
            "rpm",
        )
        self.assertEqual(
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "sles15"
            ),
            "rpm",
        )

    def test_almalinux_returns_rpm(self):
        # Test that AlmaLinux OS profiles derive package type "rpm".
        self.assertEqual(
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "almalinux9"
            ),
            "rpm",
        )

    def test_centos_returns_rpm(self):
        # Test that CentOS OS profiles derive package type "rpm".
        self.assertEqual(
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "centos7"
            ),
            "rpm",
        )

    def test_azl_returns_rpm(self):
        # Test that AZL (Azure Linux) OS profiles derive package type "rpm".
        self.assertEqual(
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "azl3"
            ),
            "rpm",
        )

    def test_unknown_profile_raises_value_error(self):
        # Test that an unsupported OS profile raises ValueError with a descriptive message.
        with self.assertRaises(ValueError) as ctx:
            native_linux_package_install_test.NativeLinuxPackageInstallTest._derive_package_type(
                "unknown"
            )
        self.assertIn("Unable to derive package type", str(ctx.exception))
        self.assertIn("unknown", str(ctx.exception))


class IsSlesTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest._is_sles()."""

    def test_sles_profile_returns_true(self):
        # Test that _is_sles() returns True for SLES profiles (sles16, SLES15).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="sles16",
        )
        self.assertTrue(t._is_sles())
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="SLES15",
        )
        self.assertTrue(t._is_sles())

    def test_non_sles_profile_returns_false(self):
        # Test that _is_sles() returns False for non-SLES profiles (ubuntu, rhel).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        self.assertFalse(t._is_sles())
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
        )
        self.assertFalse(t._is_sles())


class NativeLinuxPackageInstallTestInitTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest __init__ and derived attributes."""

    def test_default_gfx_arch_and_package_names(self):
        # Test that when gfx_arch is omitted, default is gfx94x and package_names are correct.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        self.assertEqual(t.gfx_arch, "gfx94x")
        self.assertEqual(
            t.package_names,
            ["amdrocm-gfx94x", "amdrocm-core-sdk-gfx94x"],
        )

    def test_gfx_arch_string_normalized_to_list(self):
        # Test that a single gfx_arch string is used and package_names include that arch.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
            gfx_arch="gfx110x",
        )
        self.assertEqual(t.gfx_arch, "gfx110x")
        self.assertEqual(
            t.package_names,
            ["amdrocm-gfx110x", "amdrocm-core-sdk-gfx110x"],
        )

    def test_gfx_arch_list_uses_first_element(self):
        # Test that when gfx_arch is a list, only the first element is used for package names.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch=["gfx1151", "gfx94x"],
        )
        self.assertEqual(t.gfx_arch, "gfx1151")
        self.assertEqual(
            t.package_names,
            ["amdrocm-gfx1151", "amdrocm-core-sdk-gfx1151"],
        )

    def test_gfx_arch_empty_string_falls_back_to_default(self):
        # Test that empty gfx_arch string falls back to default gfx94x.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="",
        )
        self.assertEqual(t.gfx_arch, "gfx94x")

    def test_os_profile_and_release_type_normalized_lower(self):
        # Test that os_profile, release_type, and repo_url (trailing slash) are normalized.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com/",
            os_profile="Ubuntu2404",
            release_type="NIGHTLY",
        )
        self.assertEqual(t.os_profile, "ubuntu2404")
        self.assertEqual(t.release_type, "nightly")
        self.assertEqual(t.repo_url, "https://example.com")

    def test_install_prefix_default(self):
        # Test that install_prefix is None when not provided.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        self.assertIsNone(t.install_prefix)

    def test_install_prefix_custom(self):
        # Test that a provided install_prefix is stored as given.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            install_prefix="/opt/rocm/core",
        )
        self.assertEqual(t.install_prefix, "/opt/rocm/core")


class RunSimulateInstallTestTest(unittest.TestCase):
    """Tests for run_simulate_install_test()."""

    def test_not_a_directory_returns_false(self):
        # Test that run_simulate_install_test returns False when path is a file, not a directory.
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            self.assertFalse(
                native_linux_package_install_test.run_simulate_install_test("deb", path)
            )
        finally:
            os.unlink(path)

    def test_nonexistent_path_returns_false(self):
        # Test that run_simulate_install_test returns False when path does not exist.
        self.assertFalse(
            native_linux_package_install_test.run_simulate_install_test(
                "deb", "/nonexistent/dir/path"
            )
        )

    def test_deb_empty_directory_returns_false(self):
        # Test that run_simulate_install_test returns False for deb when directory has no .deb files.
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(
                native_linux_package_install_test.run_simulate_install_test("deb", d)
            )

    def test_rpm_empty_directory_returns_false(self):
        # Test that run_simulate_install_test returns False for rpm when directory has no .rpm files.
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(
                native_linux_package_install_test.run_simulate_install_test("rpm", d)
            )

    def test_unsupported_pkg_type_returns_false(self):
        # Test that run_simulate_install_test returns False for unsupported pkg_type (e.g. tgz).
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(
                native_linux_package_install_test.run_simulate_install_test("tgz", d)
            )

    @patch("native_linux_package_install_test.subprocess.run")
    def test_deb_with_files_success_when_subprocess_succeeds(self, mock_run):
        # Test that run_simulate_install_test returns True for deb when dir has .deb and apt succeeds.
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "fake.deb").write_text("")
            result = native_linux_package_install_test.run_simulate_install_test(
                "deb", d
            )
            self.assertTrue(result)
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            self.assertEqual(call_args[0], "apt")
            self.assertEqual(call_args[1], "install")
            self.assertEqual(call_args[2], "--simulate")

    @patch("native_linux_package_install_test.subprocess.run")
    def test_rpm_with_files_success_when_subprocess_succeeds(self, mock_run):
        # Test that run_simulate_install_test returns True for rpm when dir has .rpm and rpm succeeds.
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "fake.rpm").write_text("")
            result = native_linux_package_install_test.run_simulate_install_test(
                "rpm", d
            )
            self.assertTrue(result)
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            self.assertEqual(call_args[0], "rpm")
            self.assertIn("--test", call_args)
            self.assertIn("--nodeps", call_args)

    @patch("native_linux_package_install_test.subprocess.run")
    def test_deb_subprocess_failure_returns_false(self, mock_run):
        # We mock subprocess.run to raise CalledProcessError (as if "apt install --simulate"
        # failed). With a temp dir containing a .deb, the code runs apt; we assert that
        # run_simulate_install_test returns False when that subprocess call fails.
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "apt")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "fake.deb").write_text("")
            self.assertFalse(
                native_linux_package_install_test.run_simulate_install_test("deb", d)
            )

    @patch("native_linux_package_install_test.subprocess.run")
    def test_deb_command_not_found_returns_false(self, mock_run):
        # Test that run_simulate_install_test returns False when the apt command is not found.
        mock_run.side_effect = FileNotFoundError("apt")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "fake.deb").write_text("")
            self.assertFalse(
                native_linux_package_install_test.run_simulate_install_test("deb", d)
            )


class MainValidationTest(unittest.TestCase):
    """Tests for main() CLI validation (required args per --test-type)."""

    def test_simulate_requires_packages_dir(self):
        # Test that main() exits with error when --test-type simulate but --packages-dir is missing.
        with patch("sys.argv", ["prog", "--test-type", "simulate"]):
            with self.assertRaises(SystemExit) as cm:
                native_linux_package_install_test.main()
            self.assertEqual(cm.exception.code, 2)

    def test_sanity_requires_os_profile(self):
        # Test that main() exits with error when --test-type sanity but --os-profile is missing.
        with patch(
            "sys.argv",
            [
                "prog",
                "--test-type",
                "sanity",
                "--repo-url",
                "https://x.com",
                "--gfx-arch",
                "gfx94x",
            ],
        ):
            with self.assertRaises(SystemExit) as cm:
                native_linux_package_install_test.main()
            self.assertEqual(cm.exception.code, 2)

    def test_sanity_requires_repo_url(self):
        # Test that main() exits with error when --test-type sanity but --repo-url is missing.
        with patch(
            "sys.argv",
            [
                "prog",
                "--test-type",
                "sanity",
                "--os-profile",
                "ubuntu2404",
                "--gfx-arch",
                "gfx94x",
            ],
        ):
            with self.assertRaises(SystemExit) as cm:
                native_linux_package_install_test.main()
            self.assertEqual(cm.exception.code, 2)

    def test_sanity_requires_gfx_arch(self):
        # Test that main() exits with error when --test-type sanity but --gfx-arch is missing.
        with patch(
            "sys.argv",
            [
                "prog",
                "--test-type",
                "sanity",
                "--os-profile",
                "ubuntu2404",
                "--repo-url",
                "https://x.com",
            ],
        ):
            with self.assertRaises(SystemExit) as cm:
                native_linux_package_install_test.main()
            self.assertEqual(cm.exception.code, 2)


class RunBasicVerificationTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.run_basic_verification()."""

    def test_returns_false_when_install_prefix_does_not_exist(self):
        # Test that run_basic_verification returns False when install_prefix path does not exist.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            install_prefix="/nonexistent/install/path",
        )
        self.assertFalse(t.run_basic_verification())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_true_when_enough_components_found(self, mock_run):
        # Test that run_basic_verification returns True when install_prefix exists and at least
        # VERIFY_MIN_COMPONENTS key components exist; subprocess (dpkg/rpm, rocminfo) is mocked.
        mock_run.return_value = MagicMock(returncode=0, stdout="ii rocm-pkg 1.0\n")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bin").mkdir()
            (Path(d) / "lib").mkdir()
            (Path(d) / "bin" / "rocminfo").write_text("")
            (Path(d) / "bin" / "hipcc").write_text("")
            t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
                repo_url="https://example.com",
                os_profile="ubuntu2404",
                install_prefix=d,
            )
            self.assertTrue(t.run_basic_verification())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_false_when_insufficient_components(self, mock_run):
        # Test that run_basic_verification returns False when fewer than VERIFY_MIN_COMPONENTS exist.
        mock_run.return_value = MagicMock(returncode=0, stdout="ii rocm 1.0\n")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bin").mkdir()
            (Path(d) / "bin" / "rocminfo").write_text("")  # only 1 component
            t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
                repo_url="https://example.com",
                os_profile="ubuntu2404",
                install_prefix=d,
            )
            self.assertFalse(t.run_basic_verification())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_handles_called_process_error_when_querying_packages(self, mock_run):
        # Test that run_basic_verification handles CalledProcessError when querying packages (continues, then passes if enough components).
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "dpkg")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bin").mkdir()
            (Path(d) / "lib").mkdir()
            (Path(d) / "bin" / "rocminfo").write_text("")
            (Path(d) / "bin" / "hipcc").write_text("")
            t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
                repo_url="https://example.com",
                os_profile="ubuntu2404",
                install_prefix=d,
            )
            self.assertTrue(t.run_basic_verification())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_handles_rocminfo_timeout(self, mock_run):
        # Test that run_basic_verification handles rocminfo TimeoutExpired (warns but still passes if enough components).
        import subprocess

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="ii rocm 1.0\n"),
            subprocess.TimeoutExpired("rocminfo", 30),
        ]
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bin").mkdir()
            (Path(d) / "lib").mkdir()
            (Path(d) / "bin" / "rocminfo").write_text("")
            (Path(d) / "bin" / "hipcc").write_text("")
            t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
                repo_url="https://example.com",
                os_profile="ubuntu2404",
                install_prefix=d,
            )
            self.assertTrue(t.run_basic_verification())


class SetupGpgKeyTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.setup_gpg_key()."""

    def test_returns_true_when_no_gpg_key_url(self):
        # Test that setup_gpg_key returns True when gpg_key_url is not set (no-op).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
        )
        self.assertTrue(t.setup_gpg_key())

    def test_returns_true_for_rpm_with_gpg_key_url(self):
        # Test that for RPM (including SLES), setup_gpg_key returns True without downloading (handled in repo file).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
            gpg_key_url="https://example.com/rocm.gpg",
        )
        self.assertTrue(t.setup_gpg_key())

    @patch("native_linux_package_install_test.os.chmod")
    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_true_for_deb_when_mock_succeeds(self, mock_run, mock_chmod):
        # Test that for DEB with gpg_key_url, setup_gpg_key returns True when mkdir and pipeline succeed.
        mock_run.return_value = MagicMock(returncode=0)
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gpg_key_url="https://example.com/rocm.gpg",
        )
        self.assertTrue(t.setup_gpg_key())
        self.assertEqual(mock_run.call_count, 2)  # mkdir, then pipeline

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_false_for_deb_when_subprocess_fails(self, mock_run):
        # Test that setup_gpg_key returns False when subprocess raises CalledProcessError.
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "mkdir", stderr=b"permission denied"
        )
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gpg_key_url="https://example.com/rocm.gpg",
        )
        self.assertFalse(t.setup_gpg_key())


class SetupDebRepositoryTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.setup_deb_repository()."""

    @patch("native_linux_package_install_test._run_streaming")
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_true_when_apt_update_succeeds_no_gpg(
        self, mock_file, mock_streaming
    ):
        # Test that setup_deb_repository writes repo entry (trusted=yes) and returns True when apt update returns 0.
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
        )
        self.assertTrue(t.setup_deb_repository())
        mock_file().write.assert_called_once()
        written = mock_file().write.call_args[0][0]
        self.assertIn("trusted=yes", written)
        self.assertIn("https://repo.example.com", written)

    @patch("native_linux_package_install_test._run_streaming")
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "setup_gpg_key",
        return_value=True,
    )
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_true_with_gpg_when_apt_update_succeeds(
        self, mock_file, mock_gpg, mock_streaming
    ):
        # Test that with gpg_key_url, setup_gpg_key is called and repo entry uses signed-by.
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url="https://example.com/rocm.gpg",
        )
        self.assertTrue(t.setup_deb_repository())
        mock_gpg.assert_called_once()
        written = mock_file().write.call_args[0][0]
        self.assertIn("signed-by", written)

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "setup_gpg_key",
        return_value=False,
    )
    def test_returns_false_when_setup_gpg_key_fails(self, mock_gpg):
        # Test that setup_deb_repository returns False when setup_gpg_key returns False.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url="https://example.com/rocm.gpg",
        )
        self.assertFalse(t.setup_deb_repository())

    @patch("native_linux_package_install_test._run_streaming")
    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_returns_false_when_open_raises(self, mock_file, mock_streaming):
        # Test that setup_deb_repository returns False when writing sources list raises OSError.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
        )
        self.assertFalse(t.setup_deb_repository())

    @patch("native_linux_package_install_test._run_streaming")
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_false_when_apt_update_fails(self, mock_file, mock_streaming):
        # Test that setup_deb_repository returns False when apt update returns non-zero.
        mock_streaming.return_value = 1
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
        )
        self.assertFalse(t.setup_deb_repository())

    @patch("native_linux_package_install_test._run_streaming")
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_false_when_apt_update_times_out(self, mock_file, mock_streaming):
        # Test that setup_deb_repository returns False when _run_streaming raises TimeoutExpired.
        import subprocess

        mock_streaming.side_effect = subprocess.TimeoutExpired("apt", 120)
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
        )
        self.assertFalse(t.setup_deb_repository())


class SetupSlesRepositoryTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest._setup_sles_repository()."""

    @patch("native_linux_package_install_test._run_streaming")
    @patch("native_linux_package_install_test.subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_true_when_refresh_succeeds(
        self, mock_file, mock_run, mock_streaming
    ):
        # Test that _setup_sles_repository writes repo file and returns True when zypper refresh returns 0.
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="sles16",
        )
        self.assertTrue(t._setup_sles_repository())
        written = mock_file().write.call_args[0][0]
        self.assertIn("baseurl=https://repo.example.com", written)
        self.assertIn("sles16", t.os_profile)


class SetupDnfRepositoryTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest._setup_dnf_repository()."""

    @patch("native_linux_package_install_test.subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    def test_returns_true_after_writing_repo_file(self, mock_file, mock_run):
        # Test that _setup_dnf_repository writes repo file and returns True (dnf clean may be mocked).
        mock_run.side_effect = None
        mock_run.return_value = MagicMock(returncode=0)
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="rhel8",
        )
        self.assertTrue(t._setup_dnf_repository())
        written = mock_file().write.call_args[0][0]
        self.assertIn("baseurl=https://repo.example.com", written)


class SetupRpmRepositoryTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.setup_rpm_repository()."""

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "_setup_dnf_repository",
        return_value=True,
    )
    def test_calls_setup_dnf_for_rhel(self, mock_dnf):
        # Test that for non-SLES RPM (e.g. rhel8), setup_rpm_repository calls _setup_dnf_repository.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
        )
        self.assertTrue(t.setup_rpm_repository())
        mock_dnf.assert_called_once()

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "_setup_sles_repository",
        return_value=True,
    )
    def test_calls_setup_sles_for_sles(self, mock_sles):
        # Test that for SLES, setup_rpm_repository calls _setup_sles_repository.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="sles16",
        )
        self.assertTrue(t.setup_rpm_repository())
        mock_sles.assert_called_once()


class InstallDebPackagesTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.install_deb_packages()."""

    @patch("native_linux_package_install_test._run_streaming")
    def test_returns_true_when_apt_install_succeeds(self, mock_streaming):
        # Test that install_deb_packages returns True when _run_streaming (apt install) returns 0.
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx94x",
        )
        self.assertTrue(t.install_deb_packages())
        call_args = mock_streaming.call_args[0][0]
        self.assertEqual(call_args[0], "apt")
        self.assertIn("amdrocm-gfx94x", call_args)

    @patch("native_linux_package_install_test._run_streaming")
    def test_returns_false_when_apt_install_fails(self, mock_streaming):
        # Test that install_deb_packages returns False when _run_streaming returns non-zero.
        mock_streaming.return_value = 1
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx94x",
        )
        self.assertFalse(t.install_deb_packages())

    @patch("native_linux_package_install_test._run_streaming")
    def test_returns_false_when_apt_install_times_out(self, mock_streaming):
        # Test that install_deb_packages returns False when _run_streaming raises TimeoutExpired.
        import subprocess

        mock_streaming.side_effect = subprocess.TimeoutExpired("apt", 1800)
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx94x",
        )
        self.assertFalse(t.install_deb_packages())


class InstallRpmPackagesTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.install_rpm_packages()."""

    @patch("native_linux_package_install_test._run_streaming")
    def test_returns_true_when_dnf_install_succeeds(self, mock_streaming):
        # Test that install_rpm_packages returns True for RHEL when _run_streaming (dnf install) returns 0.
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
            gfx_arch="gfx94x",
        )
        self.assertTrue(t.install_rpm_packages())
        call_args = mock_streaming.call_args[0][0]
        self.assertEqual(call_args[0], "dnf")

    @patch("native_linux_package_install_test._run_streaming")
    def test_returns_true_when_zypper_install_succeeds(self, mock_streaming):
        # Test that install_rpm_packages returns True for SLES when _run_streaming (zypper install) returns 0.
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="sles16",
            gfx_arch="gfx94x",
        )
        self.assertTrue(t.install_rpm_packages())
        call_args = mock_streaming.call_args[0][0]
        self.assertEqual(call_args[0], "zypper")


class RunRepoSetupAndInstallTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.run_repo_setup_and_install()."""

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "install_deb_packages",
        return_value=True,
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "setup_deb_repository",
        return_value=True,
    )
    def test_returns_true_for_deb_when_setup_and_install_succeed(
        self, mock_setup, mock_install
    ):
        # Test that run_repo_setup_and_install returns True when setup and install both succeed (deb).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        self.assertTrue(t.run_repo_setup_and_install())
        mock_setup.assert_called_once()
        mock_install.assert_called_once()

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "setup_deb_repository",
        return_value=False,
    )
    def test_returns_false_when_setup_deb_fails(self, mock_setup):
        # Test that run_repo_setup_and_install returns False when setup_deb_repository returns False.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        self.assertFalse(t.run_repo_setup_and_install())
        mock_setup.assert_called_once()

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "install_rpm_packages",
        return_value=True,
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "setup_rpm_repository",
        return_value=True,
    )
    def test_returns_true_for_rpm_when_setup_and_install_succeed(
        self, mock_setup, mock_install
    ):
        # Test that run_repo_setup_and_install returns True when setup and install both succeed (rpm).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
        )
        self.assertTrue(t.run_repo_setup_and_install())
        mock_setup.assert_called_once()
        mock_install.assert_called_once()


class RunFullVerificationTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.run_full_verification()."""

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "test_rdhc",
        return_value=True,
    )
    def test_returns_test_rdhc_result(self, mock_rdhc):
        # Test that run_full_verification returns whatever test_rdhc returns.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            install_prefix="/opt/rocm/core",
        )
        self.assertTrue(t.run_full_verification())
        mock_rdhc.assert_called_once()


class TestRdhcTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.test_rdhc()."""

    def test_returns_false_when_rdhc_script_missing(self):
        # Test that test_rdhc returns False when install_prefix path has no rdhc.py at RDHC_REL_PATH.
        with tempfile.TemporaryDirectory() as d:
            t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
                repo_url="https://example.com",
                os_profile="ubuntu2404",
                install_prefix=d,
            )
            self.assertFalse(t.test_rdhc())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_true_when_script_exists_and_run_succeeds(self, mock_run):
        # Test that test_rdhc returns True when rdhc.py exists and subprocess run succeeds.
        mock_run.return_value = MagicMock(returncode=0, stdout="ok")
        with tempfile.TemporaryDirectory() as d:
            libexec = Path(d) / "libexec" / "rocm-core"
            libexec.mkdir(parents=True)
            (libexec / "rdhc.py").write_text("")
            t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
                repo_url="https://example.com",
                os_profile="ubuntu2404",
                install_prefix=d,
            )
            self.assertTrue(t.test_rdhc())
            call_args = mock_run.call_args[0][0]
            self.assertIn("rdhc.py", str(call_args[0]))
            self.assertIn("--rocm-install-prefix", call_args)

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_false_when_rdhc_times_out(self, mock_run):
        # Test that test_rdhc returns False when subprocess raises TimeoutExpired.
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("rdhc", 30)
        with tempfile.TemporaryDirectory() as d:
            libexec = Path(d) / "libexec" / "rocm-core"
            libexec.mkdir(parents=True)
            (libexec / "rdhc.py").write_text("")
            t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
                repo_url="https://example.com",
                os_profile="ubuntu2404",
                install_prefix=d,
            )
            self.assertFalse(t.test_rdhc())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_false_when_rdhc_fails(self, mock_run):
        # Test that test_rdhc returns False when subprocess raises CalledProcessError.
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "rdhc")
        with tempfile.TemporaryDirectory() as d:
            libexec = Path(d) / "libexec" / "rocm-core"
            libexec.mkdir(parents=True)
            (libexec / "rdhc.py").write_text("")
            t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
                repo_url="https://example.com",
                os_profile="ubuntu2404",
                install_prefix=d,
            )
            self.assertFalse(t.test_rdhc())


class RunStreamingTest(unittest.TestCase):
    """Tests for _run_streaming()."""

    @patch("native_linux_package_install_test.subprocess.Popen")
    def test_returns_process_exit_code(self, mock_popen):
        # Test that _run_streaming returns the process exit code when process exits normally.
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["line1\n", "line2\n"])
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc
        code = native_linux_package_install_test._run_streaming(["echo", "hi"], 30)
        self.assertEqual(code, 0)
        mock_proc.wait.assert_called_once()
        self.assertEqual(mock_proc.wait.call_args[1]["timeout"], 30)

    @patch("native_linux_package_install_test.subprocess.Popen")
    def test_kills_process_on_timeout(self, mock_popen):
        # Test that _run_streaming kills the process when wait() raises TimeoutExpired.
        import subprocess as sp

        mock_proc = MagicMock()
        mock_proc.stdout = iter(["line1\n"])
        mock_proc.wait.side_effect = sp.TimeoutExpired("cmd", 30)
        mock_popen.return_value = mock_proc
        with self.assertRaises(sp.TimeoutExpired):
            native_linux_package_install_test._run_streaming(["slow-cmd"], 30)
        mock_proc.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
