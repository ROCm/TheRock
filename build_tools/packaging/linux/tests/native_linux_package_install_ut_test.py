# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Unit test coverage for native_linux_package_install_test.py:
#   All testable behaviour is covered with unit tests (pure logic or mocked I/O/subprocess),
#   including optional Step 4 uninstall (--with-uninstall / RUN_UNINSTALL).
#   Integration-only (real apt/rpm/zypper, network, root): main() and pytest CI entry paths.

import contextlib
import importlib.util
import io
import os
import stat
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

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
_packaging_utils_path = _module_path.parent / "packaging_utils.py"
_pu_spec = importlib.util.spec_from_file_location(
    "packaging_utils", _packaging_utils_path
)
packaging_utils = importlib.util.module_from_spec(_pu_spec)
_pu_spec.loader.exec_module(packaging_utils)
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


class NormalizeTestTypeTest(unittest.TestCase):
    """Tests for _normalize_test_type()."""

    def test_empty_quick_and_standard_map_to_sanity(self):
        for test_type in ("", None, "quick", "standard"):
            with self.subTest(test_type=test_type):
                self.assertEqual(
                    native_linux_package_install_test._normalize_test_type(test_type),
                    "sanity",
                )

    def test_comprehensive_and_full_map_to_full(self):
        for test_type in ("comprehensive", "full"):
            with self.subTest(test_type=test_type):
                self.assertEqual(
                    native_linux_package_install_test._normalize_test_type(test_type),
                    "full",
                )

    def test_native_modes_are_accepted(self):
        for test_type in ("install", "sanity", "full", "simulate"):
            with self.subTest(test_type=test_type):
                self.assertEqual(
                    native_linux_package_install_test._normalize_test_type(test_type),
                    test_type,
                )

    def test_strips_whitespace_and_lowercases(self):
        self.assertEqual(
            native_linux_package_install_test._normalize_test_type("  Quick  "),
            "sanity",
        )

    def test_invalid_test_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            native_linux_package_install_test._normalize_test_type("standrd")
        self.assertIn("Unsupported test_type", str(ctx.exception))


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

    def test_omitted_gfx_arch_uses_generic_package_names(self):
        # Test that when gfx_arch is omitted, generic amdrocm packages are used (no arch suffix).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        self.assertEqual(t.gfx_arch_list, [])
        self.assertIsNone(t.gfx_arch)
        self.assertEqual(
            t.package_names,
            ["amdrocm", "amdrocm-core-sdk"],
        )

    def test_gfx_arch_without_rocm_version_ignored_for_package_names(self):
        # gfx_arch is stored but not used in package names without rocm_version.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
            gfx_arch="gfx110x",
        )
        self.assertEqual(t.gfx_arch, "gfx110x")
        self.assertIsNone(t.rocm_version_major_minor)
        self.assertEqual(
            t.package_names,
            ["amdrocm", "amdrocm-core-sdk"],
        )

    def test_gfx_arch_with_rocm_version_uses_versioned_package_names(self):
        # Version in package name is major.minor only (7.13.1 -> amdrocm7.13-gfx1100).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx1100",
            rocm_version="7.13.1",
        )
        self.assertEqual(t.rocm_version_major_minor, "7.13")
        self.assertEqual(
            t.package_names,
            ["amdrocm7.13-gfx1100", "amdrocm-core-sdk7.13-gfx1100"],
        )

    def test_rocm_version_generic_uses_versioned_package_names(self):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            rocm_version="7.13.0",
        )
        self.assertEqual(t.rocm_version_major_minor, "7.13")
        self.assertEqual(
            t.package_names,
            ["amdrocm7.13", "amdrocm-core-sdk7.13"],
        )

    def test_major_minor_rocm_version_from_input(self):
        m = (
            native_linux_package_install_test.NativeLinuxPackageInstallTest._major_minor_rocm_version_from_input
        )
        self.assertIsNone(m(None))
        self.assertIsNone(m(""))
        self.assertEqual(m("7.13"), "7.13")
        self.assertEqual(m("7.13.1"), "7.13")
        self.assertEqual(m("v7.13.2"), "7.13")
        # Debian/RPM package version strings: major.minor only used in metapackage names.
        for version in (
            "7.14.0~20260520",
            "7.14.0~20260520-123456",
            "7.14.0~rc1",
            "7.14.0~rc1-123456",
        ):
            with self.subTest(version=version):
                self.assertEqual(m(version), "7.14")
        with self.assertRaises(ValueError):
            m("not-a-version")

    def test_gfx_arch_list_without_rocm_version_uses_generic_packages(self):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch=["gfx1151", "gfx94x"],
        )
        self.assertEqual(t.gfx_arch, "gfx1151")
        self.assertEqual(
            t.package_names,
            ["amdrocm", "amdrocm-core-sdk"],
        )

    def test_gfx_arch_list_with_rocm_version_multi_arch_package_names(self):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch=["gfx1151", "gfx94x"],
            rocm_version="7.13",
        )
        self.assertEqual(
            t.package_names,
            [
                "amdrocm7.13-gfx1151",
                "amdrocm-core-sdk7.13-gfx1151",
                "amdrocm7.13-gfx94x",
                "amdrocm-core-sdk7.13-gfx94x",
            ],
        )

    def test_gfx_arch_empty_string_uses_generic_packages(self):
        # Test that empty gfx_arch string yields generic package names (same as omitted).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="",
        )
        self.assertIsNone(t.gfx_arch)
        self.assertEqual(
            t.package_names,
            ["amdrocm", "amdrocm-core-sdk"],
        )

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

    def test_gfx_arch_comma_string_with_rocm_version_expands_packages(self):
        # Comma-separated arch in one string splits in normalization (same as CLI single token).
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx94x, GFX1100 ",
            rocm_version="7.13",
        )
        self.assertEqual(t.gfx_arch_list, ["gfx94x", "gfx1100"])
        self.assertEqual(
            t.package_names,
            [
                "amdrocm7.13-gfx94x",
                "amdrocm-core-sdk7.13-gfx94x",
                "amdrocm7.13-gfx1100",
                "amdrocm-core-sdk7.13-gfx1100",
            ],
        )

    def test_gfx_arch_semicolon_string_with_rocm_version_expands_packages(self):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx94x; GFX1100 ",
            rocm_version="7.13",
        )
        self.assertEqual(t.gfx_arch_list, ["gfx94x", "gfx1100"])
        self.assertEqual(
            t.package_names,
            [
                "amdrocm7.13-gfx94x",
                "amdrocm-core-sdk7.13-gfx94x",
                "amdrocm7.13-gfx1100",
                "amdrocm-core-sdk7.13-gfx1100",
            ],
        )


class NormalizeTargetListTest(unittest.TestCase):
    """Tests for normalize_target_list default behavior (preserve casing, no dedupe)."""

    def test_space_comma_and_semicolon_formats(self):
        n = packaging_utils.normalize_target_list
        self.assertEqual(
            n(["gfx94X-dcgpu", "gfx120X-all"]),
            ["gfx94X-dcgpu", "gfx120X-all"],
        )
        self.assertEqual(
            n(["gfx94X-dcgpu,gfx120X-all,gfx1151"]),
            ["gfx94X-dcgpu", "gfx120X-all", "gfx1151"],
        )
        self.assertEqual(
            n(["gfx94X-dcgpu;gfx120X-all;gfx1151"]),
            ["gfx94X-dcgpu", "gfx120X-all", "gfx1151"],
        )
        self.assertEqual(
            n(["gfx94X-dcgpu;gfx120X-all", "gfx1151"]),
            ["gfx94X-dcgpu", "gfx120X-all", "gfx1151"],
        )

    def test_preserves_casing_without_dedupe(self):
        self.assertEqual(
            packaging_utils.normalize_target_list(["gfx94X-dcgpu"]),
            ["gfx94X-dcgpu"],
        )


class NormalizedGfxArchsFromInputTest(unittest.TestCase):
    """Tests for packaging_utils.normalize_target_list (install-test options)."""

    def setUp(self):
        self.n = lambda value: packaging_utils.normalize_target_list(
            value, lowercase=True, dedupe=True
        )

    def test_none_and_blank_yield_empty(self):
        self.assertEqual(self.n(None), [])
        self.assertEqual(self.n(""), [])

    def test_list_with_commas_and_whitespace(self):
        self.assertEqual(self.n(["gfx94x,gfx1100 ", ""]), ["gfx94x", "gfx1100"])

    def test_semicolon_and_comma_mixed(self):
        self.assertEqual(
            self.n("gfx94x; GFX1100 ,gfx1200"),
            ["gfx94x", "gfx1100", "gfx1200"],
        )

    def test_semicolon_in_list_entries(self):
        self.assertEqual(self.n(["gfx94x;gfx1100"]), ["gfx94x", "gfx1100"])

    def test_dedupe_case_insensitive_order_preserved(self):
        self.assertEqual(
            self.n(["gfx94x", "GFX94X", "gfx1100"]),
            ["gfx94x", "gfx1100"],
        )


class ArgvFromCiEnvTest(unittest.TestCase):
    """Tests for _argv_from_ci_env() (workflow env → CLI argv)."""

    _base_sanity_env = {
        "TEST_TYPE": "sanity",
        "OS_PROFILE": "ubuntu2404",
        "REPO_URL": "https://repo.example.com/deb/",
        "RELEASE_TYPE": "nightly",
        "INSTALL_PREFIX": "/opt/rocm/core",
    }

    def test_returns_none_when_required_var_missing(self):
        with patch.dict(os.environ, {"OS_PROFILE": "ubuntu2404"}, clear=False):
            self.assertIsNone(native_linux_package_install_test._argv_from_ci_env())

    def test_multi_gfx_arch_whitespace_and_rocm_version_in_argv(self):
        env = {
            **self._base_sanity_env,
            "GFX_ARCH": "gfx94x gfx1100",
            "NATIVE_LINUX_INSTALL_ROCM_VERSION": "7.13.1",
        }
        with patch.dict(os.environ, env, clear=False):
            argv = native_linux_package_install_test._argv_from_ci_env()
        self.assertIsNotNone(argv)
        self.assertIn("--gfx-arch", argv)
        i = argv.index("--gfx-arch")
        self.assertEqual(argv[i + 1 : i + 3], ["gfx94x", "gfx1100"])
        self.assertIn("--rocm-version", argv)
        j = argv.index("--rocm-version")
        self.assertEqual(argv[j + 1], "7.13.1")

    def test_gfx_arch_comma_single_token_in_argv(self):
        env = {**self._base_sanity_env, "GFX_ARCH": "gfx94x,gfx1100"}
        with patch.dict(os.environ, env, clear=False):
            argv = native_linux_package_install_test._argv_from_ci_env()
        self.assertIsNotNone(argv)
        i = argv.index("--gfx-arch")
        self.assertEqual(argv[i + 1], "gfx94x,gfx1100")

    def test_gfx_arch_semicolon_splits_to_multiple_argv_tokens(self):
        env = {**self._base_sanity_env, "GFX_ARCH": "gfx94x; gfx1100"}
        with patch.dict(os.environ, env, clear=False):
            argv = native_linux_package_install_test._argv_from_ci_env()
        self.assertIsNotNone(argv)
        i = argv.index("--gfx-arch")
        self.assertEqual(argv[i + 1 : i + 3], ["gfx94x", "gfx1100"])

    def test_gfx_arch_semicolon_single_token_in_argv(self):
        env = {**self._base_sanity_env, "GFX_ARCH": "gfx94x;gfx1100"}
        with patch.dict(os.environ, env, clear=False):
            argv = native_linux_package_install_test._argv_from_ci_env()
        self.assertIsNotNone(argv)
        i = argv.index("--gfx-arch")
        self.assertEqual(argv[i + 1 : i + 3], ["gfx94x", "gfx1100"])
        env = {
            **self._base_sanity_env,
            "GFX_ARCH": "gfx94x gfx1100",
            "NATIVE_LINUX_INSTALL_ROCM_VERSION": "7.13",
        }
        with patch.dict(os.environ, env, clear=False):
            argv = native_linux_package_install_test._argv_from_ci_env()
        args = native_linux_package_install_test.parse_cli_arguments(
            argv, raise_instead_of_exit=True
        )
        self.assertEqual(args.gfx_arch, ["gfx94x", "gfx1100"])
        self.assertEqual(args.rocm_version, "7.13")
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url=args.repo_url,
            os_profile=args.os_profile,
            release_type=args.release_type,
            install_prefix=args.install_prefix,
            gfx_arch=args.gfx_arch,
            rocm_version=args.rocm_version,
        )
        self.assertEqual(len(t.package_names), 4)


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
                "https://repo_url.com",
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

    def test_sanity_parse_without_gfx_arch(self):
        # Test that sanity/full CLI accepts omitting --gfx-arch (generic amdrocm packages).
        args = native_linux_package_install_test.parse_cli_arguments(
            [
                "--test-type",
                "sanity",
                "--os-profile",
                "ubuntu2404",
                "--repo-url",
                "https://repo_url.com",
            ],
            raise_instead_of_exit=True,
        )
        self.assertIsNone(args.gfx_arch)

    def test_invalid_rocm_version_rejected(self):
        with self.assertRaises(ValueError):
            native_linux_package_install_test.parse_cli_arguments(
                [
                    "--test-type",
                    "sanity",
                    "--os-profile",
                    "ubuntu2404",
                    "--repo-url",
                    "https://repo_url.com",
                    "--rocm-version",
                    "bogus",
                ],
                raise_instead_of_exit=True,
            )

    def test_install_requires_os_profile(self):
        # install uses the same required args as sanity (no verification step).
        with patch(
            "sys.argv",
            [
                "prog",
                "--test-type",
                "install",
                "--repo-url",
                "https://repo_url.com",
                "--gfx-arch",
                "gfx94x",
            ],
        ):
            with self.assertRaises(SystemExit) as cm:
                native_linux_package_install_test.main()
            self.assertEqual(cm.exception.code, 2)

    def test_parse_cli_maps_quick_to_sanity(self):
        args = native_linux_package_install_test.parse_cli_arguments(
            [
                "--test-type",
                "quick",
                "--os-profile",
                "ubuntu2404",
                "--repo-url",
                "https://repo_url.com",
                "--gfx-arch",
                "gfx94x",
            ],
            raise_instead_of_exit=True,
        )
        self.assertEqual(args.test_type, "sanity")

    def test_parse_cli_maps_comprehensive_to_full(self):
        args = native_linux_package_install_test.parse_cli_arguments(
            [
                "--test-type",
                "comprehensive",
                "--os-profile",
                "ubuntu2404",
                "--repo-url",
                "https://repo_url.com",
                "--gfx-arch",
                "gfx94x",
            ],
            raise_instead_of_exit=True,
        )
        self.assertEqual(args.test_type, "full")

    def test_parse_cli_rejects_invalid_test_type(self):
        with self.assertRaises(ValueError) as ctx:
            native_linux_package_install_test.parse_cli_arguments(
                ["--test-type", "standrd"],
                raise_instead_of_exit=True,
            )
        self.assertIn("Unsupported test_type", str(ctx.exception))


class ArgvFromCiEnvTest(unittest.TestCase):
    """Tests for _argv_from_ci_env() (CI workflow env → CLI argv)."""

    def test_builds_argv_for_install_test_type(self):
        env = {
            "TEST_TYPE": "install",
            "OS_PROFILE": "ubuntu2404",
            "REPO_URL": "https://example.com/deb",
            "GFX_ARCH": "gfx94x",
            "RELEASE_TYPE": "dev",
            "INSTALL_PREFIX": "/opt/rocm/core",
        }
        with patch.dict(os.environ, env, clear=False):
            argv = native_linux_package_install_test._argv_from_ci_env()
        self.assertIsNotNone(argv)
        self.assertIn("--test-type", argv)
        self.assertEqual(argv[argv.index("--test-type") + 1], "install")
        self.assertEqual(argv[argv.index("--os-profile") + 1], "ubuntu2404")
        self.assertEqual(argv[argv.index("--repo-url") + 1], "https://example.com/deb")

    def test_ci_env_passes_shared_test_type_to_parser(self):
        env = {
            "TEST_TYPE": "comprehensive",
            "OS_PROFILE": "ubuntu2404",
            "REPO_URL": "https://example.com/deb",
            "GFX_ARCH": "gfx94x",
            "RELEASE_TYPE": "dev",
            "INSTALL_PREFIX": "/opt/rocm/core",
        }
        with patch.dict(os.environ, env, clear=False):
            argv = native_linux_package_install_test._argv_from_ci_env()
        self.assertIsNotNone(argv)
        self.assertEqual(argv[argv.index("--test-type") + 1], "comprehensive")
        args = native_linux_package_install_test.parse_cli_arguments(
            argv,
            raise_instead_of_exit=True,
        )
        self.assertEqual(args.test_type, "full")

    def test_returns_none_when_required_env_missing(self):
        with patch.dict(os.environ, {"TEST_TYPE": "install"}, clear=True):
            self.assertIsNone(native_linux_package_install_test._argv_from_ci_env())

    def test_adds_with_uninstall_when_run_uninstall_set(self):
        env = {
            "TEST_TYPE": "sanity",
            "OS_PROFILE": "ubuntu2404",
            "REPO_URL": "https://example.com/deb",
            "RELEASE_TYPE": "dev",
            "INSTALL_PREFIX": "/opt/rocm/core",
            "RUN_UNINSTALL": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            argv = native_linux_package_install_test._argv_from_ci_env()
        self.assertIsNotNone(argv)
        self.assertIn("--with-uninstall", argv)

    def test_adds_with_uninstall_for_true_and_yes_values(self):
        base_env = {
            "TEST_TYPE": "sanity",
            "OS_PROFILE": "ubuntu2404",
            "REPO_URL": "https://example.com/deb",
            "RELEASE_TYPE": "dev",
            "INSTALL_PREFIX": "/opt/rocm/core",
        }
        for value in ("true", "YES"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {**base_env, "RUN_UNINSTALL": value}, clear=False):
                    argv = native_linux_package_install_test._argv_from_ci_env()
                self.assertIsNotNone(argv)
                self.assertIn("--with-uninstall", argv)

    def test_run_uninstall_false_values_do_not_add_flag(self):
        base_env = {
            "TEST_TYPE": "sanity",
            "OS_PROFILE": "ubuntu2404",
            "REPO_URL": "https://example.com/deb",
            "RELEASE_TYPE": "dev",
            "INSTALL_PREFIX": "/opt/rocm/core",
        }
        for value in ("0", "false", "NO"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {**base_env, "RUN_UNINSTALL": value}, clear=False):
                    argv = native_linux_package_install_test._argv_from_ci_env()
                self.assertIsNotNone(argv)
                self.assertNotIn("--with-uninstall", argv)

    def test_raises_for_invalid_run_uninstall_value(self):
        env = {
            "TEST_TYPE": "sanity",
            "OS_PROFILE": "ubuntu2404",
            "REPO_URL": "https://example.com/deb",
            "RELEASE_TYPE": "dev",
            "INSTALL_PREFIX": "/opt/rocm/core",
            "RUN_UNINSTALL": "maybe",
        }
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(ValueError, "Invalid RUN_UNINSTALL value"):
                native_linux_package_install_test._argv_from_ci_env()


class RunTestsTestTypeTest(unittest.TestCase):
    """Tests for run_tests() early exit paths for install."""

    def _base_args(self, test_type: str):
        from argparse import Namespace

        return Namespace(
            test_type=test_type,
            os_profile="ubuntu2404",
            repo_url="https://example.com",
            release_type="dev",
            install_prefix="/opt/rocm/core",
            gfx_arch=["gfx94x"],
            gpg_key_url=None,
            packages_dir=None,
            pkg_type=None,
            rocm_version=None,
            build_variant="",
            with_uninstall=False,
        )

    def _args_for_uninstall_run_tests(
        self, test_type: str, *, with_uninstall: bool = False
    ):
        """Build args for ``run_tests()`` uninstall integration tests."""
        args = self._base_args(test_type)
        args.rocm_version = None
        args.with_uninstall = with_uninstall
        return args

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_repo_setup_and_install",
        return_value=True,
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_basic_verification",
    )
    def test_install_skips_basic_verification(self, mock_basic, mock_repo_setup):
        args = self._base_args("install")
        with _suppress_script_output():
            rc = native_linux_package_install_test.run_tests(args)
        self.assertEqual(rc, 0)
        mock_repo_setup.assert_called_once()
        mock_basic.assert_not_called()

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_repo_setup_and_install",
        return_value=False,
    )
    def test_install_fails_when_repo_setup_fails(self, mock_repo_setup):
        args = self._base_args("install")
        with _suppress_script_output():
            rc = native_linux_package_install_test.run_tests(args)
        self.assertEqual(rc, 1)

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_uninstall_and_verify",
        return_value=True,
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_basic_verification",
        return_value=True,
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_repo_setup_and_install",
        return_value=True,
    )
    def test_sanity_runs_uninstall_when_flag_set(
        self, mock_repo_setup, mock_basic, mock_uninstall
    ):
        args = self._args_for_uninstall_run_tests("sanity", with_uninstall=True)
        with _suppress_script_output():
            rc = native_linux_package_install_test.run_tests(args)
        self.assertEqual(rc, 0)
        mock_uninstall.assert_called_once()

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_uninstall_and_verify",
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_basic_verification",
        return_value=True,
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_repo_setup_and_install",
        return_value=True,
    )
    def test_sanity_skips_uninstall_when_flag_not_set(
        self, mock_repo_setup, mock_basic, mock_uninstall
    ):
        args = self._args_for_uninstall_run_tests("sanity")
        with _suppress_script_output():
            rc = native_linux_package_install_test.run_tests(args)
        self.assertEqual(rc, 0)
        mock_uninstall.assert_not_called()

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_uninstall_and_verify",
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_basic_verification",
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_repo_setup_and_install",
        return_value=True,
    )
    def test_install_skips_uninstall_even_when_flag_set(
        self, mock_repo_setup, mock_basic, mock_uninstall
    ):
        args = self._args_for_uninstall_run_tests("install", with_uninstall=True)
        with _suppress_script_output():
            rc = native_linux_package_install_test.run_tests(args)
        self.assertEqual(rc, 0)
        mock_uninstall.assert_not_called()
        mock_basic.assert_not_called()

    def test_parse_cli_with_uninstall_flag(self):
        args = native_linux_package_install_test.parse_cli_arguments(
            [
                "--test-type",
                "sanity",
                "--os-profile",
                "ubuntu2404",
                "--repo-url",
                "https://repo_url.com",
                "--release-type",
                "nightly",
                "--install-prefix",
                "/opt/rocm/core",
                "--with-uninstall",
            ],
            raise_instead_of_exit=True,
        )
        self.assertTrue(args.with_uninstall)

    def test_parse_cli_rocm_version_with_multiple_gfx_arch(self):
        args = native_linux_package_install_test.parse_cli_arguments(
            [
                "--test-type",
                "sanity",
                "--os-profile",
                "ubuntu2404",
                "--repo-url",
                "https://repo_url.com",
                "--release-type",
                "nightly",
                "--install-prefix",
                "/opt/rocm/core",
                "--rocm-version",
                "7.13.1",
                "--gfx-arch",
                "gfx94x",
                "gfx1100",
            ],
            raise_instead_of_exit=True,
        )
        self.assertEqual(args.rocm_version, "7.13.1")
        self.assertEqual(args.gfx_arch, ["gfx94x", "gfx1100"])

    def test_parse_cli_rocm_version_with_comma_single_gfx_arch_token(self):
        args = native_linux_package_install_test.parse_cli_arguments(
            [
                "--test-type",
                "sanity",
                "--os-profile",
                "ubuntu2404",
                "--repo-url",
                "https://repo_url.com",
                "--release-type",
                "nightly",
                "--install-prefix",
                "/opt/rocm/core",
                "--rocm-version",
                "7.13",
                "--gfx-arch",
                "gfx94x,gfx1100",
            ],
            raise_instead_of_exit=True,
        )
        self.assertEqual(args.gfx_arch, ["gfx94x,gfx1100"])

    def test_parse_cli_rocm_version_with_semicolon_single_gfx_arch_token(self):
        args = native_linux_package_install_test.parse_cli_arguments(
            [
                "--test-type",
                "sanity",
                "--os-profile",
                "ubuntu2404",
                "--repo-url",
                "https://example.com",
                "--release-type",
                "nightly",
                "--install-prefix",
                "/opt/rocm/core",
                "--rocm-version",
                "7.13",
                "--gfx-arch",
                "gfx94x;gfx1100",
            ],
            raise_instead_of_exit=True,
        )
        self.assertEqual(args.gfx_arch, ["gfx94x;gfx1100"])


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

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "verify_installed_file_security",
        return_value=True,
    )
    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_true_when_enough_components_found(self, mock_run, mock_security):
        # Test that run_basic_verification returns True when install_prefix exists and at least
        # VERIFY_MIN_COMPONENTS key components exist; subprocess (dpkg/rpm, rocminfo) is mocked.
        # The file-security check is stubbed here (covered separately in its own tests).
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

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "verify_installed_file_security",
        return_value=True,
    )
    @patch("native_linux_package_install_test.subprocess.run")
    def test_handles_called_process_error_when_querying_packages(
        self, mock_run, mock_security
    ):
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

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "verify_installed_file_security",
        return_value=True,
    )
    @patch("native_linux_package_install_test.subprocess.run")
    def test_handles_rocminfo_timeout(self, mock_run, mock_security):
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


class VerifyInstalledFileSecurityTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.verify_installed_file_security()."""

    def _make(self, install_prefix="/opt/rocm/core"):
        return native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            install_prefix=install_prefix,
        )

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_true_when_find_reports_no_offending_paths(self, mock_run):
        # Empty find output means every path is root-owned with safe permissions.
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with _suppress_script_output():
            self.assertTrue(self._make().verify_installed_file_security())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_false_when_find_reports_offending_paths(self, mock_run):
        # Non-empty find output lists non-root-owned or insecure paths -> failure.
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/opt/rocm/core/bin/foo\n/opt/rocm/core/bin/setuid-tool\n",
            stderr="",
        )
        with _suppress_script_output():
            self.assertFalse(self._make().verify_installed_file_security())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_ignores_blank_lines_in_find_output(self, mock_run):
        # Trailing/blank lines alone should not be treated as offending paths.
        mock_run.return_value = MagicMock(returncode=0, stdout="\n\n", stderr="")
        with _suppress_script_output():
            self.assertTrue(self._make().verify_installed_file_security())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_nonzero_find_returncode_warns_but_still_evaluates_output(self, mock_run):
        # find can exit non-zero (e.g. permission denied on a subtree) while
        # still printing partial output; that output is still evaluated.
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="/opt/rocm/core/bin/foo\n",
            stderr="find: '/opt/rocm/core/x': Permission denied\n",
        )
        with _suppress_script_output():
            self.assertFalse(self._make().verify_installed_file_security())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_builds_expected_find_command(self, mock_run):
        # Verify the single find invocation follows the prefix symlink (-H),
        # stays on one filesystem (-xdev), checks ownership (uid/gid), writable
        # (0o022) on non-symlinks (! -type l) and setid (0o6000) on regular
        # files only (-type f).
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with _suppress_script_output():
            self._make("/opt/rocm/core").verify_installed_file_security()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "find")
        self.assertEqual(cmd[1], "-H")
        self.assertEqual(cmd[2], "/opt/rocm/core")
        self.assertIn("-xdev", cmd)
        self.assertIn("-uid", cmd)
        self.assertIn("-gid", cmd)
        self.assertIn("/022", cmd)
        self.assertIn("/6000", cmd)
        self.assertIn("!", cmd)
        # symlinks are excluded from the writable portion (! -type l)
        self.assertIn("-type", cmd)
        self.assertIn("l", cmd)
        # setuid/setgid is scoped to regular files (-type f) so benign setgid
        # directories (drwxr-sr-x) are not flagged.
        self.assertIn("f", cmd)
        setid_idx = cmd.index("/6000")
        self.assertEqual(cmd[setid_idx - 3 : setid_idx], ["-type", "f", "-perm"])
        # no sticky-bit special-casing anymore
        self.assertNotIn("-1000", cmd)
        # no in-process timeout (style guide: no timeouts on basic binutils)
        self.assertNotIn("timeout", mock_run.call_args[1])

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "_verify_installed_file_security_python",
        return_value=True,
    )
    @patch("native_linux_package_install_test.subprocess.run")
    def test_falls_back_to_python_when_find_not_available(
        self, mock_run, mock_fallback
    ):
        # If find cannot be executed (OSError), fall back to the Python scan
        # rather than silently skipping; the fallback result is returned.
        mock_run.side_effect = OSError("find not found")
        with _suppress_script_output():
            self.assertTrue(self._make().verify_installed_file_security())
        mock_fallback.assert_called_once()

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "_verify_installed_file_security_python",
        return_value=False,
    )
    @patch("native_linux_package_install_test.subprocess.run")
    def test_find_missing_fallback_can_fail(self, mock_run, mock_fallback):
        # When find is missing and the Python fallback finds offenders, the
        # overall check fails.
        mock_run.side_effect = OSError("find not found")
        with _suppress_script_output():
            self.assertFalse(self._make().verify_installed_file_security())
        mock_fallback.assert_called_once()

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_passes_for_safe_root_owned_tree(
        self, mock_walk, mock_lstat
    ):
        # Python fallback returns True for root-owned 0755 dir / 0644 file modes.
        mock_walk.return_value = [
            ("/opt/rocm/core", ["bin"], ["a.txt"]),
            ("/opt/rocm/core/bin", [], ["rocminfo"]),
        ]
        mock_lstat.side_effect = [
            MagicMock(st_uid=0, st_gid=0, st_mode=stat.S_IFDIR | 0o755),  # bin dir
            MagicMock(st_uid=0, st_gid=0, st_mode=stat.S_IFREG | 0o644),  # a.txt
            MagicMock(st_uid=0, st_gid=0, st_mode=stat.S_IFREG | 0o755),  # rocminfo
        ]
        with _suppress_script_output():
            self.assertTrue(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_fails_on_non_root_entry(self, mock_walk, mock_lstat):
        # Python fallback returns False when any entry is not owned by root.
        mock_walk.return_value = [("/opt/rocm/core", [], ["a.txt", "b.txt"])]
        mock_lstat.side_effect = [
            MagicMock(st_uid=0, st_gid=0, st_mode=stat.S_IFREG | 0o644),
            MagicMock(st_uid=1000, st_gid=1000, st_mode=stat.S_IFREG | 0o644),
        ]
        with _suppress_script_output():
            self.assertFalse(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_fails_on_world_writable(self, mock_walk, mock_lstat):
        # A root-owned but world-writable directory (PATH-hijack case) is flagged.
        mock_walk.return_value = [("/opt/rocm/core", ["bin"], [])]
        mock_lstat.return_value = MagicMock(
            st_uid=0, st_gid=0, st_mode=stat.S_IFDIR | 0o777
        )
        with _suppress_script_output():
            self.assertFalse(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_flags_sticky_world_writable_dir(
        self, mock_walk, mock_lstat
    ):
        # ROCm ships no world-writable dirs, so even a sticky one (mode 1777)
        # is flagged rather than exempted.
        mock_walk.return_value = [("/opt/rocm/core", ["tmp"], [])]
        mock_lstat.return_value = MagicMock(
            st_uid=0, st_gid=0, st_mode=stat.S_IFDIR | stat.S_ISVTX | 0o777
        )
        with _suppress_script_output():
            self.assertFalse(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_fails_on_setuid(self, mock_walk, mock_lstat):
        # A root-owned setuid binary is flagged as a privilege-escalation surface.
        mock_walk.return_value = [("/opt/rocm/core", [], ["setuid-tool"])]
        mock_lstat.return_value = MagicMock(
            st_uid=0, st_gid=0, st_mode=stat.S_IFREG | stat.S_ISUID | 0o755
        )
        with _suppress_script_output():
            self.assertFalse(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_fails_on_setgid_file(self, mock_walk, mock_lstat):
        # A root-owned setgid *regular file* is still flagged.
        mock_walk.return_value = [("/opt/rocm/core", [], ["setgid-tool"])]
        mock_lstat.return_value = MagicMock(
            st_uid=0, st_gid=0, st_mode=stat.S_IFREG | stat.S_ISGID | 0o755
        )
        with _suppress_script_output():
            self.assertFalse(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_allows_setgid_directory(self, mock_walk, mock_lstat):
        # A root-owned setgid *directory* (drwxr-sr-x, mode 2755) is a benign
        # group-inheritance pattern and must NOT be flagged. This is the common
        # ROCm install-tree case (2287 such dirs surfaced in CI).
        mock_walk.return_value = [("/opt/rocm/core", ["libexec"], [])]
        mock_lstat.return_value = MagicMock(
            st_uid=0, st_gid=0, st_mode=stat.S_IFDIR | stat.S_ISGID | 0o755
        )
        with _suppress_script_output():
            self.assertTrue(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_allows_root_owned_symlink(self, mock_walk, mock_lstat):
        # A root-owned symlink (mode 0o777, but bits are meaningless) is allowed.
        # This is the /opt/rocm/core -> /opt/rocm/core-X.Y case from CI.
        mock_walk.return_value = [("/opt/rocm/core", [], ["link"])]
        mock_lstat.return_value = MagicMock(
            st_uid=0, st_gid=0, st_mode=stat.S_IFLNK | 0o777
        )
        with _suppress_script_output():
            self.assertTrue(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_fails_on_non_root_symlink(self, mock_walk, mock_lstat):
        # A non-root-owned symlink is still flagged (ownership is meaningful).
        mock_walk.return_value = [("/opt/rocm/core", [], ["link"])]
        mock_lstat.return_value = MagicMock(
            st_uid=1000, st_gid=1000, st_mode=stat.S_IFLNK | 0o777
        )
        with _suppress_script_output():
            self.assertFalse(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch("native_linux_package_install_test.os.lstat")
    @patch("native_linux_package_install_test.os.walk")
    def test_python_fallback_skips_unstatable_entries(self, mock_walk, mock_lstat):
        # Entries that cannot be lstat'd (e.g. race/permission) are skipped, not fatal.
        mock_walk.return_value = [("/opt/rocm/core", [], ["gone.txt"])]
        mock_lstat.side_effect = OSError("no such file")
        with _suppress_script_output():
            self.assertTrue(
                self._make()._verify_installed_file_security_python(
                    Path("/opt/rocm/core")
                )
            )

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "verify_installed_file_security",
        return_value=False,
    )
    @patch("native_linux_package_install_test.subprocess.run")
    def test_basic_verification_fails_when_security_check_fails(
        self, mock_run, mock_security
    ):
        # Even with enough key components present, a failed file-security check
        # (bad ownership or insecure permissions) fails Step 2.
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
            with _suppress_script_output():
                self.assertFalse(t.run_basic_verification())
        mock_security.assert_called_once()


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

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_true_for_deb_when_mock_succeeds(self, mock_run):
        # Test that for DEB with gpg_key_url, setup_gpg_key returns True when mkdir, pipeline, and chmod succeed.
        mock_run.return_value = MagicMock(returncode=0)
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gpg_key_url="https://example.com/rocm.gpg",
        )
        self.assertTrue(t.setup_gpg_key())
        self.assertEqual(mock_run.call_count, 3)  # mkdir, pipeline, chmod

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
    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_true_when_apt_update_succeeds_no_gpg(
        self, mock_run, mock_streaming
    ):
        # Test that setup_deb_repository writes repo entry via sudo tee (trusted=yes) and returns True when apt update returns 0.
        mock_run.return_value = MagicMock(returncode=0)
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
            gfx_arch="gfx94x",
        )
        self.assertTrue(t.setup_deb_repository())
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args[0][0][:2], ["sudo", "tee"])
        written = mock_run.call_args.kwargs["input"].decode()
        self.assertIn("trusted=yes", written)
        self.assertIn("https://repo.example.com", written)

    @patch("native_linux_package_install_test._run_streaming")
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "setup_gpg_key",
        return_value=True,
    )
    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_true_with_gpg_when_apt_update_succeeds(
        self, mock_run, mock_gpg, mock_streaming
    ):
        # Test that with gpg_key_url, setup_gpg_key is called and repo entry uses signed-by.
        mock_run.return_value = MagicMock(returncode=0)
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url="https://example.com/rocm.gpg",
            gfx_arch="gfx94x",
        )
        self.assertTrue(t.setup_deb_repository())
        mock_gpg.assert_called_once()
        written = mock_run.call_args.kwargs["input"].decode()
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

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_false_when_open_raises(self, mock_run):
        # Test that setup_deb_repository returns False when sudo tee fails.
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "tee", stderr=b"permission denied"
        )
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
            gfx_arch="gfx94x",
        )
        self.assertFalse(t.setup_deb_repository())

    @patch("native_linux_package_install_test._run_streaming")
    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_false_when_apt_update_fails(self, mock_run, mock_streaming):
        # Test that setup_deb_repository returns False when apt update returns non-zero.
        mock_run.return_value = MagicMock(returncode=0)
        mock_streaming.return_value = 1
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
            gfx_arch="gfx94x",
        )
        self.assertFalse(t.setup_deb_repository())

    @patch("native_linux_package_install_test._run_streaming")
    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_false_when_apt_update_times_out(self, mock_run, mock_streaming):
        # Test that setup_deb_repository returns False when _run_streaming raises TimeoutExpired.
        import subprocess

        mock_run.return_value = MagicMock(returncode=0)
        mock_streaming.side_effect = subprocess.TimeoutExpired("apt", 120)
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
            gfx_arch="gfx94x",
        )
        self.assertFalse(t.setup_deb_repository())


class SetupSlesRepositoryTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest._setup_sles_repository()."""

    @patch("native_linux_package_install_test._run_streaming")
    @patch("native_linux_package_install_test.subprocess.run")
    @patch("native_linux_package_install_test.Path.write_text")
    def test_returns_true_when_refresh_succeeds(
        self, mock_write_text, mock_run, mock_streaming
    ):
        # Test that _setup_sles_repository writes repo file and returns True when zypper refresh returns 0.
        # Implementation uses Path.write_text (not open); mock that so /etc is not touched.
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="sles16",
            gfx_arch="gfx94x",
        )
        self.assertTrue(t._setup_sles_repository())
        written = mock_write_text.call_args[0][0]
        self.assertIn("baseurl=https://repo.example.com", written)
        self.assertIn("sles16", t.os_profile)


class SetupDnfRepositoryTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest._setup_dnf_repository()."""

    @patch("native_linux_package_install_test.subprocess.run")
    @patch("native_linux_package_install_test.Path.write_text")
    def test_returns_true_after_writing_repo_file(self, mock_write_text, mock_run):
        # Test that _setup_dnf_repository writes repo file and returns True (dnf clean may be mocked).
        mock_run.return_value = MagicMock(returncode=0)
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="rhel8",
            gfx_arch="gfx94x",
        )
        self.assertTrue(t._setup_dnf_repository())
        written = mock_write_text.call_args[0][0]
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
        self.assertEqual(call_args[:4], ["sudo", "apt", "install", "-y"])
        self.assertIn("amdrocm", call_args)

    @patch("native_linux_package_install_test._run_streaming")
    def test_apt_install_includes_versioned_multi_arch_package_names(
        self, mock_streaming
    ):
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch=["gfx94x", "gfx1100"],
            rocm_version="7.13",
        )
        with _suppress_script_output():
            self.assertTrue(t.install_deb_packages())
        cmd = mock_streaming.call_args[0][0]
        self.assertEqual(cmd[:4], ["sudo", "apt", "install", "-y"])
        self.assertIn("amdrocm7.13-gfx94x", cmd)
        self.assertIn("amdrocm-core-sdk7.13-gfx94x", cmd)
        self.assertIn("amdrocm7.13-gfx1100", cmd)
        self.assertIn("amdrocm-core-sdk7.13-gfx1100", cmd)

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
            self.assertIn("rdhc.py", str(call_args[1]))
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


class ListInstalledRocmPackagesTest(unittest.TestCase):
    """Tests for Step 4b query helper ``list_installed_rocm_packages()``."""

    @patch("native_linux_package_install_test.subprocess.run")
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
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        names = t.list_installed_rocm_packages()
        self.assertEqual(names, ["amdrocm", "rocm-dev"])

    @patch("native_linux_package_install_test.subprocess.run")
    def test_rpm_parses_installed_package_names(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="amdrocm-7.13-1.x86_64\nkernel-5.14-1.x86_64\n",
        )
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
        )
        names = t.list_installed_rocm_packages()
        self.assertEqual(names, ["amdrocm-7.13-1.x86_64"])

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_empty_list_on_query_failure(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "dpkg")
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        self.assertEqual(t.list_installed_rocm_packages(), [])


class UninstallDebPackagesTest(unittest.TestCase):
    """Tests for Step 4a ``uninstall_packages()`` on deb (apt remove + autoremove)."""

    @patch("native_linux_package_install_test._run_streaming")
    def test_remove_and_autoremove_in_reverse_order(self, mock_streaming):
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
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

    @patch("native_linux_package_install_test._run_streaming")
    def test_returns_false_when_remove_fails(self, mock_streaming):
        mock_streaming.return_value = 1
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx94x",
        )
        with _suppress_script_output():
            self.assertFalse(t.uninstall_packages())


class UninstallRpmPackagesTest(unittest.TestCase):
    """Tests for Step 4a ``uninstall_packages()`` on rpm (dnf / zypper --clean-deps)."""

    @patch("native_linux_package_install_test._run_streaming")
    def test_dnf_remove_for_rhel(self, mock_streaming):
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
            gfx_arch="gfx94x",
        )
        with _suppress_script_output():
            self.assertTrue(t.uninstall_packages())
        cmd = mock_streaming.call_args[0][0]
        self.assertEqual(cmd[:3], ["dnf", "remove", "-y"])
        self.assertIn("amdrocm-core-sdk", cmd)
        self.assertIn("amdrocm", cmd)

    @patch("native_linux_package_install_test._run_streaming")
    def test_zypper_remove_for_sles(self, mock_streaming):
        """SLES must pass --clean-deps so dependency packages are removed."""
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
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
    """Tests for Step 4b ``run_uninstall_verification()``."""

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "list_installed_rocm_packages",
        return_value=[],
    )
    def test_passes_when_no_packages_remain(self, mock_list):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            install_prefix="/nonexistent/prefix",
        )
        with _suppress_script_output():
            self.assertTrue(t.run_uninstall_verification())

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "list_installed_rocm_packages",
        return_value=["amdrocm"],
    )
    def test_fails_when_packages_remain(self, mock_list):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        with _suppress_script_output():
            self.assertFalse(t.run_uninstall_verification())


class RunUninstallAndVerifyTest(unittest.TestCase):
    """Tests for Step 4 ``run_uninstall_and_verify()`` orchestration."""

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_uninstall_verification",
        return_value=True,
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "uninstall_packages",
        return_value=True,
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "list_installed_rocm_packages",
        return_value=["amdrocm"],
    )
    def test_orchestrates_uninstall_and_verify(
        self, mock_list, mock_uninstall, mock_verify
    ):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        with _suppress_script_output():
            self.assertTrue(t.run_uninstall_and_verify())
        mock_list.assert_called()
        mock_uninstall.assert_called_once()
        mock_verify.assert_called_once()

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "run_uninstall_verification",
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "uninstall_packages",
        return_value=False,
    )
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "list_installed_rocm_packages",
        return_value=[],
    )
    def test_returns_false_when_uninstall_fails(
        self, mock_list, mock_uninstall, mock_verify
    ):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        with _suppress_script_output():
            self.assertFalse(t.run_uninstall_and_verify())
        mock_verify.assert_not_called()


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


class BuildVariantPackageNamesTest(unittest.TestCase):
    """Verify that build_variant='asan' inserts '-asan' before version in package names."""

    def test_asan_with_gfx_arch_and_rocm_version(self):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx942",
            rocm_version="7.15.0",
            build_variant="asan",
        )
        self.assertEqual(
            t.package_names,
            ["amdrocm-asan7.15-gfx942", "amdrocm-core-sdk-asan7.15-gfx942"],
        )

    def test_asan_with_rocm_version_only(self):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel10",
            rocm_version="7.15.0",
            build_variant="asan",
        )
        self.assertEqual(
            t.package_names,
            ["amdrocm-asan7.15", "amdrocm-core-sdk-asan7.15"],
        )

    def test_asan_without_version_or_arch(self):
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            build_variant="asan",
        )
        self.assertEqual(
            t.package_names,
            ["amdrocm-asan", "amdrocm-core-sdk-asan"],
        )

    def test_no_build_variant_unchanged(self):
        # Verify default (no build_variant) is unaffected.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx942",
            rocm_version="7.15.0",
        )
        self.assertEqual(
            t.package_names,
            ["amdrocm7.15-gfx942", "amdrocm-core-sdk7.15-gfx942"],
        )

    def test_release_build_variant_does_not_alter_package_names(self):
        # 'release' is the default build type label passed from CI build_variant_label.
        # It must NOT insert '-release' into package names — there is no
        # amdrocm-release7.15 package.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            gfx_arch="gfx942",
            rocm_version="7.15.0",
            build_variant="release",
        )
        self.assertEqual(
            t.package_names,
            ["amdrocm7.15-gfx942", "amdrocm-core-sdk7.15-gfx942"],
        )


class VerifyNoRunpathTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest.verify_no_runpath()."""

    def _make(self, install_prefix="/opt/rocm/core"):
        return native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            install_prefix=install_prefix,
        )

    def _elf_tree(self, d):
        # Create two files with ELF magic and one non-ELF file under d.
        (Path(d) / "bin").mkdir()
        (Path(d) / "lib").mkdir()
        (Path(d) / "bin" / "hipcc").write_bytes(b"\x7fELF" + b"\x00" * 32)
        (Path(d) / "lib" / "libamdhip64.so").write_bytes(b"\x7fELF" + b"\x00" * 32)
        (Path(d) / "readme.txt").write_text("not an elf file")

    @patch("native_linux_package_install_test.subprocess.run")
    def test_passes_when_no_elf_uses_runpath(self, mock_run):
        # readelf output showing (RPATH) for every ELF -> pass.
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" 0x000000000000000f (RPATH) Library rpath: [$ORIGIN/../lib]\n",
        )
        with tempfile.TemporaryDirectory() as d:
            self._elf_tree(d)
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())
        # readelf invoked for the two ELF files only, not the .txt file.
        self.assertEqual(mock_run.call_count, 2)

    @patch("native_linux_package_install_test.subprocess.run")
    def test_fails_when_an_elf_uses_runpath(self, mock_run):
        # Any (RUNPATH) line marks the file as offending -> fail.
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" 0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN/../lib]\n",
        )
        with tempfile.TemporaryDirectory() as d:
            self._elf_tree(d)
            with _suppress_script_output():
                self.assertFalse(self._make(d).verify_no_runpath())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_passes_when_elf_has_no_dynamic_path(self, mock_run):
        # ELF with neither RPATH nor RUNPATH is acceptable.
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" 0x0000000000000001 (NEEDED) Shared library: [libc.so.6]\n",
        )
        with tempfile.TemporaryDirectory() as d:
            self._elf_tree(d)
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_skips_non_elf_files(self, mock_run):
        # A tree with no ELF files performs no readelf calls and passes.
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text("plain")
            (Path(d) / "b.json").write_text("{}")
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())
        mock_run.assert_not_called()

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_true_when_readelf_unavailable(self, mock_run):
        # If readelf cannot be executed (OSError), the check is skipped (non-fatal).
        mock_run.side_effect = OSError("readelf not found")
        with tempfile.TemporaryDirectory() as d:
            self._elf_tree(d)
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_skips_file_when_readelf_returns_nonzero(self, mock_run):
        # readelf returning non-zero (corrupt ELF, etc.) is non-fatal; the file
        # is skipped with a warning and the overall check still passes.
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="readelf: Error: Not an ELF file\n",
        )
        with tempfile.TemporaryDirectory() as d:
            self._elf_tree(d)
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())
        # readelf was invoked for both ELF files despite the non-zero exit.
        self.assertEqual(mock_run.call_count, 2)

    @patch("native_linux_package_install_test.subprocess.run")
    def test_rpath_present_is_not_flagged_even_with_runpath(self, mock_run):
        # DT_RPATH is checked first; a file that reports both tags is treated as
        # converted (RPATH present) and is not flagged.
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                " 0x000000000000000f (RPATH) Library rpath: [$ORIGIN/../lib]\n"
                " 0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN/../lib]\n"
            ),
        )
        with tempfile.TemporaryDirectory() as d:
            self._elf_tree(d)
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_fails_when_only_some_files_are_runpath_only(self, mock_run):
        # A mixed tree (one RPATH file, one RUNPATH-only file) still fails
        # because at least one ELF is missing RPATH and carries RUNPATH.
        def fake_readelf(cmd, **kwargs):
            filepath = cmd[-1]
            if filepath.endswith("libamdhip64.so"):
                stdout = (
                    " 0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN/../lib]\n"
                )
            else:
                stdout = " 0x000000000000000f (RPATH) Library rpath: [$ORIGIN/../lib]\n"
            return MagicMock(returncode=0, stdout=stdout)

        mock_run.side_effect = fake_readelf
        with tempfile.TemporaryDirectory() as d:
            self._elf_tree(d)
            with _suppress_script_output():
                self.assertFalse(self._make(d).verify_no_runpath())

    @patch("native_linux_package_install_test.subprocess.run")
    def test_files_with_neither_tag_pass_and_are_counted(self, mock_run):
        # ELFs with neither DT_RPATH nor DT_RUNPATH are reported but not fatal.
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" 0x0000000000000001 (NEEDED) Shared library: [libc.so.6]\n",
        )
        with tempfile.TemporaryDirectory() as d:
            self._elf_tree(d)
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())
        self.assertEqual(mock_run.call_count, 2)

    @patch("native_linux_package_install_test.subprocess.run")
    def test_fixed_path_rpath_is_reported_but_not_fatal(self, mock_run):
        # An rpath with a non-$ORIGIN (fixed/absolute) entry is warned about but
        # does not fail the check.
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" 0x000000000000000f (RPATH) Library rpath: [/opt/rocm/lib]\n",
        )
        with tempfile.TemporaryDirectory() as d:
            self._elf_tree(d)
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())

    def test_fixed_rpath_entries_helper(self):
        # Helper flags only entries that lack $ORIGIN, preserving order.
        fn = (
            native_linux_package_install_test.NativeLinuxPackageInstallTest._fixed_rpath_entries
        )
        # Pure $ORIGIN-relative rpath -> nothing flagged.
        self.assertEqual(
            fn(" 0x0f (RPATH) Library rpath: [$ORIGIN/../lib:$ORIGIN/../lib64]\n"),
            [],
        )
        # Mixed rpath -> only the fixed entries are returned.
        self.assertEqual(
            fn(
                " 0x0f (RPATH) Library rpath: [$ORIGIN/../lib:/opt/rocm/lib:/usr/lib]\n"
            ),
            ["/opt/rocm/lib", "/usr/lib"],
        )
        # No rpath value present -> empty.
        self.assertEqual(fn(" 0x01 (NEEDED) Shared library: [libc.so.6]\n"), [])

    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "verify_no_runpath",
        return_value=False,
    )
    @patch("native_linux_package_install_test.subprocess.run")
    def test_basic_verification_fails_when_runpath_check_fails(
        self, mock_run, mock_rpath
    ):
        # Even with enough components present, a failed RUNPATH check fails Step 2.
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
            with _suppress_script_output():
                self.assertFalse(t.run_basic_verification())
        mock_rpath.assert_called_once()


# These fixtures build real ELF shared objects with GNU-ld options like
# -Wl,-rpath and --disable-new-dtags. That only works with a Linux-targeting
# toolchain: on Windows the runner ships MinGW (cc/readelf are on PATH) but its
# ld targets PE/COFF and rejects those ELF-only options, so gate on Linux too.
_ELF_TOOLCHAIN_AVAILABLE = bool(
    sys.platform.startswith("linux") and shutil.which("cc") and shutil.which("readelf")
)


@unittest.skipUnless(
    _ELF_TOOLCHAIN_AVAILABLE,
    "real-ELF fixtures are Linux-only and require cc + readelf",
)
class VerifyNoRunpathRealElfTest(unittest.TestCase):
    """End-to-end tests for verify_no_runpath() against real ELF files.

    Unlike VerifyNoRunpathTest, which mocks readelf, these compile small shared
    objects with known DT_RPATH/DT_RUNPATH tags and run the real ``readelf`` so
    the actual subprocess code path is exercised. They only run on Linux with a
    C compiler and readelf, and are skipped elsewhere (for example the
    windows-2022 CI leg, whose MinGW ld cannot produce ELF with these options).
    Error paths that cannot be produced from a real file (readelf missing,
    non-zero exit) remain covered by the mocked VerifyNoRunpathTest.
    """

    def _compile_so(self, directory, name, *link_flags):
        # $ORIGIN is passed literally (no shell), so the linker records it as-is.
        src = Path(directory) / "src.c"
        if not src.exists():
            src.write_text("int f(void) { return 0; }\n")
        out = Path(directory) / name
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-o", str(out), *link_flags, str(src)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return out

    def _make(self, install_prefix):
        return native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
            install_prefix=install_prefix,
        )

    def test_real_rpath_origin_relative_passes(self):
        # DT_RPATH with an $ORIGIN-relative value -> pass.
        with tempfile.TemporaryDirectory() as d:
            self._compile_so(
                d,
                "librpath.so",
                "-Wl,-rpath,$ORIGIN/../lib",
                "-Wl,--disable-new-dtags",
            )
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())

    def test_real_runpath_fails(self):
        # DT_RUNPATH (new dtags) with no DT_RPATH -> conversion missed -> fail.
        with tempfile.TemporaryDirectory() as d:
            self._compile_so(
                d,
                "librunpath.so",
                "-Wl,-rpath,$ORIGIN/../lib",
                "-Wl,--enable-new-dtags",
            )
            with _suppress_script_output():
                self.assertFalse(self._make(d).verify_no_runpath())

    def test_real_neither_tag_passes(self):
        # No rpath/runpath tag at all -> pass (nothing to convert).
        with tempfile.TemporaryDirectory() as d:
            self._compile_so(d, "libnone.so")
            with _suppress_script_output():
                self.assertTrue(self._make(d).verify_no_runpath())

    def test_real_fixed_rpath_is_reported_but_not_fatal(self):
        # DT_RPATH with a non-$ORIGIN (absolute) entry -> warned, not fatal.
        with tempfile.TemporaryDirectory() as d:
            self._compile_so(
                d,
                "libfixed.so",
                "-Wl,-rpath,/opt/rocm/lib",
                "-Wl,--disable-new-dtags",
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = self._make(d).verify_no_runpath()
        output = buf.getvalue()
        self.assertTrue(result)
        self.assertIn("not $ORIGIN-relative", output)
        self.assertIn("/opt/rocm/lib", output)


if __name__ == "__main__":
    unittest.main()
