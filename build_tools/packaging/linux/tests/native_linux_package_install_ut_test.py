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
from unittest.mock import ANY, patch, MagicMock

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
        )

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
    """Tests for NativeLinuxPackageInstallTest.run_basic_verification().

    run_basic_verification() now invokes the post-install transitive dependency
    check (Step 2). These tests target component/rocminfo verification only, so the
    dependency walk is disabled via NATIVE_LINUX_SKIP_DEP_VERIFY to avoid consuming
    the mocked subprocess side effects.
    """

    def setUp(self):
        self._dep_env = patch.dict(
            os.environ,
            {native_linux_package_install_test.ENV_NATIVE_LINUX_SKIP_DEP_VERIFY: "1"},
            clear=False,
        )
        self._dep_env.start()

    def tearDown(self):
        self._dep_env.stop()

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
    @patch("native_linux_package_install_test.Path.write_text")
    def test_returns_true_when_apt_update_succeeds_no_gpg(
        self, mock_write_text, mock_streaming
    ):
        # Test that setup_deb_repository writes repo entry (trusted=yes) and returns True when apt update returns 0.
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
            gfx_arch="gfx94x",
        )
        self.assertTrue(t.setup_deb_repository())
        mock_write_text.assert_called_once()
        written = mock_write_text.call_args[0][0]
        self.assertIn("trusted=yes", written)
        self.assertIn("https://repo.example.com", written)

    @patch("native_linux_package_install_test._run_streaming")
    @patch.object(
        native_linux_package_install_test.NativeLinuxPackageInstallTest,
        "setup_gpg_key",
        return_value=True,
    )
    @patch("native_linux_package_install_test.Path.write_text")
    def test_returns_true_with_gpg_when_apt_update_succeeds(
        self, mock_write_text, mock_gpg, mock_streaming
    ):
        # Test that with gpg_key_url, setup_gpg_key is called and repo entry uses signed-by.
        mock_streaming.return_value = 0
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url="https://example.com/rocm.gpg",
            gfx_arch="gfx94x",
        )
        self.assertTrue(t.setup_deb_repository())
        mock_gpg.assert_called_once()
        written = mock_write_text.call_args[0][0]
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
    @patch(
        "native_linux_package_install_test.Path.write_text",
        side_effect=OSError("Permission denied"),
    )
    def test_returns_false_when_open_raises(self, mock_write_text, mock_streaming):
        # Test that setup_deb_repository returns False when writing sources list raises OSError.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
            gfx_arch="gfx94x",
        )
        self.assertFalse(t.setup_deb_repository())

    @patch("native_linux_package_install_test._run_streaming")
    @patch("native_linux_package_install_test.Path.write_text")
    def test_returns_false_when_apt_update_fails(self, mock_write_text, mock_streaming):
        # Test that setup_deb_repository returns False when apt update returns non-zero.
        mock_streaming.return_value = 1
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://repo.example.com",
            os_profile="ubuntu2404",
            gpg_key_url=None,
            gfx_arch="gfx94x",
        )
        self.assertFalse(t.setup_deb_repository())

    @patch("native_linux_package_install_test._run_streaming")
    @patch("native_linux_package_install_test.Path.write_text")
    def test_returns_false_when_apt_update_times_out(
        self, mock_write_text, mock_streaming
    ):
        # Test that setup_deb_repository returns False when _run_streaming raises TimeoutExpired.
        import subprocess

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
        self.assertEqual(call_args[0], "apt")
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
        self.assertEqual(cmd[:3], ["apt", "install", "-y"])
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


class DebPkgNameFromDepTokenTest(unittest.TestCase):
    """Tests for _deb_pkg_name_from_dep_token()."""

    def test_plain_name_returned_as_is(self):
        # Test that a plain package name is returned unchanged.
        self.assertEqual(
            native_linux_package_install_test._deb_pkg_name_from_dep_token("libfoo"),
            "libfoo",
        )

    def test_strips_version_constraint(self):
        # Test that a trailing version constraint in parentheses is stripped.
        self.assertEqual(
            native_linux_package_install_test._deb_pkg_name_from_dep_token(
                "libfoo (>= 1.2.3)"
            ),
            "libfoo",
        )

    def test_strips_known_arch_qualifier(self):
        # Test that a known architecture qualifier (e.g. :amd64) is stripped.
        self.assertEqual(
            native_linux_package_install_test._deb_pkg_name_from_dep_token(
                "libfoo:amd64"
            ),
            "libfoo",
        )

    def test_strips_surrounding_quotes(self):
        # Test that surrounding single or double quotes are stripped.
        self.assertEqual(
            native_linux_package_install_test._deb_pkg_name_from_dep_token('"libfoo"'),
            "libfoo",
        )

    def test_substvar_returns_none(self):
        # Test that a dpkg substvar token (e.g. ${shlibs:Depends}) returns None.
        self.assertIsNone(
            native_linux_package_install_test._deb_pkg_name_from_dep_token(
                "${shlibs:Depends}"
            )
        )

    def test_file_path_returns_none(self):
        # Test that a token starting with '/' (file path) returns None.
        self.assertIsNone(
            native_linux_package_install_test._deb_pkg_name_from_dep_token(
                "/usr/bin/foo"
            )
        )

    def test_empty_token_returns_none(self):
        # Test that an empty or whitespace-only token returns None.
        self.assertIsNone(
            native_linux_package_install_test._deb_pkg_name_from_dep_token("   ")
        )

    def test_unknown_qualifier_kept(self):
        # Test that an unknown qualifier (e.g. :any) is not treated as an arch and is kept.
        self.assertEqual(
            native_linux_package_install_test._deb_pkg_name_from_dep_token(
                "libfoo:any"
            ),
            "libfoo:any",
        )


class ParseDebianDepFieldTest(unittest.TestCase):
    """Tests for _parse_debian_dep_field()."""

    def test_empty_field_returns_empty_list(self):
        # Test that an empty Depends field parses to an empty list of groups.
        self.assertEqual(
            native_linux_package_install_test._parse_debian_dep_field(""), []
        )

    def test_comma_separates_and_groups(self):
        # Test that commas split into separate AND groups (each a single-name frozenset).
        result = native_linux_package_install_test._parse_debian_dep_field(
            "liba, libb (>= 1.0), libc"
        )
        self.assertEqual(
            result,
            [frozenset({"liba"}), frozenset({"libb"}), frozenset({"libc"})],
        )

    def test_pipe_creates_or_group(self):
        # Test that '|' within one AND term produces a single OR-group frozenset.
        result = native_linux_package_install_test._parse_debian_dep_field(
            "liba | libb"
        )
        self.assertEqual(result, [frozenset({"liba", "libb"})])

    def test_mixed_and_or(self):
        # Test a mixed expression: comma-separated AND groups, one containing an OR alternative.
        result = native_linux_package_install_test._parse_debian_dep_field(
            "liba, libb | libc (>= 2.0)"
        )
        self.assertEqual(
            result,
            [frozenset({"liba"}), frozenset({"libb", "libc"})],
        )

    def test_substvars_dropped(self):
        # Test that substvar-only OR-groups are dropped (no resolvable package names).
        result = native_linux_package_install_test._parse_debian_dep_field(
            "liba, ${shlibs:Depends}"
        )
        self.assertEqual(result, [frozenset({"liba"})])


class AptCacheShowFirstStanzaTest(unittest.TestCase):
    """Tests for _apt_cache_show_first_stanza()."""

    @patch("native_linux_package_install_test.subprocess.run")
    def test_parses_first_stanza_and_merges_continuations(self, mock_run):
        # Test that only the first stanza is parsed and folded continuation lines are merged.
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "Package: libfoo\n"
                "Depends: liba,\n"
                " libb\n"
                "\n"
                "Package: other\n"
                "Depends: zzz\n"
            ),
        )
        stanza = native_linux_package_install_test._apt_cache_show_first_stanza(
            "libfoo"
        )
        self.assertEqual(stanza.get("package"), "libfoo")
        # Folded continuation lines are concatenated verbatim (no inserted space);
        # the dependency-field parser later splits on commas regardless of spacing.
        self.assertEqual(stanza.get("depends"), "liba,libb")
        self.assertNotIn("zzz", stanza.get("depends", ""))

    @patch("native_linux_package_install_test.subprocess.run")
    def test_returns_empty_when_command_fails(self, mock_run):
        # Test that a non-zero apt-cache return code yields an empty dict.
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        self.assertEqual(
            native_linux_package_install_test._apt_cache_show_first_stanza("libfoo"),
            {},
        )


class VerifyDebianTransitiveDependenciesTest(unittest.TestCase):
    """Tests for verify_debian_transitive_dependencies()."""

    @patch("native_linux_package_install_test.subprocess.run")
    def test_apt_get_check_failure_returns_false(self, mock_run):
        # Test that a failing 'apt-get check' short-circuits to a failure result.
        mock_run.return_value = MagicMock(
            returncode=100, stdout="", stderr="broken deps"
        )
        ok, errs = (
            native_linux_package_install_test.verify_debian_transitive_dependencies(
                ["amdrocm-gfx94x"]
            )
        )
        self.assertFalse(ok)
        self.assertTrue(any("apt-get check failed" in e for e in errs))

    @patch("native_linux_package_install_test._apt_cache_show_first_stanza")
    @patch("native_linux_package_install_test._deb_is_pkg_installed")
    @patch("native_linux_package_install_test.subprocess.run")
    def test_happy_path_no_dependencies(self, mock_run, mock_installed, mock_stanza):
        # Test that an installed root with no Depends/Pre-Depends passes.
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_installed.return_value = True
        mock_stanza.return_value = {}  # no depends fields
        ok, errs = (
            native_linux_package_install_test.verify_debian_transitive_dependencies(
                ["amdrocm-gfx94x"]
            )
        )
        self.assertTrue(ok)
        self.assertEqual(errs, [])

    @patch("native_linux_package_install_test._deb_is_pkg_installed")
    @patch("native_linux_package_install_test.subprocess.run")
    def test_root_not_installed_returns_false(self, mock_run, mock_installed):
        # Test that a root package missing from dpkg yields a failure.
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_installed.return_value = False
        ok, errs = (
            native_linux_package_install_test.verify_debian_transitive_dependencies(
                ["amdrocm-gfx94x"]
            )
        )
        self.assertFalse(ok)
        self.assertTrue(any("not installed" in e for e in errs))

    @patch("native_linux_package_install_test._deb_pick_installed_rep")
    @patch("native_linux_package_install_test._apt_cache_show_first_stanza")
    @patch("native_linux_package_install_test._deb_is_pkg_installed")
    @patch("native_linux_package_install_test.subprocess.run")
    def test_unsatisfied_dependency_group_returns_false(
        self, mock_run, mock_installed, mock_stanza, mock_pick
    ):
        # Test that an unsatisfied Depends OR-group (no installed alternative) fails.
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_installed.return_value = True
        mock_stanza.return_value = {"depends": "libmissing"}
        mock_pick.return_value = None
        ok, errs = (
            native_linux_package_install_test.verify_debian_transitive_dependencies(
                ["amdrocm-gfx94x"]
            )
        )
        self.assertFalse(ok)
        self.assertTrue(any("unsatisfied dependency group" in e for e in errs))

    @patch("native_linux_package_install_test._deb_is_pkg_installed")
    @patch("native_linux_package_install_test.subprocess.run")
    def test_max_closure_exceeded_returns_false(self, mock_run, mock_installed):
        # Test that exceeding max_closure aborts the walk with an error.
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_installed.return_value = True
        ok, errs = (
            native_linux_package_install_test.verify_debian_transitive_dependencies(
                ["amdrocm-gfx94x"], max_closure=0
            )
        )
        self.assertFalse(ok)
        self.assertTrue(any("exceeded max steps" in e for e in errs))


class RpmRequireTokenClassificationTest(unittest.TestCase):
    """Tests for _rpm_require_token_is_file_or_rpmlib()."""

    def test_file_path_is_skipped(self):
        # Test that an absolute file-path requirement is classified as skippable.
        self.assertTrue(
            native_linux_package_install_test._rpm_require_token_is_file_or_rpmlib(
                "/usr/lib/libc.so.6"
            )
        )

    def test_rpmlib_is_skipped(self):
        # Test that an rpmlib(...) requirement is classified as skippable.
        self.assertTrue(
            native_linux_package_install_test._rpm_require_token_is_file_or_rpmlib(
                "rpmlib(CompressedFileNames)"
            )
        )

    def test_config_is_skipped(self):
        # Test that a config(...) requirement is classified as skippable.
        self.assertTrue(
            native_linux_package_install_test._rpm_require_token_is_file_or_rpmlib(
                "config(foo)"
            )
        )

    def test_empty_is_skipped(self):
        # Test that an empty requirement token is classified as skippable.
        self.assertTrue(
            native_linux_package_install_test._rpm_require_token_is_file_or_rpmlib("")
        )

    def test_normal_name_not_skipped(self):
        # Test that a normal capability/package name is not skippable.
        self.assertFalse(
            native_linux_package_install_test._rpm_require_token_is_file_or_rpmlib(
                "libstdc++"
            )
        )


class VerifyRpmTransitiveDependenciesTest(unittest.TestCase):
    """Tests for verify_rpm_transitive_dependencies()."""

    @patch("native_linux_package_install_test.subprocess.run")
    def test_root_not_installed_returns_false(self, mock_run):
        # Test that a root package not installed (rpm -q non-zero) yields a failure.
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="package amdrocm-gfx94x is not installed"
        )
        ok, errs = native_linux_package_install_test.verify_rpm_transitive_dependencies(
            ["amdrocm-gfx94x"]
        )
        self.assertFalse(ok)
        self.assertTrue(any("root package not installed" in e for e in errs))

    @patch("native_linux_package_install_test._rpm_requires_tokens")
    @patch("native_linux_package_install_test._rpm_name_from_installed_nevra")
    @patch("native_linux_package_install_test.subprocess.run")
    def test_happy_path_no_requires(self, mock_run, mock_name, mock_requires):
        # Test that an installed root with no (non-file) requires passes.
        mock_run.return_value = MagicMock(
            returncode=0, stdout="amdrocm-gfx94x-1.0-1.x86_64\n", stderr=""
        )
        mock_name.return_value = "amdrocm-gfx94x"
        mock_requires.return_value = []
        ok, errs = native_linux_package_install_test.verify_rpm_transitive_dependencies(
            ["amdrocm-gfx94x"]
        )
        self.assertTrue(ok)
        self.assertEqual(errs, [])

    @patch("native_linux_package_install_test._rpm_whatprovides_nevras")
    @patch("native_linux_package_install_test._rpm_requires_tokens")
    @patch("native_linux_package_install_test._rpm_name_from_installed_nevra")
    @patch("native_linux_package_install_test.subprocess.run")
    def test_unprovided_requirement_returns_false(
        self, mock_run, mock_name, mock_requires, mock_provides
    ):
        # Test that a requirement with no providing package yields a failure.
        mock_run.return_value = MagicMock(
            returncode=0, stdout="amdrocm-gfx94x-1.0-1.x86_64\n", stderr=""
        )
        mock_name.return_value = "amdrocm-gfx94x"
        mock_requires.return_value = ["libmissing.so.1"]
        mock_provides.return_value = []
        ok, errs = native_linux_package_install_test.verify_rpm_transitive_dependencies(
            ["amdrocm-gfx94x"]
        )
        self.assertFalse(ok)
        self.assertTrue(any("no installed package provides" in e for e in errs))


class VerifyTransitiveDependenciesInstalledTest(unittest.TestCase):
    """Tests for NativeLinuxPackageInstallTest._verify_transitive_dependencies_installed()."""

    def test_skipped_when_env_set(self):
        # Test that the skip env var short-circuits to success without invoking verifiers.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        with patch.dict(
            os.environ,
            {native_linux_package_install_test.ENV_NATIVE_LINUX_SKIP_DEP_VERIFY: "1"},
            clear=False,
        ):
            ok, errs = t._verify_transitive_dependencies_installed()
        self.assertTrue(ok)
        self.assertEqual(errs, [])

    @patch(
        "native_linux_package_install_test.verify_debian_transitive_dependencies",
        return_value=(True, []),
    )
    def test_routes_to_debian_for_deb(self, mock_deb):
        # Test that a deb profile routes to the Debian verifier with the package names.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="ubuntu2404",
        )
        with patch.dict(
            os.environ,
            {native_linux_package_install_test.ENV_NATIVE_LINUX_SKIP_DEP_VERIFY: ""},
            clear=False,
        ):
            ok, _ = t._verify_transitive_dependencies_installed()
        self.assertTrue(ok)
        mock_deb.assert_called_once_with(t.package_names, report=ANY)

    @patch(
        "native_linux_package_install_test.verify_rpm_transitive_dependencies",
        return_value=(True, []),
    )
    def test_routes_to_rpm_for_rpm(self, mock_rpm):
        # Test that an rpm profile routes to the RPM verifier with the package names.
        t = native_linux_package_install_test.NativeLinuxPackageInstallTest(
            repo_url="https://example.com",
            os_profile="rhel8",
        )
        with patch.dict(
            os.environ,
            {native_linux_package_install_test.ENV_NATIVE_LINUX_SKIP_DEP_VERIFY: ""},
            clear=False,
        ):
            ok, _ = t._verify_transitive_dependencies_installed()
        self.assertTrue(ok)
        mock_rpm.assert_called_once_with(t.package_names, report=ANY)


if __name__ == "__main__":
    unittest.main()
