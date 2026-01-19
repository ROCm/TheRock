# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

# Add the linux directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "linux"))

# Import the module to be tested
import packaging_utils
from packaging_utils import PackageConfig


class TestPrintFunctionName(unittest.TestCase):
    """Test cases for print_function_name function."""

    @patch("builtins.print")
    def test_print_function_name(self, mock_print):
        """Test that print_function_name prints the calling function name."""
        def caller_function():
            packaging_utils.print_function_name()

        caller_function()
        mock_print.assert_called_once()
        # Check that the printed string contains "In function:"
        call_args = mock_print.call_args[0][0]
        self.assertIn("In function:", call_args)


class TestReadPackageJsonFile(unittest.TestCase):
    """Test cases for read_package_json_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = [
            {
                "Package": "test-package",
                "Version": "1.0",
                "Gfxarch": "True"
            }
        ]

    @patch("packaging_utils.SCRIPT_DIR", Path("/test/dir"))
    @patch("pathlib.Path.open", new_callable=mock_open, read_data='[{"Package": "test-pkg"}]')
    def test_read_package_json_file_success(self, mock_file, ):
        """Test successful reading of package.json file."""
        result = packaging_utils.read_package_json_file()
        self.assertEqual(result, [{"Package": "test-pkg"}])


class TestIsKeyDefined(unittest.TestCase):
    """Test cases for is_key_defined function."""

    def test_key_true_values(self):
        """Test keys with true-like values."""
        true_values = ["1", "true", "t", "yes", "y", "on", "enable", "enabled", "found", "TRUE", "True", "YES"]
        for val in true_values:
            pkg_info = {"TestKey": val}
            self.assertTrue(packaging_utils.is_key_defined(pkg_info, "TestKey"), f"Failed for value: {val}")

    def test_key_false_values(self):
        """Test keys with false-like values."""
        false_values = ["", "0", "false", "f", "no", "n", "off", "disable", "disabled", 
                       "notfound", "none", "null", "nil", "undefined", "n/a", "FALSE", "False", "NO"]
        for val in false_values:
            pkg_info = {"TestKey": val}
            self.assertFalse(packaging_utils.is_key_defined(pkg_info, "TestKey"), f"Failed for value: {val}")

    def test_case_insensitive_key(self):
        """Test that key lookup is case insensitive."""
        pkg_info = {"testkey": "true"}
        self.assertTrue(packaging_utils.is_key_defined(pkg_info, "TESTKEY"))
        self.assertTrue(packaging_utils.is_key_defined(pkg_info, "TestKey"))

    def test_key_not_present(self):
        """Test when key is not present in dictionary."""
        pkg_info = {"OtherKey": "value"}
        result = packaging_utils.is_key_defined(pkg_info, "TestKey")
        # When key is not found, empty string is used which evaluates to False
        self.assertFalse(result)


class TestIsPostinstallscriptsAvailable(unittest.TestCase):
    """Test cases for is_postinstallscripts_available function."""

    def test_postinstall_true(self):
        """Test when Postinstall is enabled."""
        pkg_info = {"Postinstall": "true"}
        self.assertTrue(packaging_utils.is_postinstallscripts_available(pkg_info))

    def test_postinstall_false(self):
        """Test when Postinstall is disabled."""
        pkg_info = {"Postinstall": "false"}
        self.assertFalse(packaging_utils.is_postinstallscripts_available(pkg_info))


class TestIsMetaPackage(unittest.TestCase):
    """Test cases for is_meta_package function."""

    def test_meta_package_true(self):
        """Test when Metapackage is enabled."""
        pkg_info = {"Metapackage": "True"}
        self.assertTrue(packaging_utils.is_meta_package(pkg_info))

    def test_meta_package_false(self):
        """Test when Metapackage is disabled."""
        pkg_info = {"Metapackage": "False"}
        self.assertFalse(packaging_utils.is_meta_package(pkg_info))


class TestIsCompositePackage(unittest.TestCase):
    """Test cases for is_composite_package function."""

    def test_composite_true(self):
        """Test when composite is enabled."""
        pkg_info = {"composite": "yes"}
        self.assertTrue(packaging_utils.is_composite_package(pkg_info))

    def test_composite_false(self):
        """Test when composite is disabled."""
        pkg_info = {"composite": "no"}
        self.assertFalse(packaging_utils.is_composite_package(pkg_info))


class TestIsRpmStrippingDisabled(unittest.TestCase):
    """Test cases for is_rpm_stripping_disabled function."""

    def test_rpm_strip_disabled_true(self):
        """Test when RPM stripping is disabled."""
        pkg_info = {"Disable_RPM_STRIP": "True"}
        self.assertTrue(packaging_utils.is_rpm_stripping_disabled(pkg_info))

    def test_rpm_strip_disabled_false(self):
        """Test when RPM stripping is not disabled."""
        pkg_info = {"Disable_RPM_STRIP": "False"}
        self.assertFalse(packaging_utils.is_rpm_stripping_disabled(pkg_info))


class TestIsDebugPackageDisabled(unittest.TestCase):
    """Test cases for is_debug_package_disabled function."""

    def test_debug_disabled_true(self):
        """Test when debug package is disabled."""
        pkg_info = {"Disable_Debug_Package": "enabled"}
        self.assertTrue(packaging_utils.is_debug_package_disabled(pkg_info))

    def test_debug_disabled_false(self):
        """Test when debug package is not disabled."""
        pkg_info = {"Disable_Debug_Package": "disabled"}
        self.assertFalse(packaging_utils.is_debug_package_disabled(pkg_info))


class TestIsPackagingDisabled(unittest.TestCase):
    """Test cases for is_packaging_disabled function."""

    def test_packaging_disabled_true(self):
        """Test when packaging is disabled."""
        pkg_info = {"Disablepackaging": "1"}
        self.assertTrue(packaging_utils.is_packaging_disabled(pkg_info))

    def test_packaging_disabled_false(self):
        """Test when packaging is not disabled."""
        pkg_info = {"Disablepackaging": "0"}
        self.assertFalse(packaging_utils.is_packaging_disabled(pkg_info))


class TestIsGfxarchPackage(unittest.TestCase):
    """Test cases for is_gfxarch_package function."""

    def test_gfxarch_true(self):
        """Test when Gfxarch is enabled."""
        pkg_info = {"Gfxarch": "True"}
        self.assertTrue(packaging_utils.is_gfxarch_package(pkg_info))

    def test_gfxarch_false(self):
        """Test when Gfxarch is disabled."""
        pkg_info = {"Gfxarch": "False"}
        self.assertFalse(packaging_utils.is_gfxarch_package(pkg_info))


class TestGetPackageInfo(unittest.TestCase):
    """Test cases for get_package_info function."""

    @patch("packaging_utils.read_package_json_file")
    def test_get_package_info_exists(self, mock_read):
        """Test getting package info when package exists."""
        mock_read.return_value = [
            {"Package": "amdrocm-llvm", "Version": "1.0"},
            {"Package": "amdrocm-runtime", "Version": "2.0"}
        ]
        result = packaging_utils.get_package_info("amdrocm-runtime")
        self.assertEqual(result, {"Package": "amdrocm-runtime", "Version": "2.0"})

    @patch("packaging_utils.read_package_json_file")
    def test_get_package_info_not_exists(self, mock_read):
        """Test getting package info when package does not exist."""
        mock_read.return_value = [
            {"Package": "amdrocm-llvm", "Version": "1.0"}
        ]
        result = packaging_utils.get_package_info("non-existent-package")
        self.assertIsNone(result)


class TestGetPackageList(unittest.TestCase):
    """Test cases for get_package_list function."""

    @patch("packaging_utils.read_package_json_file")
    def test_get_package_list_filters_disabled(self, mock_read):
        """Test that get_package_list filters out disabled packages."""
        mock_read.return_value = [
            {"Package": "pkg1", "Disablepackaging": "false"},
            {"Package": "pkg2", "Disablepackaging": "true"},
            {"Package": "pkg3", "Disablepackaging": "no"}
        ]
        result = packaging_utils.get_package_list()
        self.assertEqual(result, ["pkg1", "pkg3"])

    @patch("packaging_utils.read_package_json_file")
    def test_get_package_list_all_enabled(self, mock_read):
        """Test get_package_list when all packages are enabled."""
        mock_read.return_value = [
            {"Package": "pkg1"},
            {"Package": "pkg2"}
        ]
        result = packaging_utils.get_package_list()
        self.assertEqual(result, ["pkg1", "pkg2"])


class TestRemoveDir(unittest.TestCase):
    """Test cases for remove_dir function."""

    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory if it still exists."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("builtins.print")
    def test_remove_existing_dir(self, mock_print):
        """Test removing an existing directory."""
        test_dir = Path(self.temp_dir) / "test_dir"
        test_dir.mkdir()
        self.assertTrue(test_dir.exists())

        packaging_utils.remove_dir(test_dir)
        self.assertFalse(test_dir.exists())
        mock_print.assert_called_once()
        self.assertIn("Removed directory", mock_print.call_args[0][0])

    @patch("builtins.print")
    def test_remove_nonexistent_dir(self, mock_print):
        """Test removing a non-existent directory."""
        test_dir = Path(self.temp_dir) / "nonexistent"
        packaging_utils.remove_dir(test_dir)
        mock_print.assert_called_once()
        self.assertIn("does not exist", mock_print.call_args[0][0])


class TestVersionToStr(unittest.TestCase):
    """Test cases for version_to_str function."""

    def test_version_three_parts(self):
        """Test version with three parts."""
        self.assertEqual(packaging_utils.version_to_str("7.1.0"), "70100")
        self.assertEqual(packaging_utils.version_to_str("10.1.0"), "100100")
        self.assertEqual(packaging_utils.version_to_str("1.2.3"), "10203")

    def test_version_two_parts(self):
        """Test version with two parts (should append .0)."""
        self.assertEqual(packaging_utils.version_to_str("7.1"), "70100")
        self.assertEqual(packaging_utils.version_to_str("10.5"), "100500")

    def test_version_one_part(self):
        """Test version with one part (should append .0.0)."""
        self.assertEqual(packaging_utils.version_to_str("7"), "70000")

    def test_version_four_parts(self):
        """Test version with four parts (should ignore extra parts)."""
        self.assertEqual(packaging_utils.version_to_str("7.1.1.1"), "70101")

    def test_version_large_numbers(self):
        """Test version with large numbers."""
        self.assertEqual(packaging_utils.version_to_str("7.10.0"), "71000")
        self.assertEqual(packaging_utils.version_to_str("15.20.30"), "152030")


class TestUpdatePackageName(unittest.TestCase):
    """Test cases for update_package_name function."""

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_update_package_name_versioned_no_rpath_no_gfx(self, mock_print, mock_get_info):
        """Test package name update with version, no rpath, no gfx."""
        mock_get_info.return_value = {"Package": "test-pkg", "Gfxarch": "False"}
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )
        result = packaging_utils.update_package_name("test-pkg", config)
        self.assertEqual(result, "test-pkg7.1")

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_update_package_name_with_rpath(self, mock_print, mock_get_info):
        """Test package name update with rpath enabled."""
        mock_get_info.return_value = {"Package": "test-pkg", "Gfxarch": "False"}
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=True,
            versioned_pkg=True
        )
        result = packaging_utils.update_package_name("test-pkg", config)
        self.assertEqual(result, "test-pkg-rpath7.1")

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_update_package_name_with_gfxarch(self, mock_print, mock_get_info):
        """Test package name update with gfxarch."""
        mock_get_info.return_value = {"Package": "test-pkg", "Gfxarch": "True"}
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900-dcgpu",
            enable_rpath=False,
            versioned_pkg=True
        )
        result = packaging_utils.update_package_name("test-pkg", config)
        self.assertEqual(result, "test-pkg7.1-gfx900")

    @patch("packaging_utils.get_package_info")
    @patch("packaging_utils.debian_replace_devel_name")
    @patch("builtins.print")
    def test_update_package_name_deb_devel(self, mock_print, mock_debian_replace, mock_get_info):
        """Test package name update for debian devel package."""
        mock_get_info.return_value = {"Package": "test-pkg-devel", "Gfxarch": "False"}
        mock_debian_replace.return_value = "test-pkg-dev"
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )
        result = packaging_utils.update_package_name("test-pkg-devel", config)
        self.assertEqual(result, "test-pkg-dev7.1")

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_update_package_name_non_versioned(self, mock_print, mock_get_info):
        """Test package name update without version."""
        mock_get_info.return_value = {"Package": "test-pkg", "Gfxarch": "False"}
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=False
        )
        result = packaging_utils.update_package_name("test-pkg", config)
        self.assertEqual(result, "test-pkg")

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_update_package_name_invalid_version(self, mock_print, mock_get_info):
        """Test package name update with invalid version."""
        mock_get_info.return_value = {"Package": "test-pkg", "Gfxarch": "False"}
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )
        with self.assertRaises(ValueError):
            packaging_utils.update_package_name("test-pkg", config)


class TestDebianReplaceDevelName(unittest.TestCase):
    """Test cases for debian_replace_devel_name function."""

    @patch("builtins.print")
    def test_replace_devel_with_dev(self, mock_print):
        """Test replacing -devel with -dev."""
        result = packaging_utils.debian_replace_devel_name("amdrocm-llvm-devel")
        self.assertEqual(result, "amdrocm-llvm-dev")

    @patch("builtins.print")
    def test_no_devel_suffix(self, mock_print):
        """Test package name without -devel suffix."""
        result = packaging_utils.debian_replace_devel_name("amdrocm-llvm")
        self.assertEqual(result, "amdrocm-llvm")

    @patch("builtins.print")
    def test_devel_in_middle(self, mock_print):
        """Test package name with devel in the middle (not at end)."""
        result = packaging_utils.debian_replace_devel_name("amdrocm-devel-tools")
        self.assertEqual(result, "amdrocm-devel-tools")


class TestConvertToVersionDependency(unittest.TestCase):
    """Test cases for convert_to_versiondependency function."""

    @patch("packaging_utils.get_package_list")
    @patch("packaging_utils.update_package_name")
    @patch("builtins.print")
    def test_convert_with_rocm_packages(self, mock_print, mock_update, mock_get_list):
        """Test converting dependencies that are ROCm packages."""
        mock_get_list.return_value = ["amdrocm-llvm", "amdrocm-runtime"]
        mock_update.side_effect = lambda pkg, cfg: f"{pkg}7.1"
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )
        
        dep_list = ["amdrocm-llvm", "libc6", "amdrocm-runtime"]
        result = packaging_utils.convert_to_versiondependency(dep_list, config)
        self.assertEqual(result, "amdrocm-llvm7.1, libc6, amdrocm-runtime7.1")

    @patch("packaging_utils.get_package_list")
    @patch("builtins.print")
    def test_convert_no_rocm_packages(self, mock_print, mock_get_list):
        """Test converting dependencies with no ROCm packages."""
        mock_get_list.return_value = ["amdrocm-llvm"]
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )
        
        dep_list = ["libc6", "libstdc++"]
        result = packaging_utils.convert_to_versiondependency(dep_list, config)
        self.assertEqual(result, "libc6, libstdc++")


class TestAppendVersionSuffix(unittest.TestCase):
    """Test cases for append_version_suffix function."""

    @patch("packaging_utils.get_package_list")
    @patch("builtins.print")
    def test_append_version_rpm(self, mock_print, mock_get_list):
        """Test appending version suffix for RPM."""
        mock_get_list.return_value = ["amdrocm-llvm", "amdrocm-runtime"]
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="50",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )
        
        dep_string = "amdrocm-llvm, libc6, amdrocm-runtime"
        result = packaging_utils.append_version_suffix(dep_string, config)
        self.assertEqual(result, "amdrocm-llvm = 7.1.0-50, libc6, amdrocm-runtime = 7.1.0-50")

    @patch("packaging_utils.get_package_list")
    @patch("builtins.print")
    def test_append_version_deb(self, mock_print, mock_get_list):
        """Test appending version suffix for DEB."""
        mock_get_list.return_value = ["amdrocm-llvm"]
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="50",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )
        
        dep_string = "amdrocm-llvm, libc6"
        result = packaging_utils.append_version_suffix(dep_string, config)
        self.assertEqual(result, "amdrocm-llvm( = 7.1.0-50), libc6")

    @patch("packaging_utils.get_package_list")
    @patch("builtins.print")
    def test_append_version_no_suffix(self, mock_print, mock_get_list):
        """Test appending version with no suffix."""
        mock_get_list.return_value = ["amdrocm-llvm"]
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )
        
        dep_string = "amdrocm-llvm"
        result = packaging_utils.append_version_suffix(dep_string, config)
        self.assertEqual(result, "amdrocm-llvm = 7.1.0")


class TestMovePackagesToDestination(unittest.TestCase):
    """Test cases for move_packages_to_destination function."""

    def setUp(self):
        """Create temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.dest_dir = Path(self.temp_dir) / "dest"

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("builtins.print")
    def test_move_deb_packages(self, mock_print):
        """Test moving DEB packages."""
        # Create test structure - files should be in dest_dir/pkg_type/
        deb_dir = self.dest_dir / "deb"
        deb_dir.mkdir(parents=True)
        test_deb = deb_dir / "test-pkg_1.0_amd64.deb"
        test_deb.touch()

        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=self.dest_dir,
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )

        packaging_utils.move_packages_to_destination("test-pkg", config)
        
        # Check that destination directory exists
        self.assertTrue(self.dest_dir.exists())
        # Check that file was moved to dest_dir root
        self.assertTrue((self.dest_dir / "test-pkg_1.0_amd64.deb").exists())
        self.assertFalse(test_deb.exists())

    @patch("platform.machine", return_value="x86_64")
    @patch("builtins.print")
    def test_move_rpm_packages(self, mock_print, mock_machine):
        """Test moving RPM packages."""
        # Create test structure - RPMs are in dest_dir/pkg_type/*/RPMS/arch/
        rpm_dir = self.dest_dir / "rpm" / "BUILD" / "RPMS" / "x86_64"
        rpm_dir.mkdir(parents=True)
        test_rpm = rpm_dir / "test-pkg-1.0.x86_64.rpm"
        test_rpm.touch()

        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=self.dest_dir,
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False,
            versioned_pkg=True
        )

        packaging_utils.move_packages_to_destination("test-pkg", config)
        
        # Check that destination directory exists
        self.assertTrue(self.dest_dir.exists())
        # Check that file was moved to dest_dir root
        self.assertTrue((self.dest_dir / "test-pkg-1.0.x86_64.rpm").exists())
        self.assertFalse(test_rpm.exists())


class TestFilterComponentsFromArtifactory(unittest.TestCase):
    """Test cases for filter_components_fromartifactory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifacts_dir = Path(self.temp_dir) / "artifacts"
        self.artifacts_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_filter_components_generic(self, mock_print, mock_get_info):
        """Test filtering components for generic (non-gfxarch) package."""
        mock_get_info.return_value = {
            "Package": "test-pkg",
            "Gfxarch": "False",
            "Artifactory": [
                {
                    "Artifact": "base",
                    "Artifact_Subdir": [
                        {
                            "Name": "test-component",
                            "Components": ["lib"]
                        }
                    ]
                }
            ]
        }

        # Create artifact directory and manifest
        artifact_dir = self.artifacts_dir / "base_lib_generic"
        artifact_dir.mkdir()
        manifest_file = artifact_dir / "artifact_manifest.txt"
        manifest_file.write_text("test-component/libtest.so\n")

        result = packaging_utils.filter_components_fromartifactory(
            "test-pkg", self.artifacts_dir, "gfx900"
        )
        
        self.assertEqual(len(result), 1)
        self.assertTrue(str(result[0]).endswith("test-component/libtest.so"))

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_filter_components_gfxarch(self, mock_print, mock_get_info):
        """Test filtering components for gfxarch package."""
        mock_get_info.return_value = {
            "Package": "test-pkg",
            "Gfxarch": "True",
            "Artifactory": [
                {
                    "Artifact": "blas",
                    "Artifact_Subdir": [
                        {
                            "Name": "rocBLAS",
                            "Components": ["lib"]
                        }
                    ]
                }
            ]
        }

        # Create artifact directory and manifest
        artifact_dir = self.artifacts_dir / "blas_lib_gfx900"
        artifact_dir.mkdir()
        manifest_file = artifact_dir / "artifact_manifest.txt"
        manifest_file.write_text("rocBLAS/librocblas.so\n")

        result = packaging_utils.filter_components_fromartifactory(
            "test-pkg", self.artifacts_dir, "gfx900"
        )
        
        self.assertEqual(len(result), 1)
        self.assertTrue(str(result[0]).endswith("rocBLAS/librocblas.so"))

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_filter_components_artifact_gfxarch_override(self, mock_print, mock_get_info):
        """Test filtering with Artifact_Gfxarch override."""
        mock_get_info.return_value = {
            "Package": "test-pkg",
            "Gfxarch": "True",
            "Artifactory": [
                {
                    "Artifact": "hipdnn",
                    "Artifact_Gfxarch": "False",
                    "Artifact_Subdir": [
                        {
                            "Name": "hipDNN",
                            "Components": ["lib"]
                        }
                    ]
                }
            ]
        }

        # Create artifact directory with generic suffix
        artifact_dir = self.artifacts_dir / "hipdnn_lib_generic"
        artifact_dir.mkdir()
        manifest_file = artifact_dir / "artifact_manifest.txt"
        manifest_file.write_text("hipDNN/libhipdnn.so\n")

        result = packaging_utils.filter_components_fromartifactory(
            "test-pkg", self.artifacts_dir, "gfx900"
        )
        
        self.assertEqual(len(result), 1)
        self.assertTrue(str(result[0]).endswith("hipDNN/libhipdnn.so"))

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_filter_components_no_artifactory(self, mock_print, mock_get_info):
        """Test filtering when package has no Artifactory key (meta package)."""
        mock_get_info.return_value = {
            "Package": "meta-pkg",
            "Metapackage": "True"
        }

        result = packaging_utils.filter_components_fromartifactory(
            "meta-pkg", self.artifacts_dir, "gfx900"
        )
        
        self.assertEqual(result, [])

    @patch("packaging_utils.get_package_info")
    @patch("builtins.print")
    def test_filter_components_no_matching_lines(self, mock_print, mock_get_info):
        """Test filtering when manifest has no matching lines."""
        mock_get_info.return_value = {
            "Package": "test-pkg",
            "Gfxarch": "False",
            "Artifactory": [
                {
                    "Artifact": "base",
                    "Artifact_Subdir": [
                        {
                            "Name": "test-component",
                            "Components": ["lib"]
                        }
                    ]
                }
            ]
        }

        # Create artifact directory and manifest with non-matching content
        artifact_dir = self.artifacts_dir / "base_lib_generic"
        artifact_dir.mkdir()
        manifest_file = artifact_dir / "artifact_manifest.txt"
        manifest_file.write_text("other-component/libother.so\n")

        result = packaging_utils.filter_components_fromartifactory(
            "test-pkg", self.artifacts_dir, "gfx900"
        )
        
        self.assertEqual(result, [])


class TestPackageConfig(unittest.TestCase):
    """Test cases for PackageConfig dataclass."""

    def test_package_config_creation(self):
        """Test creating PackageConfig with all parameters."""
        config = PackageConfig(
            artifacts_dir=Path("/tmp/artifacts"),
            dest_dir=Path("/tmp/dest"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="50",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=True,
            versioned_pkg=True
        )
        
        self.assertEqual(config.artifacts_dir, Path("/tmp/artifacts"))
        self.assertEqual(config.dest_dir, Path("/tmp/dest"))
        self.assertEqual(config.pkg_type, "rpm")
        self.assertEqual(config.rocm_version, "7.1.0")
        self.assertEqual(config.version_suffix, "50")
        self.assertEqual(config.install_prefix, "/opt/rocm")
        self.assertEqual(config.gfx_arch, "gfx900")
        self.assertTrue(config.enable_rpath)
        self.assertTrue(config.versioned_pkg)

    def test_package_config_defaults(self):
        """Test PackageConfig default values."""
        config = PackageConfig(
            artifacts_dir=Path("/tmp/artifacts"),
            dest_dir=Path("/tmp/dest"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="50",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900"
        )
        
        self.assertFalse(config.enable_rpath)
        self.assertTrue(config.versioned_pkg)


if __name__ == "__main__":
    unittest.main()
