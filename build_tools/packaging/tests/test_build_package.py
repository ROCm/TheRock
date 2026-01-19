# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

# Add the linux directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "linux"))

# Import the modules to be tested
import build_package
from packaging_utils import PackageConfig


class TestCopyPackageContents(unittest.TestCase):
    """Test cases for copy_package_contents function."""

    def setUp(self):
        """Create temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.source_dir = Path(self.temp_dir) / "source"
        self.dest_dir = Path(self.temp_dir) / "dest"
        self.source_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("builtins.print")
    def test_copy_regular_files(self, mock_print):
        """Test copying regular files."""
        # Create test files
        test_file = self.source_dir / "test.txt"
        test_file.write_text("test content")
        
        build_package.copy_package_contents(self.source_dir, self.dest_dir)
        
        # Verify file was copied
        self.assertTrue((self.dest_dir / "test.txt").exists())
        self.assertEqual((self.dest_dir / "test.txt").read_text(), "test content")

    @patch("builtins.print")
    def test_copy_directories(self, mock_print):
        """Test copying directories."""
        # Create test directory with files
        test_subdir = self.source_dir / "subdir"
        test_subdir.mkdir()
        (test_subdir / "file.txt").write_text("content")
        
        build_package.copy_package_contents(self.source_dir, self.dest_dir)
        
        # Verify directory and file were copied
        self.assertTrue((self.dest_dir / "subdir").exists())
        self.assertTrue((self.dest_dir / "subdir" / "file.txt").exists())

    @patch("builtins.print")
    def test_copy_symlinks(self, mock_print):
        """Test copying symlinks."""
        # Create a file and a symlink to it
        test_file = self.source_dir / "original.txt"
        test_file.write_text("original")
        test_link = self.source_dir / "link.txt"
        test_link.symlink_to("original.txt")
        
        build_package.copy_package_contents(self.source_dir, self.dest_dir)
        
        # Verify symlink was copied
        dest_link = self.dest_dir / "link.txt"
        self.assertTrue(dest_link.is_symlink())

    @patch("builtins.print")
    def test_copy_nonexistent_source(self, mock_print):
        """Test copying from non-existent source."""
        nonexistent = Path(self.temp_dir) / "nonexistent"
        
        build_package.copy_package_contents(nonexistent, self.dest_dir)
        
        # Should print error message but not crash
        mock_print.assert_called()


class TestGenerateChangelogFile(unittest.TestCase):
    """Test cases for generate_changelog_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.deb_dir = Path(self.temp_dir) / "debian"
        self.deb_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("build_package.update_package_name")
    @patch("builtins.print")
    def test_generate_changelog_file(self, mock_print, mock_update_name):
        """Test generating changelog file."""
        mock_update_name.return_value = "test-pkg7.1"
        
        pkg_info = {
            "Package": "test-pkg",
            "Maintainer": "John Doe <john@example.com>"
        }
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="50",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900"
        )
        
        build_package.generate_changelog_file(pkg_info, self.deb_dir, config)
        
        changelog_file = self.deb_dir / "changelog"
        self.assertTrue(changelog_file.exists())
        content = changelog_file.read_text()
        self.assertIn("test-pkg7.1", content)
        self.assertIn("7.1.0-50", content)


class TestGenerateInstallFile(unittest.TestCase):
    """Test cases for generate_install_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.deb_dir = Path(self.temp_dir) / "debian"
        self.deb_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("builtins.print")
    def test_generate_install_file(self, mock_print):
        """Test generating install file."""
        pkg_info = {"Package": "test-pkg"}
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm/core",
            gfx_arch="gfx900"
        )
        
        build_package.generate_install_file(pkg_info, self.deb_dir, config)
        
        install_file = self.deb_dir / "install"
        self.assertTrue(install_file.exists())
        content = install_file.read_text()
        self.assertIn("/opt/rocm/core", content)


class TestGenerateRulesFile(unittest.TestCase):
    """Test cases for generate_rules_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.deb_dir = Path(self.temp_dir) / "debian"
        self.deb_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("build_package.update_package_name")
    @patch("build_package.is_key_defined")
    @patch("builtins.print")
    def test_generate_rules_file(self, mock_print, mock_is_key, mock_update_name):
        """Test generating rules file."""
        mock_is_key.return_value = False
        mock_update_name.return_value = "test-pkg7.1"
        
        pkg_info = {"Package": "test-pkg"}
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900"
        )
        
        build_package.generate_rules_file(pkg_info, self.deb_dir, config)
        
        rules_file = self.deb_dir / "rules"
        self.assertTrue(rules_file.exists())
        # Check file is executable
        self.assertTrue(os.access(rules_file, os.X_OK))


class TestGenerateControlFile(unittest.TestCase):
    """Test cases for generate_control_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.deb_dir = Path(self.temp_dir) / "debian"
        self.deb_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("build_package.append_version_suffix")
    @patch("build_package.is_meta_package")
    @patch("build_package.convert_to_versiondependency")
    @patch("build_package.update_package_name")
    @patch("builtins.print")
    def test_generate_control_file_versioned(self, mock_print, mock_update, mock_convert, 
                                            mock_is_meta, mock_append):
        """Test generating control file for versioned package."""
        mock_update.return_value = "test-pkg7.1"
        mock_convert.return_value = "libc6, libstdc++6"
        mock_is_meta.return_value = False
        
        pkg_info = {
            "Package": "test-pkg",
            "Architecture": "amd64",
            "Maintainer": "Test <test@example.com>",
            "Description_Short": "Test package",
            "Description_Long": "Test package description",
            "Homepage": "https://example.com",
            "Priority": "optional",
            "Section": "devel",
            "DEBDepends": ["libc6"]
        }
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            versioned_pkg=True
        )
        
        build_package.generate_control_file(pkg_info, self.deb_dir, config)
        
        control_file = self.deb_dir / "control"
        self.assertTrue(control_file.exists())
        content = control_file.read_text()
        self.assertIn("test-pkg7.1", content)


class TestPackageWithDpkgBuild(unittest.TestCase):
    """Test cases for package_with_dpkg_build function."""

    @patch("os.chdir")
    @patch("pathlib.Path.cwd", return_value=Path("/tmp"))
    @patch("subprocess.run")
    @patch("builtins.print")
    def test_package_with_dpkg_build_success(self, mock_print, mock_run, mock_cwd, mock_chdir):
        """Test successful dpkg-buildpackage execution."""
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pkg_dir = Path(temp_dir) / "package"
            pkg_dir.mkdir()
            
            build_package.package_with_dpkg_build(pkg_dir)
            
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            self.assertEqual(args[0], "dpkg-buildpackage")

    @patch("os.chdir")
    @patch("pathlib.Path.cwd", return_value=Path("/tmp"))
    @patch("subprocess.run")
    @patch("builtins.print")
    def test_package_with_dpkg_build_failure(self, mock_print, mock_run, mock_cwd, mock_chdir):
        """Test dpkg-buildpackage failure handling."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "dpkg-buildpackage")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pkg_dir = Path(temp_dir) / "package"
            pkg_dir.mkdir()
            
            with self.assertRaises(SystemExit):
                build_package.package_with_dpkg_build(pkg_dir)


class TestGenerateSpecFile(unittest.TestCase):
    """Test cases for generate_spec_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("build_package.filter_components_fromartifactory")
    @patch("build_package.append_version_suffix")
    @patch("build_package.is_meta_package")
    @patch("build_package.convert_to_versiondependency")
    @patch("build_package.update_package_name")
    @patch("build_package.get_package_info")
    @patch("build_package.is_rpm_stripping_disabled")
    @patch("build_package.is_debug_package_disabled")
    @patch("builtins.print")
    def test_generate_spec_file_versioned(self, mock_print, mock_debug, mock_strip,
                                          mock_get_info, mock_update, mock_convert,
                                          mock_is_meta, mock_append, mock_filter):
        """Test generating spec file for versioned package."""
        mock_get_info.return_value = {
            "Package": "test-pkg",
            "BuildArch": "x86_64",
            "Description_Short": "Test",
            "Description_Long": "Test desc",
            "Group": "Development",
            "License": "MIT",
            "Vendor": "AMD",
            "RPMRequires": ["glibc"]
        }
        mock_update.return_value = "test-pkg7.1"
        mock_convert.return_value = "glibc"
        mock_is_meta.return_value = False
        mock_filter.return_value = []
        mock_strip.return_value = False
        mock_debug.return_value = False
        
        specfile = Path(self.temp_dir) / "test.spec"
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="50",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            versioned_pkg=True
        )
        
        build_package.generate_spec_file("test-pkg", specfile, config)
        
        self.assertTrue(specfile.exists())
        content = specfile.read_text()
        self.assertIn("test-pkg7.1", content)


class TestPackageWithRpmbuild(unittest.TestCase):
    """Test cases for package_with_rpmbuild function."""

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_package_with_rpmbuild_success(self, mock_print, mock_run):
        """Test successful rpmbuild execution."""
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_file = Path(temp_dir) / "test.spec"
            spec_file.touch()
            
            build_package.package_with_rpmbuild(spec_file)
            
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            self.assertEqual(args[0], "rpmbuild")

    @patch("subprocess.run")
    @patch("builtins.print")
    def test_package_with_rpmbuild_failure(self, mock_print, mock_run):
        """Test rpmbuild failure handling."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "rpmbuild")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_file = Path(temp_dir) / "test.spec"
            spec_file.touch()
            
            with self.assertRaises(SystemExit):
                build_package.package_with_rpmbuild(spec_file)


class TestParseInputPackageList(unittest.TestCase):
    """Test cases for parse_input_package_list function."""

    @patch("build_package.get_package_list")
    @patch("builtins.print")
    def test_parse_none_returns_all_packages(self, mock_print, mock_get_list):
        """Test that None input returns all packages."""
        mock_get_list.return_value = ["pkg1", "pkg2", "pkg3"]
        
        result = build_package.parse_input_package_list(None)
        
        self.assertEqual(result, ["pkg1", "pkg2", "pkg3"])

    @patch("build_package.read_package_json_file")
    @patch("build_package.is_packaging_disabled")
    @patch("builtins.print")
    def test_parse_specific_packages(self, mock_print, mock_disabled, mock_read):
        """Test parsing specific package names."""
        mock_read.return_value = [
            {"Package": "pkg1"},
            {"Package": "pkg2"},
            {"Package": "pkg3"}
        ]
        mock_disabled.return_value = False
        
        result = build_package.parse_input_package_list(["pkg1", "pkg3"])
        
        self.assertEqual(result, ["pkg1", "pkg3"])

    @patch("build_package.read_package_json_file")
    @patch("build_package.is_packaging_disabled")
    @patch("builtins.print")
    def test_parse_filters_disabled_packages(self, mock_print, mock_disabled, mock_read):
        """Test that disabled packages are filtered out."""
        mock_read.return_value = [
            {"Package": "pkg1"},
            {"Package": "pkg2"}
        ]
        mock_disabled.side_effect = [False, True]
        
        result = build_package.parse_input_package_list(["pkg1", "pkg2"])
        
        self.assertEqual(result, ["pkg1"])


class TestCleanPackageBuildDir(unittest.TestCase):
    """Test cases for clean_package_build_dir function."""

    @patch("build_package.remove_dir")
    @patch("builtins.print")
    def test_clean_package_build_dir(self, mock_print, mock_remove):
        """Test cleaning package build directories."""
        config = PackageConfig(
            artifacts_dir=Path("/tmp/artifacts"),
            dest_dir=Path("/tmp/dest"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900"
        )
        
        build_package.clean_package_build_dir(config)
        
        # Should be called at least once (for pkg_type directory)
        self.assertTrue(mock_remove.called)


class TestCreateDebPackage(unittest.TestCase):
    """Test cases for create_deb_package function."""

    @patch("build_package.remove_dir")
    @patch("build_package.move_packages_to_destination")
    @patch("build_package.create_versioned_deb_package")
    @patch("build_package.create_nonversioned_deb_package")
    @patch("builtins.print")
    def test_create_deb_package_with_rpath(self, mock_print, mock_nonver, 
                                           mock_ver, mock_move, mock_remove):
        """Test creating DEB package with rpath enabled (skip non-versioned)."""
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=True
        )
        
        build_package.create_deb_package("test-pkg", config)
        
        # Non-versioned should not be called when rpath is enabled
        mock_nonver.assert_not_called()
        mock_ver.assert_called_once()
        mock_move.assert_called_once()

    @patch("build_package.remove_dir")
    @patch("build_package.move_packages_to_destination")
    @patch("build_package.create_versioned_deb_package")
    @patch("build_package.create_nonversioned_deb_package")
    @patch("builtins.print")
    def test_create_deb_package_without_rpath(self, mock_print, mock_nonver,
                                              mock_ver, mock_move, mock_remove):
        """Test creating DEB package without rpath (both versioned and non-versioned)."""
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False
        )
        
        build_package.create_deb_package("test-pkg", config)
        
        # Both should be called when rpath is disabled
        mock_nonver.assert_called_once()
        mock_ver.assert_called_once()
        mock_move.assert_called_once()


class TestCreateRpmPackage(unittest.TestCase):
    """Test cases for create_rpm_package function."""

    @patch("build_package.remove_dir")
    @patch("build_package.move_packages_to_destination")
    @patch("build_package.create_versioned_rpm_package")
    @patch("build_package.create_nonversioned_rpm_package")
    @patch("builtins.print")
    def test_create_rpm_package_with_rpath(self, mock_print, mock_nonver,
                                           mock_ver, mock_move, mock_remove):
        """Test creating RPM package with rpath enabled."""
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=True
        )
        
        build_package.create_rpm_package("test-pkg", config)
        
        mock_nonver.assert_not_called()
        mock_ver.assert_called_once()
        mock_move.assert_called_once()

    @patch("build_package.remove_dir")
    @patch("build_package.move_packages_to_destination")
    @patch("build_package.create_versioned_rpm_package")
    @patch("build_package.create_nonversioned_rpm_package")
    @patch("builtins.print")
    def test_create_rpm_package_without_rpath(self, mock_print, mock_nonver,
                                              mock_ver, mock_move, mock_remove):
        """Test creating RPM package without rpath."""
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=False
        )
        
        build_package.create_rpm_package("test-pkg", config)
        
        mock_nonver.assert_called_once()
        mock_ver.assert_called_once()
        mock_move.assert_called_once()


class TestRun(unittest.TestCase):
    """Test cases for run function."""

    @patch("build_package.clean_package_build_dir")
    @patch("build_package.parse_input_package_list")
    @patch("build_package.create_deb_package")
    def test_run_deb_packages(self, mock_create_deb, mock_parse, mock_clean):
        """Test run function for DEB packages."""
        mock_parse.return_value = ["pkg1", "pkg2"]
        
        args = argparse.Namespace(
            artifacts_dir="/tmp/artifacts",
            dest_dir="/tmp/dest",
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="50",
            install_prefix="/opt/rocm/core",
            target="gfx900",
            rpath_pkg=False,
            pkg_names=None
        )
        
        build_package.run(args)
        
        # Should be called twice for two packages
        self.assertEqual(mock_create_deb.call_count, 2)

    @patch("build_package.clean_package_build_dir")
    @patch("build_package.parse_input_package_list")
    @patch("build_package.create_rpm_package")
    def test_run_rpm_packages(self, mock_create_rpm, mock_parse, mock_clean):
        """Test run function for RPM packages."""
        mock_parse.return_value = ["pkg1"]
        
        args = argparse.Namespace(
            artifacts_dir="/tmp/artifacts",
            dest_dir="/tmp/dest",
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="50",
            install_prefix="/opt/rocm/core",
            target="gfx900",
            rpath_pkg=False,
            pkg_names=None
        )
        
        build_package.run(args)
        
        mock_create_rpm.assert_called_once()

    def test_run_invalid_version(self):
        """Test run function with invalid version format."""
        args = argparse.Namespace(
            artifacts_dir="/tmp/artifacts",
            dest_dir="/tmp/dest",
            pkg_type="deb",
            rocm_version="7",  # Invalid: missing minor version
            version_suffix="50",
            install_prefix="/opt/rocm/core",
            target="gfx900",
            rpath_pkg=False,
            pkg_names=None
        )
        
        with self.assertRaises(ValueError):
            build_package.run(args)


class TestMain(unittest.TestCase):
    """Test cases for main function."""

    @patch("build_package.run")
    def test_main_parses_arguments(self, mock_run):
        """Test that main parses arguments correctly."""
        argv = [
            "--artifacts-dir", "/tmp/artifacts",
            "--dest-dir", "/tmp/dest",
            "--target", "gfx900",
            "--pkg-type", "deb",
            "--rocm-version", "7.1.0"
        ]
        
        build_package.main(argv)
        
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args.rocm_version, "7.1.0")
        self.assertEqual(args.pkg_type, "deb")
        self.assertEqual(args.target, "gfx900")


class TestCreateNonversionedDebPackage(unittest.TestCase):
    """Test cases for create_nonversioned_deb_package function."""

    @patch("build_package.package_with_dpkg_build")
    @patch("build_package.generate_control_file")
    @patch("build_package.generate_rules_file")
    @patch("build_package.generate_changelog_file")
    @patch("build_package.get_package_info")
    @patch("os.makedirs")
    @patch("builtins.print")
    def test_create_nonversioned_deb_package(self, mock_print, mock_makedirs,
                                            mock_get_info, mock_changelog,
                                            mock_rules, mock_control, mock_dpkg):
        """Test creating non-versioned DEB package."""
        mock_get_info.return_value = {"Package": "test-pkg"}
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            versioned_pkg=True
        )
        
        build_package.create_nonversioned_deb_package("test-pkg", config)
        
        # Verify all generation functions were called
        mock_get_info.assert_called_once_with("test-pkg")
        mock_changelog.assert_called_once()
        mock_rules.assert_called_once()
        mock_control.assert_called_once()
        mock_dpkg.assert_called_once()
        # Verify versioned_pkg was reset to True
        self.assertTrue(config.versioned_pkg)


class TestCreateVersionedDebPackage(unittest.TestCase):
    """Test cases for create_versioned_deb_package function."""

    @patch("build_package.convert_runpath_to_rpath")
    @patch("build_package.package_with_dpkg_build")
    @patch("build_package.copy_package_contents")
    @patch("build_package.generate_install_file")
    @patch("build_package.generate_debian_postscripts")
    @patch("build_package.filter_components_fromartifactory")
    @patch("build_package.generate_control_file")
    @patch("build_package.generate_rules_file")
    @patch("build_package.generate_changelog_file")
    @patch("build_package.is_postinstallscripts_available")
    @patch("build_package.is_meta_package")
    @patch("build_package.get_package_info")
    @patch("os.makedirs")
    @patch("builtins.print")
    def test_create_versioned_deb_with_artifacts(self, mock_print, mock_makedirs,
                                                 mock_get_info, mock_is_meta,
                                                 mock_postinstall, mock_changelog,
                                                 mock_rules, mock_control,
                                                 mock_filter, mock_postscripts,
                                                 mock_install, mock_copy, mock_dpkg,
                                                 mock_rpath):
        """Test creating versioned DEB package with artifacts and RPATH."""
        mock_get_info.return_value = {"Package": "test-pkg"}
        mock_is_meta.return_value = False
        mock_postinstall.return_value = True
        mock_filter.return_value = [Path("/tmp/artifact1")]
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp/artifacts"),
            dest_dir=Path("/tmp/dest"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            enable_rpath=True
        )
        
        build_package.create_versioned_deb_package("test-pkg", config)
        
        # Verify post-install scripts were generated
        mock_postscripts.assert_called_once()
        # Verify install file was generated
        mock_install.assert_called_once()
        # Verify contents were copied
        mock_copy.assert_called_once()
        # Verify rpath conversion was called
        mock_rpath.assert_called_once()
        # Verify dpkg-build was called
        mock_dpkg.assert_called_once()

    @patch("build_package.package_with_dpkg_build")
    @patch("build_package.filter_components_fromartifactory")
    @patch("build_package.generate_control_file")
    @patch("build_package.generate_rules_file")
    @patch("build_package.generate_changelog_file")
    @patch("build_package.is_postinstallscripts_available")
    @patch("build_package.is_meta_package")
    @patch("build_package.get_package_info")
    @patch("os.makedirs")
    @patch("builtins.print")
    def test_create_versioned_deb_meta_package(self, mock_print, mock_makedirs,
                                               mock_get_info, mock_is_meta,
                                               mock_postinstall, mock_changelog,
                                               mock_rules, mock_control,
                                               mock_filter, mock_dpkg):
        """Test creating versioned DEB meta package (no artifacts)."""
        mock_get_info.return_value = {"Package": "meta-pkg", "Metapackage": "True"}
        mock_is_meta.return_value = True
        mock_postinstall.return_value = False
        mock_filter.return_value = []
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp/artifacts"),
            dest_dir=Path("/tmp/dest"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900"
        )
        
        build_package.create_versioned_deb_package("meta-pkg", config)
        
        # Verify dpkg-build was called even for meta package
        mock_dpkg.assert_called_once()

    @patch("build_package.package_with_dpkg_build")
    @patch("build_package.filter_components_fromartifactory")
    @patch("build_package.generate_control_file")
    @patch("build_package.generate_rules_file")
    @patch("build_package.generate_changelog_file")
    @patch("build_package.is_postinstallscripts_available")
    @patch("build_package.is_meta_package")
    @patch("build_package.get_package_info")
    @patch("os.makedirs")
    @patch("builtins.print")
    def test_create_versioned_deb_empty_nonmeta_exits(self, mock_print, mock_makedirs,
                                                      mock_get_info, mock_is_meta,
                                                      mock_postinstall, mock_changelog,
                                                      mock_rules, mock_control,
                                                      mock_filter, mock_dpkg):
        """Test that empty sourcedir for non-meta package causes exit."""
        mock_get_info.return_value = {"Package": "test-pkg"}
        mock_is_meta.return_value = False
        mock_postinstall.return_value = False
        mock_filter.return_value = []  # Empty list
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp/artifacts"),
            dest_dir=Path("/tmp/dest"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900"
        )
        
        # This should call sys.exit internally
        with self.assertRaises(SystemExit):
            build_package.create_versioned_deb_package("test-pkg", config)


class TestGenerateDebianPostscripts(unittest.TestCase):
    """Test cases for generate_debian_postscripts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.deb_dir = Path(self.temp_dir) / "debian"
        self.deb_dir.mkdir(parents=True)
        
        # Create a mock template directory structure
        self.template_dir = Path(self.temp_dir) / "template" / "scripts"
        self.template_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("build_package.SCRIPT_DIR")
    @patch("builtins.print")
    def test_generate_debian_postscripts(self, mock_print, mock_script_dir):
        """Test generating Debian post-installation scripts."""
        # Create mock template files
        (self.template_dir / "amdrocm-core-postinst.j2").write_text(
            "#!/bin/bash\n# Post install for version {{ version_major }}.{{ version_minor }}"
        )
        (self.template_dir / "amdrocm-core-prerm.j2").write_text(
            "#!/bin/bash\n# Pre remove"
        )
        
        pkg_info = {"Package": "amdrocm-core"}
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900"
        )
        
        with patch("build_package.SCRIPT_DIR", Path(self.template_dir.parent)):
            build_package.generate_debian_postscripts(pkg_info, self.deb_dir, config)
        
        # Verify scripts were created if templates exist
        postinst = self.deb_dir / "postinst"
        prerm = self.deb_dir / "prerm"
        
        if postinst.exists():
            self.assertTrue(os.access(postinst, os.X_OK))
            content = postinst.read_text()
            self.assertIn("7.1", content)


class TestCreateNonversionedRpmPackage(unittest.TestCase):
    """Test cases for create_nonversioned_rpm_package function."""

    @patch("build_package.package_with_rpmbuild")
    @patch("build_package.generate_spec_file")
    @patch("builtins.print")
    def test_create_nonversioned_rpm_package(self, mock_print, mock_spec, mock_rpmbuild):
        """Test creating non-versioned RPM package."""
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            versioned_pkg=True
        )
        
        build_package.create_nonversioned_rpm_package("test-pkg", config)
        
        # Verify spec file generation was called
        mock_spec.assert_called_once()
        # Verify rpmbuild was called
        mock_rpmbuild.assert_called_once()
        # Verify config.versioned_pkg was reset to True
        self.assertTrue(config.versioned_pkg)


class TestCreateVersionedRpmPackage(unittest.TestCase):
    """Test cases for create_versioned_rpm_package function."""

    @patch("build_package.package_with_rpmbuild")
    @patch("build_package.generate_spec_file")
    @patch("builtins.print")
    def test_create_versioned_rpm_package(self, mock_print, mock_spec, mock_rpmbuild):
        """Test creating versioned RPM package."""
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900"
        )
        
        build_package.create_versioned_rpm_package("test-pkg", config)
        
        # Verify spec file generation was called
        mock_spec.assert_called_once()
        # Verify rpmbuild was called
        mock_rpmbuild.assert_called_once()
        # Verify versioned_pkg flag is True
        self.assertTrue(config.versioned_pkg)


class TestGenerateRpmPostscripts(unittest.TestCase):
    """Test cases for generate_rpm_postscripts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.template_dir = Path(self.temp_dir) / "template" / "scripts"
        self.template_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("build_package.SCRIPT_DIR")
    def test_generate_rpm_postscripts(self, mock_script_dir):
        """Test generating RPM post-installation scripts."""
        # Create mock template files
        (self.template_dir / "amdrocm-postinst.j2").write_text(
            "# RPM post install\necho 'Version {{ version_major }}.{{ version_minor }}'"
        )
        (self.template_dir / "amdrocm-prerm.j2").write_text(
            "# RPM pre remove"
        )
        
        pkg_info = {"Package": "amdrocm"}
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="rpm",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900"
        )
        
        with patch("build_package.SCRIPT_DIR", Path(self.template_dir.parent)):
            result = build_package.generate_rpm_postscripts(pkg_info, config)
        
        # Verify result is a dictionary
        self.assertIsInstance(result, dict)


class TestGenerateControlFileProvides(unittest.TestCase):
    """Test cases for generate_control_file with Provides/Replaces/Conflicts."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.deb_dir = Path(self.temp_dir) / "debian"
        self.deb_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("build_package.append_version_suffix")
    @patch("build_package.is_meta_package")
    @patch("build_package.convert_to_versiondependency")
    @patch("build_package.update_package_name")
    @patch("builtins.print")
    def test_generate_control_file_with_provides(self, mock_print, mock_update,
                                                 mock_convert, mock_is_meta, mock_append):
        """Test generating control file with Provides/Replaces/Conflicts."""
        mock_update.return_value = "test-pkg7.1"
        mock_convert.return_value = "libc6"
        mock_is_meta.return_value = False
        
        pkg_info = {
            "Package": "test-pkg",
            "Architecture": "amd64",
            "Maintainer": "Test <test@example.com>",
            "Description_Short": "Test package",
            "Description_Long": "Test package description",
            "Provides": ["test-pkg-old"],
            "Replaces": ["test-pkg-old"],
            "Conflicts": ["test-pkg-bad"],
            "DEBDepends": ["libc6"]
        }
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            versioned_pkg=False
        )
        
        build_package.generate_control_file(pkg_info, self.deb_dir, config)
        
        control_file = self.deb_dir / "control"
        self.assertTrue(control_file.exists())
        content = control_file.read_text()
        self.assertIn("test-pkg7.1", content)

    @patch("build_package.append_version_suffix")
    @patch("build_package.is_meta_package")
    @patch("build_package.convert_to_versiondependency")
    @patch("build_package.update_package_name")
    @patch("builtins.print")
    def test_generate_control_file_meta_package(self, mock_print, mock_update,
                                                mock_convert, mock_is_meta, mock_append):
        """Test generating control file for meta package (appends version suffix)."""
        mock_update.return_value = "meta-pkg7.1"
        mock_convert.return_value = "dep1, dep2"
        mock_is_meta.return_value = True
        mock_append.return_value = "dep1 (= 7.1.0), dep2 (= 7.1.0)"
        
        pkg_info = {
            "Package": "meta-pkg",
            "Architecture": "all",
            "Maintainer": "Test <test@example.com>",
            "Description_Short": "Meta package",
            "Description_Long": "Meta package description",
            "Metapackage": "True",
            "DEBDepends": ["dep1", "dep2"]
        }
        
        config = PackageConfig(
            artifacts_dir=Path("/tmp"),
            dest_dir=Path("/tmp"),
            pkg_type="deb",
            rocm_version="7.1.0",
            version_suffix="",
            install_prefix="/opt/rocm",
            gfx_arch="gfx900",
            versioned_pkg=True
        )
        
        build_package.generate_control_file(pkg_info, self.deb_dir, config)
        
        # Verify append_version_suffix was called for meta package
        mock_append.assert_called_once()


class TestRunBothPackageTypes(unittest.TestCase):
    """Test cases for run function with both package types."""

    @patch("build_package.clean_package_build_dir")
    @patch("build_package.parse_input_package_list")
    @patch("build_package.create_deb_package")
    @patch("build_package.create_rpm_package")
    def test_run_without_pkg_type(self, mock_create_rpm, mock_create_deb,
                                  mock_parse, mock_clean):
        """Test run function without pkg_type (creates both DEB and RPM)."""
        mock_parse.return_value = ["pkg1"]
        
        args = argparse.Namespace(
            artifacts_dir="/tmp/artifacts",
            dest_dir="/tmp/dest",
            pkg_type=None,  # No package type specified
            rocm_version="7.1.0",
            version_suffix="50",
            install_prefix="/opt/rocm/core",
            target="gfx900",
            rpath_pkg=False,
            pkg_names=None
        )
        
        build_package.run(args)
        
        # Both DEB and RPM should be created
        mock_create_deb.assert_called_once()
        mock_create_rpm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
