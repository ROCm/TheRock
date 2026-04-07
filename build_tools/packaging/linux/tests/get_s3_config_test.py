#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path to import the module
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import get_s3_config


class GeneratePackageRepositoryUrlTest(unittest.TestCase):
    """Tests for package repository URL generation."""

    def test_ci_url(self):
        """Test CI build URL format."""
        url = get_s3_config.generate_package_repository_url(
            release_type="ci",
            pkg_type="deb",
            yyyymmdd="20260320",
            artifact_id="12345678",
            platform="linux",
        )
        self.assertEqual(
            url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345678-linux/packages/deb",
        )

    def test_nightly_url(self):
        """Test nightly build URL format (RPM includes /x86_64/)."""
        url = get_s3_config.generate_package_repository_url(
            release_type="nightly",
            pkg_type="rpm",
            yyyymmdd="20260320",
            artifact_id="87654321",
        )
        self.assertEqual(
            url, "https://rocm.nightlies.amd.com/rpm/20260320-87654321/x86_64/"
        )

    def test_prerelease_url(self):
        """Test prerelease URL format with default OS profile (ubuntu2404 for deb)."""
        url = get_s3_config.generate_package_repository_url(
            release_type="prerelease",
            pkg_type="deb",
            yyyymmdd="20260320",
            artifact_id="12345678",
        )
        self.assertEqual(url, "https://rocm.prereleases.amd.com/packages/ubuntu2404")

    def test_release_url(self):
        """Test release URL format with default OS profile (rhel10 for rpm includes /x86_64/)."""
        url = get_s3_config.generate_package_repository_url(
            release_type="release",
            pkg_type="rpm",
            yyyymmdd="20260320",
            artifact_id="12345678",
        )
        self.assertEqual(url, "https://repo.amd.com/rocm/packages/rhel10/x86_64/")

    def test_dev_url(self):
        """Test dev build URL format."""
        url = get_s3_config.generate_package_repository_url(
            release_type="dev",
            pkg_type="deb",
            yyyymmdd="20260320",
            artifact_id="11111111",
        )
        self.assertEqual(
            url,
            "https://rocm.devreleases.amd.com/deb/20260320-11111111",
        )

    def test_empty_release_type_uses_ci(self):
        """Test that empty release type falls back to CI URL."""
        url = get_s3_config.generate_package_repository_url(
            release_type="",
            pkg_type="deb",
            yyyymmdd="20260320",
            artifact_id="99999999",
            platform="linux",
        )
        self.assertEqual(
            url,
            "https://therock-ci-artifacts.s3.amazonaws.com/99999999-linux/packages/deb",
        )

    def test_external_repo_url(self):
        """Test external repository URL includes repository name in path."""
        url = get_s3_config.generate_package_repository_url(
            release_type="ci",
            pkg_type="deb",
            yyyymmdd="20260320",
            artifact_id="12345678",
            platform="linux",
            s3_bucket="therock-ci-artifacts-external",
            repository="someone/fork",
        )
        self.assertEqual(
            url,
            "https://therock-ci-artifacts-external.s3.amazonaws.com/someone-fork/12345678-linux/packages/deb",
        )


class ExtractDateFromVersionTest(unittest.TestCase):
    """Tests for date extraction from ROCm package versions."""

    def test_deb_dev_version(self):
        """Test extracting date from Debian dev version."""
        date = get_s3_config.extract_date_from_version("8.1.0~dev20251203")
        self.assertEqual(date, "20251203")

    def test_deb_nightly_version(self):
        """Test extracting date from Debian nightly version."""
        date = get_s3_config.extract_date_from_version("8.1.0~20251203")
        self.assertEqual(date, "20251203")

    def test_rpm_dev_version(self):
        """Test extracting date from RPM dev version with git SHA."""
        date = get_s3_config.extract_date_from_version("8.1.0~20251203gf689a8e")
        self.assertEqual(date, "20251203")

    def test_wheel_nightly_version(self):
        """Test extracting date from wheel nightly (alpha) version."""
        date = get_s3_config.extract_date_from_version("7.10.0a20251021")
        self.assertEqual(date, "20251021")

    @patch("get_s3_config.datetime")
    def test_version_without_date_uses_current(self, mock_datetime):
        """Test fallback to current date when version has no date."""
        mock_now = mock_datetime.now.return_value
        mock_now.strftime.return_value = "20260312"

        date = get_s3_config.extract_date_from_version("8.1.0")
        self.assertEqual(date, "20260312")
        mock_now.strftime.assert_called_once_with("%Y%m%d")

    def test_prerelease_version_without_date(self):
        """Test prerelease version without date falls back to current date."""
        date = get_s3_config.extract_date_from_version("8.1.0~pre2")
        # Should be 8 digits (YYYYMMDD)
        self.assertEqual(len(date), 8)
        self.assertTrue(date.isdigit())

    def test_release_version_without_date(self):
        """Test release version without date falls back to current date."""
        date = get_s3_config.extract_date_from_version("8.1.0")
        # Should be 8 digits (YYYYMMDD)
        self.assertEqual(len(date), 8)
        self.assertTrue(date.isdigit())


class DetermineS3ConfigReleaseTypeTest(unittest.TestCase):
    """Tests for S3 config with different release types."""

    def test_dev_release_type(self):
        """Test dev release type uses dev-packages bucket."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="dev",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertEqual(bucket, "therock-dev-packages")
        self.assertEqual(prefix, "deb/20251203-12345678")
        self.assertEqual(job_type, "dev")

    def test_nightly_release_type(self):
        """Test nightly release type uses nightly-packages bucket."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="nightly",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="87654321",
            rocm_version="8.1.0~20251203",
        )
        self.assertEqual(bucket, "therock-nightly-packages")
        self.assertEqual(prefix, "rpm/20251203-87654321")
        self.assertEqual(job_type, "nightly")

    def test_prerelease_release_type(self):
        """Test prerelease uses prerelease-packages bucket without date."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="prerelease",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~pre2",
        )
        self.assertEqual(bucket, "therock-prerelease-packages")
        self.assertEqual(prefix, "v3/packages/deb")
        self.assertEqual(job_type, "prerelease")

    def test_release_release_type(self):
        """Test release uses release-packages bucket without date."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="release",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version="8.1.0",
        )
        self.assertEqual(bucket, "therock-release-packages")
        self.assertEqual(prefix, "v3/packages/rpm")
        self.assertEqual(job_type, "release")

    def test_ci_release_type(self):
        """Test 'ci' release type falls through to CI bucket logic."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="ci",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=None,
        )
        self.assertEqual(bucket, "therock-ci-artifacts")
        self.assertEqual(prefix, "12345678-linux/packages/deb")
        self.assertEqual(job_type, "ci")

    def test_empty_release_type(self):
        """Test empty release type falls through to CI bucket logic."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=None,
        )
        self.assertEqual(bucket, "therock-ci-artifacts")
        self.assertEqual(prefix, "12345678-linux/packages/deb")
        self.assertEqual(job_type, "ci")


class DetermineS3ConfigRepositoryTest(unittest.TestCase):
    """Tests for S3 config with different repositories."""

    def test_fork_pr(self):
        """Test fork PR uses external bucket with repository prefix."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="",
            repository="ROCm/TheRock",
            is_fork=True,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertEqual(bucket, "therock-ci-artifacts-external")
        self.assertEqual(prefix, "ROCm-TheRock/12345678-linux/packages/rpm")
        self.assertEqual(job_type, "ci")

    def test_external_repository(self):
        """Test external repository uses external bucket with repository prefix."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="",
            repository="someone/fork",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertEqual(bucket, "therock-ci-artifacts-external")
        self.assertEqual(prefix, "someone-fork/12345678-linux/packages/deb")
        self.assertEqual(job_type, "ci")

    def test_default_rocm_therock(self):
        """Test default ROCm/TheRock uses ci-artifacts bucket."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertEqual(bucket, "therock-ci-artifacts")
        self.assertEqual(prefix, "12345678-linux/packages/deb")
        self.assertEqual(job_type, "ci")


class DetermineS3ConfigPackageTypeTest(unittest.TestCase):
    """Tests for S3 config with different package types."""

    def test_deb_package_type(self):
        """Test deb package type in prefix."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="dev",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertIn("deb", prefix)

    def test_rpm_package_type(self):
        """Test rpm package type in prefix."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="dev",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version="8.1.0~20251203gf689a8e",
        )
        self.assertIn("rpm", prefix)


class DetermineS3ConfigDateConsistencyTest(unittest.TestCase):
    """Tests to ensure date consistency between version and S3 path."""

    def test_date_extracted_from_deb_version(self):
        """Test date in S3 path matches date in deb version."""
        version = "8.1.0~dev20251203"
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="dev",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=version,
        )
        # Date from version should be in the prefix
        self.assertIn("20251203", prefix)

    def test_date_extracted_from_rpm_version(self):
        """Test date in S3 path matches date in rpm version."""
        version = "8.1.0~20251203gf689a8e"
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="dev",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version=version,
        )
        # Date from version should be in the prefix
        self.assertIn("20251203", prefix)

    def test_date_extracted_from_wheel_version(self):
        """Test date in S3 path matches date in wheel version."""
        version = "7.10.0a20251021"
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="nightly",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=version,
        )
        # Date from version should be in the prefix
        self.assertIn("20251021", prefix)

    @patch("get_s3_config.datetime")
    def test_fallback_to_current_date_when_no_version(self, mock_datetime):
        """Test fallback to current date when version is not provided."""
        mock_now = mock_datetime.now.return_value
        mock_now.strftime.return_value = "20260312"

        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="dev",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=None,
        )
        # Should use current date
        self.assertIn("20260312", prefix)


class PlatformValidationTest(unittest.TestCase):
    """Tests for platform validation with native packages."""

    def test_deb_on_windows_raises_error(self):
        """Test that deb packages on Windows platform raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_s3_config.determine_s3_config(
                release_type="ci",
                repository="ROCm/TheRock",
                is_fork=False,
                pkg_type="deb",
                artifact_id="12345678",
                rocm_version=None,
                platform="windows",
            )
        self.assertIn("deb", str(context.exception))
        self.assertIn("Linux", str(context.exception))
        self.assertIn("windows", str(context.exception))

    def test_rpm_on_windows_raises_error(self):
        """Test that rpm packages on Windows platform raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_s3_config.determine_s3_config(
                release_type="dev",
                repository="ROCm/TheRock",
                is_fork=False,
                pkg_type="rpm",
                artifact_id="12345678",
                rocm_version="8.1.0~dev20251203",
                platform="windows",
            )
        self.assertIn("rpm", str(context.exception))
        self.assertIn("Linux", str(context.exception))
        self.assertIn("windows", str(context.exception))

    def test_deb_on_linux_succeeds(self):
        """Test that deb packages on Linux platform works correctly."""
        bucket, prefix, job_type, _ = get_s3_config.determine_s3_config(
            release_type="ci",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=None,
            platform="linux",
        )
        self.assertEqual(bucket, "therock-ci-artifacts")
        self.assertEqual(prefix, "12345678-linux/packages/deb")
        self.assertEqual(job_type, "ci")


class PackageRepositoryUrlRpmArchTest(unittest.TestCase):
    """Tests to ensure RPM URLs include /x86_64/ suffix."""

    def test_nightly_rpm_includes_x86_64(self):
        """Test nightly RPM URL includes /x86_64/ suffix."""
        _, _, _, repo_url = get_s3_config.determine_s3_config(
            release_type="nightly",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version="8.1.0~20251203",
        )
        self.assertTrue(repo_url.endswith("/x86_64/"))
        self.assertIn("rpm/20251203-12345678/x86_64/", repo_url)

    def test_nightly_deb_no_x86_64(self):
        """Test nightly DEB URL does not include /x86_64/."""
        _, _, _, repo_url = get_s3_config.determine_s3_config(
            release_type="nightly",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~20251203",
        )
        self.assertNotIn("/x86_64/", repo_url)
        self.assertIn("deb/20251203-12345678", repo_url)

    def test_dev_rpm_includes_x86_64(self):
        """Test dev RPM URL includes /x86_64/ suffix."""
        _, _, _, repo_url = get_s3_config.determine_s3_config(
            release_type="dev",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertTrue(repo_url.endswith("/x86_64/"))
        self.assertIn("rpm/20251203-12345678/x86_64/", repo_url)

    def test_ci_rpm_includes_x86_64(self):
        """Test CI RPM URL includes /x86_64/ suffix."""
        _, _, _, repo_url = get_s3_config.determine_s3_config(
            release_type="ci",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version=None,
        )
        self.assertTrue(repo_url.endswith("/x86_64/"))
        self.assertIn("packages/rpm/x86_64/", repo_url)

    def test_ci_deb_no_x86_64(self):
        """Test CI DEB URL does not include /x86_64/."""
        _, _, _, repo_url = get_s3_config.determine_s3_config(
            release_type="ci",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=None,
        )
        self.assertNotIn("/x86_64/", repo_url)
        self.assertIn("packages/deb", repo_url)

    def test_prerelease_rpm_includes_x86_64(self):
        """Test prerelease RPM URL includes /x86_64/ suffix."""
        _, _, _, repo_url = get_s3_config.determine_s3_config(
            release_type="prerelease",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version="8.1.0~pre2",
        )
        self.assertTrue(repo_url.endswith("/x86_64/"))
        self.assertIn("rhel10/x86_64/", repo_url)

    def test_prerelease_deb_no_x86_64(self):
        """Test prerelease DEB URL does not include /x86_64/."""
        _, _, _, repo_url = get_s3_config.determine_s3_config(
            release_type="prerelease",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~pre2",
        )
        self.assertNotIn("/x86_64/", repo_url)
        self.assertIn("ubuntu2404", repo_url)

    def test_release_rpm_includes_x86_64(self):
        """Test release RPM URL includes /x86_64/ suffix."""
        _, _, _, repo_url = get_s3_config.determine_s3_config(
            release_type="release",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version="8.1.0",
        )
        self.assertTrue(repo_url.endswith("/x86_64/"))
        self.assertIn("rhel10/x86_64/", repo_url)


if __name__ == "__main__":
    unittest.main()
