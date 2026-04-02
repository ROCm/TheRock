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
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
            release_type="dev",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertEqual(bucket, "therock-dev-packages")
        self.assertEqual(prefix, "v3/packages/deb/20251203-12345678")
        self.assertEqual(job_type, "dev")

    def test_nightly_release_type(self):
        """Test nightly release type uses nightly-packages bucket."""
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
            release_type="nightly",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="rpm",
            artifact_id="87654321",
            rocm_version="8.1.0~20251203",
        )
        self.assertEqual(bucket, "therock-nightly-packages")
        self.assertEqual(prefix, "v3/packages/rpm/20251203-87654321")
        self.assertEqual(job_type, "nightly")

    def test_prerelease_release_type(self):
        """Test prerelease uses prerelease-packages bucket without date."""
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
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
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
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
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
            release_type="ci",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=None,
        )
        self.assertEqual(bucket, "therock-ci-artifacts")
        self.assertIn("v3/packages/deb/", prefix)
        self.assertEqual(job_type, "ci")

    def test_empty_release_type(self):
        """Test empty release type falls through to CI bucket logic."""
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
            release_type="",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=None,
        )
        self.assertEqual(bucket, "therock-ci-artifacts")
        self.assertIn("v3/packages/deb/", prefix)
        self.assertEqual(job_type, "ci")


class DetermineS3ConfigRepositoryTest(unittest.TestCase):
    """Tests for S3 config with different repositories."""

    def test_fork_pr(self):
        """Test fork PR uses external bucket."""
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
            release_type="",
            repository="ROCm/TheRock",
            is_fork=True,
            pkg_type="rpm",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertEqual(bucket, "therock-ci-artifacts-external")
        self.assertEqual(prefix, "v3/packages/rpm/20251203-12345678")
        self.assertEqual(job_type, "ci")

    def test_external_repository(self):
        """Test external repository uses external bucket."""
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
            release_type="",
            repository="someone/fork",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertEqual(bucket, "therock-ci-artifacts-external")
        self.assertEqual(prefix, "v3/packages/deb/20251203-12345678")
        self.assertEqual(job_type, "ci")

    def test_default_rocm_therock(self):
        """Test default ROCm/TheRock uses ci-artifacts bucket."""
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
            release_type="",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version="8.1.0~dev20251203",
        )
        self.assertEqual(bucket, "therock-ci-artifacts")
        self.assertEqual(prefix, "v3/packages/deb/20251203-12345678")
        self.assertEqual(job_type, "ci")


class DetermineS3ConfigPackageTypeTest(unittest.TestCase):
    """Tests for S3 config with different package types."""

    def test_deb_package_type(self):
        """Test deb package type in prefix."""
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
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
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
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
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
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
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
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
        bucket, prefix, job_type = get_s3_config.determine_s3_config(
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

        bucket, prefix, job_type = get_s3_config.determine_s3_config(
            release_type="dev",
            repository="ROCm/TheRock",
            is_fork=False,
            pkg_type="deb",
            artifact_id="12345678",
            rocm_version=None,
        )
        # Should use current date
        self.assertIn("20260312", prefix)


if __name__ == "__main__":
    unittest.main()
