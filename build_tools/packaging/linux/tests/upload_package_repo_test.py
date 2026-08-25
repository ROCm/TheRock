#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``upload_package_repo.py`` (main-line helpers).

Lightweight coverage of pure-Python helpers that do not need createrepo_c,
mergerepo_c, or live S3.

Coverage:

  - ``regenerate_repo_metadata_from_s3`` — RPM/DEB dispatch and invalid type
  - ``s3_object_exists`` — head_object success and 404 handling
  - ``_package_install_url`` — RPM baseurl includes ``x86_64/``

Run::

    python3.12 build_tools/packaging/linux/tests/upload_package_repo_test.py -v

    python3.12 -m unittest discover -s build_tools/packaging/linux/tests \\
        -p 'upload_package_repo_test.py' -v
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent
BUILD_TOOLS_DIR = LINUX_DIR.parent.parent

for path in (BUILD_TOOLS_DIR, LINUX_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _import_upload_package_repo():
    """Import upload_package_repo with a temporary boto3 stub."""
    boto3_stub = types.ModuleType("boto3")
    saved_boto3 = sys.modules.get("boto3")
    sys.modules["boto3"] = boto3_stub
    try:
        import upload_package_repo as upload_repo  # noqa: WPS433

        return upload_repo
    finally:
        if saved_boto3 is not None:
            sys.modules["boto3"] = saved_boto3
        else:
            sys.modules.pop("boto3", None)


upload_repo = _import_upload_package_repo()

TEST_BUCKET = "therock-test-bucket"
TEST_PREFIX = "run-linux/packages/rpm/20250825-12345"
TEST_JOB_TYPE = "nightly"


class RegenerateRepoMetadataFromS3Test(unittest.TestCase):
    """Tests for ``regenerate_repo_metadata_from_s3()`` dispatch."""

    @patch.object(upload_repo, "regenerate_deb_metadata_from_s3")
    @patch.object(upload_repo, "regenerate_rpm_metadata_from_s3")
    def test_dispatches_rpm_to_merge_helper(
        self, mock_rpm: MagicMock, mock_deb: MagicMock
    ) -> None:
        s3 = object()
        packages = ["/tmp/pkg.rpm"]

        upload_repo.regenerate_repo_metadata_from_s3(
            s3, TEST_BUCKET, TEST_PREFIX, "rpm", packages, TEST_JOB_TYPE
        )

        mock_rpm.assert_called_once_with(s3, TEST_BUCKET, TEST_PREFIX, packages)
        mock_deb.assert_not_called()

    @patch.object(upload_repo, "regenerate_deb_metadata_from_s3")
    @patch.object(upload_repo, "regenerate_rpm_metadata_from_s3")
    def test_dispatches_deb_to_merge_helper(
        self, mock_rpm: MagicMock, mock_deb: MagicMock
    ) -> None:
        s3 = object()
        packages = ["/tmp/pkg.deb"]

        upload_repo.regenerate_repo_metadata_from_s3(
            s3, TEST_BUCKET, TEST_PREFIX, "deb", packages, TEST_JOB_TYPE
        )

        mock_deb.assert_called_once_with(
            s3, TEST_BUCKET, TEST_PREFIX, packages, TEST_JOB_TYPE
        )
        mock_rpm.assert_not_called()

    def test_raises_for_unsupported_package_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported package type"):
            upload_repo.regenerate_repo_metadata_from_s3(
                object(), TEST_BUCKET, TEST_PREFIX, "apk", [], TEST_JOB_TYPE
            )


class S3ObjectExistsTest(unittest.TestCase):
    """Tests for ``s3_object_exists()``."""

    def test_returns_true_when_object_exists(self) -> None:
        s3 = MagicMock()
        s3.head_object.return_value = {"ContentLength": 1}

        self.assertTrue(
            upload_repo.s3_object_exists(s3, TEST_BUCKET, f"{TEST_PREFIX}/pkg.rpm")
        )
        s3.head_object.assert_called_once_with(
            Bucket=TEST_BUCKET, Key=f"{TEST_PREFIX}/pkg.rpm"
        )

    def test_returns_false_on_404(self) -> None:
        s3 = MagicMock()
        error = s3.exceptions.ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject",
        )
        s3.head_object.side_effect = error

        self.assertFalse(
            upload_repo.s3_object_exists(s3, TEST_BUCKET, f"{TEST_PREFIX}/missing.rpm")
        )


class PackageInstallUrlTest(unittest.TestCase):
    """Tests for ``_package_install_url()``."""

    def test_rpm_url_includes_x86_64_subdirectory(self) -> None:
        url = upload_repo._package_install_url(TEST_BUCKET, TEST_PREFIX, "rpm")
        self.assertTrue(url.endswith("/x86_64"))
        self.assertIn(TEST_PREFIX, url)

    def test_deb_url_uses_repo_root(self) -> None:
        deb_prefix = "run-linux/packages/deb/20250825-12345"
        url = upload_repo._package_install_url(TEST_BUCKET, deb_prefix, "deb")
        self.assertNotIn("/x86_64", url)
        self.assertIn(deb_prefix, url)


if __name__ == "__main__":
    unittest.main()
