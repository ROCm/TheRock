#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``upload_package_repo.py``.

Lightweight coverage of pure-Python helpers that do not need createrepo_c,
dpkg-scanpackages, or live S3.

Coverage:

  - ``generate_release_file_with_checksums`` — Release includes checksum sections
  - ``upload_to_s3`` — uploads repodata; dedupes packages only
  - ``s3_object_exists`` — head_object success and 404 handling
  - ``_package_install_url`` — RPM baseurl includes ``x86_64/``

Run::

    python3.12 build_tools/packaging/linux/tests/upload_package_repo_test.py -v
"""

import gzip
import sys
import tempfile
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
    boto3_stub.client = MagicMock()
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
TEST_PREFIX = "12345-linux/packages/rpm/20250825-12345"
TEST_JOB_TYPE = "nightly"


class GenerateReleaseFileTest(unittest.TestCase):
    """Tests for ``generate_release_file_with_checksums()``."""

    def test_release_includes_checksum_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dists_dir = Path(temp_dir) / "main" / "binary-amd64"
            dists_dir.mkdir(parents=True)
            (dists_dir / "Packages").write_text("Package: demo\n", encoding="utf-8")
            with gzip.open(dists_dir / "Packages.gz", "wb") as handle:
                handle.write(b"Package: demo\n")

            release_file = Path(temp_dir) / "Release"
            upload_repo.generate_release_file_with_checksums(
                release_file, TEST_JOB_TYPE, dists_dir
            )
            release_text = release_file.read_text(encoding="utf-8")

        self.assertIn("MD5Sum:", release_text)
        self.assertIn("SHA256:", release_text)
        self.assertIn("main/binary-amd64/Packages", release_text)


class UploadToS3Test(unittest.TestCase):
    """Tests for ``upload_to_s3()`` upload walk."""

    @patch("upload_package_repo.boto3.client")
    @patch.object(upload_repo, "s3_object_exists", return_value=True)
    def test_uploads_repodata_and_skips_deduped_packages(
        self, _mock_exists: MagicMock, mock_boto_client: MagicMock
    ) -> None:
        s3 = MagicMock()
        mock_boto_client.return_value = s3

        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir)
            repodata_dir = package_dir / "x86_64" / "repodata"
            repodata_dir.mkdir(parents=True)
            (repodata_dir / "repomd.xml").write_text("<repomd/>", encoding="utf-8")
            (package_dir / "x86_64" / "pkg-a.rpm").write_bytes(b"rpm")

            upload_repo.upload_to_s3(
                str(package_dir), TEST_BUCKET, TEST_PREFIX, dedupe=True
            )

        uploaded_keys = [call.args[2] for call in s3.upload_file.call_args_list]
        self.assertIn(f"{TEST_PREFIX}/x86_64/repodata/repomd.xml", uploaded_keys)
        self.assertNotIn(f"{TEST_PREFIX}/x86_64/pkg-a.rpm", uploaded_keys)


class S3ObjectExistsTest(unittest.TestCase):
    """Tests for ``s3_object_exists()``."""

    def test_returns_true_when_object_exists(self) -> None:
        s3 = MagicMock()
        s3.head_object.return_value = {"ContentLength": 1}

        self.assertTrue(
            upload_repo.s3_object_exists(s3, TEST_BUCKET, f"{TEST_PREFIX}/pkg.rpm")
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
        deb_prefix = "12345-linux/packages/deb/20250825-12345"
        url = upload_repo._package_install_url(TEST_BUCKET, deb_prefix, "deb")
        self.assertNotIn("/x86_64", url)
        self.assertIn(deb_prefix, url)


if __name__ == "__main__":
    unittest.main()
