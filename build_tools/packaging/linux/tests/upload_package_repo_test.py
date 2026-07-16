#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``upload_package_repo.py`` RPM repodata helpers.

Tests cover S3 RPM key listing, local+S3 arch-dir materialization, and
fail-fast repodata validation. S3 is stubbed via ``FakeS3``; filesystem
layout uses real temp directories.

Run::

    python3.12 build_tools/packaging/linux/tests/upload_package_repo_test.py -v
"""

import gzip
import sys
import tempfile
import types
import unittest
from collections.abc import Iterator
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent
BUILD_TOOLS_DIR = LINUX_DIR.parent.parent

# upload_package_repo imports boto3 at module load; stub it so tests run
# without AWS credentials or the boto3 package installed.
_boto3_stub = types.ModuleType("boto3")
sys.modules["boto3"] = _boto3_stub

# Resolve packaging modules from any working directory (style guide).
for path in (BUILD_TOOLS_DIR, LINUX_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import upload_package_repo as upload_repo  # noqa: E402

# Test fixture defaults (avoid unexplained literals in assertions and helpers).
TEST_BUCKET = "therock-test-bucket"
TEST_RPM_PREFIX = "run-linux/packages/rpm"
TEST_LEGACY_PREFIX = "prefix"
RPM_ARCH_SUBDIR = "x86_64"

LOCAL_ONLY_RPM = "local-only.rpm"
SHARED_RPM = "shared.rpm"
S3_ONLY_RPM = "s3-only.rpm"
PKG_A_RPM = "pkg-a.rpm"
PKG_B_RPM = "pkg-b.rpm"

EXPECTED_LOCAL_RPM_COUNT = 2
EXPECTED_DOWNLOADED_RPM_COUNT = 1
EXPECTED_S3_DOWNLOAD_CALLS = 1

FAKE_RPM_BYTES = b"fake-rpm"
LOCAL_RPM_BYTES = b"local"
SHARED_LOCAL_RPM_BYTES = b"shared-local"
S3_RPM_BYTES = b"from-s3"
PKG_A_RPM_BYTES = b"a"
PKG_B_RPM_BYTES = b"b"


class FakeS3:
    """Minimal S3 client stub matching ``_S3Client`` for RPM helper tests."""

    def __init__(
        self,
        rpm_keys: list[str],
        downloads: dict[str, bytes] | None = None,
    ) -> None:
        """Configure listed RPM keys and optional per-key download payloads."""
        self._rpm_keys = rpm_keys
        self._downloads = downloads or {}
        # Record download_file calls so tests can assert S3 backfill behavior.
        self.download_calls: list[tuple[str, str, str]] = []

    def get_paginator(self, op_name: str) -> object:
        """Return a paginator stub that only supports list_objects_v2."""
        if op_name != "list_objects_v2":
            raise ValueError(f"Unexpected paginator: {op_name!r}")
        s3 = self

        class Dispatcher:
            """Yields S3 list pages for prefix/x86_64/ queries only."""

            def paginate(
                self, *, Bucket: str, Prefix: str
            ) -> Iterator[dict[str, list[dict[str, str]]]]:
                # Mirror production: only keys under prefix/x86_64/ are RPM candidates.
                del Bucket
                if Prefix.endswith(f"/{RPM_ARCH_SUBDIR}/"):
                    yield {
                        "Contents": [{"Key": key} for key in s3._rpm_keys],
                    }
                else:
                    yield {}

        return Dispatcher()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        """Write stub RPM bytes to filename and record the S3 key requested."""
        self.download_calls.append((bucket, key, filename))
        Path(filename).write_bytes(self._downloads.get(key, FAKE_RPM_BYTES))


class UploadPackageRepoTestCase(unittest.TestCase):
    """Base test case with a per-test temporary directory."""

    def setUp(self) -> None:
        """Create an isolated temp dir for filesystem layout under test."""
        self._temp_context = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_context.name)

    def tearDown(self) -> None:
        """Remove the per-test temp dir."""
        self._temp_context.cleanup()


# ---------------------------------------------------------------------------
# _list_s3_rpm_keys — S3 listing must exclude repodata metadata keys
# ---------------------------------------------------------------------------
class ListS3RpmKeysTest(unittest.TestCase):
    """Tests for ``_list_s3_rpm_keys()``."""

    def test_filters_non_rpm_keys(self) -> None:
        """Only ``.rpm`` keys under prefix/x86_64/ are returned.

        Repodata files (e.g. primary.xml.gz) live in the same S3 prefix but
        must not be downloaded as packages during full repodata regeneration.
        """
        s3 = FakeS3(
            rpm_keys=[
                f"{TEST_RPM_PREFIX}/{RPM_ARCH_SUBDIR}/{PKG_A_RPM}",
                f"{TEST_RPM_PREFIX}/{RPM_ARCH_SUBDIR}/repodata/primary.xml.gz",
                f"{TEST_RPM_PREFIX}/{RPM_ARCH_SUBDIR}/{PKG_B_RPM}",
            ]
        )
        keys = upload_repo._list_s3_rpm_keys(s3, TEST_BUCKET, TEST_RPM_PREFIX)
        self.assertEqual(
            keys,
            [
                f"{TEST_RPM_PREFIX}/{RPM_ARCH_SUBDIR}/{PKG_A_RPM}",
                f"{TEST_RPM_PREFIX}/{RPM_ARCH_SUBDIR}/{PKG_B_RPM}",
            ],
        )


# ---------------------------------------------------------------------------
# _prepare_rpm_arch_dir_for_repodata — local tree + S3 backfill
# ---------------------------------------------------------------------------
class PrepareRpmArchDirTest(UploadPackageRepoTestCase):
    """Tests for ``_prepare_rpm_arch_dir_for_repodata()``."""

    def test_uses_local_tree_and_downloads_missing_s3_rpms(self) -> None:
        """Local build RPMs are copied first; only missing S3 RPMs are downloaded.

        Regression guard for upload dedupe: local-only RPMs must appear in the
        working arch dir even when upload skipped them. Shared names prefer the
        local copy; S3-only packages are fetched once for createrepo_c input.
        """
        package_dir = self.temp_dir / "packages"
        arch_dir = package_dir / RPM_ARCH_SUBDIR
        arch_dir.mkdir(parents=True)
        # Two RPMs built locally this run (one will also exist on S3).
        (arch_dir / LOCAL_ONLY_RPM).write_bytes(LOCAL_RPM_BYTES)
        (arch_dir / SHARED_RPM).write_bytes(SHARED_LOCAL_RPM_BYTES)

        s3_key_shared = f"{TEST_LEGACY_PREFIX}/{RPM_ARCH_SUBDIR}/{SHARED_RPM}"
        s3_key_only = f"{TEST_LEGACY_PREFIX}/{RPM_ARCH_SUBDIR}/{S3_ONLY_RPM}"
        s3 = FakeS3(
            rpm_keys=[s3_key_shared, s3_key_only],
            downloads={s3_key_only: S3_RPM_BYTES},
        )

        prep = upload_repo._prepare_rpm_arch_dir_for_repodata(
            s3=s3,
            bucket=TEST_BUCKET,
            prefix=TEST_LEGACY_PREFIX,
            package_dir=package_dir,
            work_dir=self.temp_dir / "work",
        )

        self.assertEqual(prep.local_rpm_count, EXPECTED_LOCAL_RPM_COUNT)
        self.assertEqual(prep.downloaded_rpm_count, EXPECTED_DOWNLOADED_RPM_COUNT)
        self.assertTrue((prep.arch_dir / LOCAL_ONLY_RPM).exists())
        self.assertTrue((prep.arch_dir / SHARED_RPM).exists())
        self.assertTrue((prep.arch_dir / S3_ONLY_RPM).exists())
        # shared.rpm must not be re-downloaded when already present locally.
        self.assertEqual(len(s3.download_calls), EXPECTED_S3_DOWNLOAD_CALLS)
        self.assertEqual(s3.download_calls[0][1], s3_key_only)


# ---------------------------------------------------------------------------
# _validate_rpm_repodata — fail-fast after createrepo_c
# ---------------------------------------------------------------------------
class ValidateRpmRepodataTest(UploadPackageRepoTestCase):
    """Tests for ``_validate_rpm_repodata()``."""

    def test_raises_when_primary_xml_indexes_fewer_packages_than_rpms(self) -> None:
        """Reject repodata when primary.xml.gz indexes fewer packages than on disk.

        Catches the stale-repodata bug: mergerepo_c on a partial upload set could
        leave repomd.xml pointing at fewer RPMs than dnf/zypper need at install time.
        """
        arch_dir = self.temp_dir / RPM_ARCH_SUBDIR
        arch_dir.mkdir(parents=True)
        (arch_dir / PKG_A_RPM).write_bytes(PKG_A_RPM_BYTES)
        (arch_dir / PKG_B_RPM).write_bytes(PKG_B_RPM_BYTES)

        repodata_dir = arch_dir / "repodata"
        repodata_dir.mkdir(parents=True)
        # primary.xml.gz claims 1 package but two .rpm files are present.
        primary_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<metadata xmlns="http://linux.duke.edu/metadata/common" '
            'xmlns:rpm="http://linux.duke.edu/metadata/rpm" packages="1">'
            '<package type="rpm">'
            "<name>pkg-a</name>"
            "</package>"
            "</metadata>"
        )
        with gzip.open(repodata_dir / "primary.xml.gz", "wb") as handle:
            handle.write(primary_xml.encode("utf-8"))

        with self.assertRaisesRegex(RuntimeError, "RPM repodata is incomplete"):
            upload_repo._validate_rpm_repodata(arch_dir)


if __name__ == "__main__":
    unittest.main()
