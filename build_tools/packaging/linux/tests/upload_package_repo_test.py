#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``upload_package_repo.py`` RPM repodata helpers.

Regression guards for Issue #6540: repodata must index every RPM in the local
build tree, then upload repodata/ after package upload (dedupe OK).

Coverage:

  - ``_assert_rpm_lead_magic`` — reject non-RPM content before createrepo_c
  - ``_validate_rpm_repodata`` — fail-fast when ``primary.xml.gz`` indexes fewer
    packages than ``.rpm`` files on disk
  - ``upload_rpm_repodata_to_s3`` — upload local ``repodata/`` files to S3

Not covered here: ``create_rpm_repo`` end-to-end (requires ``createrepo_c``);
DEB merge path in the same module.

Prerequisites:

  - Python 3.10 or newer
  - Run from TheROCK repository root (or any cwd — modules resolved via ``__file__``)
  - Stdlib only for these tests; boto3 is stubbed only during module import
    (restored afterward so pytest collection of other build_tools tests is safe)

Run::

    python3.12 build_tools/packaging/linux/tests/upload_package_repo_test.py -v

    python3.12 -m unittest discover -s build_tools/packaging/linux/tests \\
        -p 'upload_package_repo_test.py' -v
"""

import gzip
import sys
import tempfile
import types
import unittest
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent
BUILD_TOOLS_DIR = LINUX_DIR.parent.parent

# Resolve packaging modules from any working directory (style guide).
for path in (BUILD_TOOLS_DIR, LINUX_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _import_upload_package_repo():
    """Import upload_package_repo with a temporary boto3 stub.

    upload_package_repo imports boto3 at module load. Stub only for the import
    so pytest can collect other build_tools tests (e.g. manage_test.py) that
    need the real boto3 package.
    """
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

# Test fixture defaults (avoid unexplained literals in assertions and helpers).
TEST_BUCKET = "therock-test-bucket"
TEST_RPM_PREFIX = "run-linux/packages/rpm"
RPM_ARCH_SUBDIR = "x86_64"

PKG_A_RPM = "pkg-a.rpm"
PKG_B_RPM = "pkg-b.rpm"
LOCAL_ONLY_RPM = "local-only.rpm"

RPM_LEAD_MAGIC = b"\xed\xab\xee\xdb"


def _stub_rpm_bytes(payload: bytes) -> bytes:
    """Return bytes with a valid RPM lead header prefix."""
    return RPM_LEAD_MAGIC + payload


LOCAL_RPM_BYTES = _stub_rpm_bytes(b"local")
PKG_A_RPM_BYTES = _stub_rpm_bytes(b"a")
PKG_B_RPM_BYTES = _stub_rpm_bytes(b"b")
INVALID_RPM_BYTES = b"not-rpm"


class FakeS3:
    """Minimal S3 client stub for ``upload_rpm_repodata_to_s3`` tests."""

    def __init__(self) -> None:
        self.upload_calls: list[tuple[str, str, str]] = []

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, str] | None = None,
    ) -> None:
        del ExtraArgs
        self.upload_calls.append((filename, bucket, key))


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
# _assert_rpm_lead_magic — RPM lead header sanity check
# ---------------------------------------------------------------------------
class AssertRpmLeadMagicTest(UploadPackageRepoTestCase):
    """Tests for ``_assert_rpm_lead_magic()``."""

    def test_accepts_valid_rpm_lead_magic(self) -> None:
        """Files beginning with the RPM lead magic pass validation."""
        rpm_path = self.temp_dir / LOCAL_ONLY_RPM
        rpm_path.write_bytes(LOCAL_RPM_BYTES)
        upload_repo._assert_rpm_lead_magic(rpm_path)

    def test_raises_on_invalid_lead_magic(self) -> None:
        """Non-RPM content is rejected before repodata generation."""
        rpm_path = self.temp_dir / "bad.rpm"
        rpm_path.write_bytes(INVALID_RPM_BYTES)
        with self.assertRaisesRegex(RuntimeError, "Not a valid RPM file"):
            upload_repo._assert_rpm_lead_magic(rpm_path)


# ---------------------------------------------------------------------------
# _validate_rpm_repodata — fail-fast after createrepo_c
# ---------------------------------------------------------------------------
class ValidateRpmRepodataTest(UploadPackageRepoTestCase):
    """Tests for ``_validate_rpm_repodata()``."""

    def test_raises_when_primary_xml_indexes_fewer_packages_than_rpms(self) -> None:
        """Reject repodata when primary.xml.gz indexes fewer packages than on disk.

        Guards the createrepo_c step: every local .rpm must appear in primary.xml.gz
        before repodata is uploaded to S3 (Issue #6540).
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


# ---------------------------------------------------------------------------
# upload_rpm_repodata_to_s3 — push local repodata/ after package upload
# ---------------------------------------------------------------------------
class UploadRpmRepodataToS3Test(UploadPackageRepoTestCase):
    """Tests for ``upload_rpm_repodata_to_s3()``."""

    def test_uploads_all_repodata_files_under_x86_64(self) -> None:
        """Every file in local repodata/ is uploaded under prefix/x86_64/repodata/."""
        package_dir = self.temp_dir / "packages"
        repodata_dir = package_dir / RPM_ARCH_SUBDIR / "repodata"
        repodata_dir.mkdir(parents=True)
        (repodata_dir / "primary.xml.gz").write_bytes(b"primary")
        (repodata_dir / "repomd.xml").write_text(
            '<?xml version="1.0"?><repomd/>', encoding="utf-8"
        )

        s3 = FakeS3()
        upload_repo.upload_rpm_repodata_to_s3(
            s3, TEST_BUCKET, TEST_RPM_PREFIX, package_dir
        )

        uploaded_keys = {call[2] for call in s3.upload_calls}
        self.assertEqual(
            uploaded_keys,
            {
                f"{TEST_RPM_PREFIX}/{RPM_ARCH_SUBDIR}/repodata/primary.xml.gz",
                f"{TEST_RPM_PREFIX}/{RPM_ARCH_SUBDIR}/repodata/repomd.xml",
            },
        )

    def test_skips_when_x86_64_directory_missing(self) -> None:
        """No S3 uploads when the local package tree has no x86_64/ directory."""
        package_dir = self.temp_dir / "packages"
        package_dir.mkdir(parents=True)

        s3 = FakeS3()
        upload_repo.upload_rpm_repodata_to_s3(
            s3, TEST_BUCKET, TEST_RPM_PREFIX, package_dir
        )
        self.assertEqual(s3.upload_calls, [])


if __name__ == "__main__":
    unittest.main()
