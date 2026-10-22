#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``verify_package_repo_checksums.py``.

These tests use real temporary repository files. Network access is mocked only
for the published-index check. A one-byte archive change must fail, and a wrong
MD5 must fail even when SHA256 still matches.

Run from the repository root::

    python3.12 -m unittest discover -s build_tools/packaging/linux/tests \\
        -p 'verify_package_repo_checksums_test.py' -v
"""

import gzip
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.sax.saxutils import escape

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_SCRIPT_DIR.parent

if os.fspath(LINUX_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(LINUX_DIR))

import build_package_repo as repo_builder  # noqa: E402
import verify_package_repo_checksums as checksums  # noqa: E402

_PACKAGE_BYTES = b"native-package-payload"
_NS_COMMON = "http://linux.duke.edu/metadata/common"
_NS_REPO = "http://linux.duke.edu/metadata/repo"


def _md5(payload: bytes) -> str:
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha512(payload: bytes) -> str:
    return hashlib.sha512(payload).hexdigest()


def _write_deb_repo(
    root: Path,
    payload: bytes = _PACKAGE_BYTES,
    *,
    fields: dict[str, str] | None = None,
    omit: tuple[str, ...] = (),
    extra_payload: bytes | None = None,
) -> Path:
    """Build a one-package DEB repository and return the archive path."""
    pool = root / "pool" / "main"
    index = root / "dists" / "stable" / "main" / "binary-amd64"
    pool.mkdir(parents=True)
    index.mkdir(parents=True)
    archive = pool / "demo_1_amd64.deb"
    archive.write_bytes(payload)
    stanza = {
        "Package": "demo",
        "Version": "1",
        "Filename": "pool/main/demo_1_amd64.deb",
        "Size": str(len(payload)),
        "MD5sum": _md5(payload),
        "SHA256": _sha256(payload),
        "Description": "demo package",
    }
    if fields:
        stanza.update(fields)
    for field_name in omit:
        stanza.pop(field_name, None)
    lines = [f"{name}: {value}" for name, value in stanza.items()]
    description_at = next(
        index for index, line in enumerate(lines) if line.startswith("Description:")
    )
    # Real Packages indexes fold the Description field across lines.
    lines.insert(description_at + 1, " continued description")
    packages_text = "\n".join(lines) + "\n\n"
    packages_path = index / "Packages"
    packages_path.write_text(packages_text, encoding="utf-8")
    with gzip.open(index / "Packages.gz", "wb") as handle:
        handle.write(packages_text.encode("utf-8"))
    repo_builder.generate_release_file_with_checksums(
        root / "dists" / "stable" / "Release",
        "ci",
        index,
    )
    if extra_payload is not None:
        (pool / "extra_1_amd64.deb").write_bytes(extra_payload)
    return archive


def _primary_xml(
    payload: bytes,
    *,
    checksum_type: str = "sha256",
    checksum_value: str | None = None,
    href: str = "demo.rpm",
    size: int | None = None,
    include_checksum: bool = True,
    include_location: bool = True,
    include_size: bool = True,
) -> bytes:
    digest = checksum_value
    if digest is None:
        if checksum_type == "sha512":
            digest = _sha512(payload)
        elif checksum_type == "sha1":
            digest = hashlib.sha1(payload, usedforsecurity=False).hexdigest()
        else:
            digest = _sha256(payload)
    package_size = len(payload) if size is None else size
    parts = [
        f'<metadata xmlns="{_NS_COMMON}">',
        '<package type="rpm">',
        "<name>demo</name>",
        '<version ver="1" rel="1"/>',
    ]
    if include_checksum:
        parts.append(
            f'<checksum type="{escape(checksum_type)}">{escape(digest or "")}</checksum>'
        )
    if include_size:
        parts.append(f'<size package="{package_size}"/>')
    if include_location:
        parts.append(f'<location href="{escape(href)}"/>')
    parts.extend(["</package>", "</metadata>"])
    return "".join(parts).encode("utf-8")


def _write_rpm_repo(
    root: Path,
    payload: bytes = _PACKAGE_BYTES,
    *,
    primary: bytes | None = None,
    repomd_digest: str | None = None,
    extra_payload: bytes | None = None,
    write_archive: bool = True,
) -> Path:
    """Build a one-package RPM repository and return the archive path."""
    arch = root / "x86_64"
    repodata = arch / "repodata"
    repodata.mkdir(parents=True)
    archive = arch / "demo.rpm"
    if write_archive:
        archive.write_bytes(payload)
    primary_xml = primary if primary is not None else _primary_xml(payload)
    primary_gz = gzip.compress(primary_xml)
    (repodata / "primary.xml.gz").write_bytes(primary_gz)
    digest = repomd_digest if repomd_digest is not None else _sha256(primary_gz)
    repomd = (
        f'<repomd xmlns="{_NS_REPO}">'
        '<data type="primary">'
        f'<checksum type="sha256">{digest}</checksum>'
        '<location href="repodata/primary.xml.gz"/>'
        f"<size>{len(primary_gz)}</size>"
        "</data>"
        "</repomd>"
    )
    (repodata / "repomd.xml").write_bytes(repomd.encode("utf-8"))
    if extra_payload is not None:
        (arch / "extra.rpm").write_bytes(extra_payload)
    return archive


class DebRepositoryChecksumTest(unittest.TestCase):
    def test_valid_repository_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_deb_repo(Path(temp_dir))
            checksums.verify_package_repository(Path(temp_dir), "deb")

    def test_corrupt_archive_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = _write_deb_repo(root)
            corrupted = bytearray(archive.read_bytes())
            corrupted[-1] ^= 0x01
            archive.write_bytes(bytes(corrupted))
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        message = str(caught.exception)
        self.assertIn("MD5 mismatch", message)
        self.assertIn("SHA256 mismatch", message)

    def test_wrong_md5_fails_when_sha256_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root, fields={"MD5sum": "a" * 32})
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        message = str(caught.exception)
        self.assertIn("MD5 mismatch", message)
        self.assertNotIn("SHA256 mismatch", message)

    def test_wrong_sha256_fails_when_md5_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root, fields={"SHA256": "b" * 64})
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        message = str(caught.exception)
        self.assertIn("SHA256 mismatch", message)
        self.assertNotIn("MD5 mismatch", message)

    def test_missing_sha256_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root, omit=("SHA256",))
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        self.assertIn("missing SHA256", str(caught.exception))

    def test_malformed_md5_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root, fields={"MD5sum": "not-a-digest"})
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        self.assertIn("MD5sum is malformed", str(caught.exception))

    def test_size_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root, fields={"Size": "1"})
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        self.assertIn("size mismatch", str(caught.exception))

    def test_missing_indexed_package_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = _write_deb_repo(
                root, fields={"Filename": "pool/main/missing.deb"}
            )
            archive.unlink()
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        self.assertIn("missing package file", str(caught.exception))

    def test_unindexed_package_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root, extra_payload=b"extra")
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        self.assertIn("not in the repository index", str(caught.exception))

    def test_unsafe_filename_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = _write_deb_repo(root, fields={"Filename": "../outside.deb"})
            archive.unlink()
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        self.assertIn("escapes the repository root", str(caught.exception))

    def test_corrupt_packages_gz_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root)
            packages_gz = (
                root / "dists" / "stable" / "main" / "binary-amd64" / "Packages.gz"
            )
            payload = bytearray(packages_gz.read_bytes())
            payload[-1] ^= 0x01
            packages_gz.write_bytes(bytes(payload))
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        self.assertIn("Packages.gz", str(caught.exception))

    def test_duplicate_filename_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root)
            packages = root / "dists" / "stable" / "main" / "binary-amd64" / "Packages"
            packages.write_text(
                packages.read_text(encoding="utf-8") * 2, encoding="utf-8"
            )
            with gzip.open(
                root / "dists" / "stable" / "main" / "binary-amd64" / "Packages.gz",
                "wb",
            ) as handle:
                handle.write(packages.read_bytes())
            repo_builder.generate_release_file_with_checksums(
                root / "dists" / "stable" / "Release",
                "ci",
                packages.parent,
            )
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "deb")
        self.assertIn("indexed more than once", str(caught.exception))


class RpmRepositoryChecksumTest(unittest.TestCase):
    def test_valid_repository_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_rpm_repo(Path(temp_dir))
            checksums.verify_package_repository(Path(temp_dir), "rpm")

    def test_sha512_package_checksum_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = _PACKAGE_BYTES
            primary = _primary_xml(payload, checksum_type="sha512")
            _write_rpm_repo(Path(temp_dir), payload, primary=primary)
            checksums.verify_package_repository(Path(temp_dir), "rpm")

    def test_corrupt_archive_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = _write_rpm_repo(root)
            corrupted = bytearray(archive.read_bytes())
            corrupted[-1] ^= 0x01
            archive.write_bytes(bytes(corrupted))
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "rpm")
        self.assertIn("SHA256 mismatch", str(caught.exception))

    def test_sha1_package_checksum_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            payload = _PACKAGE_BYTES
            _write_rpm_repo(
                root, payload, primary=_primary_xml(payload, checksum_type="sha1")
            )
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "rpm")
        self.assertIn("SHA256 or stronger", str(caught.exception))

    def test_wrong_primary_checksum_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_rpm_repo(root, repomd_digest="c" * 64)
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "rpm")
        self.assertIn("SHA256 mismatch", str(caught.exception))

    def test_missing_package_checksum_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary = _primary_xml(_PACKAGE_BYTES, include_checksum=False)
            _write_rpm_repo(root, primary=primary)
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "rpm")
        self.assertIn("incomplete", str(caught.exception))

    def test_missing_indexed_package_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_rpm_repo(root, write_archive=False)
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "rpm")
        self.assertIn("missing package file", str(caught.exception))

    def test_unindexed_package_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_rpm_repo(root, extra_payload=b"extra-rpm")
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "rpm")
        self.assertIn("not in the repository index", str(caught.exception))

    def test_unsafe_location_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary = _primary_xml(_PACKAGE_BYTES, href="../outside.rpm")
            _write_rpm_repo(root, primary=primary, write_archive=False)
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "rpm")
        self.assertIn("escapes the repository root", str(caught.exception))

    def test_empty_primary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary = f'<metadata xmlns="{_NS_COMMON}"></metadata>'.encode("utf-8")
            _write_rpm_repo(root, primary=primary, write_archive=False)
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_package_repository(root, "rpm")
        self.assertIn("no packages", str(caught.exception))


class PublishedChecksumTest(unittest.TestCase):
    def test_published_deb_requires_https(self) -> None:
        with self.assertRaises(checksums.PackageChecksumError) as caught:
            checksums.verify_published_repository_checksums(
                "http://example.invalid/deb", "deb"
            )
        self.assertIn("unreadable", str(caught.exception))

    def test_published_deb_checks_release_and_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root)
            release = (root / "dists" / "stable" / "Release").read_bytes()
            packages_gz = (
                root / "dists" / "stable" / "main" / "binary-amd64" / "Packages.gz"
            ).read_bytes()

        def fetch(url: str) -> bytes:
            if url.endswith("/Release"):
                return release
            if url.endswith("/Packages.gz"):
                return packages_gz
            raise AssertionError(url)

        with patch.object(checksums, "_fetch_https_bytes", side_effect=fetch):
            checksums.verify_published_repository_checksums(
                "https://example.invalid/deb", "deb"
            )

    def test_published_deb_detects_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_deb_repo(root)
            release = (root / "dists" / "stable" / "Release").read_bytes()
            packages_gz = bytearray(
                (
                    root / "dists" / "stable" / "main" / "binary-amd64" / "Packages.gz"
                ).read_bytes()
            )
        packages_gz[-1] ^= 0x01

        def fetch(url: str) -> bytes:
            if url.endswith("/Release"):
                return release
            return bytes(packages_gz)

        with patch.object(checksums, "_fetch_https_bytes", side_effect=fetch):
            with self.assertRaises(checksums.PackageChecksumError) as caught:
                checksums.verify_published_repository_checksums(
                    "https://example.invalid/deb", "deb"
                )
        self.assertIn("mismatch", str(caught.exception).lower())


class CliTest(unittest.TestCase):
    def test_main_passes_for_valid_deb_repo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_deb_repo(Path(temp_dir))
            exit_code = checksums.main(["--pkg-type", "deb", "--package-dir", temp_dir])
        self.assertEqual(exit_code, 0)

    def test_main_fails_for_corrupt_rpm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = _write_rpm_repo(root)
            archive.write_bytes(b"corrupted-rpm")
            exit_code = checksums.main(["--pkg-type", "rpm", "--package-dir", temp_dir])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
