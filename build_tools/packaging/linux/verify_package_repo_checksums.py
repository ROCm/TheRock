#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify native Linux repository checksums.

The package build job runs this after ``build_package_repo.py`` and before
``upload_package_repo.py``. It hashes every local ``.deb`` or ``.rpm`` and
compares the result with repository metadata. A missing digest, malformed
metadata entry, size mismatch, digest mismatch, missing archive, or unindexed
archive fails the process before upload.

The install test calls :func:`verify_published_repository_checksums` after
repository setup and before ``apt``, ``dnf``, or ``zypper`` install. That check
confirms the published index matches its ``Release`` or ``repomd.xml`` checksum
and that every package entry advertises the required digest. It does not
download every package archive; the package manager compares those bytes again
while installing.

DEB archives require both ``MD5sum`` and ``SHA256``. MD5 remains because Debian
repository metadata still publishes it; SHA256 is the integrity check. RPM
archive checksums must be SHA256 or stronger (SHA384 or SHA512).

Usage:
  python ./build_tools/packaging/linux/verify_package_repo_checksums.py \\
    --pkg-type deb \\
    --package-dir output/packages

  python ./build_tools/packaging/linux/verify_package_repo_checksums.py \\
    --pkg-type rpm \\
    --package-dir output/packages

``--package-dir`` is the repository root from ``build_package_repo.py``:
``dists/`` and ``pool/`` for DEB, or ``x86_64/`` for RPM. Exit status is 0 when
every check passes and 1 when any check fails.
"""

import argparse
import gzip
import hashlib
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

_READ_CHUNK_BYTES = 1024 * 1024
_FETCH_TIMEOUT_SEC = 60
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_MD5_OCTETS = 16
_SHA1_OCTETS = 20
_SHA256_OCTETS = 32
_SHA512_OCTETS = 64
# SHA1 and MD5 are not accepted as the RPM package digest.
_SHA_BITS = {"sha256": 256, "sha384": 384, "sha512": 512}

_CLI_EPILOG = """
examples:
  Verify a local DEB repository before upload:
    python ./build_tools/packaging/linux/verify_package_repo_checksums.py \\
      --pkg-type deb --package-dir output/packages

  Verify a local RPM repository before upload:
    python ./build_tools/packaging/linux/verify_package_repo_checksums.py \\
      --pkg-type rpm --package-dir output/packages

DEB checks MD5sum and SHA256 on every pool archive, and checks Release against
Packages and Packages.gz. RPM checks repomd.xml against primary.xml.gz, then
checks each package SHA256 (or stronger) against the archive under x86_64/.
Exit status is 0 on success and 1 on any mismatch, missing file, or corrupt
metadata.
"""
_DEB_INDEX_DIR = Path("dists/stable/main/binary-amd64")
_DEB_RELEASE = Path("dists/stable/Release")
_DEB_PACKAGES_REL = "main/binary-amd64/Packages"
_DEB_PACKAGES_GZ_REL = "main/binary-amd64/Packages.gz"
_PUBLISHED_DEB_RELEASE = "dists/stable/Release"
_PUBLISHED_DEB_PACKAGES_GZ = "dists/stable/main/binary-amd64/Packages.gz"
_RPM_ARCH_DIR = Path("x86_64")
_PUBLISHED_RPM_REPOMD = "repodata/repomd.xml"


@dataclass(frozen=True)
class ContentDigest:
    """Algorithm name and hex digest for one repository file."""

    algorithm: str
    hex_digest: str


@dataclass(frozen=True)
class PrimaryMetadata:
    """Compressed RPM primary metadata named by ``repomd.xml``."""

    href: str
    digest: ContentDigest
    size: int


@dataclass(frozen=True)
class IndexedRpm:
    """One package row from RPM ``primary.xml``."""

    label: str
    href: str | None
    digest: ContentDigest | None
    size: int | None


class PackageChecksumError(Exception):
    """One or more repository checksum checks failed.

    ``errors`` lists every problem found in the repository. The process should
    stop before upload or install; individual lines are safe to print.
    """

    def __init__(self, errors: list[str]) -> None:
        if not errors:
            raise ValueError("PackageChecksumError requires at least one error")
        self.errors = errors
        super().__init__("\n".join(errors))


def verify_package_repository(package_dir: Path, pkg_type: str) -> None:
    """Hash every local package and compare it with repository metadata.

    Args:
        package_dir: Repository root produced by ``build_package_repo.py``.
            DEB roots contain ``dists/`` and ``pool/``. RPM roots contain
            ``x86_64/``.
        pkg_type: ``deb`` or ``rpm``.

    Raises:
        PackageChecksumError: When any required checksum check fails. The
            exception text lists every failure found in this pass.
    """
    root = package_dir.resolve()
    if pkg_type == "deb":
        _verify_local_deb_repository(root)
        return
    if pkg_type == "rpm":
        _verify_local_rpm_repository(root)
        return
    raise PackageChecksumError([f"unsupported package type: {pkg_type}"])


def verify_published_repository_checksums(repo_url: str, pkg_type: str) -> None:
    """Check the published index without downloading every package archive.

    Args:
        repo_url: HTTPS APT repository root for DEB, or HTTPS RPM ``baseurl``.
            DEB reads ``dists/stable/Release`` and ``Packages.gz``. RPM reads
            ``repodata/repomd.xml`` and the primary metadata it names.
        pkg_type: ``deb`` or ``rpm``.

    Raises:
        PackageChecksumError: When the index checksum does not match, a package
            entry omits the required digest, or the metadata cannot be read.
            Network and parse errors are reported through this exception so the
            install test has one fail-closed path. The original exception is
            preserved as ``__cause__``.
    """
    try:
        if pkg_type == "deb":
            _verify_published_deb_repository(repo_url)
            return
        if pkg_type == "rpm":
            _verify_published_rpm_repository(repo_url)
            return
        raise PackageChecksumError([f"unsupported package type: {pkg_type}"])
    except PackageChecksumError:
        raise
    except (
        ET.ParseError,
        OSError,
        TimeoutError,
        UnicodeError,
        urllib.error.URLError,
        ValueError,
    ) as error:
        raise PackageChecksumError(
            [f"checksum metadata is unreadable: {error}"]
        ) from error


def main(argv: list[str] | None = None) -> int:
    """Verify a local DEB or RPM repository from the command line.

    Args:
        argv: Command-line arguments. ``None`` uses ``sys.argv[1:]``.

    Returns:
        ``0`` when every checksum matches. ``1`` when verification fails.
        Invalid arguments exit through ``argparse``.

    CLI arguments:

        --pkg-type:
            Required. ``deb`` or ``rpm``.
        --package-dir:
            Required. Repository root produced by ``build_package_repo.py``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Verify DEB or RPM repository checksums against local package files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG,
    )
    parser.add_argument(
        "--pkg-type",
        required=True,
        choices=["deb", "rpm"],
        help="deb requires MD5 and SHA256. rpm requires SHA256 or stronger.",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        required=True,
        help="Root of the repository tree produced by build_package_repo.py.",
    )
    args = parser.parse_args(argv)
    package_dir = args.package_dir
    if not package_dir.is_dir():
        parser.error(f"--package-dir is not a directory: {package_dir}")
    try:
        verify_package_repository(package_dir, args.pkg_type)
    except PackageChecksumError as error:
        print("[FAIL] Package repository checksum verification failed", file=sys.stderr)
        for line in error.errors:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("[PASS] Package repository checksums match")
    return 0


def _verify_local_deb_repository(root: Path) -> None:
    """Fail unless the local DEB index and every pool archive agree.

    ``Release`` must match ``Packages`` and ``Packages.gz`` with MD5 and SHA256.
    Each stanza must match the archive size, ``MD5sum``, and ``SHA256``. An
    extra ``.deb`` that is not indexed also fails.
    """
    failures = _Failures()
    index_dir = root / _DEB_INDEX_DIR
    packages_path = index_dir / "Packages"
    packages_gz_path = index_dir / "Packages.gz"
    packages_bytes = _read_required(packages_path, failures)
    packages_gz_bytes = _read_required(packages_gz_path, failures)
    if packages_bytes is None or packages_gz_bytes is None:
        failures.raise_if_any()
        return
    try:
        decompressed = gzip.decompress(packages_gz_bytes)
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        failures.add(f"Packages.gz is corrupt: {error}")
        decompressed = None
    if decompressed is not None and decompressed != packages_bytes:
        failures.add("Packages.gz does not decompress to Packages")
    release_text = _read_required_text(root / _DEB_RELEASE, failures)
    if release_text is not None:
        _verify_release_checksums(
            release_text,
            {
                _DEB_PACKAGES_REL: packages_bytes,
                _DEB_PACKAGES_GZ_REL: packages_gz_bytes,
            },
            failures,
        )
    stanzas = _parse_deb_packages(packages_bytes.decode("utf-8"), failures)
    indexed = _verify_deb_stanzas(root, stanzas, failures)
    _verify_archive_set(root, indexed, ".deb", failures)
    failures.raise_if_any()


def _verify_local_rpm_repository(root: Path) -> None:
    """Fail unless ``repomd.xml``, primary metadata, and every RPM agree.

    The primary checksum and each package checksum must be SHA256 or stronger.
    An extra ``.rpm`` under ``x86_64/`` that is not indexed also fails.
    """
    failures = _Failures()
    arch_dir = root / _RPM_ARCH_DIR
    repomd_path = arch_dir / "repodata" / "repomd.xml"
    repomd_bytes = _read_required(repomd_path, failures)
    if repomd_bytes is None:
        failures.raise_if_any()
        return
    try:
        primary = _rpm_primary_metadata(repomd_bytes)
    except (ET.ParseError, PackageChecksumError) as error:
        failures.add(str(error))
        failures.raise_if_any()
        return
    try:
        primary_path = _resolve_inside(arch_dir, primary.href)
    except ValueError as error:
        failures.add(str(error))
        failures.raise_if_any()
        return
    primary_bytes = _read_required(primary_path, failures)
    if primary_bytes is None:
        failures.raise_if_any()
        return
    _compare_bytes(
        payload=primary_bytes,
        digest=primary.digest,
        expected_size=primary.size,
        label=primary.href,
        failures=failures,
    )
    try:
        primary_xml = gzip.decompress(primary_bytes)
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        failures.add(f"primary metadata is corrupt: {error}")
        failures.raise_if_any()
        return
    indexed = _verify_rpm_packages(arch_dir, primary_xml, failures)
    _verify_archive_set(arch_dir, indexed, ".rpm", failures)
    failures.raise_if_any()


def _verify_published_deb_repository(repo_url: str) -> None:
    """Check published DEB metadata. APT checks the downloaded archives later."""
    release_bytes = _fetch_https_bytes(_repo_child(repo_url, _PUBLISHED_DEB_RELEASE))
    packages_gz = _fetch_https_bytes(_repo_child(repo_url, _PUBLISHED_DEB_PACKAGES_GZ))
    failures = _Failures()
    _verify_release_checksums(
        release_bytes.decode("utf-8"),
        {_DEB_PACKAGES_GZ_REL: packages_gz},
        failures,
    )
    try:
        packages_bytes = gzip.decompress(packages_gz)
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        failures.add(f"Packages.gz is corrupt: {error}")
        failures.raise_if_any()
        return
    stanzas = _parse_deb_packages(packages_bytes.decode("utf-8"), failures)
    _require_deb_digest_fields(stanzas, failures)
    failures.raise_if_any()


def _verify_published_rpm_repository(repo_url: str) -> None:
    """Check published RPM metadata. DNF or Zypper checks the archives later."""
    repomd_bytes = _fetch_https_bytes(_repo_child(repo_url, _PUBLISHED_RPM_REPOMD))
    primary = _rpm_primary_metadata(repomd_bytes)
    if urlsplit(primary.href).scheme or primary.href.startswith("/"):
        raise PackageChecksumError(
            [f"primary metadata location escapes the repository: {primary.href}"]
        )
    primary_bytes = _fetch_https_bytes(_repo_child(repo_url, primary.href))
    failures = _Failures()
    _compare_bytes(
        payload=primary_bytes,
        digest=primary.digest,
        expected_size=primary.size,
        label=primary.href,
        failures=failures,
    )
    try:
        primary_xml = gzip.decompress(primary_bytes)
    except (OSError, EOFError, gzip.BadGzipFile) as error:
        failures.add(f"primary metadata is corrupt: {error}")
        failures.raise_if_any()
        return
    _require_rpm_digest_fields(primary_xml, failures)
    failures.raise_if_any()


def _verify_release_checksums(
    release_text: str,
    files: dict[str, bytes],
    failures: "_Failures",
) -> None:
    """Require MD5Sum and SHA256 entries in a Debian ``Release`` file."""
    sections = _parse_release_sections(release_text)
    for algorithm in ("md5", "sha256"):
        section_name = "MD5Sum" if algorithm == "md5" else "SHA256"
        entries = sections.get(section_name)
        if not entries:
            failures.add(f"Release is missing {section_name} checksums")
            continue
        for rel_path, payload in files.items():
            entry = entries.get(rel_path)
            if entry is None:
                failures.add(f"Release {section_name} is missing {rel_path}")
                continue
            expected_hash, expected_size = entry
            _compare_bytes(
                payload=payload,
                digest=ContentDigest(algorithm=algorithm, hex_digest=expected_hash),
                expected_size=expected_size,
                label=rel_path,
                failures=failures,
            )


def _verify_deb_stanzas(
    root: Path,
    stanzas: list[dict[str, str]],
    failures: "_Failures",
) -> set[Path]:
    """Check every DEB stanza, then return the archives that were indexed.

    One package can fail size, MD5, and SHA256 together. Those mismatches stay
    in this loop so the report lists every problem before the run stops.
    """
    if not stanzas:
        failures.add("no packages found in Packages")
        return set()
    indexed: set[Path] = set()
    seen: set[str] = set()
    for stanza in stanzas:
        name = stanza.get("Package", "<unknown>")
        version = stanza.get("Version", "<unknown>")
        label = f"{name} {version}"
        filename = stanza.get("Filename", "")
        if not filename:
            failures.add(f"{label}: missing Filename")
            continue
        if filename in seen:
            failures.add(f"{label}: {filename} is indexed more than once")
            continue
        seen.add(filename)
        md5 = _required_hex(
            stanza=stanza,
            field_name="MD5sum",
            octets=_MD5_OCTETS,
            label=label,
            failures=failures,
        )
        sha256 = _required_hex(
            stanza=stanza,
            field_name="SHA256",
            octets=_SHA256_OCTETS,
            label=label,
            failures=failures,
        )
        size = _required_size(
            value=stanza.get("Size", ""), label=label, failures=failures
        )
        optional = _optional_deb_digests(stanza, label, failures)
        if md5 is None or sha256 is None or size is None:
            continue
        try:
            archive = _resolve_inside(root, filename)
        except ValueError as error:
            failures.add(f"{label}: {error}")
            continue
        if not archive.is_file():
            failures.add(f"{label}: missing package file {filename}")
            continue
        algorithms = ["md5", "sha256", *optional]
        _size, digests = _hash_file(archive, algorithms)
        if _size != size:
            failures.add(
                f"{label}: size mismatch for {filename}: expected {size}, actual {_size}"
            )
        if digests["md5"].lower() != md5.lower():
            failures.add(
                f"{label}: MD5 mismatch for {filename}: "
                f"expected {md5}, actual {digests['md5']}"
            )
        if digests["sha256"].lower() != sha256.lower():
            failures.add(
                f"{label}: SHA256 mismatch for {filename}: "
                f"expected {sha256}, actual {digests['sha256']}"
            )
        for algorithm, expected in optional.items():
            actual = digests[algorithm]
            if actual.lower() != expected.lower():
                failures.add(
                    f"{label}: {algorithm.upper()} mismatch for {filename}: "
                    f"expected {expected}, actual {actual}"
                )
        indexed.add(archive)
    return indexed


def _require_deb_digest_fields(
    stanzas: list[dict[str, str]],
    failures: "_Failures",
) -> None:
    """Require ``MD5sum``, ``SHA256``, and ``Size`` on every Packages stanza."""
    if not stanzas:
        failures.add("no packages found in Packages")
        return
    for stanza in stanzas:
        label = (
            f"{stanza.get('Package', '<unknown>')} {stanza.get('Version', '<unknown>')}"
        )
        _required_hex(
            stanza=stanza,
            field_name="MD5sum",
            octets=_MD5_OCTETS,
            label=label,
            failures=failures,
        )
        _required_hex(
            stanza=stanza,
            field_name="SHA256",
            octets=_SHA256_OCTETS,
            label=label,
            failures=failures,
        )
        _required_size(value=stanza.get("Size", ""), label=label, failures=failures)


def _verify_rpm_packages(
    arch_dir: Path,
    primary_xml: bytes,
    failures: "_Failures",
) -> set[Path]:
    """Hash each indexed RPM and return the archives that were checked."""
    try:
        packages = _rpm_package_elements(primary_xml)
    except ET.ParseError as error:
        failures.add(f"primary metadata is malformed: {error}")
        return set()
    if not packages:
        failures.add("no packages found in primary metadata")
        return set()
    indexed: set[Path] = set()
    seen: set[str] = set()
    for package in packages:
        if package.href is None or package.digest is None or package.size is None:
            failures.add(f"{package.label}: primary package metadata is incomplete")
            continue
        if package.href in seen:
            failures.add(f"{package.label}: {package.href} is indexed more than once")
            continue
        seen.add(package.href)
        try:
            archive = _resolve_inside(arch_dir, package.href)
        except ValueError as error:
            failures.add(f"{package.label}: {error}")
            continue
        if not archive.is_file():
            failures.add(f"{package.label}: missing package file {package.href}")
            continue
        actual_size, digests = _hash_file(archive, [package.digest.algorithm])
        if actual_size != package.size:
            failures.add(
                f"{package.label}: size mismatch for {package.href}: "
                f"expected {package.size}, actual {actual_size}"
            )
        actual = digests[package.digest.algorithm]
        if actual.lower() != package.digest.hex_digest.lower():
            failures.add(
                f"{package.label}: {package.digest.algorithm.upper()} mismatch for "
                f"{package.href}: expected {package.digest.hex_digest}, actual {actual}"
            )
        indexed.add(archive)
    return indexed


def _require_rpm_digest_fields(primary_xml: bytes, failures: "_Failures") -> None:
    """Require a SHA256-or-stronger digest, location, and size on every package."""
    try:
        packages = _rpm_package_elements(primary_xml)
    except ET.ParseError as error:
        failures.add(f"primary metadata is malformed: {error}")
        return
    if not packages:
        failures.add("no packages found in primary metadata")
        return
    for package in packages:
        if package.href is None or package.digest is None or package.size is None:
            failures.add(f"{package.label}: primary package metadata is incomplete")


def _rpm_primary_metadata(repomd_bytes: bytes) -> PrimaryMetadata:
    root = ET.fromstring(repomd_bytes)
    primary = None
    for child in list(root):
        if _local_name(child.tag) == "data" and child.get("type") == "primary":
            primary = child
            break
    if primary is None:
        raise PackageChecksumError(["repomd.xml has no primary metadata"])
    checksum = _child(primary, "checksum")
    location = _child(primary, "location")
    size = _child(primary, "size")
    if checksum is None or not checksum.text or location is None or size is None:
        raise PackageChecksumError(["repomd.xml primary metadata is incomplete"])
    href = location.get("href", "").strip()
    if not href:
        raise PackageChecksumError(["repomd.xml primary location is missing"])
    algorithm = _strong_sha(checksum.get("type", ""), "repomd.xml primary")
    expected = checksum.text.strip()
    if not _is_hex(expected, _digest_octets(algorithm)):
        raise PackageChecksumError(
            [f"repomd.xml primary checksum is malformed: {expected}"]
        )
    size_text = (size.text or "").strip()
    if not size_text.isdecimal():
        raise PackageChecksumError(
            [f"repomd.xml primary size is malformed: {size_text!r}"]
        )
    return PrimaryMetadata(
        href=href,
        digest=ContentDigest(algorithm=algorithm, hex_digest=expected),
        size=int(size_text),
    )


def _rpm_package_elements(primary_xml: bytes) -> list[IndexedRpm]:
    root = ET.fromstring(primary_xml)
    packages = [child for child in list(root) if _local_name(child.tag) == "package"]
    parsed: list[IndexedRpm] = []
    for package in packages:
        name = _child_text(package, "name") or "<unknown>"
        version_node = _child(package, "version")
        version = (
            version_node.get("ver", "<unknown>")
            if version_node is not None
            else "<unknown>"
        )
        label = f"{name} {version}"
        checksum = _child(package, "checksum")
        location = _child(package, "location")
        size_node = _child(package, "size")
        href = location.get("href", "").strip() if location is not None else ""
        digest: ContentDigest | None = None
        size = None
        if checksum is not None and (checksum.text or "").strip():
            algorithm = _strong_sha(checksum.get("type", ""), label)
            expected = checksum.text.strip()
            if not _is_hex(expected, _digest_octets(algorithm)):
                raise PackageChecksumError(
                    [f"{label}: checksum is malformed: {expected}"]
                )
            digest = ContentDigest(algorithm=algorithm, hex_digest=expected)
        package_size = (
            size_node.get("package", "").strip() if size_node is not None else ""
        )
        if package_size.isdecimal():
            size = int(package_size)
        parsed.append(
            IndexedRpm(
                label=label,
                href=href or None,
                digest=digest,
                size=size,
            )
        )
    return parsed


def _parse_deb_packages(text: str, failures: "_Failures") -> list[dict[str, str]]:
    stanzas: list[dict[str, str]] = []
    current: dict[str, str] = {}
    last_field = ""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            if current:
                stanzas.append(current)
                current = {}
            last_field = ""
            continue
        if line.startswith((" ", "\t")):
            if not last_field:
                failures.add(f"Packages line {line_number} continues an unknown field")
                continue
            current[last_field] = f"{current[last_field]}\n{line.strip()}"
            continue
        if ":" not in line:
            failures.add(f"Packages line {line_number} is malformed")
            continue
        field_name, value = line.split(":", 1)
        last_field = field_name.strip()
        current[last_field] = value.strip()
    if current:
        stanzas.append(current)
    return stanzas


def _parse_release_sections(text: str) -> dict[str, dict[str, tuple[str, int]]]:
    sections: dict[str, dict[str, tuple[str, int]]] = {}
    current_name = ""
    for raw_line in text.splitlines():
        if not raw_line:
            current_name = ""
            continue
        if raw_line[0].isspace():
            if not current_name:
                continue
            parts = raw_line.split(None, 2)
            if len(parts) != 3 or not parts[1].isdecimal():
                continue
            sections.setdefault(current_name, {})[parts[2]] = (parts[0], int(parts[1]))
            continue
        if raw_line.endswith(":"):
            current_name = raw_line[:-1]
            sections.setdefault(current_name, {})
            continue
        current_name = ""
    return sections


def _optional_deb_digests(
    stanza: dict[str, str],
    label: str,
    failures: "_Failures",
) -> dict[str, str]:
    """Verify SHA1 and SHA512 only when a stanza publishes those fields."""
    found: dict[str, str] = {}
    for field_name, algorithm, octets in (
        ("SHA1", "sha1", _SHA1_OCTETS),
        ("SHA512", "sha512", _SHA512_OCTETS),
    ):
        if field_name not in stanza:
            continue
        value = stanza[field_name]
        if not _is_hex(value, octets):
            failures.add(f"{label}: {field_name} is malformed: {value}")
            continue
        found[algorithm] = value
    return found


def _required_hex(
    *,
    stanza: dict[str, str],
    field_name: str,
    octets: int,
    label: str,
    failures: "_Failures",
) -> str | None:
    if field_name not in stanza:
        failures.add(f"{label}: missing {field_name}")
        return None
    value = stanza[field_name]
    if not _is_hex(value, octets):
        failures.add(f"{label}: {field_name} is malformed: {value}")
        return None
    return value


def _required_size(*, value: str, label: str, failures: "_Failures") -> int | None:
    if not value.isdecimal():
        failures.add(f"{label}: size is malformed: {value!r}")
        return None
    return int(value)


def _compare_bytes(
    *,
    payload: bytes,
    digest: ContentDigest,
    expected_size: int,
    label: str,
    failures: "_Failures",
) -> None:
    """Record a size or digest mismatch for one already-loaded file."""
    if len(payload) != expected_size:
        failures.add(
            f"{label}: size mismatch: expected {expected_size}, actual {len(payload)}"
        )
    actual = _hash_bytes(payload, digest.algorithm)
    if actual.lower() != digest.hex_digest.lower():
        failures.add(
            f"{label}: {digest.algorithm.upper()} mismatch: "
            f"expected {digest.hex_digest}, actual {actual}"
        )


def _verify_archive_set(
    root: Path,
    indexed: set[Path],
    suffix: str,
    failures: "_Failures",
) -> None:
    """Fail when a package archive on disk is missing from the index."""
    for path in root.rglob(f"*{suffix}"):
        if not path.is_file():
            continue
        try:
            resolved = _resolve_inside(root, path.relative_to(root).as_posix())
        except ValueError as error:
            failures.add(str(error))
            continue
        if resolved not in indexed:
            failures.add(
                f"{path.relative_to(root).as_posix()} is not in the repository index"
            )


def _resolve_inside(root: Path, relative: str) -> Path:
    """Resolve ``relative`` and reject paths that leave ``root``.

    Metadata filenames are untrusted. Absolute paths, backslashes, and ``..``
    segments that escape the repository are rejected before the file is opened.
    """
    raw = relative.strip()
    if not raw or raw.startswith(("/", "\\")) or "\\" in raw or "\x00" in raw:
        raise ValueError(f"{relative} escapes the repository root")
    root_resolved = root.resolve()
    candidate = (root_resolved / raw).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{relative} escapes the repository root") from error
    return candidate


def _hash_file(path: Path, algorithms: list[str]) -> tuple[int, dict[str, str]]:
    hashers = {name: _hasher(name) for name in algorithms}
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            for hasher in hashers.values():
                hasher.update(chunk)
    return size, {name: hasher.hexdigest() for name, hasher in hashers.items()}


def _hash_bytes(payload: bytes, algorithm: str) -> str:
    hasher = _hasher(algorithm)
    hasher.update(payload)
    return hasher.hexdigest()


def _hasher(algorithm: str):
    if algorithm in ("md5", "sha1"):
        return getattr(hashlib, algorithm)(usedforsecurity=False)
    if algorithm in _SHA_BITS:
        return getattr(hashlib, algorithm)()
    raise PackageChecksumError([f"unsupported checksum type: {algorithm}"])


def _strong_sha(algorithm: str, label: str) -> str:
    """Accept SHA256, SHA384, and SHA512. Reject MD5 and SHA1."""
    normalized = algorithm.strip().lower()
    if _SHA_BITS.get(normalized, 0) < 256:
        raise PackageChecksumError(
            [f"{label}: checksum type must be SHA256 or stronger, got {algorithm!r}"]
        )
    return normalized


def _digest_octets(algorithm: str) -> int:
    return _SHA_BITS[algorithm] // 8


def _is_hex(value: str, octets: int) -> bool:
    return len(value) == octets * 2 and all(
        character in _HEX_DIGITS for character in value
    )


def _read_required(path: Path, failures: "_Failures") -> bytes | None:
    if not path.is_file():
        failures.add(f"missing metadata file {path}")
        return None
    return path.read_bytes()


def _read_required_text(path: Path, failures: "_Failures") -> str | None:
    payload = _read_required(path, failures)
    if payload is None:
        return None
    return payload.decode("utf-8")


def _repo_child(repo_url: str, relative: str) -> str:
    parts = urlsplit(repo_url.strip())
    if parts.scheme != "https" or not parts.netloc:
        raise ValueError(f"Expected an HTTPS repository URL: {repo_url}")
    return urljoin(repo_url.strip().rstrip("/") + "/", relative.lstrip("/"))


def _fetch_https_bytes(url: str) -> bytes:
    """Download one metadata URL. HTTP and redirects off HTTPS are rejected."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.netloc:
        raise ValueError(f"Expected an HTTPS URL: {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "amdrocm-package-checksum/1.0"}
    )
    with urllib.request.urlopen(
        request, timeout=_FETCH_TIMEOUT_SEC
    ) as response:  # nosec B310
        final_url = response.geturl()
        if urlsplit(final_url).scheme != "https":
            raise ValueError(f"Checksum metadata redirected off HTTPS: {final_url}")
        return response.read()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(parent: ET.Element, name: str) -> ET.Element | None:
    for child in list(parent):
        if _local_name(child.tag) == name:
            return child
    return None


def _child_text(parent: ET.Element, name: str) -> str:
    child = _child(parent, name)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


class _Failures:
    """Collect every mismatch, then raise one ``PackageChecksumError``."""

    def __init__(self) -> None:
        self._errors: list[str] = []

    def add(self, message: str) -> None:
        self._errors.append(message)

    def raise_if_any(self) -> None:
        if self._errors:
            raise PackageChecksumError(self._errors)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
