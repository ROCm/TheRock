#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage ASan Python packages in a strictly local package index.

The output layout intentionally mirrors the proposed ASan publication prefix::

    <output-root>/whl-asan/gfx942-all/
      *.whl
      *.tar.gz
      index.html
      index-manifest.json
      simple/
        index.html
        <normalized-project>/index.html

``index.html`` is suitable for ``pip --find-links``. The ``simple`` subtree is
a PEP 503-style repository suitable for ``pip --index-url``. This tool only
accepts filesystem paths and has no upload or network code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from email.parser import BytesParser
from html import escape
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
from urllib.parse import quote
import zipfile

from generate_local_index import generate_simple_index


DEFAULT_FAMILY = "gfx942-all"
DEFAULT_VERSION_PREFIX = "10.1.0+asan."
PHASE1_PROJECTS = frozenset(
    {
        "rocm",
        "rocm-sdk-core",
        "rocm-sdk-libraries",
        "rocm-sdk-device-gfx942",
        "rocm-sdk-devel",
    }
)
PHASE1_ARTIFACT_TYPES = {
    project: "sdist" if project == "rocm" else "wheel" for project in PHASE1_PROJECTS
}
_FAMILY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class PackageRecord:
    filename: str
    artifact_type: str
    project: str
    normalized_project: str
    version: str
    sha256: str
    size: int


def normalize_project_name(name: str) -> str:
    """Normalize a Python distribution name as specified by PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sdist_metadata(path: Path) -> bytes:
    """Read the authoritative top-level PKG-INFO from an sdist.

    Setuptools sdists can also contain a generated ``*.egg-info/PKG-INFO``.
    The root ``<sdist-root>/PKG-INFO`` is authoritative. Auxiliary copies are
    accepted only when their normalized contents agree with it.
    """
    try:
        with tarfile.open(path, "r:gz") as sdist:
            metadata_files = sorted(
                (
                    member
                    for member in sdist.getmembers()
                    if member.isfile() and member.name.endswith("/PKG-INFO")
                ),
                key=lambda member: member.name,
            )
            canonical_files = [
                member
                for member in metadata_files
                if len(PurePosixPath(member.name).parts) == 2
                and PurePosixPath(member.name).parts[0] not in {"", ".", "..", "/"}
            ]
            if len(canonical_files) != 1:
                raise ValueError(
                    f"Expected exactly one top-level <sdist-root>/PKG-INFO in "
                    f"{path}, found {len(canonical_files)}"
                )

            canonical_file = sdist.extractfile(canonical_files[0])
            if canonical_file is None:
                raise ValueError(f"Cannot read PKG-INFO in {path}")
            canonical_bytes = canonical_file.read()
            normalized_canonical = canonical_bytes.replace(b"\r\n", b"\n").rstrip(
                b"\n"
            )

            for member in metadata_files:
                if member is canonical_files[0]:
                    continue
                metadata_file = sdist.extractfile(member)
                if metadata_file is None:
                    raise ValueError(f"Cannot read {member.name} in {path}")
                nested_bytes = metadata_file.read()
                normalized_nested = nested_bytes.replace(b"\r\n", b"\n").rstrip(
                    b"\n"
                )
                if normalized_nested != normalized_canonical:
                    raise ValueError(
                        f"Auxiliary PKG-INFO {member.name} disagrees with "
                        f"{canonical_files[0].name} in {path}"
                    )
            return canonical_bytes
    except tarfile.TarError as exc:
        raise ValueError(f"Invalid sdist archive: {path}") from exc


def inspect_package(path: Path, version_prefix: str) -> PackageRecord:
    """Read and validate identity from wheel METADATA or sdist PKG-INFO."""
    if not path.is_file():
        raise ValueError(f"Not a package file: {path}")

    if path.suffix == ".whl":
        artifact_type = "wheel"
        try:
            with zipfile.ZipFile(path) as wheel:
                metadata_files = [
                    name
                    for name in wheel.namelist()
                    if name.endswith(".dist-info/METADATA")
                ]
                if len(metadata_files) != 1:
                    raise ValueError(
                        f"Expected exactly one .dist-info/METADATA in {path}, "
                        f"found {len(metadata_files)}"
                    )
                metadata_bytes = wheel.read(metadata_files[0])
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid wheel archive: {path}") from exc
    elif path.name.endswith(".tar.gz"):
        artifact_type = "sdist"
        metadata_bytes = _read_sdist_metadata(path)
    else:
        raise ValueError(f"Unsupported package file: {path}")

    metadata = BytesParser().parsebytes(metadata_bytes)

    project = metadata.get("Name")
    version = metadata.get("Version")
    if not project or not version:
        raise ValueError(f"Package metadata lacks Name or Version: {path}")
    if not version.lower().startswith(version_prefix.lower()):
        raise ValueError(
            f"Package {path.name} has version {version!r}; expected prefix "
            f"{version_prefix!r}"
        )

    return PackageRecord(
        filename=path.name,
        artifact_type=artifact_type,
        project=project,
        normalized_project=normalize_project_name(project),
        version=version,
        sha256=sha256_file(path),
        size=path.stat().st_size,
    )


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temp:
        temp.write(content)
        temp_path = Path(temp.name)
    temp_path.replace(path)


def _html_page(title: str, links: list[tuple[str, str]]) -> str:
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8">',
        f"  <title>{escape(title)}</title>",
        "</head>",
        "<body>",
        f"  <h1>{escape(title)}</h1>",
    ]
    lines.extend(
        f'  <a href="{escape(href, quote=True)}">{escape(label)}</a><br>'
        for href, label in links
    )
    lines.extend(["</body>", "</html>", ""])
    return "\n".join(lines)


def _write_indexes(
    index_dir: Path, family: str, records: list[PackageRecord]
) -> None:
    """Write flat find-links and PEP 503 indexes for staged packages."""
    generate_simple_index(
        output_path=index_dir / "index.html",
        local_files=[index_dir / record.filename for record in records],
        title=f"ROCm 10.1 ASan Python Packages - {family}",
    )

    by_project: dict[str, list[PackageRecord]] = {}
    for record in records:
        by_project.setdefault(record.normalized_project, []).append(record)

    simple_dir = index_dir / "simple"
    root_links = [(f"./{quote(project)}/", project) for project in sorted(by_project)]
    _write_text_atomic(
        simple_dir / "index.html", _html_page("Simple index", root_links)
    )

    for project, project_records in sorted(by_project.items()):
        links = [
            (
                f"../../{quote(record.filename)}#sha256={record.sha256}",
                record.filename,
            )
            for record in sorted(project_records, key=lambda item: item.filename)
        ]
        _write_text_atomic(
            simple_dir / project / "index.html",
            _html_page(f"Links for {project}", links),
        )


def _write_manifest(index_dir: Path, family: str, records: list[PackageRecord]) -> None:
    manifest = {
        "schema_version": 1,
        "index_kind": "local-only",
        "relative_path": f"whl-asan/{family}",
        "package_count": len(records),
        "packages": [asdict(record) for record in records],
    }
    _write_text_atomic(
        index_dir / "index-manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


def _validate_family(family: str) -> None:
    if not _FAMILY_RE.fullmatch(family) or family in {".", ".."}:
        raise ValueError(f"Invalid family name: {family!r}")


def _validate_phase1_set(records: list[PackageRecord]) -> None:
    found_projects = {record.normalized_project for record in records}
    missing = sorted(PHASE1_PROJECTS - found_projects)
    if missing:
        raise ValueError(
            "Local index is missing required Phase 1 projects: " + ", ".join(missing)
        )
    for project, artifact_type in PHASE1_ARTIFACT_TYPES.items():
        if not any(
            record.normalized_project == project
            and record.artifact_type == artifact_type
            for record in records
        ):
            raise ValueError(
                f"Phase 1 project {project} requires a {artifact_type} artifact"
            )


def _validate_consistent_versions(records: list[PackageRecord]) -> None:
    versions = sorted({record.version for record in records})
    if len(versions) != 1:
        raise ValueError(
            "Local index must contain exactly one package version; found: "
            + ", ".join(versions)
        )


def stage_index(
    input_dir: Path,
    output_root: Path,
    *,
    family: str = DEFAULT_FAMILY,
    version_prefix: str = DEFAULT_VERSION_PREFIX,
    require_phase1_set: bool = False,
) -> Path:
    """Validate and copy packages into a local ``whl-asan`` index."""
    _validate_family(family)
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    wheel_dir = input_dir / "dist" if (input_dir / "dist").is_dir() else input_dir
    source_packages = sorted(
        path
        for pattern in ("*.whl", "*.tar.gz")
        for path in wheel_dir.glob(pattern)
        if path.is_file()
    )
    if not source_packages:
        raise FileNotFoundError(f"No wheel or sdist files found in {wheel_dir}")

    source_records = {
        package.name: inspect_package(package, version_prefix)
        for package in source_packages
    }
    index_dir = output_root.resolve() / "whl-asan" / family
    index_dir.mkdir(parents=True, exist_ok=True)

    # Never silently replace a differently-built package with the same filename.
    for source in source_packages:
        destination = index_dir / source.name
        if (
            destination.exists()
            and sha256_file(destination) != source_records[source.name].sha256
        ):
            raise FileExistsError(
                "Refusing to overwrite package with different contents: "
                f"{destination}"
            )

    for source in source_packages:
        destination = index_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)

    staged_packages = sorted(
        path
        for pattern in ("*.whl", "*.tar.gz")
        for path in index_dir.glob(pattern)
    )
    records = [
        inspect_package(package, version_prefix) for package in staged_packages
    ]
    records.sort(key=lambda item: (item.normalized_project, item.filename))
    _validate_consistent_versions(records)

    if require_phase1_set:
        _validate_phase1_set(records)

    _write_indexes(index_dir, family, records)
    _write_manifest(index_dir, family, records)
    return index_dir


def verify_index(
    output_root: Path,
    *,
    family: str = DEFAULT_FAMILY,
    version_prefix: str = DEFAULT_VERSION_PREFIX,
    require_phase1_set: bool = False,
) -> Path:
    """Verify wheel hashes, metadata, completeness, and generated indexes."""
    _validate_family(family)
    index_dir = output_root.resolve() / "whl-asan" / family
    manifest_path = index_dir / "index-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("index_kind") != "local-only":
        raise ValueError(f"Unexpected index kind in {manifest_path}")
    if manifest.get("relative_path") != f"whl-asan/{family}":
        raise ValueError(f"Unexpected index path in {manifest_path}")

    expected = {item["filename"]: item for item in manifest.get("packages", [])}
    actual_names = {
        path.name
        for pattern in ("*.whl", "*.tar.gz")
        for path in index_dir.glob(pattern)
    }
    if actual_names != set(expected):
        raise ValueError(
            "Staged packages differ from manifest: "
            f"expected={sorted(expected)}, actual={sorted(actual_names)}"
        )

    records = []
    for filename in sorted(expected):
        record = inspect_package(index_dir / filename, version_prefix)
        if asdict(record) != expected[filename]:
            raise ValueError(f"Package does not match manifest: {filename}")
        records.append(record)

    _validate_consistent_versions(records)
    if require_phase1_set:
        _validate_phase1_set(records)

    required_indexes = [index_dir / "index.html", index_dir / "simple" / "index.html"]
    required_indexes.extend(
        index_dir / "simple" / record.normalized_project / "index.html"
        for record in records
    )
    missing_indexes = sorted(
        str(path) for path in required_indexes if not path.is_file()
    )
    if missing_indexes:
        raise FileNotFoundError("Missing index files: " + ", ".join(missing_indexes))

    flat_index = (index_dir / "index.html").read_text(encoding="utf-8")
    simple_root = (index_dir / "simple" / "index.html").read_text(
        encoding="utf-8"
    )
    for record in records:
        if f'./{quote(record.filename)}' not in flat_index:
            raise ValueError(
                f"Package missing from find-links index: {record.filename}"
            )
        if f'./{quote(record.normalized_project)}/' not in simple_root:
            raise ValueError(
                f"Project missing from simple index: {record.normalized_project}"
            )
        project_index = (
            index_dir / "simple" / record.normalized_project / "index.html"
        ).read_text(encoding="utf-8")
        expected_link = (
            f"../../{quote(record.filename)}#sha256={record.sha256}"
        )
        if expected_link not in project_index:
            raise ValueError(
                f"Package missing from project index: {record.filename}"
            )
    return index_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser(
        "stage", help="Validate, copy, and index wheels and sdists"
    )
    stage.add_argument("--input-dir", type=Path, required=True)
    stage.add_argument("--output-root", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Verify an existing local index")
    verify.add_argument("--output-root", type=Path, required=True)

    for command in (stage, verify):
        command.add_argument("--family", default=DEFAULT_FAMILY)
        command.add_argument("--version-prefix", default=DEFAULT_VERSION_PREFIX)
        command.add_argument(
            "--require-phase1-set",
            action="store_true",
            help=(
                "Require the rocm selector sdist plus core, libraries, gfx942 "
                "device, and devel wheels"
            ),
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "stage":
            index_dir = stage_index(
                args.input_dir,
                args.output_root,
                family=args.family,
                version_prefix=args.version_prefix,
                require_phase1_set=args.require_phase1_set,
            )
        else:
            index_dir = verify_index(
                args.output_root,
                family=args.family,
                version_prefix=args.version_prefix,
                require_phase1_set=args.require_phase1_set,
            )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(index_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
