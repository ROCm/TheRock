# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve and mirror third-party Python wheels into a structured S3 index.

This tool is intentionally independent of the legacy PyTorch test-infra fork
under ``build_tools/third_party/s3_management``. It supports local use and CI:

* ``resolve`` downloads one deterministic dependency snapshot from PyPI.
* ``publish`` uploads or refreshes that snapshot in one S3 bucket.
* ``mirror`` performs both operations for convenient one-bucket local use.
"""

from argparse import ArgumentParser, Namespace
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Protocol, runtime_checkable
from urllib.request import Request, urlopen

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version


_ALLOWED_PLATFORM_TAGS = frozenset({"any", "linux_x86_64", "win_amd64"})
_ALLOWED_CPYTHON_TAGS = frozenset(
    {"cp310", "cp311", "cp312", "cp313", "cp314", "cp315"}
)
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_INDEX_NAME = "whl-next"
_MANIFEST_FILENAME = "manifest.json"
_MANIFEST_SCHEMA_VERSION = 1
_PYPI_ACCEPT = "application/json"
_S3_CONTENT_TYPE = "application/octet-stream"
_NORMALIZED_PACKAGE_PATTERN = re.compile(r"[-_.]+")
_PACKAGE_NAME_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_LOWERCASE_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class DependencyPolicy:
    """Configured versions of one dependency belonging to one project."""

    project: str
    versions: tuple[str, ...]


DEPENDENCIES: dict[str, DependencyPolicy] = {
    "ml_dtypes": DependencyPolicy(project="jax", versions=("latest",)),
    "opt_einsum": DependencyPolicy(project="jax", versions=("latest",)),
    "tomli": DependencyPolicy(project="jax", versions=("latest",)),
    "sympy": DependencyPolicy(project="torch", versions=("latest",)),
    "mpmath": DependencyPolicy(project="torch", versions=("1.3.0",)),
    "pillow": DependencyPolicy(project="torch", versions=("latest",)),
    # 3.4.2 supports Python 3.10; latest supports newer Python versions.
    "networkx": DependencyPolicy(project="torch", versions=("3.4.2", "latest")),
    # 2.2.6 supports Python 3.10 and 3.11; latest supports newer versions.
    "numpy": DependencyPolicy(project="torch", versions=("2.2.6", "latest")),
    "jinja2": DependencyPolicy(project="torch", versions=("latest",)),
    "markupsafe": DependencyPolicy(project="torch", versions=("latest",)),
    "filelock": DependencyPolicy(project="torch", versions=("latest",)),
    "fsspec": DependencyPolicy(project="torch", versions=("latest",)),
    "typing-extensions": DependencyPolicy(project="torch", versions=("latest",)),
    "rocm-bootstrap": DependencyPolicy(project="rocm", versions=("latest",)),
    "setuptools": DependencyPolicy(project="rocm", versions=("81.0.0",)),
}


@dataclass(frozen=True)
class PypiWheel:
    """One allowed wheel described by PyPI project metadata."""

    filename: str
    url: str
    sha256: str
    size: int
    version: Version


@dataclass(frozen=True)
class SnapshotWheel:
    """One downloaded wheel and its destination identity."""

    package: str
    project: str
    requested_version: str
    selected_version: str
    filename: str
    source_url: str
    size: int
    sha256: str
    relative_path: str
    destination_key: str


@dataclass(frozen=True)
class DependencySnapshot:
    """A complete, locally materialized dependency resolution."""

    schema_version: int
    created_at: str
    source_revision: str | None
    wheels: tuple[SnapshotWheel, ...]


@dataclass(frozen=True)
class PublishSummary:
    """Counts for one bucket publication."""

    uploaded: int
    refreshed: int
    skipped: int


class S3Client(Protocol):
    """Subset of the boto3 S3 client used by this tool."""

    def head_object(self, **kwargs: object) -> dict[str, object]: ...

    def get_object(self, **kwargs: object) -> dict[str, object]: ...

    def put_object(self, **kwargs: object) -> dict[str, object]: ...

    def copy_object(self, **kwargs: object) -> dict[str, object]: ...


@runtime_checkable
class S3Body(Protocol):
    """Readable response body returned by the S3 client."""

    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


def normalize_package_name(name: str) -> str:
    """Return the PEP 503-normalized distribution name."""
    return _NORMALIZED_PACKAGE_PATTERN.sub("-", name).lower()


def dependency_destination_key(package: str, filename: str) -> str:
    """Return the structured core ``whl-next`` destination key."""
    normalized_package = normalize_package_name(package)
    return f"v5/rocm/core/{_INDEX_NAME}/{normalized_package}/{filename}"


def is_wheel_allowed(filename: str) -> bool:
    """Return whether a wheel has a supported Python and platform tag."""
    filename_parts = filename.removesuffix(".whl").rsplit("-", maxsplit=3)
    if not filename.endswith(".whl") or len(filename_parts) != 4:
        return False
    python_tag = filename_parts[1]
    if python_tag not in _ALLOWED_CPYTHON_TAGS and python_tag != "py3":
        return False

    try:
        _, _, _, tags = parse_wheel_filename(filename)
    except ValueError:
        return False

    return any(
        not tag.abi.endswith("t") and _is_platform_tag_allowed(tag.platform)
        for tag in tags
    )


def _is_platform_tag_allowed(platform_tag: str) -> bool:
    return platform_tag in _ALLOWED_PLATFORM_TAGS or (
        platform_tag.startswith("manylinux") and platform_tag.endswith("_x86_64")
    )


def select_dependencies(
    *,
    project: str = "all",
    dependency_names: frozenset[str] | None = None,
) -> dict[str, DependencyPolicy]:
    """Select configured dependencies by project and optional package names."""
    projects = sorted({policy.project for policy in DEPENDENCIES.values()})
    if project != "all" and project not in projects:
        raise ValueError(
            f"project={project!r} is invalid; expected 'all' or one of {projects}"
        )

    selected = {
        name: policy
        for name, policy in DEPENDENCIES.items()
        if project == "all" or policy.project == project
    }
    if dependency_names is None:
        return selected

    requested = {normalize_package_name(name) for name in dependency_names}
    selected_by_normalized_name = {
        normalize_package_name(name): (name, policy)
        for name, policy in selected.items()
    }
    unknown = requested - selected_by_normalized_name.keys()
    if unknown:
        raise ValueError(
            f"Unknown dependency package(s) for project {project!r}: "
            f"{sorted(unknown)}"
        )
    return {
        selected_by_normalized_name[name][0]: selected_by_normalized_name[name][1]
        for name in sorted(requested)
    }


def fetch_pypi_project(package: str) -> dict[str, object]:
    """Fetch and validate top-level PyPI JSON project metadata."""
    url = f"https://pypi.org/pypi/{package}/json"
    request = Request(url, headers={"Accept": _PYPI_ACCEPT})
    with urlopen(request) as response:
        raw_data: object = json.load(response)
    return _require_mapping(raw_data, f"PyPI response for {package}")


def parse_pypi_wheels(package: str, metadata: dict[str, object]) -> list[PypiWheel]:
    """Parse allowed, non-yanked wheels from PyPI project metadata."""
    releases = _require_mapping(
        metadata.get("releases"), f"PyPI response for {package}.releases"
    )
    wheels: list[PypiWheel] = []
    for raw_version, raw_files in releases.items():
        try:
            release_version = Version(raw_version)
        except InvalidVersion:
            continue
        files = _require_list(
            raw_files, f"PyPI response for {package}.releases[{raw_version!r}]"
        )
        for file_index, raw_file in enumerate(files):
            context = (
                f"PyPI response for {package}.releases[{raw_version!r}]"
                f"[{file_index}]"
            )
            file_info = _require_mapping(raw_file, context)
            wheel = _parse_pypi_wheel(
                package=package,
                release_version=release_version,
                file_info=file_info,
                context=context,
            )
            if wheel is not None:
                wheels.append(wheel)
    return sorted(wheels, key=lambda wheel: (wheel.version, wheel.filename))


def _parse_pypi_wheel(
    *,
    package: str,
    release_version: Version,
    file_info: dict[str, object],
    context: str,
) -> PypiWheel | None:
    filename_value = file_info.get("filename")
    if not isinstance(filename_value, str) or not is_wheel_allowed(filename_value):
        return None
    if file_info.get("yanked", False):
        return None
    if file_info.get("packagetype") != "bdist_wheel":
        return None

    try:
        distribution, wheel_version, _, _ = parse_wheel_filename(filename_value)
    except ValueError as exc:
        raise ValueError(
            f"{context} has invalid wheel filename {filename_value!r}"
        ) from exc
    if canonicalize_name(str(distribution)) != canonicalize_name(package):
        raise ValueError(
            f"{context} filename distribution {distribution!s} does not match "
            f"package {package!r}"
        )
    if wheel_version != release_version:
        raise ValueError(
            f"{context} filename version {wheel_version!s} does not match "
            f"release {release_version!s}"
        )

    digests = _require_mapping(file_info.get("digests"), f"{context}.digests")
    sha256 = _require_nonempty_string(digests.get("sha256"), f"{context}.sha256")
    if _SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError(f"{context}.sha256 is not a SHA256 hex digest")
    url = _require_nonempty_string(file_info.get("url"), f"{context}.url")
    size = _require_positive_int(file_info.get("size"), f"{context}.size")
    return PypiWheel(
        filename=filename_value,
        url=url,
        sha256=sha256.lower(),
        size=size,
        version=release_version,
    )


def select_release_wheels(
    *,
    package: str,
    requested_version: str,
    candidates: list[PypiWheel],
) -> list[PypiWheel]:
    """Select one configured release and all of its allowed wheels."""
    by_version: dict[Version, list[PypiWheel]] = {}
    for wheel in candidates:
        by_version.setdefault(wheel.version, []).append(wheel)

    if requested_version == "latest":
        stable_versions = [
            version
            for version in by_version
            if not version.is_prerelease and not version.is_devrelease
        ]
        if not stable_versions:
            raise ValueError(f"No allowed stable wheels found for {package}")
        selected_version = max(stable_versions)
    else:
        try:
            selected_version = Version(requested_version)
        except InvalidVersion as exc:
            raise ValueError(
                f"Configured version {requested_version!r} for {package} is invalid"
            ) from exc
        if selected_version not in by_version:
            raise ValueError(
                f"Configured version {requested_version!r} for {package} has no "
                "allowed non-yanked wheels"
            )
    return sorted(by_version[selected_version], key=lambda wheel: wheel.filename)


def download_wheel(wheel: PypiWheel, output_path: Path) -> None:
    """Download one wheel and validate its size and PyPI SHA256."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    request = Request(wheel.url, headers={"Accept": "application/octet-stream"})
    with urlopen(request) as response, output_path.open("wb") as output_file:
        while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
            output_file.write(chunk)
            digest.update(chunk)
            size += len(chunk)

    if size != wheel.size:
        raise ValueError(
            f"Downloaded {wheel.filename} has size {size}, expected {wheel.size}"
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != wheel.sha256:
        raise ValueError(
            f"Downloaded {wheel.filename} has SHA256 {actual_sha256}, "
            f"expected {wheel.sha256}"
        )


def resolve_snapshot(
    *,
    output_dir: Path,
    selected_dependencies: dict[str, DependencyPolicy],
    source_revision: str | None = None,
) -> DependencySnapshot:
    """Resolve and download one complete dependency snapshot."""
    if output_dir.exists():
        raise FileExistsError(f"Snapshot output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temp_dir:
        staging_dir = Path(temp_dir) / "snapshot"
        staging_dir.mkdir()
        snapshot = _resolve_snapshot_into(
            output_dir=staging_dir,
            selected_dependencies=selected_dependencies,
            source_revision=source_revision,
        )
        staging_dir.rename(output_dir)

    print(
        f"Resolved {len(snapshot.wheels)} wheels into "
        f"{output_dir / _MANIFEST_FILENAME}"
    )
    return snapshot


def _resolve_snapshot_into(
    *,
    output_dir: Path,
    selected_dependencies: dict[str, DependencyPolicy],
    source_revision: str | None,
) -> DependencySnapshot:
    records_by_key: dict[str, SnapshotWheel] = {}
    for package, policy in sorted(selected_dependencies.items()):
        print(f"Resolving {package}")
        candidates = parse_pypi_wheels(package, fetch_pypi_project(package))
        for requested_version in policy.versions:
            selected_wheels = select_release_wheels(
                package=package,
                requested_version=requested_version,
                candidates=candidates,
            )
            for wheel in selected_wheels:
                record = _download_snapshot_wheel(
                    output_dir=output_dir,
                    package=package,
                    project=policy.project,
                    requested_version=requested_version,
                    wheel=wheel,
                )
                existing = records_by_key.get(record.destination_key)
                if existing is not None and existing.sha256 != record.sha256:
                    raise ValueError(
                        f"Conflicting content selected for {record.destination_key}"
                    )
                records_by_key.setdefault(record.destination_key, record)

    if not records_by_key:
        raise ValueError("Dependency resolution selected no wheels")
    snapshot = DependencySnapshot(
        schema_version=_MANIFEST_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_revision=source_revision,
        wheels=tuple(records_by_key[key] for key in sorted(records_by_key)),
    )
    write_snapshot_manifest(snapshot, output_dir / _MANIFEST_FILENAME)
    return snapshot


def _download_snapshot_wheel(
    *,
    output_dir: Path,
    package: str,
    project: str,
    requested_version: str,
    wheel: PypiWheel,
) -> SnapshotWheel:
    normalized_package = normalize_package_name(package)
    relative_path = Path("wheels", normalized_package, wheel.filename)
    output_path = output_dir / relative_path
    if not output_path.exists():
        print(f"Downloading {wheel.filename}")
        download_wheel(wheel, output_path)
    else:
        _validate_local_file(output_path, wheel.size, wheel.sha256)
    return SnapshotWheel(
        package=package,
        project=project,
        requested_version=requested_version,
        selected_version=str(wheel.version),
        filename=wheel.filename,
        source_url=wheel.url,
        size=wheel.size,
        sha256=wheel.sha256,
        relative_path=relative_path.as_posix(),
        destination_key=dependency_destination_key(package, wheel.filename),
    )


def write_snapshot_manifest(snapshot: DependencySnapshot, manifest_path: Path) -> None:
    """Write a deterministic JSON snapshot manifest."""
    data = {
        "schema_version": snapshot.schema_version,
        "created_at": snapshot.created_at,
        "source_revision": snapshot.source_revision,
        "wheels": [asdict(wheel) for wheel in snapshot.wheels],
    }
    manifest_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if manifest_path.stat().st_size == 0:
        raise RuntimeError(f"Snapshot manifest is empty: {manifest_path}")


def append_snapshot_summary(snapshot: DependencySnapshot, summary_path: Path) -> None:
    """Append a Markdown snapshot summary for CI or local inspection."""
    package_versions = sorted(
        {(wheel.package, wheel.selected_version) for wheel in snapshot.wheels}
    )
    lines = ["## Python dependency mirror", ""]
    if snapshot.source_revision:
        lines.extend([f"TheRock revision: `{snapshot.source_revision}`", ""])
    lines.extend(["| Package | Version |", "| --- | --- |"])
    lines.extend(
        f"| `{package}` | `{version}` |" for package, version in package_versions
    )
    lines.append("")
    _append_lines(summary_path, lines)


def append_publish_summary(
    *,
    bucket: str,
    summary: PublishSummary,
    dry_run: bool,
    summary_path: Path,
) -> None:
    """Append a concise Markdown result for one bucket publication."""
    operation = "dry-run" if dry_run else "published"
    _append_lines(
        summary_path,
        [
            f"- `{bucket}`: {operation}; uploaded {summary.uploaded}, "
            f"refreshed {summary.refreshed}, skipped {summary.skipped}",
        ],
    )


def _append_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def load_snapshot(snapshot_dir: Path) -> DependencySnapshot:
    """Load and validate a dependency snapshot and all local wheel files."""
    manifest_path = snapshot_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")
    raw_data: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = _require_mapping(raw_data, "snapshot manifest")
    schema_version = _require_positive_int(
        data.get("schema_version"), "snapshot manifest.schema_version"
    )
    if schema_version != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported snapshot schema {schema_version}; "
            f"expected {_MANIFEST_SCHEMA_VERSION}"
        )
    created_at = _require_nonempty_string(
        data.get("created_at"), "snapshot manifest.created_at"
    )
    source_revision_value = data.get("source_revision")
    if source_revision_value is not None and not isinstance(source_revision_value, str):
        raise ValueError("snapshot manifest.source_revision must be a string or null")

    raw_wheels = _require_list(data.get("wheels"), "snapshot manifest.wheels")
    if not raw_wheels:
        raise ValueError("snapshot manifest.wheels must not be empty")
    wheels = tuple(
        _parse_snapshot_wheel(raw_wheel, index)
        for index, raw_wheel in enumerate(raw_wheels)
    )
    destination_keys = [wheel.destination_key for wheel in wheels]
    if len(destination_keys) != len(set(destination_keys)):
        raise ValueError("snapshot manifest contains duplicate destination keys")

    snapshot = DependencySnapshot(
        schema_version=schema_version,
        created_at=created_at,
        source_revision=source_revision_value,
        wheels=wheels,
    )
    for wheel in snapshot.wheels:
        _validate_snapshot_wheel_file(snapshot_dir, wheel)
    return snapshot


def _parse_snapshot_wheel(raw_wheel: object, index: int) -> SnapshotWheel:
    context = f"snapshot manifest.wheels[{index}]"
    data = _require_mapping(raw_wheel, context)
    package = _require_package_name(data.get("package"), f"{context}.package")
    filename = _require_nonempty_string(data.get("filename"), f"{context}.filename")
    selected_version = _require_nonempty_string(
        data.get("selected_version"), f"{context}.selected_version"
    )
    _validate_snapshot_wheel_identity(
        package=package,
        selected_version=selected_version,
        filename=filename,
        context=context,
    )
    relative_path = _require_nonempty_string(
        data.get("relative_path"), f"{context}.relative_path"
    )
    expected_relative_path = Path(
        "wheels", normalize_package_name(package), filename
    ).as_posix()
    if relative_path != expected_relative_path:
        raise ValueError(
            f"{context}.relative_path is {relative_path!r}, "
            f"expected {expected_relative_path!r}"
        )
    destination_key = _require_nonempty_string(
        data.get("destination_key"), f"{context}.destination_key"
    )
    expected_key = dependency_destination_key(package, filename)
    if destination_key != expected_key:
        raise ValueError(
            f"{context}.destination_key is {destination_key!r}, "
            f"expected {expected_key!r}"
        )
    sha256 = _require_nonempty_string(data.get("sha256"), f"{context}.sha256")
    if _LOWERCASE_SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError(f"{context}.sha256 is not a lowercase SHA256 hex digest")
    return SnapshotWheel(
        package=package,
        project=_require_nonempty_string(data.get("project"), f"{context}.project"),
        requested_version=_require_nonempty_string(
            data.get("requested_version"), f"{context}.requested_version"
        ),
        selected_version=selected_version,
        filename=filename,
        source_url=_require_nonempty_string(
            data.get("source_url"), f"{context}.source_url"
        ),
        size=_require_positive_int(data.get("size"), f"{context}.size"),
        sha256=sha256,
        relative_path=relative_path,
        destination_key=destination_key,
    )


def _validate_snapshot_wheel_identity(
    *,
    package: str,
    selected_version: str,
    filename: str,
    context: str,
) -> None:
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError(f"{context}.filename must be a wheel basename")
    try:
        distribution, wheel_version, _, _ = parse_wheel_filename(filename)
    except ValueError as exc:
        raise ValueError(f"{context}.filename is not a valid wheel filename") from exc
    if not is_wheel_allowed(filename):
        raise ValueError(f"{context}.filename has unsupported wheel tags")
    if canonicalize_name(str(distribution)) != canonicalize_name(package):
        raise ValueError(
            f"{context}.filename distribution {distribution!s} does not match "
            f"package {package!r}"
        )
    try:
        manifest_version = Version(selected_version)
    except InvalidVersion as exc:
        raise ValueError(f"{context}.selected_version is invalid") from exc
    if wheel_version != manifest_version:
        raise ValueError(
            f"{context}.filename version {wheel_version!s} does not match "
            f"selected_version {selected_version!r}"
        )


def _validate_snapshot_wheel_file(snapshot_dir: Path, wheel: SnapshotWheel) -> None:
    path = _resolve_snapshot_wheel_path(snapshot_dir, wheel)
    _validate_local_file(path, wheel.size, wheel.sha256)


def _resolve_snapshot_wheel_path(
    snapshot_dir: Path,
    wheel: SnapshotWheel,
) -> Path:
    snapshot_root = snapshot_dir.resolve(strict=True)
    path = (snapshot_root / wheel.relative_path).resolve(strict=False)
    if not path.is_relative_to(snapshot_root):
        raise ValueError(
            f"Snapshot wheel path escapes snapshot directory: {wheel.relative_path}"
        )
    return path


def _validate_local_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot wheel not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Snapshot wheel {path} has size {actual_size}, expected {expected_size}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(_DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Snapshot wheel {path} has SHA256 {actual_sha256}, "
            f"expected {expected_sha256}"
        )


def publish_snapshot(
    *,
    snapshot_dir: Path,
    bucket: str,
    refresh_existing: bool,
    dry_run: bool = False,
    s3_client: S3Client | None = None,
) -> PublishSummary:
    """Publish one validated local snapshot into one S3 bucket."""
    snapshot = load_snapshot(snapshot_dir)
    client = s3_client if s3_client is not None else boto3.client("s3")
    uploaded = 0
    refreshed = 0
    skipped = 0

    for wheel in snapshot.wheels:
        path = _resolve_snapshot_wheel_path(snapshot_dir, wheel)
        head = _head_object(client, bucket, wheel.destination_key)
        if head is None:
            print(f"Uploading s3://{bucket}/{wheel.destination_key}")
            if not dry_run:
                _upload_wheel(client, bucket, wheel, path)
                _verify_s3_object(client, bucket, wheel)
            uploaded += 1
        elif refresh_existing:
            _verify_existing_s3_object(client, bucket, wheel, head)
            print(f"Refreshing s3://{bucket}/{wheel.destination_key}")
            if not dry_run:
                _refresh_s3_object(client, bucket, wheel, head)
                _verify_s3_object(client, bucket, wheel)
            refreshed += 1
        else:
            _verify_existing_s3_object(client, bucket, wheel, head)
            print(f"Skipping existing s3://{bucket}/{wheel.destination_key}")
            skipped += 1

    summary = PublishSummary(
        uploaded=uploaded,
        refreshed=refreshed,
        skipped=skipped,
    )
    print(
        f"Publication summary for {bucket}: uploaded={summary.uploaded}, "
        f"refreshed={summary.refreshed}, skipped={summary.skipped}"
    )
    return summary


def _head_object(client: S3Client, bucket: str, key: str) -> dict[str, object] | None:
    try:
        return client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def _upload_wheel(
    client: S3Client,
    bucket: str,
    wheel: SnapshotWheel,
    path: Path,
) -> None:
    with path.open("rb") as input_file:
        client.put_object(
            Bucket=bucket,
            Key=wheel.destination_key,
            Body=input_file,
            ContentLength=wheel.size,
            ContentType=_S3_CONTENT_TYPE,
            ChecksumSHA256=_expected_s3_checksum(wheel),
        )


def _refresh_s3_object(
    client: S3Client,
    bucket: str,
    wheel: SnapshotWheel,
    head: dict[str, object],
) -> None:
    etag = _require_nonempty_string(head.get("ETag"), "S3 object ETag")
    metadata_value = head.get("Metadata", {})
    metadata = _require_string_mapping(metadata_value, "S3 object Metadata")
    copy_arguments: dict[str, object] = {
        "Bucket": bucket,
        "Key": wheel.destination_key,
        "CopySource": {"Bucket": bucket, "Key": wheel.destination_key},
        "CopySourceIfMatch": etag,
        "Metadata": metadata,
        "MetadataDirective": "REPLACE",
        "ChecksumAlgorithm": "SHA256",
        "ContentType": head.get("ContentType", _S3_CONTENT_TYPE),
    }
    copy_fields = {
        "CacheControl": "CacheControl",
        "ContentDisposition": "ContentDisposition",
        "ContentEncoding": "ContentEncoding",
        "ContentLanguage": "ContentLanguage",
        "Expires": "Expires",
        "WebsiteRedirectLocation": "WebsiteRedirectLocation",
    }
    for source_name, destination_name in copy_fields.items():
        if source_name in head:
            copy_arguments[destination_name] = head[source_name]
    client.copy_object(**copy_arguments)


def _verify_s3_object(
    client: S3Client,
    bucket: str,
    wheel: SnapshotWheel,
) -> None:
    head = client.head_object(
        Bucket=bucket,
        Key=wheel.destination_key,
        ChecksumMode="ENABLED",
    )
    _verify_s3_size(head, bucket, wheel)
    if not _verify_stored_s3_checksum(head, bucket, wheel):
        raise ValueError(
            f"s3://{bucket}/{wheel.destination_key} does not provide a "
            "full-object SHA256 checksum"
        )


def _verify_existing_s3_object(
    client: S3Client,
    bucket: str,
    wheel: SnapshotWheel,
    head: dict[str, object],
) -> None:
    _verify_s3_size(head, bucket, wheel)
    if not _verify_stored_s3_checksum(head, bucket, wheel):
        _verify_downloaded_s3_object(client, bucket, wheel)


def _verify_stored_s3_checksum(
    head: dict[str, object],
    bucket: str,
    wheel: SnapshotWheel,
) -> bool:
    actual_checksum = head.get("ChecksumSHA256")
    if actual_checksum is None or head.get("ChecksumType") == "COMPOSITE":
        return False
    expected_checksum = _expected_s3_checksum(wheel)
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"s3://{bucket}/{wheel.destination_key} has SHA256 checksum "
            f"{actual_checksum!r}, expected {expected_checksum!r}"
        )
    return True


def _verify_downloaded_s3_object(
    client: S3Client,
    bucket: str,
    wheel: SnapshotWheel,
) -> None:
    response = client.get_object(Bucket=bucket, Key=wheel.destination_key)
    body_value = response.get("Body")
    if not isinstance(body_value, S3Body):
        raise ValueError(
            f"s3://{bucket}/{wheel.destination_key} returned an unreadable body"
        )

    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := body_value.read(_DOWNLOAD_CHUNK_SIZE):
            if not isinstance(chunk, bytes):
                raise ValueError(
                    f"s3://{bucket}/{wheel.destination_key} returned non-byte data"
                )
            digest.update(chunk)
            size += len(chunk)
    finally:
        body_value.close()

    if size != wheel.size:
        raise ValueError(
            f"s3://{bucket}/{wheel.destination_key} downloaded size {size}, "
            f"expected {wheel.size}"
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != wheel.sha256:
        raise ValueError(
            f"s3://{bucket}/{wheel.destination_key} has SHA256 "
            f"{actual_sha256}, expected {wheel.sha256}"
        )


def _expected_s3_checksum(wheel: SnapshotWheel) -> str:
    return base64.b64encode(bytes.fromhex(wheel.sha256)).decode("ascii")


def _verify_s3_size(head: dict[str, object], bucket: str, wheel: SnapshotWheel) -> None:
    actual_size = head.get("ContentLength")
    if actual_size != wheel.size:
        raise ValueError(
            f"s3://{bucket}/{wheel.destination_key} has size {actual_size!r}, "
            f"expected {wheel.size}"
        )


def _require_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value.keys()
    ):
        raise ValueError(f"{context} must be an object with string keys")
    return value


def _require_string_mapping(value: object, context: str) -> dict[str, str]:
    mapping = _require_mapping(value, context)
    if not all(isinstance(item, str) for item in mapping.values()):
        raise ValueError(f"{context} values must be strings")
    return {key: item for key, item in mapping.items() if isinstance(item, str)}


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _require_nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_package_name(value: object, context: str) -> str:
    name = _require_nonempty_string(value, context)
    if _PACKAGE_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"{context} is not a valid Python package name")
    return name


def _require_positive_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _add_selection_arguments(parser: ArgumentParser) -> None:
    projects = sorted({policy.project for policy in DEPENDENCIES.values()})
    parser.add_argument(
        "--project",
        choices=["all", *projects],
        default="all",
        help="Configured dependency project to mirror (default: all).",
    )
    parser.add_argument(
        "--dependency-package",
        action="append",
        dest="dependency_packages",
        help="Limit to one configured dependency; may be passed repeatedly.",
    )
    parser.add_argument(
        "--source-revision",
        help="Optional TheRock revision to record in the snapshot manifest.",
    )


def _add_publish_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--bucket", required=True, help="Destination S3 bucket.")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Refresh existing keys with a conditional same-key S3 copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect S3 and report actions without writing.",
    )


def _add_summary_argument(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Append a Markdown operation summary to this file.",
    )


def create_parser() -> ArgumentParser:
    """Create the command-line parser."""
    parser = ArgumentParser(
        description="Resolve and mirror third-party Python wheel snapshots."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser(
        "resolve", help="Resolve and download a local dependency snapshot."
    )
    resolve_parser.add_argument("--output-dir", type=Path, required=True)
    _add_selection_arguments(resolve_parser)
    _add_summary_argument(resolve_parser)

    publish_parser = subparsers.add_parser(
        "publish", help="Publish one existing local snapshot to S3."
    )
    publish_parser.add_argument("--snapshot-dir", type=Path, required=True)
    _add_publish_arguments(publish_parser)
    _add_summary_argument(publish_parser)

    mirror_parser = subparsers.add_parser(
        "mirror", help="Resolve and publish to one S3 bucket."
    )
    mirror_parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help=(
            "Keep the resolved snapshot at this path instead of using a "
            "temporary directory."
        ),
    )
    _add_selection_arguments(mirror_parser)
    _add_publish_arguments(mirror_parser)
    _add_summary_argument(mirror_parser)
    return parser


def _selected_dependencies_from_args(args: Namespace) -> dict[str, DependencyPolicy]:
    dependency_names = (
        frozenset(args.dependency_packages) if args.dependency_packages else None
    )
    return select_dependencies(
        project=args.project,
        dependency_names=dependency_names,
    )


def main(argv: list[str]) -> int:
    """Run the local/CI command-line interface."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.command == "resolve":
        snapshot = resolve_snapshot(
            output_dir=args.output_dir,
            selected_dependencies=_selected_dependencies_from_args(args),
            source_revision=args.source_revision,
        )
        if args.summary_file is not None:
            append_snapshot_summary(snapshot, args.summary_file)
        return 0
    if args.command == "publish":
        summary = publish_snapshot(
            snapshot_dir=args.snapshot_dir,
            bucket=args.bucket,
            refresh_existing=args.refresh_existing,
            dry_run=args.dry_run,
        )
        if args.summary_file is not None:
            append_publish_summary(
                bucket=args.bucket,
                summary=summary,
                dry_run=args.dry_run,
                summary_path=args.summary_file,
            )
        return 0
    if args.command == "mirror":
        _run_mirror(args)
        return 0
    parser.error(f"Unsupported command: {args.command}")


def _run_mirror(args: Namespace) -> None:
    selected_dependencies = _selected_dependencies_from_args(args)
    if args.snapshot_dir is not None:
        _resolve_and_publish(
            args=args,
            snapshot_dir=args.snapshot_dir,
            selected_dependencies=selected_dependencies,
        )
        return

    with tempfile.TemporaryDirectory(prefix="rocm-dependency-snapshot-") as temp_dir:
        _resolve_and_publish(
            args=args,
            snapshot_dir=Path(temp_dir) / "snapshot",
            selected_dependencies=selected_dependencies,
        )


def _resolve_and_publish(
    *,
    args: Namespace,
    snapshot_dir: Path,
    selected_dependencies: dict[str, DependencyPolicy],
) -> None:
    snapshot = resolve_snapshot(
        output_dir=snapshot_dir,
        selected_dependencies=selected_dependencies,
        source_revision=args.source_revision,
    )
    if args.summary_file is not None:
        append_snapshot_summary(snapshot, args.summary_file)
    summary = publish_snapshot(
        snapshot_dir=snapshot_dir,
        bucket=args.bucket,
        refresh_existing=args.refresh_existing,
        dry_run=args.dry_run,
    )
    if args.summary_file is not None:
        append_publish_summary(
            bucket=args.bucket,
            summary=summary,
            dry_run=args.dry_run,
            summary_path=args.summary_file,
        )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
