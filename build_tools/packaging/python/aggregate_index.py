#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCm aggregate Python Simple API index tooling."""

import argparse
import dataclasses
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_PUBLIC_BASE = "/rocm/whl-next"

_NORMALIZE_RE = re.compile(r"[-_.]+")
_PUBLIC_PATH_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ManifestError(ValueError):
    """Raised when an ownership manifest is malformed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_mapping_without_duplicate_keys(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_without_duplicate_keys,
)


@dataclasses.dataclass(frozen=True)
class PackageOwnership:
    """Ownership declaration for one normalized package."""

    name: str
    owner_path: str

    @property
    def owner_public_base(self) -> str:
        """Return the owner root as an absolute public path."""
        return f"/rocm/{self.owner_path}"


@dataclasses.dataclass(frozen=True)
class PythonIndexOwnership:
    """Ownership declarations for one aggregate Python index."""

    public_base: str
    packages: dict[str, PackageOwnership]

    def ordered_packages(self) -> list[PackageOwnership]:
        """Return packages in route JSON order: owner_path, then package name."""
        return sorted(self.packages.values(), key=lambda p: (p.owner_path, p.name))


@dataclasses.dataclass(frozen=True)
class OwnershipManifest:
    """Parsed ownership manifest."""

    schema_version: int
    python_indexes: list[PythonIndexOwnership]


def pep503_normalize(name: str) -> str:
    """Normalize a package name per PEP 503."""
    return _NORMALIZE_RE.sub("-", name.lower())


def load_ownership_manifest(path: Path) -> OwnershipManifest:
    """Load and validate an ownership manifest from ``path``."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=_UniqueKeyLoader)
    except ConstructorError as e:
        raise ManifestError(f"{path}: {e}") from e
    return parse_ownership_manifest(data, context=str(path))


def parse_ownership_manifest(
    data: object, *, context: str = "manifest"
) -> OwnershipManifest:
    """Parse and validate ownership manifest data."""
    manifest = _require_mapping(data, context)
    _require_keys(manifest, {"schema_version", "python_indexes"}, context)

    schema_version = _require_int(
        manifest["schema_version"], f"{context}.schema_version"
    )
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ManifestError(
            f"{context}.schema_version must be {SUPPORTED_SCHEMA_VERSION}, "
            f"got {schema_version}"
        )

    index_items = _require_list(manifest["python_indexes"], f"{context}.python_indexes")
    if len(index_items) != 1:
        raise ManifestError(
            f"{context}.python_indexes must contain exactly one /rocm/whl-next index"
        )

    python_indexes = [
        _parse_python_index(item, f"{context}.python_indexes[{i}]")
        for i, item in enumerate(index_items)
    ]
    return OwnershipManifest(
        schema_version=schema_version,
        python_indexes=python_indexes,
    )


def _parse_python_index(data: object, context: str) -> PythonIndexOwnership:
    index = _require_mapping(data, context)
    _require_keys(index, {"public_base", "packages"}, context)

    public_base = _require_str(index["public_base"], f"{context}.public_base")
    _validate_public_base(public_base, f"{context}.public_base")
    if public_base != SUPPORTED_PUBLIC_BASE:
        raise ManifestError(
            f"{context}.public_base must be {SUPPORTED_PUBLIC_BASE!r}, "
            f"got {public_base!r}"
        )

    package_items = _require_mapping(index["packages"], f"{context}.packages")
    if not package_items:
        raise ManifestError(f"{context}.packages must not be empty")

    packages: dict[str, PackageOwnership] = {}
    for raw_name, raw_config in package_items.items():
        if not isinstance(raw_name, str):
            raise ManifestError(
                f"{context}.packages contains a non-string package name"
            )
        package_name = pep503_normalize(raw_name)
        if package_name != raw_name:
            raise ManifestError(
                f"{context}.packages.{raw_name}: package name must be PEP 503 "
                f"normalized as {package_name!r}"
            )
        package_config = _require_mapping(
            raw_config, f"{context}.packages.{package_name}"
        )
        _require_keys(
            package_config,
            {"owner_path"},
            f"{context}.packages.{package_name}",
        )
        owner_path = _require_str(
            package_config["owner_path"],
            f"{context}.packages.{package_name}.owner_path",
        )
        _validate_owner_path(
            owner_path, f"{context}.packages.{package_name}.owner_path"
        )
        packages[package_name] = PackageOwnership(
            name=package_name,
            owner_path=owner_path,
        )

    route_ordered_packages = sorted(
        packages.values(), key=lambda p: (p.owner_path, p.name)
    )
    return PythonIndexOwnership(
        public_base=public_base,
        packages={package.name: package for package in route_ordered_packages},
    )


def _require_mapping(data: object, context: str) -> Mapping[object, object]:
    if not isinstance(data, Mapping):
        raise ManifestError(f"{context} must be a mapping")
    return data


def _require_list(data: object, context: str) -> list[object]:
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ManifestError(f"{context} must be a list")
    return list(data)


def _require_str(data: object, context: str) -> str:
    if not isinstance(data, str):
        raise ManifestError(f"{context} must be a string")
    if data == "":
        raise ManifestError(f"{context} must not be empty")
    return data


def _require_int(data: object, context: str) -> int:
    if not isinstance(data, int) or isinstance(data, bool):
        raise ManifestError(f"{context} must be an integer")
    return data


def _require_keys(
    data: Mapping[object, object],
    expected_keys: set[str],
    context: str,
) -> None:
    actual_keys = set(data)
    unknown_keys = actual_keys - expected_keys
    missing_keys = expected_keys - actual_keys
    if unknown_keys:
        unknown = ", ".join(repr(key) for key in sorted(unknown_keys, key=repr))
        raise ManifestError(f"{context} contains unknown key(s): {unknown}")
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ManifestError(f"{context} is missing required key(s): {missing}")


def _validate_public_base(public_base: str, context: str) -> None:
    if not public_base.startswith("/rocm/"):
        raise ManifestError(f"{context} must be contained under /rocm/")
    if public_base.endswith("/"):
        raise ManifestError(f"{context} must not end with '/'")
    _validate_public_path_segments(public_base.removeprefix("/"), context)


def _validate_owner_path(owner_path: str, context: str) -> None:
    if owner_path.startswith("/"):
        raise ManifestError(f"{context} must be relative to /rocm")
    if "://" in owner_path:
        raise ManifestError(f"{context} must not contain a URL scheme")
    _validate_public_path_segments(owner_path, context)


def _validate_public_path_segments(path: str, context: str) -> None:
    segments = path.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ManifestError(f"{context} contains an empty, '.', or '..' path segment")
    for segment in segments:
        if not _PUBLIC_PATH_SEGMENT_RE.fullmatch(segment):
            raise ManifestError(
                f"{context} contains unsupported path segment {segment!r}"
            )


def _validate_manifest_command(args: argparse.Namespace) -> int:
    manifest = load_ownership_manifest(args.manifest)
    index = manifest.python_indexes[0]
    owner_counts: dict[str, int] = {}
    for package in index.ordered_packages():
        owner_counts[package.owner_path] = owner_counts.get(package.owner_path, 0) + 1

    print(f"schema_version: {manifest.schema_version}")
    print(f"public_base: {index.public_base}")
    print(f"packages: {len(index.packages)}")
    for owner_path in sorted(owner_counts):
        print(f"{owner_path}: {owner_counts[owner_path]}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="validate an aggregate index ownership manifest",
    )
    validate_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the ownership manifest YAML file",
    )
    validate_parser.set_defaults(func=_validate_manifest_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
