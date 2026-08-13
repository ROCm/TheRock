#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCm aggregate Python Simple API index tooling."""

import argparse
import dataclasses
import html.parser
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_PUBLIC_BASE = "/rocm/whl-next"
ROUTES_FILENAME = "rocm-whl-next-routes.json"
VALIDATION_FILENAME = "validation.json"

_NORMALIZE_RE = re.compile(r"[-_.]+")
_PUBLIC_PATH_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ManifestError(ValueError):
    """Raised when an ownership manifest is malformed."""


class IndexValidationError(ValueError):
    """Raised when product-local index content is invalid."""


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


@dataclasses.dataclass(frozen=True)
class ProductRootLink:
    """A package link parsed from a product-local root index."""

    package_name: str
    href: str
    text: str


@dataclasses.dataclass(frozen=True)
class ValidatedPackage:
    """Validated product-local source for one aggregate package."""

    name: str
    owner_path: str
    product_root_index: Path
    package_index: Path


@dataclasses.dataclass(frozen=True)
class UnpublishedPackage:
    """Manifest-owned package that is not present in its product-local root."""

    name: str
    owner_path: str
    product_root_index: Path


@dataclasses.dataclass(frozen=True)
class ValidatedIndexContent:
    """Validated product-local content for one aggregate index."""

    public_base: str
    packages: dict[str, ValidatedPackage]
    unpublished_packages: dict[str, UnpublishedPackage]

    def ordered_packages(self) -> list[ValidatedPackage]:
        """Return packages in route JSON order: owner_path, then package name."""
        return sorted(self.packages.values(), key=lambda p: (p.owner_path, p.name))

    def ordered_unpublished_packages(self) -> list[UnpublishedPackage]:
        """Return unpublished packages in route JSON order."""
        return sorted(
            self.unpublished_packages.values(),
            key=lambda p: (p.owner_path, p.name),
        )


@dataclasses.dataclass(frozen=True)
class GeneratedOutputPaths:
    """Paths written by aggregate output generation."""

    aggregate_root: Path
    route_table: Path
    validation_report: Path


@dataclasses.dataclass(frozen=True)
class _Anchor:
    href: str
    text: str


class _AnchorParser(html.parser.HTMLParser):
    """Collect HTML anchor hrefs and text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[_Anchor] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href is None:
            return
        self._active_href = href
        self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._active_href is None:
            return
        self.anchors.append(
            _Anchor(
                href=self._active_href.strip(),
                text="".join(self._active_text).strip(),
            )
        )
        self._active_href = None
        self._active_text = []


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


def validate_product_indexes(
    manifest: OwnershipManifest,
    content_root: Path,
    *,
    strict_completeness: bool = False,
    require_all_manifest_packages: bool = False,
) -> ValidatedIndexContent:
    """Validate product-local root and package indexes for a manifest.

    Args:
        manifest: Parsed ownership manifest.
        content_root: Local directory mirroring public ``/rocm/...`` paths.

    Returns:
        A typed model containing every validated package source.

    Raises:
        IndexValidationError: If a referenced product-local root or package
        page is missing, empty, malformed, or invalid.
    """
    index = manifest.python_indexes[0]
    packages_by_owner_path: dict[str, list[PackageOwnership]] = {}
    for package in index.ordered_packages():
        packages_by_owner_path.setdefault(package.owner_path, []).append(package)

    validated_packages: dict[str, ValidatedPackage] = {}
    unpublished_packages: dict[str, UnpublishedPackage] = {}
    for owner_path in sorted(packages_by_owner_path):
        product_root_index = content_root / "rocm" / owner_path / "index.html"
        root_links = parse_product_root_index(product_root_index)
        root_links_by_package: dict[str, list[ProductRootLink]] = {}
        for link in root_links:
            root_links_by_package.setdefault(link.package_name, []).append(link)

        if strict_completeness:
            manifest_package_names = {
                package.name for package in packages_by_owner_path[owner_path]
            }
            extra_package_names = sorted(
                set(root_links_by_package) - manifest_package_names
            )
            if extra_package_names:
                extra_packages = ", ".join(repr(name) for name in extra_package_names)
                raise IndexValidationError(
                    f"{product_root_index}: product root contains package(s) "
                    f"absent from the ownership manifest: {extra_packages}"
                )

        for package in packages_by_owner_path[owner_path]:
            matching_links = root_links_by_package.get(package.name, [])
            if not matching_links:
                if require_all_manifest_packages:
                    raise IndexValidationError(
                        f"{product_root_index}: missing canonical package link "
                        f"for {package.name!r}"
                    )
                unpublished_packages[package.name] = UnpublishedPackage(
                    name=package.name,
                    owner_path=owner_path,
                    product_root_index=product_root_index,
                )
                continue
            if len(matching_links) != 1:
                raise IndexValidationError(
                    f"{product_root_index}: expected exactly one package link "
                    f"for {package.name!r}, found {len(matching_links)}"
                )

            package_index = (
                content_root / "rocm" / owner_path / package.name / "index.html"
            )
            _require_non_empty_file(package_index, "package index")
            validated_packages[package.name] = ValidatedPackage(
                name=package.name,
                owner_path=owner_path,
                product_root_index=product_root_index,
                package_index=package_index,
            )

    route_ordered_packages = sorted(
        validated_packages.values(), key=lambda p: (p.owner_path, p.name)
    )
    return ValidatedIndexContent(
        public_base=index.public_base,
        packages={package.name: package for package in route_ordered_packages},
        unpublished_packages={
            package.name: package
            for package in sorted(
                unpublished_packages.values(), key=lambda p: (p.owner_path, p.name)
            )
        },
    )


def generate_outputs(
    manifest: OwnershipManifest,
    content_root: Path,
    output_dir: Path,
    *,
    strict_completeness: bool = False,
    require_all_manifest_packages: bool = False,
) -> GeneratedOutputPaths:
    """Validate product-local content and write aggregate routed outputs."""
    validated = validate_product_indexes(
        manifest,
        content_root,
        strict_completeness=strict_completeness,
        require_all_manifest_packages=require_all_manifest_packages,
    )
    return write_generated_outputs(validated, output_dir)


def write_generated_outputs(
    validated: ValidatedIndexContent,
    output_dir: Path,
) -> GeneratedOutputPaths:
    """Write aggregate root, exact route table, and validation report."""
    aggregate_root_html = render_aggregate_root(validated)
    route_table = build_route_table(validated)
    validation_report = build_validation_report(validated)
    _assert_output_package_sets_match(
        validated,
        aggregate_root_html,
        route_table,
        validation_report,
    )

    aggregate_root_path = output_dir / "rocm" / "whl-next" / "index.html"
    route_table_path = output_dir / ROUTES_FILENAME
    validation_report_path = output_dir / VALIDATION_FILENAME
    _write_text_atomic(aggregate_root_path, aggregate_root_html)
    _write_text_atomic(route_table_path, _json_dumps(route_table))
    _write_text_atomic(validation_report_path, _json_dumps(validation_report))
    return GeneratedOutputPaths(
        aggregate_root=aggregate_root_path,
        route_table=route_table_path,
        validation_report=validation_report_path,
    )


def render_aggregate_root(validated: ValidatedIndexContent) -> str:
    """Render the aggregate PEP 503 root index."""
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "  <body>",
    ]
    for package_name in sorted(validated.packages):
        lines.append(f'    <a href="{package_name}/">{package_name}</a><br/>')
    lines.extend(
        [
            "  </body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(lines)


def build_route_table(validated: ValidatedIndexContent) -> dict[str, object]:
    """Build the exact aggregate package route table."""
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "public_base": validated.public_base,
        "routes": [
            {
                "package": package.name,
                "owner_path": package.owner_path,
                "target": f"/rocm/{package.owner_path}/{package.name}/",
            }
            for package in validated.ordered_packages()
        ],
    }


def build_validation_report(validated: ValidatedIndexContent) -> dict[str, object]:
    """Build deterministic CI-readable validation output."""
    owners: dict[str, int] = {}
    for package in validated.ordered_packages():
        owners[package.owner_path] = owners.get(package.owner_path, 0) + 1
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "public_base": validated.public_base,
        "package_count": len(validated.packages),
        "unpublished_package_count": len(validated.unpublished_packages),
        "owners": {
            owner_path: {"package_count": owners[owner_path]}
            for owner_path in sorted(owners)
        },
        "packages": [
            {
                "name": package.name,
                "owner_path": package.owner_path,
                "product_root_index": package.product_root_index.as_posix(),
                "package_index": package.package_index.as_posix(),
            }
            for package in validated.ordered_packages()
        ],
        "unpublished_packages": [
            {
                "name": package.name,
                "owner_path": package.owner_path,
                "product_root_index": package.product_root_index.as_posix(),
            }
            for package in validated.ordered_unpublished_packages()
        ],
    }


def parse_product_root_index(path: Path) -> list[ProductRootLink]:
    """Parse and validate package links from one product-local root index."""
    text = _read_non_empty_text(path, "product root index")
    return _parse_product_root_index_text(text, str(path))


def _parse_product_root_index_text(text: str, context: str) -> list[ProductRootLink]:
    parser = _AnchorParser()
    parser.feed(text)
    parser.close()

    links: list[ProductRootLink] = []
    for anchor in parser.anchors:
        package_name = _package_name_from_root_href(anchor.href, context)
        if anchor.text != package_name:
            raise IndexValidationError(
                f"{context}: link text {anchor.text!r} does not match "
                f"package href {anchor.href!r}"
            )
        links.append(
            ProductRootLink(
                package_name=package_name,
                href=anchor.href,
                text=anchor.text,
            )
        )
    return links


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


def _package_name_from_root_href(href: str, context: str) -> str:
    if href == "":
        raise IndexValidationError(f"{context}: package link href must not be empty")
    split_href = urlsplit(href)
    if split_href.scheme or split_href.netloc:
        raise IndexValidationError(
            f"{context}: package link href {href!r} must be relative"
        )
    if split_href.query or split_href.fragment:
        raise IndexValidationError(
            f"{context}: package link href {href!r} must not include query or fragment"
        )
    if href.startswith("/"):
        raise IndexValidationError(
            f"{context}: package link href {href!r} must not be absolute"
        )
    if not href.endswith("/"):
        raise IndexValidationError(
            f"{context}: package link href {href!r} must end with '/'"
        )

    package_name = href.removesuffix("/")
    segments = package_name.split("/")
    if len(segments) != 1:
        raise IndexValidationError(
            f"{context}: package link href {href!r} must name one package directory"
        )
    if package_name in {"", ".", ".."}:
        raise IndexValidationError(
            f"{context}: package link href {href!r} contains an unsafe path segment"
        )

    normalized = pep503_normalize(package_name)
    if normalized != package_name:
        raise IndexValidationError(
            f"{context}: package link href {href!r} must use normalized "
            f"package name {normalized!r}"
        )
    return package_name


def _read_non_empty_text(path: Path, description: str) -> str:
    _require_non_empty_file(path, description)
    return path.read_text(encoding="utf-8")


def _require_non_empty_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise IndexValidationError(f"Missing {description}: {path}")
    if path.stat().st_size == 0:
        raise IndexValidationError(f"Empty {description}: {path}")


def _assert_output_package_sets_match(
    validated: ValidatedIndexContent,
    aggregate_root_html: str,
    route_table: dict[str, object],
    validation_report: dict[str, object],
) -> None:
    expected_packages = set(validated.packages)
    aggregate_packages = {
        link.package_name
        for link in _parse_product_root_index_text(
            aggregate_root_html, "aggregate root"
        )
    }
    route_packages = _route_package_set(route_table)
    validation_packages = _validation_package_set(validation_report)
    if aggregate_packages != expected_packages:
        raise RuntimeError("Aggregate root package set does not match validated input")
    if route_packages != expected_packages:
        raise RuntimeError("Route table package set does not match validated input")
    if validation_packages != expected_packages:
        raise RuntimeError(
            "Validation report package set does not match validated input"
        )


def _route_package_set(route_table: dict[str, object]) -> set[str]:
    routes = route_table.get("routes")
    if not isinstance(routes, list):
        raise RuntimeError("Route table routes must be a list")
    packages: set[str] = set()
    for route in routes:
        if not isinstance(route, dict):
            raise RuntimeError("Route table routes must contain objects")
        package_name = route.get("package")
        if not isinstance(package_name, str):
            raise RuntimeError("Route table route package must be a string")
        packages.add(package_name)
    return packages


def _validation_package_set(validation_report: dict[str, object]) -> set[str]:
    packages = validation_report.get("packages")
    if not isinstance(packages, list):
        raise RuntimeError("Validation report packages must be a list")
    package_names: set[str] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise RuntimeError("Validation report packages must contain objects")
        package_name = package.get("name")
        if not isinstance(package_name, str):
            raise RuntimeError("Validation report package name must be a string")
        package_names.add(package_name)
    return package_names


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _json_dumps(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


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


def _generate_command(args: argparse.Namespace) -> int:
    manifest = load_ownership_manifest(args.manifest)
    outputs = generate_outputs(
        manifest,
        args.content_root,
        args.output_dir,
        strict_completeness=args.strict or args.strict_completeness,
        require_all_manifest_packages=args.strict or args.require_all_manifest_packages,
    )
    print(f"aggregate_root: {outputs.aggregate_root}")
    print(f"route_table: {outputs.route_table}")
    print(f"validation_report: {outputs.validation_report}")
    return 0


def _validate_content_command(args: argparse.Namespace) -> int:
    manifest = load_ownership_manifest(args.manifest)
    validated = validate_product_indexes(
        manifest,
        args.content_root,
        strict_completeness=args.strict or args.strict_completeness,
        require_all_manifest_packages=args.strict or args.require_all_manifest_packages,
    )
    owner_counts: dict[str, int] = {}
    for package in validated.ordered_packages():
        owner_counts[package.owner_path] = owner_counts.get(package.owner_path, 0) + 1

    print(f"public_base: {validated.public_base}")
    print(f"packages: {len(validated.packages)}")
    print(f"unpublished_packages: {len(validated.unpublished_packages)}")
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

    generate_parser = subparsers.add_parser(
        "generate",
        help="validate product-local content and generate aggregate routed outputs",
    )
    generate_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the ownership manifest YAML file",
    )
    generate_parser.add_argument(
        "--content-root",
        type=Path,
        required=True,
        help="Local directory mirroring public /rocm/... product indexes",
    )
    generate_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where aggregate outputs will be written",
    )
    generate_parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on undeclared product-root packages and unpublished "
        "manifest packages",
    )
    generate_parser.add_argument(
        "--strict-completeness",
        action="store_true",
        help="fail if referenced product roots contain packages absent from "
        "the ownership manifest",
    )
    generate_parser.add_argument(
        "--require-all-manifest-packages",
        action="store_true",
        help="fail if a manifest-owned package is absent from its product root",
    )
    generate_parser.set_defaults(func=_generate_command)

    content_parser = subparsers.add_parser(
        "validate-content",
        help="validate product-local index content for an ownership manifest",
    )
    content_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the ownership manifest YAML file",
    )
    content_parser.add_argument(
        "--content-root",
        type=Path,
        required=True,
        help="Local directory mirroring public /rocm/... product indexes",
    )
    content_parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on undeclared product-root packages and unpublished "
        "manifest packages",
    )
    content_parser.add_argument(
        "--strict-completeness",
        action="store_true",
        help="fail if referenced product roots contain packages absent from "
        "the ownership manifest",
    )
    content_parser.add_argument(
        "--require-all-manifest-packages",
        action="store_true",
        help="fail if a manifest-owned package is absent from its product root",
    )
    content_parser.set_defaults(func=_validate_content_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
