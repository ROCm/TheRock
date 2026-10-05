#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCm aggregate Python Simple API index tooling.

Validates and generates the routed aggregate Simple API index for:

    /rocm/whl-next/

The generator is storage-independent. It consumes a checked-in ownership
manifest and can optionally validate a local directory tree mirroring the
public product-local index layout:

    <content-root>/rocm/<owner_path>/index.html
    <content-root>/rocm/<owner_path>/<normalized-package>/index.html

For example, a package owned by ``pytorch/whl-next`` is validated from:

    <content-root>/rocm/pytorch/whl-next/index.html
    <content-root>/rocm/pytorch/whl-next/torch/index.html

The required routed baseline writes:

    <output>/rocm/whl-next/index.html
    <output>/rocm-whl-next-routes.json
    <output>/validation.json

The aggregate root links to canonical package subdirectories under the
aggregate namespace, such as ``torch/``. The generated route table maps those
aggregate package requests to exact product-local package pages. The generator
does not copy or rewrite product-local package pages; they remain authoritative
for artifact links and metadata in the routed design.

CLI subcommands:

    # Validate only the ownership manifest.
    python aggregate_index.py validate-manifest \
        --manifest build_tools/packaging/python/rocm_whl_next_ownership.yaml

    # Validate a local public-tree snapshot without writing outputs.
    python aggregate_index.py validate-content \
        --manifest build_tools/packaging/python/rocm_whl_next_ownership.yaml \
        --stream nightly \
        --content-root /tmp/rocm-whl-next-content

    # Generate routed aggregate outputs directly from the manifest.
    python aggregate_index.py generate \
        --manifest build_tools/packaging/python/rocm_whl_next_ownership.yaml \
        --stream nightly \
        --output-dir /tmp/rocm-whl-next-output

    # Validate a local public-tree snapshot while generating outputs.
    python aggregate_index.py generate \
        --manifest build_tools/packaging/python/rocm_whl_next_ownership.yaml \
        --stream nightly \
        --content-root /tmp/rocm-whl-next-content \
        --output-dir /tmp/rocm-whl-next-output

When ``--content-root`` is provided, the deployment validation invocation should
use ``generate --strict-completeness`` and must not use ``--allow-unpublished``.
Both set-comparison directions are deployment-visible failures: a manifest-owned
package that is not published would route to a missing product-local package
page, while a published package absent from the manifest is unreachable through
the aggregate index.

``--allow-unpublished`` is only for explicit pre-publication inventory workflows
that also provide ``--content-root``. In that mode, manifest-owned packages
absent from product roots are recorded as unpublished in ``validation.json`` and
excluded from the aggregate root and exact route table.
"""

import argparse
import dataclasses
import html
import html.parser
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote, urlsplit

from packaging.utils import canonicalize_name, is_normalized_name
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


MANIFEST_SCHEMA_VERSION = 3
ROUTES_SCHEMA_VERSION = 2
VALIDATION_SCHEMA_VERSION = 2
SUPPORTED_PUBLIC_BASE = "/rocm/whl-next"
ROUTES_FILENAME = "rocm-whl-next-routes.json"
VALIDATION_FILENAME = "validation.json"

_PUBLIC_PATH_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_STREAM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


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
        try:
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as e:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key {key!r}",
                key_node.start_mark,
            ) from e
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
    streams: frozenset[str]


@dataclasses.dataclass(frozen=True)
class StreamConfig:
    """Known streams, named groups, and the default package stream set."""

    known: tuple[str, ...]
    groups: dict[str, frozenset[str]]
    default: frozenset[str]


@dataclasses.dataclass(frozen=True)
class ResolvedOwnership:
    """Resolved ownership for one package."""

    name: str
    owner_path: str
    source: str = "exact"


@dataclasses.dataclass(frozen=True)
class PythonIndexOwnership:
    """Ownership declarations for one aggregate Python index."""

    public_base: str
    packages: dict[str, PackageOwnership]

    def ordered_packages(self) -> list[PackageOwnership]:
        """Return packages in route JSON order: owner_path, then package name."""
        return sorted(self.packages.values(), key=lambda p: (p.owner_path, p.name))

    def active_packages(self, stream: str) -> dict[str, PackageOwnership]:
        """Return packages enabled for one stream in route JSON order."""
        packages = [
            package for package in self.ordered_packages() if stream in package.streams
        ]
        return {package.name: package for package in packages}

    def owner_paths(self, stream: str) -> set[str]:
        """Return product-local roots referenced by active package ownership."""
        return {package.owner_path for package in self.active_packages(stream).values()}

    def resolve_package(
        self, package_name: str, stream: str
    ) -> ResolvedOwnership | None:
        """Resolve exact package ownership for one stream."""
        exact = self.packages.get(package_name)
        if exact is None or stream not in exact.streams:
            return None
        return ResolvedOwnership(
            name=package_name,
            owner_path=exact.owner_path,
        )


@dataclasses.dataclass(frozen=True)
class OwnershipManifest:
    """Parsed ownership manifest."""

    schema_version: int
    streams: StreamConfig
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
    source: str

    @property
    def product_root_public_path(self) -> str:
        """Return the product root index path under the public /rocm tree."""
        return _product_root_public_path(self.owner_path)

    @property
    def package_public_path(self) -> str:
        """Return the package index path under the public /rocm tree."""
        return _package_index_public_path(self.owner_path, self.name)


@dataclasses.dataclass(frozen=True)
class UnpublishedPackage:
    """Manifest-owned package that is not present in its product-local root."""

    name: str
    owner_path: str
    product_root_index: Path

    @property
    def product_root_public_path(self) -> str:
        """Return the product root index path under the public /rocm tree."""
        return _product_root_public_path(self.owner_path)


@dataclasses.dataclass(frozen=True)
class ValidatedIndexContent:
    """Validated product-local content for one aggregate index."""

    public_base: str
    stream: str
    generation_mode: str
    content_validated: bool
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
    unpublished_package_count: int


@dataclasses.dataclass(frozen=True)
class Route:
    """One exact aggregate package route."""

    package: str
    owner_path: str
    target: str

    def to_json(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RouteTable:
    """Exact aggregate package route table."""

    schema_version: int
    public_base: str
    stream: str
    routes: list[Route]

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "public_base": self.public_base,
            "stream": self.stream,
            "routes": [route.to_json() for route in self.routes],
        }


@dataclasses.dataclass(frozen=True)
class ValidationOwnerSummary:
    """Validation summary for one owner path."""

    package_count: int

    def to_json(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ValidationPackage:
    """Validation report entry for one published package."""

    name: str
    owner_path: str
    product_root_index: str
    package_index: str
    source: str

    def to_json(self) -> dict[str, str]:
        return {
            "name": self.name,
            "owner_path": self.owner_path,
            "product_root_index": self.product_root_index,
            "package_index": self.package_index,
            "source": self.source,
        }


@dataclasses.dataclass(frozen=True)
class ValidationUnpublishedPackage:
    """Validation report entry for one unpublished manifest package."""

    name: str
    owner_path: str
    product_root_index: str

    def to_json(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    """CI-readable validation report."""

    schema_version: int
    public_base: str
    stream: str
    generation_mode: str
    content_validated: bool
    package_count: int
    unpublished_package_count: int
    owners: dict[str, ValidationOwnerSummary]
    packages: list[ValidationPackage]
    unpublished_packages: list[ValidationUnpublishedPackage]

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "public_base": self.public_base,
            "stream": self.stream,
            "generation_mode": self.generation_mode,
            "content_validated": self.content_validated,
            "package_count": self.package_count,
            "unpublished_package_count": self.unpublished_package_count,
            "owners": {
                owner_path: summary.to_json()
                for owner_path, summary in sorted(self.owners.items())
            },
            "packages": [package.to_json() for package in self.packages],
            "unpublished_packages": [
                package.to_json() for package in self.unpublished_packages
            ],
        }


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
        if self._active_href is not None:
            raise IndexValidationError("Package links must not contain nested tags")
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

    def close(self) -> None:
        super().close()
        if self._active_href is not None:
            raise IndexValidationError("Unterminated package link")


def pep503_normalize(name: str) -> str:
    """Normalize a package name per PEP 503."""
    return str(canonicalize_name(name))


def load_ownership_manifest(path: Path) -> OwnershipManifest:
    """Load and validate an ownership manifest from ``path``."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as e:
        raise ManifestError(f"{path}: {e}") from e
    except (OSError, UnicodeDecodeError) as e:
        raise ManifestError(f"{path}: cannot read ownership manifest: {e}") from e
    return parse_ownership_manifest(data, context=str(path))


def validate_product_indexes(
    manifest: OwnershipManifest,
    content_root: Path,
    *,
    stream: str,
    strict_completeness: bool = False,
    require_all_manifest_packages: bool = True,
) -> ValidatedIndexContent:
    """Validate product-local root and package indexes for a manifest.

    The validator is strict by default: every manifest-owned package must be
    present in its product root. Pass ``require_all_manifest_packages=False``
    only for explicit pre-publication inventory workflows, where absent
    manifest packages are recorded as unpublished.

    Args:
        manifest: Parsed ownership manifest.
        content_root: Local directory mirroring public ``/rocm/...`` paths.

    Returns:
        A typed model containing every validated package source.

    Raises:
        IndexValidationError: If a referenced product-local root or package
        page is missing, empty, malformed, or invalid.
    """
    _validate_manifest_stream(manifest, stream, "stream")
    index = manifest.python_indexes[0]
    active_packages = index.active_packages(stream)

    validated_packages: dict[str, ValidatedPackage] = {}
    unpublished_packages: dict[str, UnpublishedPackage] = {}
    owner_paths = index.owner_paths(stream)
    if strict_completeness:
        inactive_owner_paths = {
            package.owner_path
            for package in index.packages.values()
            if package.owner_path not in owner_paths
        }
        for owner_path in inactive_owner_paths:
            product_root_index = content_root / "rocm" / owner_path / "index.html"
            if product_root_index.is_file():
                owner_paths.add(owner_path)

    for owner_path in sorted(owner_paths):
        product_root_index = content_root / "rocm" / owner_path / "index.html"
        root_links = parse_product_root_index(product_root_index)
        root_links_by_package: dict[str, list[ProductRootLink]] = {}
        for link in root_links:
            root_links_by_package.setdefault(link.package_name, []).append(link)

        for package_name, matching_links in sorted(root_links_by_package.items()):
            ownership = index.resolve_package(package_name, stream)
            if ownership is None:
                if strict_completeness:
                    raise IndexValidationError(
                        f"{product_root_index}: product root contains package "
                        f"{package_name!r} absent from the ownership manifest "
                        f"for stream {stream!r}"
                    )
                continue
            if ownership.owner_path != owner_path:
                raise IndexValidationError(
                    f"{product_root_index}: package {package_name!r} resolves to "
                    f"owner path {ownership.owner_path!r}, not {owner_path!r}"
                )
            if len(matching_links) != 1:
                raise IndexValidationError(
                    f"{product_root_index}: expected exactly one package link "
                    f"for {package_name!r}, found {len(matching_links)}"
                )

            package_index = (
                content_root / "rocm" / owner_path / package_name / "index.html"
            )
            _require_non_empty_file(package_index, "package index")
            validated_packages[package_name] = ValidatedPackage(
                name=package_name,
                owner_path=owner_path,
                product_root_index=product_root_index,
                package_index=package_index,
                source=ownership.source,
            )

    for package in sorted(
        active_packages.values(),
        key=lambda p: (p.owner_path, p.name),
    ):
        if package.name in validated_packages:
            continue
        product_root_index = content_root / "rocm" / package.owner_path / "index.html"
        if require_all_manifest_packages:
            raise IndexValidationError(
                f"{product_root_index}: missing canonical package link "
                f"for {package.name!r}"
            )
        unpublished_packages[package.name] = UnpublishedPackage(
            name=package.name,
            owner_path=package.owner_path,
            product_root_index=product_root_index,
        )

    route_ordered_packages = sorted(
        validated_packages.values(), key=lambda p: (p.owner_path, p.name)
    )
    return ValidatedIndexContent(
        public_base=index.public_base,
        stream=stream,
        generation_mode="content",
        content_validated=True,
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
    output_dir: Path,
    *,
    stream: str,
    content_root: Path | None = None,
    strict_completeness: bool = False,
    require_all_manifest_packages: bool = True,
) -> GeneratedOutputPaths:
    """Write aggregate routed outputs for one stream.

    If ``content_root`` is provided, product-local content is validated before
    outputs are written. Otherwise, outputs are generated directly from the
    stream-filtered ownership manifest.
    """
    if content_root is None:
        if strict_completeness or not require_all_manifest_packages:
            raise IndexValidationError(
                "Content validation flags require --content-root"
            )
        validated = declared_index_content(manifest, stream)
    else:
        validated = validate_product_indexes(
            manifest,
            content_root,
            stream=stream,
            strict_completeness=strict_completeness,
            require_all_manifest_packages=require_all_manifest_packages,
        )
    return write_generated_outputs(validated, output_dir)


def declared_index_content(
    manifest: OwnershipManifest,
    stream: str,
) -> ValidatedIndexContent:
    """Build aggregate output content directly from manifest ownership."""
    _validate_manifest_stream(manifest, stream, "stream")
    index = manifest.python_indexes[0]
    packages = [
        ValidatedPackage(
            name=package.name,
            owner_path=package.owner_path,
            product_root_index=Path(),
            package_index=Path(),
            source="manifest",
        )
        for package in index.active_packages(stream).values()
    ]
    route_ordered_packages = sorted(packages, key=lambda p: (p.owner_path, p.name))
    return ValidatedIndexContent(
        public_base=index.public_base,
        stream=stream,
        generation_mode="manifest",
        content_validated=False,
        packages={package.name: package for package in route_ordered_packages},
        unpublished_packages={},
    )


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
    _write_text_atomic(route_table_path, _json_dumps(route_table.to_json()))
    _write_text_atomic(validation_report_path, _json_dumps(validation_report.to_json()))
    return GeneratedOutputPaths(
        aggregate_root=aggregate_root_path,
        route_table=route_table_path,
        validation_report=validation_report_path,
        unpublished_package_count=validation_report.unpublished_package_count,
    )


def render_aggregate_root(validated: ValidatedIndexContent) -> str:
    """Render the aggregate PEP 503 root index."""
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "  <body>",
    ]
    for package_name in sorted(validated.packages):
        href = quote(package_name, safe="")
        text = html.escape(package_name)
        lines.append(f'    <a href="{href}/">{text}</a><br/>')
    lines.extend(
        [
            "  </body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(lines)


def build_route_table(validated: ValidatedIndexContent) -> RouteTable:
    """Build the exact aggregate package route table."""
    return RouteTable(
        schema_version=ROUTES_SCHEMA_VERSION,
        public_base=validated.public_base,
        stream=validated.stream,
        routes=[
            Route(
                package=package.name,
                owner_path=package.owner_path,
                target=_package_public_base(package.owner_path, package.name),
            )
            for package in validated.ordered_packages()
        ],
    )


def build_validation_report(validated: ValidatedIndexContent) -> ValidationReport:
    """Build deterministic CI-readable validation output."""
    owners: dict[str, int] = {}
    for package in validated.ordered_packages():
        owners[package.owner_path] = owners.get(package.owner_path, 0) + 1
    return ValidationReport(
        schema_version=VALIDATION_SCHEMA_VERSION,
        public_base=validated.public_base,
        stream=validated.stream,
        generation_mode=validated.generation_mode,
        content_validated=validated.content_validated,
        package_count=len(validated.packages),
        unpublished_package_count=len(validated.unpublished_packages),
        owners={
            owner_path: ValidationOwnerSummary(package_count=owners[owner_path])
            for owner_path in sorted(owners)
        },
        packages=[
            ValidationPackage(
                name=package.name,
                owner_path=package.owner_path,
                product_root_index=package.product_root_public_path,
                package_index=package.package_public_path,
                source=package.source,
            )
            for package in validated.ordered_packages()
        ],
        unpublished_packages=[
            ValidationUnpublishedPackage(
                name=package.name,
                owner_path=package.owner_path,
                product_root_index=package.product_root_public_path,
            )
            for package in validated.ordered_unpublished_packages()
        ],
    )


def parse_product_root_index(path: Path) -> list[ProductRootLink]:
    """Parse and validate package links from one product-local root index."""
    text = _read_non_empty_text(path, "product root index")
    return _parse_product_root_index_text(text, str(path))


def _parse_product_root_index_text(text: str, context: str) -> list[ProductRootLink]:
    parser = _AnchorParser()
    try:
        parser.feed(text)
        parser.close()
    except IndexValidationError as e:
        raise IndexValidationError(f"{context}: {e}") from e

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
    _require_keys(manifest, {"schema_version", "streams", "python_indexes"}, context)

    schema_version = _require_int(
        manifest["schema_version"], f"{context}.schema_version"
    )
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"{context}.schema_version must be {MANIFEST_SCHEMA_VERSION}, "
            f"got {schema_version}"
        )

    streams = _parse_stream_config(manifest["streams"], f"{context}.streams")
    index_items = _require_list(manifest["python_indexes"], f"{context}.python_indexes")
    if len(index_items) != 1:
        raise ManifestError(
            f"{context}.python_indexes must contain exactly one /rocm/whl-next index"
        )

    python_indexes = [
        _parse_python_index(item, f"{context}.python_indexes[{i}]", streams)
        for i, item in enumerate(index_items)
    ]
    return OwnershipManifest(
        schema_version=schema_version,
        streams=streams,
        python_indexes=python_indexes,
    )


def _parse_stream_config(data: object, context: str) -> StreamConfig:
    stream_config = _require_mapping(data, context)
    _require_keys(stream_config, {"known", "groups", "default"}, context)
    known = _parse_stream_list(stream_config["known"], f"{context}.known")
    groups = _parse_stream_groups(
        stream_config["groups"], frozenset(known), f"{context}.groups"
    )
    default_group = _require_str(stream_config["default"], f"{context}.default")
    _validate_stream_group_name(default_group, f"{context}.default")
    if default_group not in groups:
        raise ManifestError(
            f"{context}.default contains unknown stream group {default_group!r}"
        )
    return StreamConfig(
        known=tuple(known),
        groups=groups,
        default=groups[default_group],
    )


def _parse_stream_groups(
    data: object,
    known_streams: frozenset[str],
    context: str,
) -> dict[str, frozenset[str]]:
    group_items = _require_mapping(data, context)
    if not group_items:
        raise ManifestError(f"{context} must not be empty")

    groups: dict[str, frozenset[str]] = {}
    for raw_group_name, raw_streams in group_items.items():
        group_name = _require_str(raw_group_name, f"{context} key")
        _validate_stream_group_name(group_name, f"{context}.{group_name}")
        streams = frozenset(_parse_stream_list(raw_streams, f"{context}.{group_name}"))
        unknown_streams = sorted(streams - known_streams)
        if unknown_streams:
            raise ManifestError(
                f"{context}.{group_name} contains unknown stream(s): "
                f"{', '.join(unknown_streams)}"
            )
        groups[group_name] = streams
    return groups


def _parse_stream_list(data: object, context: str) -> list[str]:
    streams = _require_list(data, context)
    if not streams:
        raise ManifestError(f"{context} must not be empty")
    parsed_streams: list[str] = []
    seen_streams: set[str] = set()
    for i, raw_stream in enumerate(streams):
        stream = _require_str(raw_stream, f"{context}[{i}]")
        _validate_stream_name(stream, f"{context}[{i}]")
        if stream in seen_streams:
            raise ManifestError(f"{context} contains duplicate stream {stream!r}")
        seen_streams.add(stream)
        parsed_streams.append(stream)
    return parsed_streams


def _parse_python_index(
    data: object,
    context: str,
    stream_config: StreamConfig,
) -> PythonIndexOwnership:
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
        _validate_normalized_package_name(
            package_name,
            f"{context}.packages.{package_name}",
            ManifestError,
        )
        package_config = _require_mapping(
            raw_config, f"{context}.packages.{package_name}"
        )
        _require_keys(
            package_config,
            {"owner_path"},
            f"{context}.packages.{package_name}",
            optional_keys={"stream_group", "streams"},
        )
        owner_path = _require_str(
            package_config["owner_path"],
            f"{context}.packages.{package_name}.owner_path",
        )
        _validate_owner_path(
            owner_path, f"{context}.packages.{package_name}.owner_path"
        )
        streams = _parse_package_streams(
            package_config,
            stream_config,
            f"{context}.packages.{package_name}",
        )
        packages[package_name] = PackageOwnership(
            name=package_name,
            owner_path=owner_path,
            streams=streams,
        )

    route_ordered_packages = sorted(
        packages.values(), key=lambda p: (p.owner_path, p.name)
    )
    return PythonIndexOwnership(
        public_base=public_base,
        packages={package.name: package for package in route_ordered_packages},
    )


def _parse_package_streams(
    package_config: Mapping[object, object],
    stream_config: StreamConfig,
    context: str,
) -> frozenset[str]:
    has_stream_group = "stream_group" in package_config
    has_streams = "streams" in package_config
    if has_stream_group and has_streams:
        raise ManifestError(
            f"{context} must not contain both 'stream_group' and 'streams'"
        )
    if has_stream_group:
        group_name = _require_str(
            package_config["stream_group"], f"{context}.stream_group"
        )
        _validate_stream_group_name(group_name, f"{context}.stream_group")
        if group_name not in stream_config.groups:
            raise ManifestError(
                f"{context}.stream_group contains unknown stream group "
                f"{group_name!r}"
            )
        return stream_config.groups[group_name]
    if not has_streams:
        return stream_config.default
    streams = frozenset(
        _parse_stream_list(package_config["streams"], f"{context}.streams")
    )
    unknown_streams = sorted(streams - set(stream_config.known))
    if unknown_streams:
        raise ManifestError(
            f"{context}.streams contains unknown stream(s): "
            f"{', '.join(unknown_streams)}"
        )
    return streams


def _validate_manifest_stream(
    manifest: OwnershipManifest,
    stream: str,
    context: str,
) -> None:
    _validate_stream_name(stream, context)
    if stream not in manifest.streams.known:
        raise ManifestError(f"{context} contains unknown stream {stream!r}")


def _validate_stream_name(stream: str, context: str) -> None:
    if not _STREAM_RE.fullmatch(stream):
        raise ManifestError(
            f"{context} must start with lowercase alphanumeric and contain only "
            "lowercase alphanumeric, '_', or '-'"
        )


def _validate_stream_group_name(group_name: str, context: str) -> None:
    if not _STREAM_RE.fullmatch(group_name):
        raise ManifestError(
            f"{context} must start with lowercase alphanumeric and contain only "
            "lowercase alphanumeric, '_', or '-'"
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
    *,
    optional_keys: set[str] | None = None,
) -> None:
    optional_keys = optional_keys or set()
    actual_keys = set(data)
    unknown_keys = actual_keys - expected_keys - optional_keys
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


def _validate_normalized_package_name(
    package_name: str,
    context: str,
    error_type: type[ManifestError] | type[IndexValidationError],
) -> None:
    if not is_normalized_name(package_name) or not _PUBLIC_PATH_SEGMENT_RE.fullmatch(
        package_name
    ):
        raise error_type(
            f"{context}: package name must be a valid normalized Python project name"
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
    _validate_normalized_package_name(
        package_name,
        f"{context}: package link href {href!r}",
        IndexValidationError,
    )
    return package_name


def _read_non_empty_text(path: Path, description: str) -> str:
    _require_non_empty_file(path, description)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise IndexValidationError(f"Cannot read {description}: {path}: {e}") from e


def _require_non_empty_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise IndexValidationError(f"Missing {description}: {path}")
    if path.stat().st_size == 0:
        raise IndexValidationError(f"Empty {description}: {path}")


def _assert_output_package_sets_match(
    validated: ValidatedIndexContent,
    aggregate_root_html: str,
    route_table: RouteTable,
    validation_report: ValidationReport,
) -> None:
    expected_packages = set(validated.packages)
    aggregate_packages = {
        link.package_name
        for link in _parse_product_root_index_text(
            aggregate_root_html, "aggregate root"
        )
    }
    route_packages = {route.package for route in route_table.routes}
    validation_packages = {package.name for package in validation_report.packages}
    if aggregate_packages != expected_packages:
        raise RuntimeError("Aggregate root package set does not match validated input")
    if route_packages != expected_packages:
        raise RuntimeError("Route table package set does not match validated input")
    if validation_packages != expected_packages:
        raise RuntimeError(
            "Validation report package set does not match validated input"
        )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        dir=path.parent,
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        f.write(text)
        tmp_path = Path(f.name)
    tmp_path.replace(path)


def _json_dumps(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _public_path(owner_path: str, *segments: str) -> str:
    quoted_segments = [quote(segment, safe="") for segment in segments]
    return "/" + "/".join(["rocm", owner_path, *quoted_segments])


def _product_root_public_path(owner_path: str) -> str:
    return _public_path(owner_path, "index.html")


def _package_public_base(owner_path: str, package_name: str) -> str:
    return _public_path(owner_path, package_name) + "/"


def _package_index_public_path(owner_path: str, package_name: str) -> str:
    return _public_path(owner_path, package_name, "index.html")


def _validate_manifest_command(args: argparse.Namespace) -> int:
    manifest = load_ownership_manifest(args.manifest)
    index = manifest.python_indexes[0]

    print(f"schema_version: {manifest.schema_version}")
    print(f"public_base: {index.public_base}")
    print(f"streams: {', '.join(manifest.streams.known)}")
    print(f"default_streams: {', '.join(sorted(manifest.streams.default))}")
    print(f"packages: {len(index.packages)}")
    for stream in manifest.streams.known:
        owner_counts: dict[str, int] = {}
        for package in index.active_packages(stream).values():
            owner_counts[package.owner_path] = (
                owner_counts.get(package.owner_path, 0) + 1
            )
        print(f"stream {stream}: {sum(owner_counts.values())}")
        for owner_path in sorted(owner_counts):
            print(f"  {owner_path}: {owner_counts[owner_path]}")
    return 0


def _generate_command(args: argparse.Namespace) -> int:
    manifest = load_ownership_manifest(args.manifest)
    outputs = generate_outputs(
        manifest,
        args.output_dir,
        stream=args.stream,
        content_root=args.content_root,
        strict_completeness=args.strict_completeness,
        require_all_manifest_packages=not args.allow_unpublished,
    )
    _warn_unpublished_packages(outputs.unpublished_package_count)
    print(f"aggregate_root: {outputs.aggregate_root}")
    print(f"route_table: {outputs.route_table}")
    print(f"validation_report: {outputs.validation_report}")
    return 0


def _validate_content_command(args: argparse.Namespace) -> int:
    manifest = load_ownership_manifest(args.manifest)
    validated = validate_product_indexes(
        manifest,
        args.content_root,
        stream=args.stream,
        strict_completeness=args.strict_completeness,
        require_all_manifest_packages=not args.allow_unpublished,
    )
    owner_counts: dict[str, int] = {}
    for package in validated.ordered_packages():
        owner_counts[package.owner_path] = owner_counts.get(package.owner_path, 0) + 1

    print(f"public_base: {validated.public_base}")
    print(f"stream: {validated.stream}")
    print(f"packages: {len(validated.packages)}")
    print(f"unpublished_packages: {len(validated.unpublished_packages)}")
    _warn_unpublished_packages(len(validated.unpublished_packages))
    for owner_path in sorted(owner_counts):
        print(f"{owner_path}: {owner_counts[owner_path]}")
    return 0


def _warn_unpublished_packages(unpublished_count: int) -> None:
    if unpublished_count:
        print(
            f"warning: {unpublished_count} manifest package(s) are unpublished "
            "and excluded from active aggregate outputs",
            file=sys.stderr,
        )


def _add_content_validation_flags(parser: argparse.ArgumentParser) -> None:
    """Add flags controlling manifest/product-root set comparison."""
    parser.add_argument(
        "--allow-unpublished",
        action="store_true",
        help="allow manifest-owned packages absent from product roots and exclude "
        "them from validated output; requires --content-root",
    )
    parser.add_argument(
        "--strict-completeness",
        action="store_true",
        help="fail if a product root publishes a package absent from the "
        "ownership manifest; requires --content-root",
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(required=True)
    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="validate an aggregate index ownership manifest",
        allow_abbrev=False,
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
        help="generate aggregate routed outputs, optionally validating "
        "product-local content",
        allow_abbrev=False,
    )
    generate_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the ownership manifest YAML file",
    )
    generate_parser.add_argument(
        "--stream",
        required=True,
        help="Concrete stream to generate, such as dev, nightly, rc, bkc, "
        "stable, or stable-staging",
    )
    generate_parser.add_argument(
        "--content-root",
        type=Path,
        help="Optional local directory mirroring public /rocm/... product indexes",
    )
    generate_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where aggregate outputs will be written",
    )
    _add_content_validation_flags(generate_parser)
    generate_parser.set_defaults(func=_generate_command)

    content_parser = subparsers.add_parser(
        "validate-content",
        help="validate product-local index content for an ownership manifest",
        allow_abbrev=False,
    )
    content_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the ownership manifest YAML file",
    )
    content_parser.add_argument(
        "--stream",
        required=True,
        help="Concrete stream to validate, such as dev, nightly, rc, bkc, "
        "stable, or stable-staging",
    )
    content_parser.add_argument(
        "--content-root",
        type=Path,
        required=True,
        help="Local directory mirroring public /rocm/... product indexes",
    )
    _add_content_validation_flags(content_parser)
    content_parser.set_defaults(func=_validate_content_command)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ManifestError, IndexValidationError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
