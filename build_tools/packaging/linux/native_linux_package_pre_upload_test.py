#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify built native Linux packages before upload.

After ``packaging_summary.write_build_manifest``:

1. **Presence** — each Created ``.deb`` / ``.rpm`` in ``built_packages.txt``
   exists under ``--packages-dir`` (path-safe).
2. **Control fields** — for each present package, re-read control/tags from the
   artifact and compare to the claim in ``built_packages.json`` (Version,
   Depends/Requires, and related fields written at build time).
3. **package.json fidelity** — claim Version / Package|Name / Depends|Requires
   match what packaging helpers derive from ``package.json`` (same
   ``update_package_name`` / ``process_*_dependencies`` used at build time).

Example::

    ./build_tools/packaging/linux/native_linux_package_pre_upload_test.py \\
        --packages-dir output/packages \\
        --pkg-type deb \\
        --artifacts-dir output/artifacts
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

# Allow ``python build_tools/packaging/linux/native_linux_package_pre_upload_test.py`` from repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_TOOLS_DIR = _SCRIPT_DIR.parent.parent
for _path in (_BUILD_TOOLS_DIR, _SCRIPT_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from _therock_utils.log_utils import TheRockLogger, configure_logging
import packaging_summary as summary
from packaging_utils import (
    GFX_HOST,
    GFX_META,
    PackageConfig,
    get_package_info,
    process_nonversioned_dependencies,
    process_versioned_dependencies,
    read_package_json_file,
    update_package_name,
)

logger = TheRockLogger(__name__)

# Bare package name from a Depends/Requires token (drop version constraints).
_DEP_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9+.\-]*)")

# Same filenames as packaging_summary.write_build_manifest output.
MANIFEST_NAME = summary.MANIFEST_TXT_NAME
CLAIM_JSON_NAME = summary.MANIFEST_JSON_NAME
_PACKAGE_EXTENSIONS = {".deb", ".rpm"}

# Compare the exact field sets written into the JSON claim (no second table).
_DEB_COMPARE_FIELDS: tuple[str, ...] = summary.DEB_CONTROL_FIELDS
_RPM_COMPARE_FIELDS: tuple[str, ...] = tuple(
    name for name, _ in summary.RPM_SCALAR_TAGS
) + tuple(name for name, _ in summary.RPM_LIST_QUERIES)


@dataclass
class ManifestVerifyResult:
    """Outcome of checking manifest filenames against files on disk.

    Attributes:
        expected: Filenames parsed from the Created Packages section.
        found: Subset of ``expected`` that exist under ``packages-dir``.
        missing: Subset of ``expected`` that are absent.
    """

    expected: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True when there is at least one expected name and none missing."""
        return bool(self.expected) and not self.missing


@dataclass
class ControlFieldMismatch:
    """One claim vs live control-field difference.

    Attributes:
        filename: Package basename under ``packages-dir``.
        field: Control/tag name (e.g. ``Depends``, ``Version``).
        claimed: Value from ``built_packages.json``.
        actual: Value re-read from the artifact at verify time.
    """

    filename: str
    field: str
    claimed: str
    actual: str


@dataclass
class ControlVerifyResult:
    """Outcome of comparing ``built_packages.json`` claims to live package control.

    Attributes:
        checked: Filenames that completed a full field compare (may still have
            mismatches listed separately).
        mismatches: Field-level claim ≠ live differences.
        claim_errors: Build-time soft-fail ``error`` entries, or missing
            ``control`` object — hard-fail at pre-upload.
        missing_claims: On-disk packages with no JSON entry.
        live_read_errors: ``read_package_control_fields`` failures at verify time.
    """

    checked: list[str] = field(default_factory=list)
    mismatches: list[ControlFieldMismatch] = field(default_factory=list)
    claim_errors: list[str] = field(default_factory=list)
    missing_claims: list[str] = field(default_factory=list)
    live_read_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when at least one package was checked and no error lists are set."""
        return (
            bool(self.checked)
            and not self.mismatches
            and not self.claim_errors
            and not self.missing_claims
            and not self.live_read_errors
        )


@dataclass
class PackageJsonMismatch:
    """One claim vs ``package.json`` expectation difference."""

    filename: str
    field: str
    claimed: str
    expected: str


@dataclass
class PackageJsonVerifyResult:
    """Outcome of comparing the JSON claim to ``package.json`` expectations.

    Attributes:
        checked: Filenames that completed fidelity checks.
        mismatches: Version / Depends / Requires differences.
        unresolved: Claim packages whose Package/Name did not match any
            ``update_package_name`` result from ``package.json``.
    """

    checked: list[str] = field(default_factory=list)
    mismatches: list[PackageJsonMismatch] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checked) and not self.mismatches and not self.unresolved


def parse_built_package_names(
    manifest_path: Path,
    *,
    pkg_type: str | None = None,
    name_prefix: str | None = None,
) -> list[str]:
    """Parse Created package filenames from ``built_packages.txt``.

    Only non-comment lines ending in ``.deb`` / ``.rpm`` are treated as built
    package filenames. Failed/Skipped sections list base names without those
    extensions and are ignored.

    Parameters:
        manifest_path: Path to ``built_packages.txt``.
        pkg_type: Optional ``deb`` or ``rpm`` to keep only that extension.
        name_prefix: Optional filename prefix filter (e.g. ``amdrocm-core-sdk``).

    Returns:
        Sorted unique package filenames from the Created Packages section.

    Raises:
        FileNotFoundError: When the manifest file does not exist.
        ValueError: When ``pkg_type`` is not ``deb`` or ``rpm``.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"built packages manifest not found: {manifest_path}")

    allowed_ext: set[str] | None = None
    if pkg_type is not None:
        normalized = pkg_type.strip().lower()
        if normalized not in {"deb", "rpm"}:
            raise ValueError(f"Unsupported pkg_type {pkg_type!r}; expected deb or rpm")
        allowed_ext = {f".{normalized}"}

    prefix = name_prefix.strip() if name_prefix else None
    names: list[str] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        suffix = Path(line).suffix.lower()
        if suffix not in _PACKAGE_EXTENSIONS:
            continue
        if allowed_ext is not None and suffix not in allowed_ext:
            continue
        if prefix is not None and not line.startswith(prefix):
            continue
        names.append(line)
    return sorted(set(names))


def _resolve_package_path(packages_dir: Path, package_name: str) -> Path:
    """Resolve a manifest filename under ``packages_dir``, rejecting traversal.

    Parameters:
        packages_dir: Resolved packages output directory.
        package_name: Filename from ``built_packages.txt``.

    Returns:
        Absolute path to the package file under ``packages_dir``.

    Raises:
        ValueError: When ``package_name`` escapes ``packages_dir`` (e.g. ``../``).
    """
    packages_dir = packages_dir.resolve()
    candidate = (packages_dir / package_name).resolve()
    try:
        candidate.relative_to(packages_dir)
    except ValueError as exc:
        raise ValueError(
            f"package path escapes packages directory: {package_name!r}",
        ) from exc
    return candidate


def verify_manifest_packages(
    packages_dir: Path,
    package_names: list[str],
) -> ManifestVerifyResult:
    """Check that each expected package filename exists under ``packages_dir``.

    Parameters:
        packages_dir: Directory that should contain the built package files.
        package_names: Filenames from :func:`parse_built_package_names`.

    Returns:
        Result with ``expected`` / ``found`` / ``missing`` filename lists.

    Raises:
        ValueError: When a manifest entry would escape ``packages_dir``.
    """
    packages_dir = packages_dir.resolve()
    result = ManifestVerifyResult(expected=list(package_names))
    for name in package_names:
        path = _resolve_package_path(packages_dir, name)
        if path.is_file():
            result.found.append(name)
        else:
            result.missing.append(name)
    return result


def load_claim_document(claim_path: Path) -> dict[str, object]:
    """Load and validate ``built_packages.json`` (full document).

    Raises:
        FileNotFoundError: When the claim file is missing.
        ValueError: When JSON is invalid or schema is unsupported.
    """
    if not claim_path.is_file():
        raise FileNotFoundError(f"built packages claim not found: {claim_path}")
    try:
        doc = json.loads(claim_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid built packages claim JSON: {claim_path}: {exc}"
        ) from exc
    if not isinstance(doc, dict):
        raise ValueError(f"built packages claim must be a JSON object: {claim_path}")
    schema = doc.get("schema_version")
    if schema != summary.MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported claim schema_version {schema!r} in {claim_path} "
            f"(expected {summary.MANIFEST_SCHEMA_VERSION})",
        )
    packages = doc.get("packages")
    if not isinstance(packages, list):
        raise ValueError(f"built packages claim missing packages list: {claim_path}")
    return doc


def index_claims_by_filename(
    claim_doc: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Index ``claim_doc["packages"]`` by filename."""
    by_name: dict[str, dict[str, object]] = {}
    packages = claim_doc.get("packages")
    if not isinstance(packages, list):
        return by_name
    for entry in packages:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        if isinstance(filename, str) and filename:
            by_name[filename] = entry
    return by_name


def load_package_claims(claim_path: Path) -> dict[str, dict[str, object]]:
    """Load ``built_packages.json`` and index package entries by filename."""
    return index_claims_by_filename(load_claim_document(claim_path))


def expected_control_version(
    rocm_version: str,
    version_suffix: str,
    *,
    pkg_type: str,
) -> str:
    """Expected control Version (deb) or Version string (rpm Version field only).

    Mirrors ``deb_package`` / ``rpm_package`` version construction.
    """
    normalized = pkg_type.strip().lower()
    if normalized == "deb":
        if version_suffix:
            return f"{rocm_version}-{version_suffix}"
        return rocm_version
    if normalized == "rpm":
        return rocm_version
    raise ValueError(f"Unsupported pkg_type {pkg_type!r}; expected deb or rpm")


def parse_dependency_names(dep_string: str) -> set[str]:
    """Extract bare package names from a Depends/Requires field string."""
    names: set[str] = set()
    for raw in (dep_string or "").split(","):
        token = raw.strip()
        if not token:
            continue
        # DEB alternatives: take each side of |.
        for alt in token.split("|"):
            match = _DEP_NAME_RE.match(alt.strip())
            if match:
                name = match.group(1)
                if not name.startswith("rpmlib("):
                    names.add(name)
    return names


def _candidate_gfx_archs(claim_gfx_arch: str) -> list[str]:
    archs: list[str] = []
    if claim_gfx_arch and claim_gfx_arch not in archs:
        archs.append(claim_gfx_arch)
    for extra in (GFX_META, GFX_HOST):
        if extra not in archs:
            archs.append(extra)
    return archs


def match_claim_name_to_package_json(
    installed_name: str,
    *,
    pkg_type: str,
    rocm_version: str,
    version_suffix: str,
    claim_gfx_arch: str,
    artifacts_dir: Path | None,
) -> tuple[str, PackageConfig] | tuple[None, None]:
    """Find ``package.json`` base + ``PackageConfig`` that yields ``installed_name``.

    Tries versioned/non-versioned × claim gfx arch / GFX_META / GFX_HOST using
    the same ``update_package_name`` as packaging.
    """
    try:
        entries = read_package_json_file()
    except OSError as exc:
        raise ValueError(f"unable to read package.json: {exc}") from exc

    artifacts = artifacts_dir if artifacts_dir is not None else Path(".")
    gfx_list = tuple(
        a
        for a in _candidate_gfx_archs(claim_gfx_arch)
        if a not in (GFX_HOST, GFX_META, "")
    )

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        base = entry.get("Package")
        if not isinstance(base, str) or not base:
            continue
        for versioned in (True, False):
            for gfx_arch in _candidate_gfx_archs(claim_gfx_arch):
                config = PackageConfig(
                    artifacts_dir=artifacts,
                    dest_dir=artifacts,
                    pkg_type=pkg_type,
                    rocm_version=rocm_version,
                    version_suffix=version_suffix or "",
                    install_prefix="/opt/rocm",
                    gfx_arch=gfx_arch,
                    versioned_pkg=versioned,
                    gfxarch_list=gfx_list,
                )
                try:
                    if update_package_name(base, config) == installed_name:
                        return base, config
                except (ValueError, TypeError):
                    continue
    return None, None


def _expected_depends_string(pkg_info: dict, config: PackageConfig) -> str:
    """Reuse build-time dependency helpers for the matched variant."""
    if config.versioned_pkg:
        field_key = (
            "DEBDepends" if str(config.pkg_type).lower() == "deb" else "RPMRequires"
        )
        return process_versioned_dependencies(pkg_info, field_key, config)
    return process_nonversioned_dependencies(pkg_info, config)


def verify_claim_against_package_json(
    package_names: list[str],
    claims_by_filename: dict[str, dict[str, object]],
    claim_doc: dict[str, object],
    *,
    pkg_type: str,
    artifacts_dir: Path | None = None,
) -> PackageJsonVerifyResult:
    """Compare claim control fields to ``package.json``-derived expectations.

    For each present package with a control claim:
    - Version must match ``rocm_version`` / ``version_suffix`` from the claim doc
      (same construction as ``deb_package`` / ``rpm_package``).
    - Package/Name must match ``update_package_name`` for some ``package.json``
      entry under claim gfx settings.
    - Depends/Requires name-set must include every name expected from
      ``process_versioned_dependencies`` / ``process_nonversioned_dependencies``
      (claim may list extra system deps such as ``libc6``).

    RPM Release is checked against ``version_suffix`` when present.
    """
    result = PackageJsonVerifyResult()
    rocm_version = str(claim_doc.get("rocm_version") or "")
    version_suffix = str(claim_doc.get("version_suffix") or "")
    claim_gfx_arch = str(claim_doc.get("gfx_arch") or "")
    if not rocm_version:
        raise ValueError("built packages claim missing rocm_version")

    expected_version = expected_control_version(
        rocm_version, version_suffix, pkg_type=pkg_type
    )
    dep_field = "Depends" if pkg_type == "deb" else "Requires"
    name_field = "Package" if pkg_type == "deb" else "Name"

    for filename in package_names:
        claim = claims_by_filename.get(filename)
        if claim is None or claim.get("error"):
            continue
        control_obj = claim.get("control")
        if not isinstance(control_obj, dict):
            continue

        claimed_version = _normalize_field_value(control_obj.get("Version", ""))
        if claimed_version != expected_version:
            result.mismatches.append(
                PackageJsonMismatch(
                    filename=filename,
                    field="Version",
                    claimed=claimed_version,
                    expected=expected_version,
                )
            )

        if pkg_type == "rpm" and version_suffix:
            claimed_release = _normalize_field_value(control_obj.get("Release", ""))
            if claimed_release != version_suffix:
                result.mismatches.append(
                    PackageJsonMismatch(
                        filename=filename,
                        field="Release",
                        claimed=claimed_release,
                        expected=version_suffix,
                    )
                )

        installed_name = _normalize_field_value(control_obj.get(name_field, ""))
        if not installed_name:
            result.unresolved.append(f"{filename}: claim missing {name_field}")
            result.checked.append(filename)
            continue

        base, matched_config = match_claim_name_to_package_json(
            installed_name,
            pkg_type=pkg_type,
            rocm_version=rocm_version,
            version_suffix=version_suffix,
            claim_gfx_arch=claim_gfx_arch,
            artifacts_dir=artifacts_dir,
        )
        if base is None or matched_config is None:
            result.unresolved.append(
                f"{filename}: {name_field}={installed_name!r} "
                "not produced by package.json + update_package_name"
            )
            result.checked.append(filename)
            continue

        pkg_info = get_package_info(base, raise_if_missing=False)
        if pkg_info is None:
            result.unresolved.append(
                f"{filename}: package.json entry missing for {base!r}"
            )
            result.checked.append(filename)
            continue

        # Prefer claim gfx for dependency expansion when artifacts are available.
        dep_config = matched_config
        if artifacts_dir is not None:
            dep_config = replace(matched_config, artifacts_dir=artifacts_dir)

        expected_deps = _expected_depends_string(pkg_info, dep_config)
        expected_names = parse_dependency_names(expected_deps)
        claimed_deps = _normalize_field_value(control_obj.get(dep_field, ""))
        claimed_names = parse_dependency_names(claimed_deps)
        missing = sorted(expected_names - claimed_names)
        if missing:
            result.mismatches.append(
                PackageJsonMismatch(
                    filename=filename,
                    field=dep_field,
                    claimed=claimed_deps,
                    expected=f"missing names: {', '.join(missing)}",
                )
            )

        result.checked.append(filename)
    return result


def _compare_fields_for_pkg_type(pkg_type: str) -> tuple[str, ...]:
    """Return the ordered control-field names to compare for ``deb`` or ``rpm``."""
    normalized = pkg_type.strip().lower()
    if normalized == "deb":
        return _DEB_COMPARE_FIELDS
    if normalized == "rpm":
        return _RPM_COMPARE_FIELDS
    raise ValueError(f"Unsupported pkg_type {pkg_type!r}; expected deb or rpm")


def _normalize_field_value(value: object) -> str:
    """Coerce claim/live field values to stripped strings for equality checks."""
    if value is None:
        return ""
    return str(value).strip()


def verify_control_fields(
    packages_dir: Path,
    package_names: list[str],
    claims_by_filename: dict[str, dict[str, object]],
    *,
    pkg_type: str,
) -> ControlVerifyResult:
    """Compare live package control fields to the JSON claim.

    For each filename: resolve path → load claim → re-read control with the
    same :func:`packaging_summary.read_package_control_fields` used at build
    time → string-compare every field for ``pkg_type``.

    Build-time probe soft-fails (``error`` in the claim) become hard failures
    here. Does not consult ``package.json``.

    Parameters:
        packages_dir: Directory containing package files.
        package_names: Filenames to verify (typically presence ``found`` list).
        claims_by_filename: Claim entries from :func:`load_package_claims`.
        pkg_type: ``deb`` or ``rpm`` (selects compared field set).

    Returns:
        :class:`ControlVerifyResult` with mismatches and claim/live errors.
    """
    fields = _compare_fields_for_pkg_type(pkg_type)
    result = ControlVerifyResult()
    packages_dir = packages_dir.resolve()

    for name in package_names:
        claim = claims_by_filename.get(name)
        if claim is None:
            result.missing_claims.append(name)
            continue
        # Soft-fail at build time → fail closed before upload.
        if claim.get("error"):
            result.claim_errors.append(f"{name}: {claim['error']}")
            continue
        control_obj = claim.get("control")
        if not isinstance(control_obj, dict):
            result.claim_errors.append(f"{name}: claim missing control object")
            continue

        path = _resolve_package_path(packages_dir, name)
        try:
            # Same reader as packaging_summary.collect_created_package_claims.
            live = summary.read_package_control_fields(path)
        except (OSError, ValueError, RuntimeError) as exc:
            result.live_read_errors.append(f"{name}: {exc}")
            continue

        for field_name in fields:
            claimed = _normalize_field_value(control_obj.get(field_name, ""))
            actual = _normalize_field_value(live.get(field_name, ""))
            if claimed != actual:
                result.mismatches.append(
                    ControlFieldMismatch(
                        filename=name,
                        field=field_name,
                        claimed=claimed,
                        actual=actual,
                    )
                )
        result.checked.append(name)
    return result


def format_report_text(
    presence: ManifestVerifyResult,
    manifest_path: Path,
    *,
    control: ControlVerifyResult | None = None,
    package_json: PackageJsonVerifyResult | None = None,
    claim_path: Path | None = None,
) -> str:
    """Format a human-readable presence + control + package.json summary."""
    overall_ok = (
        presence.passed
        and (control is None or control.passed)
        and (package_json is None or package_json.passed)
    )
    overall = "PASS" if overall_ok else "FAIL"
    lines = [
        "ROCm build package manifest verification",
        "=" * 72,
        f"Overall result: {overall}",
        f"Manifest: {manifest_path}",
        f"Packages expected: {len(presence.expected)}",
        f"Packages found: {len(presence.found)}",
        f"Packages missing: {len(presence.missing)}",
    ]
    if presence.missing:
        lines.append("")
        lines.append("Missing packages:")
        for name in presence.missing:
            lines.append(f"  - {name}")

    if control is not None:
        lines.append("")
        lines.append("Control-field claim verification")
        if claim_path is not None:
            lines.append(f"Claim: {claim_path}")
        lines.append(f"Packages checked: {len(control.checked)}")
        lines.append(f"Field mismatches: {len(control.mismatches)}")
        lines.append(f"Claim errors: {len(control.claim_errors)}")
        lines.append(f"Missing claims: {len(control.missing_claims)}")
        lines.append(f"Live read errors: {len(control.live_read_errors)}")
        if control.mismatches:
            lines.append("")
            lines.append("Mismatched fields:")
            for m in control.mismatches:
                lines.append(
                    f"  - {m.filename} {m.field}: "
                    f"claimed={m.claimed!r} actual={m.actual!r}"
                )
        if control.claim_errors:
            lines.append("")
            lines.append("Claim errors:")
            for msg in control.claim_errors:
                lines.append(f"  - {msg}")
        if control.missing_claims:
            lines.append("")
            lines.append("Missing claim entries:")
            for name in control.missing_claims:
                lines.append(f"  - {name}")
        if control.live_read_errors:
            lines.append("")
            lines.append("Live control read errors:")
            for msg in control.live_read_errors:
                lines.append(f"  - {msg}")

    if package_json is not None:
        lines.append("")
        lines.append("package.json fidelity (claim vs packaging helpers)")
        lines.append(f"Packages checked: {len(package_json.checked)}")
        lines.append(f"Field mismatches: {len(package_json.mismatches)}")
        lines.append(f"Unresolved names: {len(package_json.unresolved)}")
        if package_json.mismatches:
            lines.append("")
            lines.append("package.json mismatches:")
            for m in package_json.mismatches:
                lines.append(
                    f"  - {m.filename} {m.field}: "
                    f"claimed={m.claimed!r} expected={m.expected!r}"
                )
        if package_json.unresolved:
            lines.append("")
            lines.append("Unresolved claim Package/Name:")
            for msg in package_json.unresolved:
                lines.append(f"  - {msg}")

    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters:
        argv: Argument list (typically ``sys.argv[1:]``).

    Returns:
        Parsed namespace for :func:`run`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Verify Created Packages from built_packages.txt exist under "
            "--packages-dir, compare control fields to built_packages.json, "
            "then check claim Version/Depends against package.json helpers."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level logging)",
    )
    parser.add_argument(
        "--packages-dir",
        type=Path,
        required=True,
        help="Directory containing built .deb or .rpm files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(f"Path to {MANIFEST_NAME} (default: <packages-dir>/{MANIFEST_NAME})"),
    )
    parser.add_argument(
        "--claim-json",
        type=Path,
        default=None,
        help=(
            f"Path to {CLAIM_JSON_NAME} " f"(default: <packages-dir>/{CLAIM_JSON_NAME})"
        ),
    )
    parser.add_argument(
        "--pkg-type",
        choices=("deb", "rpm", "DEB", "RPM"),
        required=True,
        help="Package format filter for manifest filenames",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help=(
            "Only verify filenames starting with this prefix "
            "(e.g. amdrocm-core-sdk for a smoke subset)"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Artifact tree used by packaging gfx-arch helpers when matching "
            "claim Package/Name and Depends to package.json (recommended in CI)"
        ),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Execute presence + control-field + package.json fidelity verification.

    Parameters:
        args: Parsed command-line arguments from :func:`parse_args`.

    Returns:
        ``0`` on success, ``1`` when presence, control, or package.json checks
        fail, ``2`` on configuration errors (missing packages-dir/manifest/claim,
        empty Created list, bad path).

    Note:
        To skip this entire gate in CI, disable ``enable_pre_upload_build_verify``
        on the workflow — there is no per-check CLI skip flag.
    """
    packages_dir = args.packages_dir.expanduser().resolve()
    if not packages_dir.is_dir():
        logger.error(f"packages directory not found: {packages_dir}")
        return 2

    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else packages_dir / MANIFEST_NAME
    )
    claim_path = (
        args.claim_json.expanduser().resolve()
        if args.claim_json is not None
        else packages_dir / CLAIM_JSON_NAME
    )
    artifacts_dir = (
        args.artifacts_dir.expanduser().resolve()
        if args.artifacts_dir is not None
        else None
    )

    pkg_type = (args.pkg_type or "").strip().lower()

    # --- Phase 1: presence (built_packages.txt) ---
    try:
        package_names = parse_built_package_names(
            manifest_path,
            pkg_type=pkg_type,
            name_prefix=args.name_prefix,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"{exc}")
        return 2

    if not package_names:
        logger.error(f"no package filenames found in manifest: {manifest_path}")
        return 2

    try:
        presence = verify_manifest_packages(packages_dir, package_names)
    except ValueError as exc:
        logger.error(f"{exc}")
        return 2

    # --- Phase 2: control claim (built_packages.json) ---
    # Require the claim file even when presence fails so a missing JSON is
    # reported as a config error (exit 2) rather than a silent skip.
    try:
        claim_doc = load_claim_document(claim_path)
        claims = index_claims_by_filename(claim_doc)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"{exc}")
        return 2

    if not presence.passed:
        # Do not compare control for missing artifacts; presence failure is enough.
        print(format_report_text(presence, manifest_path, claim_path=claim_path))
        logger.error(
            f"Manifest verification failed: {len(presence.missing)} missing file(s)",
        )
        return 1

    try:
        control = verify_control_fields(
            packages_dir,
            presence.found,
            claims,
            pkg_type=pkg_type,
        )
    except ValueError as exc:
        logger.error(f"{exc}")
        return 2

    # --- Phase 3: claim vs package.json (reuse packaging_utils) ---
    package_json: PackageJsonVerifyResult | None = None
    if control.passed:
        try:
            package_json = verify_claim_against_package_json(
                presence.found,
                claims,
                claim_doc,
                pkg_type=pkg_type,
                artifacts_dir=artifacts_dir,
            )
        except ValueError as exc:
            logger.error(f"{exc}")
            return 2

    print(
        format_report_text(
            presence,
            manifest_path,
            control=control,
            package_json=package_json,
            claim_path=claim_path,
        )
    )

    if not control.passed:
        logger.error(
            "Control-field verification failed: "
            f"{len(control.mismatches)} mismatch(es), "
            f"{len(control.claim_errors)} claim error(s), "
            f"{len(control.missing_claims)} missing claim(s), "
            f"{len(control.live_read_errors)} live read error(s)",
        )
        return 1

    if package_json is not None and not package_json.passed:
        logger.error(
            "package.json fidelity failed: "
            f"{len(package_json.mismatches)} mismatch(es), "
            f"{len(package_json.unresolved)} unresolved name(s)",
        )
        return 1

    logger.info("Manifest verification passed.")
    logger.info(
        f"Control-field verification passed ({len(control.checked)} package(s)).",
    )
    if package_json is not None:
        logger.info(
            "package.json fidelity passed "
            f"({len(package_json.checked)} package(s)).",
        )
    return 0


def main(argv: list[str]) -> int:
    """Program entry point: parse arguments, configure logging, and run verify.

    Parameters:
        argv: Argument list (typically ``sys.argv[1:]``).

    Returns:
        Process exit code from :func:`run`.
    """
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
