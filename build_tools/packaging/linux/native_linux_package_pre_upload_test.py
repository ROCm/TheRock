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

Does not re-derive expectations from ``package.json``.

Example::

    ./build_tools/packaging/linux/native_linux_package_pre_upload_test.py \\
        --packages-dir output/packages \\
        --pkg-type deb
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
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

logger = TheRockLogger(__name__)

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


def load_package_claims(claim_path: Path) -> dict[str, dict[str, object]]:
    """Load ``built_packages.json`` and index package entries by filename.

    Validates ``schema_version`` against
    ``packaging_summary.MANIFEST_SCHEMA_VERSION``. Does not re-probe packages;
    that happens in :func:`verify_control_fields`.

    Parameters:
        claim_path: Path to the JSON claim document from
            ``packaging_summary.write_build_manifest``.

    Returns:
        Mapping of filename → package claim dict (``control`` and/or ``error``).

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
    by_name: dict[str, dict[str, object]] = {}
    for entry in packages:
        if not isinstance(entry, dict):
            continue
        filename = entry.get("filename")
        if isinstance(filename, str) and filename:
            by_name[filename] = entry
    return by_name


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
    claim_path: Path | None = None,
) -> str:
    """Format a human-readable presence + control verification summary for stdout.

    When ``control`` is omitted (e.g. presence already failed), only the
    presence section is printed.
    """
    overall_ok = presence.passed and (control is None or control.passed)
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
            "--packages-dir, then compare control fields to built_packages.json."
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
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Execute presence + control-field verification.

    Parameters:
        args: Parsed command-line arguments from :func:`parse_args`.

    Returns:
        ``0`` on success, ``1`` when presence or control checks fail, ``2`` on
        configuration errors (missing packages-dir/manifest/claim, empty Created
        list, bad path).

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
        claims = load_package_claims(claim_path)
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

    print(
        format_report_text(
            presence,
            manifest_path,
            control=control,
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

    logger.info("Manifest verification passed.")
    logger.info(
        f"Control-field verification passed ({len(control.checked)} package(s)).",
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
