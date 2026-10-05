#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify built native Linux packages against ``built_packages.txt``.

After ``packaging_summary.write_build_manifest``, confirm each ``.deb`` /
``.rpm`` filename listed under Created Packages exists under
``--packages-dir``. Failed/Skipped sections list base names without those
extensions and are excluded from the presence check (v1).

Scope (presence only):
    Parse the existing text manifest and check files on disk. Does not change
    the ``built_packages.txt`` format, does not read package control fields,
    and does not re-derive expectations from ``package.json``.

Example::

    ./build_tools/packaging/linux/build_package_verify.py \\
        --packages-dir output/packages \\
        --pkg-type deb
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow ``python build_tools/packaging/linux/build_package_verify.py`` from repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_TOOLS_DIR = _SCRIPT_DIR.parent.parent
for _path in (_BUILD_TOOLS_DIR, _SCRIPT_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from _therock_utils.log_utils import TheRockLogger, configure_logging

logger = TheRockLogger(__name__)

MANIFEST_NAME = "built_packages.txt"
_PACKAGE_EXTENSIONS = {".deb", ".rpm"}


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


def format_report_text(result: ManifestVerifyResult, manifest_path: Path) -> str:
    """Format a human-readable verification summary for stdout.

    Parameters:
        result: Outcome from :func:`verify_manifest_packages`.
        manifest_path: Path to the manifest used for this run (for the report).

    Returns:
        Multi-line report string ending without a trailing blank line requirement.
    """
    overall = "PASS" if result.passed else "FAIL"
    lines = [
        "ROCm build package manifest verification",
        "=" * 72,
        f"Overall result: {overall}",
        f"Manifest: {manifest_path}",
        f"Packages expected: {len(result.expected)}",
        f"Packages found: {len(result.found)}",
        f"Packages missing: {len(result.missing)}",
    ]
    if result.missing:
        lines.append("")
        lines.append("Missing packages:")
        for name in result.missing:
            lines.append(f"  - {name}")
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
            "--packages-dir."
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
    """Execute manifest presence verification.

    Parameters:
        args: Parsed command-line arguments from :func:`parse_args`.

    Returns:
        ``0`` on success, ``1`` when files are missing, ``2`` on configuration
        errors (missing packages-dir/manifest, empty Created list, bad path).
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

    pkg_type = (args.pkg_type or "").strip().lower()
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
        result = verify_manifest_packages(packages_dir, package_names)
    except ValueError as exc:
        logger.error(f"{exc}")
        return 2

    print(format_report_text(result, manifest_path))

    if not result.passed:
        logger.error(
            f"Manifest verification failed: {len(result.missing)} missing file(s)",
        )
        return 1

    logger.info("Manifest verification passed.")
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
