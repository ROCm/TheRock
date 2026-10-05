#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify built native Linux packages against ``built_packages.txt``.

After ``packaging_summary.write_build_manifest``, confirm each ``.deb`` /
``.rpm`` filename listed under Created Packages exists under
``--packages-dir``, then validate control metadata (Package/Name, Version,
Depends/Requires, Architecture) against expectations from
``packaging_utils`` helpers. Failed/Skipped base-name sections are excluded
from the presence check (v1). Inventory always comes from the manifest; field
expectations reuse build helpers (no reimplemented Depends resolver).

```
./build_tools/packaging/linux/build_package_verify.py \\
    --packages-dir output/packages \\
    --pkg-type deb \\
    --rocm-version 7.15.0 \\
    --version-suffix 28484694006 \\
    --artifacts-dir output/artifacts
```
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

# Allow ``python build_tools/packaging/linux/build_package_verify.py`` from repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BUILD_TOOLS_DIR = _SCRIPT_DIR.parent.parent
for _path in (_BUILD_TOOLS_DIR, _SCRIPT_DIR):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from _therock_utils.log_utils import TheRockLogger, configure_logging
from _therock_utils.sdk_targets import group_package_targets
from packaging_utils import (
    GFX_HOST,
    GFX_META,
    PackageConfig,
    get_package_info,
    get_package_list,
    is_gfxarch_package,
    is_meta_package,
    process_nonversioned_dependencies,
    process_versioned_dependencies,
    update_package_name,
)

logger = TheRockLogger(__name__)

MANIFEST_NAME = "built_packages.txt"
_PACKAGE_EXTENSIONS = {".deb", ".rpm"}
_RPM_AUTO_REQUIRE_RE = re.compile(r"^(rpmlib\(|.*\.so)")


@dataclass
class ExpectedControl:
    """Expected control metadata for one installed package name."""

    package_name: str
    version: str
    depends: str
    architecture: str
    base_package: str
    versioned_pkg: bool
    gfx_arch: str


@dataclass
class FieldMismatch:
    """One control-field comparison failure for a package file."""

    filename: str
    package_name: str
    field: str
    expected: str
    actual: str


@dataclass
class ManifestVerifyResult:
    """Outcome of checking manifest entries against files on disk."""

    expected: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    field_mismatches: list[FieldMismatch] = field(default_factory=list)
    unknown_packages: list[str] = field(default_factory=list)
    read_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when files exist and control-field checks (if any) passed."""
        return (
            bool(self.expected)
            and not self.missing
            and not self.field_mismatches
            and not self.unknown_packages
            and not self.read_errors
        )


def parse_built_package_names(
    manifest_path: Path,
    *,
    pkg_type: str | None = None,
    name_prefix: str | None = None,
) -> list[str]:
    """Parse created package filenames from ``built_packages.txt``.

    Only non-comment lines ending in ``.deb`` / ``.rpm`` are treated as built
    package filenames. Failed/skipped sections list base names without those
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
        package_names: Filenames from ``parse_built_package_names``.

    Returns:
        Result with expected / found / missing filename lists.

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


def _run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture stdout/stderr without raising on failure."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        tool = Path(cmd[0]).name if cmd else "tool"
        raise RuntimeError(f"{tool} not found on PATH") from exc


def expected_control_version(config: PackageConfig) -> str:
    """Build the version string expected in DEB/RPM control metadata.

    DEB uses ``rocm_version`` plus optional ``version_suffix`` as the debian
    revision. RPM combines ``VERSION-RELEASE`` where release defaults to ``1``.
    """
    if config.pkg_type.lower() == "rpm":
        release = config.version_suffix or "1"
        return f"{config.rocm_version}-{release}"
    version = str(config.rocm_version)
    if config.version_suffix:
        version += f"-{config.version_suffix}"
    return version


def versions_match(expected: str, actual: str, pkg_type: str) -> bool:
    """Compare expected and actual control-field version strings.

    DEB allows ``~`` as an alternative separator to ``-`` in version revisions;
    RPM requires an exact match.
    """
    if expected == actual:
        return True
    if pkg_type.lower() == "deb":
        return expected.replace("-", "~") == actual.replace("-", "~")
    return False


def normalize_dep_tokens(depends: str, *, pkg_type: str) -> frozenset[str]:
    """Normalize a Depends/Requires string into comparable dependency tokens.

    Splits on commas; strips versions in parentheses and whitespace. For RPM,
    drops ``rpmlib(...)`` and ``*.so*`` automatic requires.
    """
    if not depends or not depends.strip():
        return frozenset()
    tokens: set[str] = set()
    for part in depends.replace("\n", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if pkg_type.lower() == "rpm" and _RPM_AUTO_REQUIRE_RE.match(token):
            continue
        # Drop version constraints: ``pkg (>= 1.0)`` → ``pkg``.
        token = re.split(r"\s*\(", token, maxsplit=1)[0].strip()
        if not token:
            continue
        tokens.add(token)
    return frozenset(tokens)


def deps_match(expected: str, actual: str, pkg_type: str) -> bool:
    """Return True when every expected dependency token appears in actual.

    Actual may contain extra tokens (e.g. RPM automatic requires that survive
    filtering); expected must be a subset of actual.
    """
    return normalize_dep_tokens(expected, pkg_type=pkg_type) <= normalize_dep_tokens(
        actual, pkg_type=pkg_type
    )


def read_package_control_fields(package_path: Path, pkg_type: str) -> dict[str, str]:
    """Read Package/Name, Version, Depends/Requires, and Architecture from a package.

    Parameters:
        package_path: Path to a built ``.deb`` or ``.rpm``.
        pkg_type: ``deb`` or ``rpm`` (case-insensitive).

    Returns:
        Dict with keys ``package_name``, ``version``, ``depends``, ``architecture``.

    Raises:
        RuntimeError: When ``dpkg-deb`` / ``rpm`` is missing or the query fails.
    """
    pkg_type = pkg_type.lower()
    if pkg_type == "deb":
        fields = {}
        for key, control_key in (
            ("package_name", "Package"),
            ("version", "Version"),
            ("depends", "Depends"),
            ("architecture", "Architecture"),
        ):
            result = _run_capture(
                ["dpkg-deb", "-f", str(package_path), control_key],
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"dpkg-deb failed for {package_path} ({control_key}): "
                    f"{result.stderr.strip()}",
                )
            fields[key] = result.stdout.strip()
        return fields

    name_r = _run_capture(
        ["rpm", "-qp", "--qf", r"%{NAME}", str(package_path)],
    )
    ver_r = _run_capture(
        ["rpm", "-qp", "--qf", r"%{VERSION}-%{RELEASE}", str(package_path)],
    )
    arch_r = _run_capture(
        ["rpm", "-qp", "--qf", r"%{ARCH}", str(package_path)],
    )
    req_r = _run_capture(
        ["rpm", "-qp", "--requires", str(package_path)],
    )
    for label, result in (
        ("NAME", name_r),
        ("VERSION-RELEASE", ver_r),
        ("ARCH", arch_r),
        ("requires", req_r),
    ):
        if result.returncode != 0:
            raise RuntimeError(
                f"rpm query failed for {package_path} ({label}): "
                f"{result.stderr.strip()}",
            )
    # ``--requires`` prints one capability per line.
    depends = ", ".join(
        line.strip() for line in req_r.stdout.splitlines() if line.strip()
    )
    return {
        "package_name": name_r.stdout.strip(),
        "version": ver_r.stdout.strip(),
        "depends": depends,
        "architecture": arch_r.stdout.strip(),
    }


def _expected_depends(pkg_info: dict, config: PackageConfig) -> str:
    """Resolve main Depends/Requires via packaging_utils (same as build)."""
    if config.pkg_type.lower() == "rpm":
        field_key = "RPMRequires"
    else:
        field_key = "DEBDepends"
    if config.versioned_pkg:
        return process_versioned_dependencies(pkg_info, field_key, config) or ""
    return process_nonversioned_dependencies(pkg_info, config) or ""


def _expected_architecture(pkg_info: dict, pkg_type: str) -> str:
    """Architecture string expected in package metadata."""
    if pkg_type.lower() == "rpm":
        return (pkg_info.get("BuildArch") or pkg_info.get("Architecture") or "").strip()
    return (pkg_info.get("Architecture") or "").strip()


def _variant_specs_for_package(
    pkg_name: str,
    config: PackageConfig,
) -> list[tuple[bool, str]]:
    """Return ``(versioned_pkg, gfx_arch)`` variants matching ``build_package``.

    Enumeration only — does not invoke package builders.
    """
    pkg_info = get_package_info(pkg_name)
    specs: list[tuple[bool, str]] = []
    if config.enable_kpack:
        if is_gfxarch_package(pkg_info, config.enable_kpack, config.artifacts_dir):
            if not is_meta_package(pkg_info):
                specs.append((True, GFX_HOST))
            for device_arch in group_package_targets(config.gfxarch_list):
                specs.append((True, device_arch))
            specs.append((True, GFX_META))
            specs.append((False, GFX_META))
        else:
            specs.append((True, ""))
            specs.append((False, ""))
    else:
        specs.append((True, config.gfx_arch))
        specs.append((False, config.gfx_arch))
    return specs


def build_expected_control_index(config: PackageConfig) -> dict[str, ExpectedControl]:
    """Map installed package name → expected control fields for all eligible bases.

    Uses the same eligibility list as ``build_package.py`` (``get_package_list``)
    and variant shapes as ``build_package_variants``, with
    ``group_package_targets`` for device owners.
    """
    pkg_list, _skipped = get_package_list(config.artifacts_dir)
    index: dict[str, ExpectedControl] = {}
    for base_pkg in pkg_list:
        pkg_info = get_package_info(base_pkg)
        arch = _expected_architecture(pkg_info, config.pkg_type)
        for versioned_pkg, gfx_arch in _variant_specs_for_package(base_pkg, config):
            local = replace(
                config,
                versioned_pkg=versioned_pkg,
                gfx_arch=gfx_arch,
            )
            installed = update_package_name(base_pkg, local)
            index[installed] = ExpectedControl(
                package_name=installed,
                version=expected_control_version(local),
                depends=_expected_depends(pkg_info, local),
                architecture=arch,
                base_package=base_pkg,
                versioned_pkg=versioned_pkg,
                gfx_arch=gfx_arch,
            )
    return index


def verify_control_fields(
    packages_dir: Path,
    filenames: list[str],
    pkg_type: str,
    expected_index: dict[str, ExpectedControl],
    result: ManifestVerifyResult,
) -> None:
    """Compare control fields for present package files; mutate ``result``."""
    packages_dir = packages_dir.resolve()
    for filename in filenames:
        path = _resolve_package_path(packages_dir, filename)
        try:
            actual = read_package_control_fields(path, pkg_type)
        except (RuntimeError, ValueError) as exc:
            result.read_errors.append(f"{filename}: {exc}")
            continue

        package_name = actual["package_name"]
        expected = expected_index.get(package_name)
        if expected is None:
            result.unknown_packages.append(f"{filename} ({package_name})")
            continue

        if not versions_match(expected.version, actual["version"], pkg_type):
            result.field_mismatches.append(
                FieldMismatch(
                    filename=filename,
                    package_name=package_name,
                    field="Version",
                    expected=expected.version,
                    actual=actual["version"],
                )
            )
        if not deps_match(expected.depends, actual["depends"], pkg_type):
            result.field_mismatches.append(
                FieldMismatch(
                    filename=filename,
                    package_name=package_name,
                    field="Depends" if pkg_type.lower() == "deb" else "Requires",
                    expected=expected.depends,
                    actual=actual["depends"],
                )
            )
        if expected.architecture and actual["architecture"] != expected.architecture:
            result.field_mismatches.append(
                FieldMismatch(
                    filename=filename,
                    package_name=package_name,
                    field="Architecture",
                    expected=expected.architecture,
                    actual=actual["architecture"],
                )
            )


def format_report_text(result: ManifestVerifyResult, manifest_path: Path) -> str:
    """Format a human-readable verification summary for stdout."""
    overall = "PASS" if result.passed else "FAIL"
    lines = [
        "ROCm build package manifest verification",
        "=" * 72,
        f"Overall result: {overall}",
        f"Manifest: {manifest_path}",
        f"Packages expected: {len(result.expected)}",
        f"Packages found: {len(result.found)}",
        f"Packages missing: {len(result.missing)}",
        f"Field mismatches: {len(result.field_mismatches)}",
        f"Unknown packages: {len(result.unknown_packages)}",
        f"Read errors: {len(result.read_errors)}",
    ]
    if result.missing:
        lines.append("")
        lines.append("Missing packages:")
        for name in result.missing:
            lines.append(f"  - {name}")
    if result.unknown_packages:
        lines.append("")
        lines.append("Unknown packages (not in expected control index):")
        for name in result.unknown_packages:
            lines.append(f"  - {name}")
    if result.field_mismatches:
        lines.append("")
        lines.append("Control field mismatches:")
        for mismatch in result.field_mismatches:
            lines.append(
                f"  - {mismatch.filename} [{mismatch.package_name}] "
                f"{mismatch.field}: expected={mismatch.expected!r} "
                f"actual={mismatch.actual!r}"
            )
    if result.read_errors:
        lines.append("")
        lines.append("Metadata read errors:")
        for err in result.read_errors:
            lines.append(f"  - {err}")
    return "\n".join(lines)


def _build_package_config(args: argparse.Namespace) -> PackageConfig:
    """Create PackageConfig via ``build_package.create_package_config``."""
    from build_package import create_package_config

    # Reuse build CLI mapping; dest_dir is unused for verify expectations.
    ns = argparse.Namespace(
        dest_dir=args.packages_dir,
        artifacts_dir=args.artifacts_dir,
        target=args.target,
        enable_kpack=args.enable_kpack,
        rocm_version=args.rocm_version,
        version_suffix=args.version_suffix or "",
        install_prefix=args.install_prefix,
        build_variant=args.build_variant or "",
        pkg_type=args.pkg_type,
    )
    return create_package_config(ns)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters:
        argv: Argument list (typically ``sys.argv[1:]``).

    Returns:
        Parsed namespace for ``run()``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Verify Created Packages from built_packages.txt exist under "
            "--packages-dir, then compare Package/Version/Depends/Architecture "
            "to packaging_utils expectations."
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
        help="Package format to verify (required for control-field checks)",
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
        required=True,
        help="Artifacts directory used to build the expected control index",
    )
    parser.add_argument(
        "--rocm-version",
        type=str,
        required=True,
        help="ROCm release version (must match build_package.py)",
    )
    parser.add_argument(
        "--version-suffix",
        type=str,
        default="",
        help="Version suffix / CI run ID (must match build_package.py)",
    )
    parser.add_argument(
        "--build-variant",
        type=str,
        default="",
        help="Build variant (e.g. asan); must match build_package.py",
    )
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        required=False,
        help="GFX target list (default: auto-detect from artifacts-dir)",
    )
    parser.add_argument(
        "--enable-kpack",
        action="store_true",
        help="Force kpack mode (default: auto-detect from therock_manifest.json)",
    )
    parser.add_argument(
        "--install-prefix",
        type=str,
        default="/opt/rocm/core",
        help="Install prefix passed through to PackageConfig",
    )
    parser.add_argument(
        "--skip-control-fields",
        action="store_true",
        help="Only check file presence; skip Package/Version/Depends/Arch checks",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Execute manifest presence + optional control-field verification.

    Parameters:
        args: Parsed command-line arguments.

    Returns:
        0 on success, 1 when files/fields fail, 2 on configuration errors.
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
        logger.error(
            f"no package filenames found in manifest: {manifest_path}",
        )
        return 2

    try:
        result = verify_manifest_packages(packages_dir, package_names)
    except ValueError as exc:
        logger.error(f"{exc}")
        return 2

    if result.missing:
        print(format_report_text(result, manifest_path))
        logger.error(
            f"Manifest verification failed: {len(result.missing)} missing file(s)",
        )
        return 1

    if not args.skip_control_fields:
        artifacts_dir = args.artifacts_dir.expanduser().resolve()
        if not artifacts_dir.is_dir():
            logger.error(f"artifacts directory not found: {artifacts_dir}")
            return 2
        try:
            config = _build_package_config(args)
            expected_index = build_expected_control_index(config)
        except Exception as exc:
            logger.error(f"failed to build expected control index: {exc}")
            return 2

        try:
            verify_control_fields(
                packages_dir,
                result.found,
                pkg_type,
                expected_index,
                result,
            )
        except ValueError as exc:
            logger.error(f"{exc}")
            return 2

    print(format_report_text(result, manifest_path))

    if not result.passed:
        logger.error(
            "Manifest verification failed: "
            f"missing={len(result.missing)} "
            f"field_mismatches={len(result.field_mismatches)} "
            f"unknown={len(result.unknown_packages)} "
            f"read_errors={len(result.read_errors)}",
        )
        return 1

    logger.info("Manifest verification passed.")
    return 0


def main(argv: list[str]) -> int:
    """Program entry point: parse arguments, configure logging, and run verify.

    Parameters:
        argv: Argument list (typically ``sys.argv[1:]``).

    Returns:
        Process exit code from ``run()``.
    """
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
