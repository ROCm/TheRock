#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify built native Linux packages against ``package.json``.

Runs after ``build_package.py`` and before simulated install or upload. Inspects
already-built ``.deb``/``.rpm`` files only; it does not build or modify packages.
For each requested entry, derives expected variant names from ``package.json`` and
the same CLI flags used at build time, then confirms files exist in
``--packages-dir`` and optionally checks control-field versions against
``--rocm-version`` and ``--version-suffix``.

Writes ``build_status_report.txt`` and ``build_status_report.json`` when
``--report-dir`` is set.

```
# Standard CI pre-upload verification (deb):
./build_tools/packaging/linux/build_package_verify.py \\
    --pkg-type deb \\
    --packages-dir output/packages \\
    --artifacts-dir output/artifacts \\
    --dest-dir output/packages \\
    --rocm-version 7.15.0 \\
    --version-suffix 28484694006 \\
    --pkg-names amdrocm-core-sdk \\
    --report-dir output/pre_upload_reports
```

``--version-suffix``: CI run ID appended to the DEB/RPM version field
  (e.g. ``7.15.0-28484694006`` for DEB, release ``28484694006`` for RPM).
  Does not affect the package name or install prefix.

``--build-variant``: Build type that modifies both the package name and
  install prefix. Currently supports ``asan``.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from build_package import (
    DEFAULT_INSTALL_PREFIX,
    create_package_config,
    parse_input_package_list,
)
from packaging_utils import (
    GFX_HOST,
    GFX_META,
    PackageConfig,
    get_package_info,
    is_gfxarch_package,
    is_meta_package,
    read_package_json_file,
    update_package_name,
)
from _therock_utils.log_utils import TheRockLogger, configure_logging

logger = TheRockLogger(__name__)

_CLI_EXAMPLES_EPILOG = """
Examples:
 # CI pre-upload verification (matches multi_arch_build_native_linux_packages.yml)
 ./build_tools/packaging/linux/build_package_verify.py \\
   --pkg-type deb \\
   --packages-dir output/packages \\
   --artifacts-dir output/artifacts \\
   --dest-dir output/packages \\
   --rocm-version 7.15.0 \\
   --version-suffix 28484694006 \\
   --pkg-names amdrocm-core-sdk \\
   --report-dir output/pre_upload_reports

 # Kpack multi-arch build with explicit GPU targets
 ./build_tools/packaging/linux/build_package_verify.py \\
   --pkg-type deb \\
   --packages-dir output/packages \\
   --artifacts-dir output/artifacts \\
   --dest-dir output/packages \\
   --rocm-version 7.15.0 \\
   --version-suffix 28484694006 \\
   --enable-kpack \\
   --target gfx1100 gfx942 \\
   --pkg-names amdrocm-core-sdk \\
   --report-dir output/pre_upload_reports

 # ASAN build variant (package name and install prefix must match build_package.py)
 ./build_tools/packaging/linux/build_package_verify.py \\
   --pkg-type deb \\
   --packages-dir output/packages \\
   --artifacts-dir output/artifacts \\
   --dest-dir output/packages \\
   --rocm-version 7.15.0 \\
   --version-suffix 28484694006 \\
   --build-variant asan \\
   --pkg-names amdrocm-core-sdk \\
   --report-dir output/pre_upload_reports

 # RPM verification with unexpected-file detection
 ./build_tools/packaging/linux/build_package_verify.py \\
   --pkg-type rpm \\
   --packages-dir output/packages \\
   --artifacts-dir output/artifacts \\
   --dest-dir output/packages \\
   --rocm-version 7.15.0 \\
   --version-suffix 28484694006 \\
   --pkg-names amdrocm-core-sdk \\
   --fail-on-extra \\
   --report-dir output/pre_upload_reports

 # Presence-only check (skip control-field version comparison)
 ./build_tools/packaging/linux/build_package_verify.py \\
   --pkg-type deb \\
   --packages-dir output/packages \\
   --artifacts-dir output/artifacts \\
   --rocm-version 7.15.0 \\
   --pkg-names amdrocm-core-sdk \\
   --no-version-check
"""


@dataclass
class VariantBuildCheck:
    """Verification result for one package variant."""

    base_package: str
    label: str
    expected_name: str
    file_path: Path | None
    found: bool
    expected_version: str
    actual_version: str | None
    version_ok: bool
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when the variant file exists and version checks succeed."""
        return self.found and self.version_ok and not self.errors


@dataclass
class BuildVerifyReport:
    """Verification results for all variants of one ``package.json`` entry."""

    base_package: str
    variants: list[VariantBuildCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every variant in this report passed."""
        return all(v.passed for v in self.variants)


@dataclass
class BuildVerifySummary:
    """Aggregate verification outcome across all requested packages."""

    packages_requested: list[str]
    reports: list[BuildVerifyReport]
    package_files_found: list[str]
    extra_package_files: list[str] = field(default_factory=list)

    @property
    def variants_expected(self) -> int:
        """Total variant count across all package reports."""
        return sum(len(r.variants) for r in self.reports)

    @property
    def variants_found(self) -> int:
        """Variants whose expected package file exists on disk."""
        return sum(1 for r in self.reports for v in r.variants if v.found)

    @property
    def variants_passed(self) -> int:
        """Variants that passed presence and version checks."""
        return sum(1 for r in self.reports for v in r.variants if v.passed)

    @property
    def variants_failed(self) -> int:
        """Variants that failed or were not found."""
        return self.variants_expected - self.variants_passed

    @property
    def passed(self) -> bool:
        """True when no variant failed and no unexpected package files were flagged."""
        if self.extra_package_files:
            return False
        return self.variants_failed == 0 and all(r.passed for r in self.reports)

    def missing_variants(self) -> list[str]:
        """Expected installed package names with no matching file on disk."""
        return [
            v.expected_name for r in self.reports for v in r.variants if not v.found
        ]

    def version_failures(self) -> list[str]:
        """Expected installed package names whose control-field version mismatched."""
        return [
            v.expected_name
            for r in self.reports
            for v in r.variants
            if v.found and not v.version_ok
        ]


@dataclass(frozen=True)
class PackageVariantSpec:
    """One expected package variant derived from ``package.json`` and build flags.

    Attributes:
        label: Human-readable variant name (e.g. ``host``, ``meta``, ``device-gfx1100``).
        versioned_pkg: Whether the ROCm version appears in the installed package name.
        gfx_arch: Gfx-arch token for kpack routing (``GFX_HOST``, ``GFX_META``, or device arch).
    """

    label: str
    versioned_pkg: bool
    gfx_arch: str


def iter_package_variant_specs(
    pkg_name: str,
    config: PackageConfig,
) -> list[PackageVariantSpec]:
    """Enumerate expected variants using the same routing as ``build_package_variants``.

    Read-only helper for verification: mirrors how ``build_package.py`` splits a
    ``package.json`` entry into variant names without building anything.

    Parameters:
        pkg_name: ``package.json`` base name.
        config: Build configuration (kpack mode, gfx targets, etc.).

    Returns:
        Ordered list of variant specifications to check on disk.
    """
    pkg_info = get_package_info(pkg_name)
    specs: list[PackageVariantSpec] = []

    if config.enable_kpack:
        if is_gfxarch_package(pkg_info, config.enable_kpack, config.artifacts_dir):
            if not is_meta_package(pkg_info):
                specs.append(
                    PackageVariantSpec(
                        label="host",
                        versioned_pkg=True,
                        gfx_arch=GFX_HOST,
                    )
                )
            for device_arch in config.gfxarch_list:
                specs.append(
                    PackageVariantSpec(
                        label=f"device-{device_arch}",
                        versioned_pkg=True,
                        gfx_arch=device_arch,
                    )
                )
            specs.append(
                PackageVariantSpec(
                    label="meta",
                    versioned_pkg=True,
                    gfx_arch=GFX_META,
                )
            )
            specs.append(
                PackageVariantSpec(
                    label="non-versioned",
                    versioned_pkg=False,
                    gfx_arch=GFX_META,
                )
            )
        else:
            specs.append(
                PackageVariantSpec(
                    label="versioned",
                    versioned_pkg=True,
                    gfx_arch="",
                )
            )
            specs.append(
                PackageVariantSpec(
                    label="non-versioned",
                    versioned_pkg=False,
                    gfx_arch="",
                )
            )
    else:
        specs.append(
            PackageVariantSpec(
                label="versioned",
                versioned_pkg=True,
                gfx_arch=config.gfx_arch,
            )
        )
        specs.append(
            PackageVariantSpec(
                label="non-versioned",
                versioned_pkg=False,
                gfx_arch=config.gfx_arch,
            )
        )

    return specs


def _run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture stdout/stderr without raising on failure."""
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def resolve_installed_name(
    pkg_name: str,
    config: PackageConfig,
    *,
    versioned_pkg: bool,
    gfx_arch: str,
) -> str:
    """Compute the on-disk package stem for one variant.

    Applies the same ``update_package_name`` rules as ``build_package.py`` for
    versioned vs non-versioned packages and gfx-arch suffixes.

    Parameters:
        pkg_name: Base name from ``package.json``.
        config: Shared build configuration.
        versioned_pkg: Whether this variant includes the ROCm version in its name.
        gfx_arch: Gfx-arch token for kpack variants (``GFX_HOST``, ``GFX_META``, etc.).

    Returns:
        Package stem without ``.deb``/``.rpm`` extension (e.g. ``amdrocm-core-sdk7.15``).
    """
    local_config = replace(
        config,
        versioned_pkg=versioned_pkg,
        gfx_arch=gfx_arch,
    )
    return update_package_name(pkg_name, local_config)


def find_package_files(packages_dir: Path, pkg_type: str) -> dict[str, Path]:
    """Index built package files in ``packages_dir`` by installed package stem.

    Parameters:
        packages_dir: Directory containing built ``.deb`` or ``.rpm`` files.
        pkg_type: ``deb`` or ``rpm`` (case-insensitive).

    Returns:
        Mapping from package stem (filename without extension) to file path.
    """
    ext = ".deb" if pkg_type.lower() == "deb" else ".rpm"
    package_files: dict[str, Path] = {}
    for path in sorted(packages_dir.iterdir()):
        if not path.is_file():
            continue
        if not path.name.lower().endswith(ext):
            continue
        # Stem matches the installed package name used by dpkg/rpm metadata.
        package_files[path.name[: -len(ext)]] = path
    return package_files


def expected_control_version(config: PackageConfig) -> str:
    """Build the version string expected in DEB/RPM control metadata.

    DEB uses ``rocm_version`` plus optional ``version_suffix`` as the debian
    revision. RPM combines ``VERSION-RELEASE`` where release defaults to ``1``.

    Parameters:
        config: Shared build configuration.

    Returns:
        Expected version string for comparison with ``dpkg-deb -f`` or ``rpm -qp``.
    """
    if config.pkg_type.lower() == "rpm":
        release = config.version_suffix or "1"
        return f"{config.rocm_version}-{release}"
    version = str(config.rocm_version)
    if config.version_suffix:
        version += f"-{config.version_suffix}"
    return version


def read_package_file_version(package_path: Path, pkg_type: str) -> str:
    """Read the version field from a built package file.

    Parameters:
        package_path: Path to a ``.deb`` or ``.rpm`` file.
        pkg_type: ``deb`` or ``rpm`` (case-insensitive).

    Returns:
        Version string from package metadata.

    Raises:
        RuntimeError: When ``dpkg-deb`` or ``rpm`` query fails.
    """
    pkg_type = pkg_type.lower()
    if pkg_type == "deb":
        result = _run_capture(
            ["dpkg-deb", "-f", str(package_path), "Version"],
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"dpkg-deb failed for {package_path}: {result.stderr.strip()}",
            )
        return result.stdout.strip()
    result = _run_capture(
        ["rpm", "-qp", "--qf", r"%{VERSION}-%{RELEASE}", str(package_path)],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rpm query failed for {package_path}: {result.stderr.strip()}",
        )
    return result.stdout.strip()


def versions_match(expected: str, actual: str, pkg_type: str) -> bool:
    """Compare expected and actual control-field version strings.

    DEB allows ``~`` as an alternative separator to ``-`` in version revisions;
    RPM requires an exact match.

    Parameters:
        expected: Version derived from ``--rocm-version`` and ``--version-suffix``.
        actual: Version read from the built package file.
        pkg_type: ``deb`` or ``rpm`` (case-insensitive).

    Returns:
        True when the versions are equivalent for the given package format.
    """
    if expected == actual:
        return True
    if pkg_type.lower() == "deb":
        # Debian revision may use ~ where we encoded - in the expected string.
        normalized_expected = expected.replace("-", "~")
        normalized_actual = actual.replace("-", "~")
        return normalized_expected == normalized_actual
    return False


def verify_variant(
    base_package: str,
    label: str,
    expected_name: str,
    package_files: dict[str, Path],
    config: PackageConfig,
    *,
    check_version: bool,
) -> VariantBuildCheck:
    """Verify one variant: file presence and optional control-field version.

    Parameters:
        base_package: ``package.json`` base name.
        label: Human-readable variant label (e.g. ``meta``, ``device-gfx1100``).
        expected_name: Installed package stem expected on disk.
        package_files: Index from ``find_package_files``.
        config: Shared build configuration.
        check_version: When False, skip ``dpkg-deb``/``rpm`` version queries.

    Returns:
        Per-variant check result with error details on failure.
    """
    expected_version = expected_control_version(config)
    path = package_files.get(expected_name)
    found = path is not None
    actual_version: str | None = None
    version_ok = not check_version
    errors: list[str] = []

    if not found:
        errors.append(f"package file not found for expected name {expected_name!r}")
    elif check_version:
        try:
            actual_version = read_package_file_version(path, config.pkg_type)
            version_ok = versions_match(
                expected_version,
                actual_version,
                config.pkg_type,
            )
            if not version_ok:
                errors.append(
                    f"version mismatch: expected {expected_version!r}, "
                    f"got {actual_version!r}",
                )
        except RuntimeError as exc:
            version_ok = False
            errors.append(str(exc))

    return VariantBuildCheck(
        base_package=base_package,
        label=label,
        expected_name=expected_name,
        file_path=path,
        found=found,
        expected_version=expected_version,
        actual_version=actual_version,
        version_ok=version_ok,
        errors=errors,
    )


def verify_package(
    pkg_name: str,
    config: PackageConfig,
    packages_dir: Path,
    *,
    check_version: bool,
) -> BuildVerifyReport:
    """Verify all variants ``build_package.py`` would produce for one package.

    Parameters:
        pkg_name: ``package.json`` base name.
        config: Shared build configuration (kpack routing, gfx targets, etc.).
        packages_dir: Directory containing built package files.
        check_version: When False, verify presence only.

    Returns:
        Report with one ``VariantBuildCheck`` per expected variant.
    """
    report = BuildVerifyReport(base_package=pkg_name)
    package_files = find_package_files(packages_dir, config.pkg_type)

    for spec in iter_package_variant_specs(pkg_name, config):
        expected_name = resolve_installed_name(
            pkg_name,
            config,
            versioned_pkg=spec.versioned_pkg,
            gfx_arch=spec.gfx_arch,
        )
        report.variants.append(
            verify_variant(
                pkg_name,
                spec.label,
                expected_name,
                package_files,
                config,
                check_version=check_version,
            ),
        )
    return report


def collect_extra_package_files(
    package_files: dict[str, Path],
    expected_names: set[str],
) -> list[str]:
    """List package stems present on disk but not expected by any variant.

    Parameters:
        package_files: Index from ``find_package_files``.
        expected_names: Installed package stems derived from variant enumeration.

    Returns:
        Sorted list of unexpected package stems.
    """
    return sorted(name for name in package_files if name not in expected_names)


def resolve_pkg_names(
    args: argparse.Namespace,
    config: PackageConfig,
) -> list[str]:
    """Resolve the package list from CLI flags.

    Parameters:
        args: Parsed command-line arguments.
        config: Shared build configuration (used for ``--all-eligible`` filtering).

    Returns:
        ``package.json`` base names to verify.

    Raises:
        ValueError: Propagated from ``parse_input_package_list`` on invalid input.
    """
    if args.all_eligible:
        pkg_list, _skipped = parse_input_package_list(None, config.artifacts_dir)
        return pkg_list
    return list(args.pkg_names)


def build_summary(
    packages_requested: list[str],
    reports: list[BuildVerifyReport],
    package_files: dict[str, Path],
    *,
    fail_on_extra: bool,
) -> BuildVerifySummary:
    """Assemble the aggregate verification summary.

    Parameters:
        packages_requested: Base names passed on the command line.
        reports: Per-package verification reports.
        package_files: Full index of files in ``--packages-dir``.
        fail_on_extra: When False, unexpected files are ignored for pass/fail.

    Returns:
        Summary used for console output and report files.
    """
    expected_names = {v.expected_name for r in reports for v in r.variants}
    extra = collect_extra_package_files(package_files, expected_names)
    if not fail_on_extra:
        extra = []
    return BuildVerifySummary(
        packages_requested=packages_requested,
        reports=reports,
        package_files_found=sorted(package_files.keys()),
        extra_package_files=extra,
    )


def _variant_to_dict(variant: VariantBuildCheck) -> dict[str, object]:
    """Serialize one variant check for JSON report output."""
    return {
        "base_package": variant.base_package,
        "label": variant.label,
        "expected_name": variant.expected_name,
        "found": variant.found,
        "file_path": str(variant.file_path) if variant.file_path else None,
        "expected_version": variant.expected_version,
        "actual_version": variant.actual_version,
        "version_ok": variant.version_ok,
        "passed": variant.passed,
        "errors": variant.errors,
    }


def format_report_json(summary: BuildVerifySummary) -> str:
    """Format the verification summary as indented JSON.

    Parameters:
        summary: Aggregate verification outcome.

    Returns:
        JSON string suitable for ``build_status_report.json``.
    """
    payload = {
        "passed": summary.passed,
        "packages_requested": summary.packages_requested,
        "variants_expected": summary.variants_expected,
        "variants_found": summary.variants_found,
        "variants_passed": summary.variants_passed,
        "variants_failed": summary.variants_failed,
        "missing_variants": summary.missing_variants(),
        "version_failures": summary.version_failures(),
        "package_files_found": summary.package_files_found,
        "extra_package_files": summary.extra_package_files,
        "reports": [
            {
                "base_package": report.base_package,
                "passed": report.passed,
                "variants": [_variant_to_dict(v) for v in report.variants],
            }
            for report in summary.reports
        ],
    }
    return json.dumps(payload, indent=2)


def format_report_text(summary: BuildVerifySummary) -> str:
    """Format the verification summary as human-readable text.

    Parameters:
        summary: Aggregate verification outcome.

    Returns:
        Multi-line report for console output and ``build_status_report.txt``.
    """
    lines: list[str] = []
    overall = "PASS" if summary.passed else "FAIL"
    lines.append("ROCm build package verification report")
    lines.append("=" * 72)
    lines.append(f"Overall result: {overall}")
    lines.append(f"Packages requested: {', '.join(summary.packages_requested)}")
    lines.append(
        f"Variants: {summary.variants_expected} expected, "
        f"{summary.variants_found} found, "
        f"{summary.variants_passed} passed, "
        f"{summary.variants_failed} failed",
    )
    lines.append(f"Package files on disk: {len(summary.package_files_found)}")
    lines.append("")

    for report in summary.reports:
        lines.append(f"Package (package.json): {report.base_package}")
        for variant in report.variants:
            status = "PASS" if variant.passed else "FAIL"
            found = "yes" if variant.found else "no"
            lines.append(f"  [{status}] {variant.label}")
            lines.append(f"         expected name: {variant.expected_name}")
            lines.append(f"         file found: {found}")
            if variant.file_path:
                lines.append(f"         path: {variant.file_path}")
            if variant.actual_version is not None:
                lines.append(
                    f"         version: {variant.actual_version} "
                    f"(expected {variant.expected_version})",
                )
            for err in variant.errors:
                lines.append(f"         error: {err}")
        lines.append("")

    if summary.missing_variants():
        lines.append(f"Missing variants: {summary.missing_variants()}")
    if summary.version_failures():
        lines.append(f"Version failures: {summary.version_failures()}")
    if summary.extra_package_files:
        lines.append(f"Unexpected extra packages: {summary.extra_package_files}")
    return "\n".join(lines)


def write_report_files(summary: BuildVerifySummary, report_dir: Path) -> None:
    """Write text and JSON verification reports under ``report_dir``.

    Parameters:
        summary: Aggregate verification outcome.
        report_dir: Output directory (created if missing).

    Raises:
        FileNotFoundError: When either report file is missing after write.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    text_path = report_dir / "build_status_report.txt"
    json_path = report_dir / "build_status_report.json"
    text_path.write_text(format_report_text(summary) + "\n", encoding="utf-8")
    json_path.write_text(format_report_json(summary) + "\n", encoding="utf-8")
    if not text_path.is_file():
        raise FileNotFoundError(f"Failed to write report: {text_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Failed to write report: {json_path}")
    logger.info(f"Build report written to: {text_path}")
    logger.info(f"Build report written to: {json_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters:
        argv: Argument list (typically ``sys.argv[1:]``).

    Returns:
        Parsed namespace for ``run()``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Verify built native Linux packages match package.json variant names "
            "and control-field versions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EXAMPLES_EPILOG,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level logging)",
    )
    parser.add_argument(
        "--pkg-type",
        required=True,
        choices=("deb", "rpm", "DEB", "RPM"),
        help="Choose the package format to be verified: DEB or RPM",
    )
    parser.add_argument(
        "--packages-dir",
        type=Path,
        required=True,
        help="Directory containing built .deb or .rpm files",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Artifact tree for kpack/gfx-arch detection",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("/tmp/rocm-build-verify"),
        help="Placeholder dest dir for PackageConfig (matches build_package)",
    )
    parser.add_argument(
        "--rocm-version",
        required=True,
        help="ROCm release version used when packages were built",
    )
    parser.add_argument(
        "--version-suffix",
        type=str,
        nargs="?",
        default="",
        help=(
            "Release identifier appended to the package version field in DEB/RPM "
            "metadata (e.g. a CI run ID like '28484694006'). "
            "For DEB this becomes the debian revision (e.g. '7.15.0-28484694006'); "
            "for RPM this sets the release field. "
            "Does not affect the package name or install prefix."
        ),
    )
    parser.add_argument(
        "--install-prefix",
        default=DEFAULT_INSTALL_PREFIX,
        help="Base directory where package will be installed",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=[],
        help="GPU targets for kpack (auto-detected from artifacts if omitted)",
    )
    parser.add_argument(
        "--enable-kpack",
        action="store_true",
        help="Enable multi-architecture package generation",
    )
    parser.add_argument(
        "--runpath-pkg",
        action="store_true",
        help="Keep RUNPATH in binaries (by default, RUNPATH is converted to RPATH)",
    )
    parser.add_argument(
        "--build-variant",
        default="",
        help=(
            "Build variant (e.g. 'asan'). When set to 'asan', the install prefix "
            "becomes DEFAULT_INSTALL_PREFIX-asan-MAJOR.MINOR "
            "(e.g. /opt/rocm/core-asan-7.15)."
        ),
    )
    parser.add_argument(
        "--pkg-names",
        nargs="+",
        default=["amdrocm-core-sdk"],
        help="package.json base names to verify (ignored if --all-eligible)",
    )
    parser.add_argument(
        "--all-eligible",
        action="store_true",
        help="Verify every package.json entry eligible for the artifact dir",
    )
    parser.add_argument(
        "--no-version-check",
        action="store_true",
        help="Skip control-field version verification (presence only)",
    )
    parser.add_argument(
        "--fail-on-extra",
        action="store_true",
        help="Fail when unexpected package files exist in packages-dir",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        metavar="DIR",
        help="Write build_status_report.txt and .json under DIR",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Execute build package verification.

    Parameters:
        args: Parsed command-line arguments.

    Returns:
        0 on success, 1 when verification failed, 2 on usage or configuration errors.
    """
    packages_dir = args.packages_dir.expanduser().resolve()
    if not packages_dir.is_dir():
        logger.error(f"packages directory not found: {packages_dir}")
        return 2

    read_package_json_file()
    try:
        config = create_package_config(args)
    except ValueError as exc:
        logger.error(f"{exc}")
        return 2

    try:
        pkg_names = resolve_pkg_names(args, config)
    except ValueError as exc:
        logger.error(f"{exc}")
        return 2

    if not pkg_names:
        logger.error("no packages to verify")
        return 2

    check_version = not args.no_version_check
    reports = [
        verify_package(name, config, packages_dir, check_version=check_version)
        for name in pkg_names
    ]
    package_files = find_package_files(packages_dir, config.pkg_type)
    summary = build_summary(
        pkg_names,
        reports,
        package_files,
        fail_on_extra=args.fail_on_extra,
    )

    print(format_report_text(summary))

    if args.report_dir is not None:
        write_report_files(summary, args.report_dir.expanduser())

    if not summary.passed:
        logger.error(
            f"Build verification failed: {summary.variants_failed} variant(s), "
            f"{len(summary.extra_package_files)} extra file(s)",
        )
        return 1
    logger.info("Build verification passed.")
    return 0


def main(argv: list[str]) -> int:
    """Program entry point: parse arguments, configure logging, and run verification."""
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
