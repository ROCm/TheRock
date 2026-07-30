#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pre-upload Step 1 — verify built packages match ``package.json`` expectations.

After ``build_package.py``, confirms that generated native Linux packages exist on
disk with the expected variant names and control-field versions. This is the
**build verification** gate from the pre-upload milestone (Step 1); dependency
field checks are handled separately by ``pkg_dependency_checker.py`` (Step 2).

Checks per ``package.json`` entry (and each kpack variant):

* **Presence** — expected ``.deb`` / ``.rpm`` file exists under ``--packages-dir``.
* **Version** — DEB ``Version`` / RPM ``VERSION-RELEASE`` matches ``--rocm-version``
  and ``--version-suffix`` (same rules as ``deb_package.py`` / ``rpm_package.py``).
* **Inventory** — optional fail on unexpected extra package files in the output dir.

Example (CI, after build)::

    python3 build_package_verify.py \\
        --pkg-type deb \\
        --packages-dir output/packages \\
        --artifacts-dir output/artifacts \\
        --dest-dir output/packages \\
        --rocm-version 7.14.0 \\
        --version-suffix "${ARTIFACT_RUN_ID}" \\
        --pkg-names amdrocm-core-sdk \\
        --report-dir output/pre_upload_reports
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_TOOLS_DIR = SCRIPT_DIR.parent.parent
if str(BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_TOOLS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from packaging_utils import PackageConfig, read_package_json_file  # noqa: E402

import pkg_dependency_checker as checker  # noqa: E402


@dataclass
class VariantBuildCheck:
    """One expected package variant and whether the built archive matches."""

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
        return self.found and self.version_ok and not self.errors


@dataclass
class BuildVerifyReport:
    """Aggregate build verification for one ``package.json`` base name."""

    base_package: str
    variants: list[VariantBuildCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.variants)


@dataclass
class BuildVerifySummary:
    """Roll-up across all packages checked in one run."""

    packages_requested: list[str]
    reports: list[BuildVerifyReport]
    package_files_found: list[str]
    extra_package_files: list[str] = field(default_factory=list)

    @property
    def variants_expected(self) -> int:
        return sum(len(r.variants) for r in self.reports)

    @property
    def variants_found(self) -> int:
        return sum(1 for r in self.reports for v in r.variants if v.found)

    @property
    def variants_passed(self) -> int:
        return sum(1 for r in self.reports for v in r.variants if v.passed)

    @property
    def variants_failed(self) -> int:
        return self.variants_expected - self.variants_passed

    @property
    def passed(self) -> bool:
        if self.extra_package_files:
            return False
        return self.variants_failed == 0 and all(r.passed for r in self.reports)

    def missing_variants(self) -> list[str]:
        return [
            v.expected_name
            for r in self.reports
            for v in r.variants
            if not v.found
        ]

    def version_failures(self) -> list[str]:
        return [
            v.expected_name
            for r in self.reports
            for v in r.variants
            if v.found and not v.version_ok
        ]


def expected_control_version(config: PackageConfig) -> str:
    """Return the version string written into DEB/RPM control metadata."""
    if config.pkg_type.lower() == "rpm":
        release = config.version_suffix or "1"
        return f"{config.rocm_version}-{release}"
    version = str(config.rocm_version)
    if config.version_suffix:
        version += f"-{config.version_suffix}"
    return version


def read_package_file_version(package_path: Path, pkg_type: str) -> str:
    """Read ``Version`` (DEB) or ``VERSION-RELEASE`` (RPM) from a package archive."""
    pkg_type = pkg_type.lower()
    if pkg_type == "deb":
        result = checker._run_capture(
            ["dpkg-deb", "-f", str(package_path), "Version"],
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"dpkg-deb failed for {package_path}: {result.stderr.strip()}",
            )
        return result.stdout.strip()
    result = checker._run_capture(
        ["rpm", "-qp", "--qf", r"%{VERSION}-%{RELEASE}", str(package_path)],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rpm query failed for {package_path}: {result.stderr.strip()}",
        )
    return result.stdout.strip()


def versions_match(expected: str, actual: str, pkg_type: str) -> bool:
    """Return True when ``actual`` matches ``expected`` (DEB allows ``~`` vs ``-``)."""
    if expected == actual:
        return True
    if pkg_type.lower() == "deb":
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
    """Verify one expected variant exists and optionally matches version."""
    expected_version = expected_control_version(config)
    path = package_files.get(expected_name)
    found = path is not None
    actual_version: str | None = None
    version_ok = not check_version
    errors: list[str] = []

    if not found:
        errors.append(f"package file not found for expected name {expected_name!r}")
    else:
        if check_version:
            try:
                actual_version = read_package_file_version(path, config.pkg_type)
                version_ok = versions_match(
                    expected_version, actual_version, config.pkg_type,
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
    """Verify all build variants for one ``package.json`` package entry."""
    report = BuildVerifyReport(base_package=pkg_name)
    package_files = checker.find_package_files(packages_dir, config.pkg_type)

    for label, versioned_pkg, gfx_arch in checker.iter_variant_configs(
        pkg_name, config,
    ):
        expected_name = checker.resolve_installed_name(
            pkg_name,
            config,
            versioned_pkg=versioned_pkg,
            gfx_arch=gfx_arch,
        )
        report.variants.append(
            verify_variant(
                pkg_name,
                label,
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
    """Return package stems present on disk but not expected by ``package.json``."""
    return sorted(name for name in package_files if name not in expected_names)


def resolve_pkg_names(
    args: argparse.Namespace,
    config: PackageConfig,
) -> list[str]:
    """Resolve which ``package.json`` entries to verify."""
    if args.all_eligible:
        from build_package import parse_input_package_list

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
    """Build the run summary including optional extra-file detection."""
    expected_names = {
        v.expected_name for r in reports for v in r.variants
    }
    extra = collect_extra_package_files(package_files, expected_names)
    if not fail_on_extra:
        extra = []
    return BuildVerifySummary(
        packages_requested=packages_requested,
        reports=reports,
        package_files_found=sorted(package_files.keys()),
        extra_package_files=extra,
    )


def _variant_to_dict(variant: VariantBuildCheck) -> dict:
    """Serialize one variant check for JSON output."""
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
    """Serialize the full build verification report as JSON."""
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
    """Build a plain-text build status report for logs or ``build_status_report.txt``."""
    lines: list[str] = []
    overall = "PASS" if summary.passed else "FAIL"
    lines.append("ROCm build package verification report (Step 1)")
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
    """Write ``build_status_report.txt`` and ``build_status_report.json``."""
    report_dir.mkdir(parents=True, exist_ok=True)
    text_path = report_dir / "build_status_report.txt"
    json_path = report_dir / "build_status_report.json"
    text_path.write_text(format_report_text(summary) + "\n", encoding="utf-8")
    json_path.write_text(format_report_json(summary) + "\n", encoding="utf-8")
    print(f"Build report written to: {text_path}")
    print(f"Build report written to: {json_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``build_package_verify``."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify built native Linux packages match package.json variant names "
            "and control-field versions (pre-upload Step 1)."
        ),
    )
    parser.add_argument(
        "--pkg-type",
        required=True,
        choices=("deb", "rpm", "DEB", "RPM"),
        help="Package format (deb or rpm)",
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
        default="",
        help="Build version suffix (e.g. CI run id)",
    )
    parser.add_argument(
        "--install-prefix",
        default="/opt/rocm/core",
        help="Install prefix recorded in PackageConfig",
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
        help="Use kpack variant rules (auto-detected from manifest if omitted)",
    )
    parser.add_argument(
        "--rpath-pkg",
        action="store_true",
        help="Package was built with --rpath-pkg",
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose packaging_utils debug prints",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: verify built packages, write report, return exit code."""
    args = parse_args(argv)
    checker._suppress_packaging_noise(args.quiet)

    packages_dir = args.packages_dir.expanduser().resolve()
    if not packages_dir.is_dir():
        print(f"Error: packages directory not found: {packages_dir}", file=sys.stderr)
        return 2

    read_package_json_file()
    try:
        config = checker.build_checker_config(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        pkg_names = resolve_pkg_names(args, config)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not pkg_names:
        print("Error: no packages to verify", file=sys.stderr)
        return 2

    check_version = not args.no_version_check
    reports = [
        verify_package(name, config, packages_dir, check_version=check_version)
        for name in pkg_names
    ]
    package_files = checker.find_package_files(packages_dir, config.pkg_type)
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
        print(
            f"\nBuild verification failed: "
            f"{summary.variants_failed} variant(s), "
            f"{len(summary.extra_package_files)} extra file(s).",
            file=sys.stderr,
        )
        return 1
    print("\nBuild verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Epilog
# -----
# Step 1 (this module): names + versions of built archives vs package.json variants.
# Step 2 (pkg_dependency_checker.py): Depends/Requires vs package.json rules.
# Step 3 (native_linux_package_install_test.py --test-type simulate): installability.
