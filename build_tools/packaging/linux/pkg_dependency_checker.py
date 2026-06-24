#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify ROCm native package dependencies against ``package.json`` expectations.

Compares dependencies that packaging would emit (via ``process_main_dependencies``)
with either:

* **installed** — packages on the host (``dpkg-query`` / ``rpm -q``), or
* **package-file** — ``Depends`` / ``Requires`` read from local ``.deb`` / ``.rpm`` files.

Use this after installing metapackages such as ``amdrocm-core-sdk`` to confirm
per-GPU variants are present (regression guard for issues like #6093).

Example (installed DEB, kpack nightly)::

    sudo python3 pkg_dependency_checker.py \\
        --pkg-type deb \\
        --rocm-version 7.14.0 \\
        --version-suffix 20260620-27854481844 \\
        --enable-kpack \\
        --target gfx1100 gfx942 \\
        --artifacts-dir /path/to/artifacts \\
        --pkg-names amdrocm-core-sdk

Example (offline ``.deb`` control fields)::

    python3 pkg_dependency_checker.py \\
        --mode package-file \\
        --pkg-type deb \\
        --packages-dir /path/to/debs \\
        --rocm-version 7.14.0 \\
        --enable-kpack \\
        --target gfx1100 \\
        --artifacts-dir /path/to/artifacts \\
        --pkg-names amdrocm-core-sdk
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_TOOLS_DIR = SCRIPT_DIR.parent.parent
if str(BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_TOOLS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from packaging_utils import (  # noqa: E402
    GFX_HOST,
    GFX_META,
    PackageConfig,
    get_package_info,
    is_gfxarch_package,
    is_meta_package,
    normalize_target_list,
    process_main_dependencies,
    read_package_json_file,
    update_package_name,
)

# DEB: "pkg ( = 1.0 ), pkg2 | pkg3"
_DEB_DEP_ALTERNATIVE_RE = re.compile(r"\s*\|\s*")
_DEB_DEP_NAME_RE = re.compile(
    r"^\s*([a-zA-Z0-9][a-zA-Z0-9+.\-]*)(?:\s*\([^)]+\))?\s*$"
)
# RPM: "pkg = 1.0" or "rpmlib(...)"
_RPM_SKIP_PREFIXES = ("rpmlib(", "rtld(", "/")


@dataclass
class VariantCheck:
    """One package variant and its resolved dependency names."""

    label: str
    installed_name: str
    package_found: bool
    expected_deps: list[str]
    actual_deps: list[str]
    missing: list[str] = field(default_factory=list)
    not_installed: list[str] = field(default_factory=list)
    extra_deps: list[str] = field(default_factory=list)
    passed: bool = True


@dataclass
class CheckReport:
    """Aggregate result for one logical ``package.json`` entry."""

    base_package: str
    variants: list[VariantCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.variants)

    @property
    def variants_checked(self) -> int:
        return len(self.variants)

    @property
    def variants_passed(self) -> int:
        return sum(1 for v in self.variants if v.passed)

    @property
    def variants_failed(self) -> int:
        return self.variants_checked - self.variants_passed


@dataclass
class RunSummary:
    """Roll-up across all ``CheckReport`` results from one checker invocation."""

    packages_requested: list[str]
    reports: list[CheckReport]

    @property
    def packages_checked(self) -> int:
        return len(self.reports)

    @property
    def variants_checked(self) -> int:
        return sum(r.variants_checked for r in self.reports)

    @property
    def variants_passed(self) -> int:
        return sum(r.variants_passed for r in self.reports)

    @property
    def variants_failed(self) -> int:
        return sum(r.variants_failed for r in self.reports)

    @property
    def passed(self) -> bool:
        return self.variants_failed == 0 and all(r.passed for r in self.reports)

    def unique_expected_deps(self) -> list[str]:
        names: set[str] = set()
        for report in self.reports:
            for variant in report.variants:
                names.update(variant.expected_deps)
        return sorted(names)

    def unique_actual_deps(self) -> list[str]:
        names: set[str] = set()
        for report in self.reports:
            for variant in report.variants:
                names.update(variant.actual_deps)
        return sorted(names)

    def unique_missing_deps(self) -> list[str]:
        names: set[str] = set()
        for report in self.reports:
            for variant in report.variants:
                names.update(variant.missing)
        return sorted(names)

    def unique_not_installed_deps(self) -> list[str]:
        names: set[str] = set()
        for report in self.reports:
            for variant in report.variants:
                names.update(variant.not_installed)
        return sorted(names)

    def packages_found(self) -> list[str]:
        return sorted(
            {
                v.installed_name
                for report in self.reports
                for v in report.variants
                if v.package_found
            }
        )

    def packages_not_found(self) -> list[str]:
        return sorted(
            {
                v.installed_name
                for report in self.reports
                for v in report.variants
                if not v.package_found
            }
        )


def _suppress_packaging_noise(quiet: bool) -> None:
    if not quiet:
        return
    import packaging_utils as pu

    pu.print_function_name = lambda: None  # type: ignore[method-assign, assignment]


def parse_dependency_names(dep_field: str, pkg_type: str) -> list[str]:
    """Extract package names from a DEB ``Depends`` or RPM requires string.

    For DEB alternatives (``a | b``), every alternative name is returned.
    RPM virtual provides and rpmlib constraints are skipped.
    """
    if not dep_field or not dep_field.strip():
        return []

    names: list[str] = []
    pkg_type = pkg_type.lower()
    for raw_token in dep_field.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if pkg_type == "rpm":
            if token.startswith(_RPM_SKIP_PREFIXES):
                continue
            name = token.split("=", 1)[0].strip()
            if name:
                names.append(name)
            continue

        for alt in _DEB_DEP_ALTERNATIVE_RE.split(token):
            alt = alt.strip()
            if not alt:
                continue
            match = _DEB_DEP_NAME_RE.match(alt)
            if match:
                names.append(match.group(1))
            else:
                names.append(alt.split("(", 1)[0].strip())
    return names


def build_checker_config(args: argparse.Namespace) -> PackageConfig:
    """Map CLI flags to ``PackageConfig`` (mirrors ``build_package.create_package_config``)."""
    parts = args.rocm_version.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Version string '{args.rocm_version}' does not have major.minor versions"
        )
    major = re.match(r"^\d+", parts[0])
    minor = re.match(r"^\d+", parts[1])
    if not major or not minor:
        raise ValueError(f"Invalid rocm version: {args.rocm_version}")
    modified = f"{major.group()}.{minor.group()}"

    install_prefix = args.install_prefix
    if install_prefix == "/opt/rocm/core":
        install_prefix = f"{install_prefix}-{modified}"

    targets = normalize_target_list(args.target)
    if args.enable_kpack:
        gfx_arch = ""
        gfxarch_list = tuple(targets)
    else:
        gfx_arch = targets[0] if targets else ""
        gfxarch_list = ()

    pkg_type = (args.pkg_type or "").lower()
    if pkg_type not in {"deb", "rpm"}:
        raise ValueError(f"Invalid package type: {args.pkg_type}")

    return PackageConfig(
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        dest_dir=Path(args.dest_dir).resolve(),
        pkg_type=pkg_type,
        rocm_version=args.rocm_version,
        version_suffix=args.version_suffix or "",
        install_prefix=install_prefix,
        gfx_arch=gfx_arch,
        enable_rpath=args.rpath_pkg,
        enable_kpack=args.enable_kpack,
        gfxarch_list=gfxarch_list,
    )


def _dependency_field_key(pkg_type: str) -> str:
    return "DEBDepends" if pkg_type.lower() == "deb" else "RPMRequires"


def expected_dependencies(
    pkg_name: str,
    config: PackageConfig,
    *,
    versioned_pkg: bool,
    gfx_arch: str,
) -> list[str]:
    """Compute expected dependency names for one package variant."""
    local = replace(
        config,
        versioned_pkg=versioned_pkg,
        gfx_arch=gfx_arch,
    )
    pkg_info = get_package_info(pkg_name)
    field_key = _dependency_field_key(config.pkg_type)
    dep_string = process_main_dependencies(pkg_info, field_key, local)
    return parse_dependency_names(dep_string, config.pkg_type)


def iter_variant_configs(
    pkg_name: str,
    config: PackageConfig,
) -> list[tuple[str, bool, str]]:
    """Return ``(label, versioned_pkg, gfx_arch)`` tuples to verify for ``pkg_name``.

    Mirrors the variant split performed by ``build_package.py`` without building.
    """
    pkg_info = get_package_info(pkg_name)
    is_meta = is_meta_package(pkg_info)
    is_gfx = is_gfxarch_package(pkg_info, config.enable_kpack, config.artifacts_dir)
    variants: list[tuple[str, bool, str]] = []

    if config.enable_kpack and is_gfx:
        if is_meta:
            if not config.enable_rpath:
                variants.append(
                    ("non-versioned metapackage", False, GFX_META),
                )
            variants.append(("versioned meta (GFX_META)", True, GFX_META))
            for arch in config.gfxarch_list:
                variants.append((f"arch-specific metapackage ({arch})", True, arch))
        else:
            variants.append(("host package", True, GFX_HOST))
            for arch in config.gfxarch_list:
                variants.append((f"device package ({arch})", True, arch))
            variants.append(("versioned meta (GFX_META)", True, GFX_META))
            if not config.enable_rpath:
                variants.append(("non-versioned metapackage", False, GFX_META))
    elif config.enable_kpack:
        variants.append(("versioned package", True, ""))
        if not config.enable_rpath:
            variants.append(("non-versioned metapackage", False, ""))
    else:
        variants.append(("versioned package", True, config.gfx_arch))
        if not config.enable_rpath:
            variants.append(("non-versioned metapackage", False, config.gfx_arch))

    return variants


def resolve_installed_name(
    pkg_name: str,
    config: PackageConfig,
    *,
    versioned_pkg: bool,
    gfx_arch: str,
) -> str:
    """Map a logical variant to the on-disk / installed package name."""
    local = replace(config, versioned_pkg=versioned_pkg, gfx_arch=gfx_arch)
    return update_package_name(pkg_name, local)


def _run_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )


def is_installed(package_name: str, pkg_type: str) -> bool:
    """Return True when ``package_name`` is installed on the system."""
    pkg_type = pkg_type.lower()
    if pkg_type == "deb":
        if not shutil.which("dpkg-query"):
            raise RuntimeError("dpkg-query not found; cannot check DEB packages")
        result = _run_capture(
            ["dpkg-query", "-W", "-f=${Status}", package_name],
        )
        return result.returncode == 0 and result.stdout.strip().startswith("install ok")
    if not shutil.which("rpm"):
        raise RuntimeError("rpm not found; cannot check RPM packages")
    result = _run_capture(["rpm", "-q", package_name])
    return result.returncode == 0


def read_installed_dependencies(package_name: str, pkg_type: str) -> list[str]:
    """Read declared dependencies of an installed package."""
    pkg_type = pkg_type.lower()
    if pkg_type == "deb":
        result = _run_capture(
            ["dpkg-query", "-W", "-f=${Depends}", package_name],
        )
        if result.returncode != 0:
            return []
        return parse_dependency_names(result.stdout, "deb")
    result = _run_capture(["rpm", "-q", "--requires", package_name])
    if result.returncode != 0:
        return []
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return parse_dependency_names(", ".join(lines), "rpm")


def read_package_file_dependencies(package_path: Path, pkg_type: str) -> list[str]:
    """Read ``Depends`` / ``Requires`` from a local package file."""
    pkg_type = pkg_type.lower()
    if pkg_type == "deb":
        if not shutil.which("dpkg-deb"):
            raise RuntimeError("dpkg-deb not found")
        result = _run_capture(["dpkg-deb", "-f", "Depends", str(package_path)])
        if result.returncode != 0:
            raise RuntimeError(
                f"dpkg-deb failed for {package_path}: {result.stderr.strip()}"
            )
        return parse_dependency_names(result.stdout, "deb")
    if not shutil.which("rpm"):
        raise RuntimeError("rpm not found")
    result = _run_capture(["rpm", "-qp", "--requires", str(package_path)])
    if result.returncode != 0:
        raise RuntimeError(f"rpm query failed for {package_path}: {result.stderr.strip()}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return parse_dependency_names(", ".join(lines), "rpm")


def find_package_files(packages_dir: Path, pkg_type: str) -> dict[str, Path]:
    """Index ``.deb`` / ``.rpm`` files by package name prefix."""
    pkg_type = pkg_type.lower()
    index: dict[str, Path] = {}
    if pkg_type == "deb":
        for path in sorted(packages_dir.glob("*.deb")):
            # amdrocm-foo7.14_7.14.0_amd64.deb → amdrocm-foo7.14
            stem = path.name.split("_", 1)[0]
            index[stem] = path
        return index
    for path in sorted(packages_dir.rglob("*.rpm")):
        # amdrocm-foo7.14-7.14.0.x86_64.rpm → amdrocm-foo7.14
        stem = path.name.rsplit(".", 2)[0]
        stem = re.sub(r"-[^-]+-[^-]+$", "", stem)
        index[stem] = path
    return index


def _compare_dep_lists(
    expected: list[str],
    actual: list[str],
    *,
    verify_installed: bool,
    pkg_type: str,
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(missing_from_actual, not_installed, extra_in_actual)``."""
    expected_set = set(expected)
    actual_set = set(actual)
    missing = [dep for dep in expected if dep not in actual_set]
    extra = [dep for dep in actual if dep not in expected_set]
    not_installed: list[str] = []
    if verify_installed:
        for dep in expected:
            if dep not in actual_set:
                continue
            if not is_installed(dep, pkg_type):
                not_installed.append(dep)
    return missing, not_installed, extra


def check_package(
    pkg_name: str,
    config: PackageConfig,
    *,
    mode: str,
    packages_dir: Path | None,
    verify_installed: bool,
    only_installed: bool,
) -> CheckReport:
    """Verify all variants of ``pkg_name`` against ``package.json`` expectations."""
    report = CheckReport(base_package=pkg_name)
    package_files = (
        find_package_files(packages_dir, config.pkg_type) if packages_dir else {}
    )

    for label, versioned_pkg, gfx_arch in iter_variant_configs(pkg_name, config):
        installed_name = resolve_installed_name(
            pkg_name,
            config,
            versioned_pkg=versioned_pkg,
            gfx_arch=gfx_arch,
        )

        if only_installed and mode == "installed" and not is_installed(
            installed_name, config.pkg_type
        ):
            continue

        expected = expected_dependencies(
            pkg_name,
            config,
            versioned_pkg=versioned_pkg,
            gfx_arch=gfx_arch,
        )

        if mode == "installed":
            if not is_installed(installed_name, config.pkg_type):
                variant = VariantCheck(
                    label=label,
                    installed_name=installed_name,
                    package_found=False,
                    expected_deps=expected,
                    actual_deps=[],
                    missing=expected,
                    passed=not expected,
                )
                report.variants.append(variant)
                continue
            actual = read_installed_dependencies(installed_name, config.pkg_type)
            package_found = True
        else:
            pkg_path = package_files.get(installed_name)
            if pkg_path is None:
                variant = VariantCheck(
                    label=label,
                    installed_name=installed_name,
                    package_found=False,
                    expected_deps=expected,
                    actual_deps=[],
                    missing=expected,
                    passed=not expected,
                )
                report.variants.append(variant)
                continue
            actual = read_package_file_dependencies(pkg_path, config.pkg_type)
            package_found = True

        missing, not_installed, extra = _compare_dep_lists(
            expected,
            actual,
            verify_installed=verify_installed and mode == "installed",
            pkg_type=config.pkg_type,
        )
        passed = not missing and not not_installed
        report.variants.append(
            VariantCheck(
                label=label,
                installed_name=installed_name,
                package_found=package_found,
                expected_deps=expected,
                actual_deps=actual,
                missing=missing,
                not_installed=not_installed,
                extra_deps=extra,
                passed=passed,
            )
        )

    return report


def check_transitive_closure(
    root_packages: list[str],
    config: PackageConfig,
    *,
    mode: str,
    packages_dir: Path | None,
) -> list[CheckReport]:
    """Recursively verify amdrocm dependencies declared in ``package.json``."""
    reports: list[CheckReport] = []
    seen: set[str] = set()
    queue = list(root_packages)

    while queue:
        pkg_name = queue.pop(0)
        if pkg_name in seen:
            continue
        seen.add(pkg_name)

        try:
            get_package_info(pkg_name)
        except ValueError:
            continue

        report = check_package(
            pkg_name,
            config,
            mode=mode,
            packages_dir=packages_dir,
            verify_installed=False,
            only_installed=True,
        )
        if report.variants:
            reports.append(report)

        for variant in report.variants:
            for dep in variant.expected_deps:
                if not dep.startswith("amdrocm"):
                    continue
                base = _guess_base_package_name(dep)
                if base and base not in seen:
                    queue.append(base)

    return reports


def _guess_base_package_name(installed_name: str) -> str | None:
    """Best-effort map from versioned installed name back to ``package.json`` key."""
    name = installed_name
    if name.endswith("-dev"):
        name = name[: -len("-dev")] + "-devel"
    name = re.sub(r"-host\d+\.\d+.*$", "", name)
    name = re.sub(r"-\d+\.\d+(-[a-z0-9]+)?$", "", name)
    name = re.sub(r"-gfx[a-z0-9]+$", "", name)
    try:
        get_package_info(name)
        return name
    except ValueError:
        return None


def build_run_summary(
    packages_requested: list[str],
    reports: list[CheckReport],
) -> RunSummary:
    """Aggregate per-package reports into a single run summary."""
    return RunSummary(packages_requested=packages_requested, reports=reports)


def _variant_to_dict(variant: VariantCheck) -> dict:
    return {
        "label": variant.label,
        "installed_name": variant.installed_name,
        "package_found": variant.package_found,
        "passed": variant.passed,
        "expected_deps": variant.expected_deps,
        "actual_deps": variant.actual_deps,
        "missing": variant.missing,
        "not_installed": variant.not_installed,
        "extra_deps": variant.extra_deps,
    }


def format_report_json(summary: RunSummary) -> str:
    """Serialize the full run report as JSON."""
    payload = {
        "packages_requested": summary.packages_requested,
        "passed": summary.passed,
        "packages_checked": summary.packages_checked,
        "variants_checked": summary.variants_checked,
        "variants_passed": summary.variants_passed,
        "variants_failed": summary.variants_failed,
        "packages_found": summary.packages_found(),
        "packages_not_found": summary.packages_not_found(),
        "unique_expected_deps": summary.unique_expected_deps(),
        "unique_actual_deps": summary.unique_actual_deps(),
        "unique_missing_deps": summary.unique_missing_deps(),
        "unique_not_installed_deps": summary.unique_not_installed_deps(),
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


def format_report_text(summary: RunSummary) -> str:
    """Build a plain-text report suitable for logs or ``--report-file``."""
    lines: list[str] = []
    lines.append("ROCm package dependency check report")
    lines.append("=" * 72)
    lines.append(f"Packages requested: {', '.join(summary.packages_requested)}")
    lines.append("")

    for report in summary.reports:
        lines.append(f"Package (package.json): {report.base_package}")
        if not report.variants:
            lines.append("  (no variants checked)")
            lines.append("")
            continue
        for variant in report.variants:
            status = "PASS" if variant.passed else "FAIL"
            found = "yes" if variant.package_found else "no"
            lines.append(f"  [{status}] {variant.label}")
            lines.append(f"         name: {variant.installed_name}")
            lines.append(f"         package found: {found}")
            lines.append(
                f"         dependencies expected ({len(variant.expected_deps)}): "
                f"{variant.expected_deps}"
            )
            lines.append(
                f"         dependencies found ({len(variant.actual_deps)}): "
                f"{variant.actual_deps}"
            )
            if variant.missing:
                lines.append(f"         missing from declares: {variant.missing}")
            if variant.not_installed:
                lines.append(
                    f"         declared but not installed: {variant.not_installed}"
                )
            if variant.extra_deps:
                lines.append(f"         extra (not in package.json rules): {variant.extra_deps}")
        lines.append("")

    lines.append(format_summary_text(summary))
    return "\n".join(lines)


def format_summary_text(summary: RunSummary) -> str:
    """Build the final summary block."""
    overall = "PASS" if summary.passed else "FAIL"
    lines = [
        "SUMMARY",
        "-" * 72,
        f"Overall result: {overall}",
        f"Packages checked: {summary.packages_checked}",
        f"Variants checked: {summary.variants_checked} "
        f"({summary.variants_passed} passed, {summary.variants_failed} failed)",
        f"Packages found: {len(summary.packages_found())} — {summary.packages_found()}",
        f"Packages not found: {len(summary.packages_not_found())} — "
        f"{summary.packages_not_found()}",
        f"Unique dependencies expected: {len(summary.unique_expected_deps())}",
        f"Unique dependencies found in declares: {len(summary.unique_actual_deps())}",
        f"Unique dependencies missing from declares: {len(summary.unique_missing_deps())} — "
        f"{summary.unique_missing_deps()}",
    ]
    not_installed = summary.unique_not_installed_deps()
    if not_installed:
        lines.append(
            f"Unique dependencies declared but not installed: {len(not_installed)} — "
            f"{not_installed}"
        )
    return "\n".join(lines)


def print_report(report: CheckReport, *, verbose: bool, show_report: bool) -> None:
    """Print human-readable results for one package."""
    print(f"\n=== {report.base_package} ===")
    if not report.variants:
        print("  (no matching installed variants to check)")
        return
    for variant in report.variants:
        status = "PASS" if variant.passed else "FAIL"
        found = "found" if variant.package_found else "NOT FOUND"
        print(f"  [{status}] {variant.label}: {variant.installed_name} ({found})")
        show_deps = verbose or show_report or not variant.passed
        if show_deps:
            print(
                f"         expected ({len(variant.expected_deps)}): "
                f"{variant.expected_deps}"
            )
            print(
                f"         actual   ({len(variant.actual_deps)}): "
                f"{variant.actual_deps}"
            )
        if variant.missing:
            print(f"         missing from declares: {variant.missing}")
        if variant.not_installed:
            print(f"         declared but not installed: {variant.not_installed}")
        if (verbose or show_report) and variant.extra_deps:
            print(f"         extra (not in package.json rules): {variant.extra_deps}")


def print_summary(summary: RunSummary) -> None:
    """Print the roll-up summary to stdout."""
    print("\n" + format_summary_text(summary))


def write_report_file(summary: RunSummary, path: Path) -> None:
    """Write report to ``path`` (``.json`` → JSON, otherwise plain text)."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(format_report_json(summary) + "\n", encoding="utf-8")
    else:
        path.write_text(format_report_text(summary) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify ROCm package dependencies match package.json expectations "
            "(installed system or local .deb/.rpm files)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("installed", "package-file"),
        default="installed",
        help="installed: query dpkg/rpm on this host; package-file: read local archives",
    )
    parser.add_argument(
        "--pkg-type",
        required=True,
        choices=("deb", "rpm", "DEB", "RPM"),
        help="Package format (deb or rpm)",
    )
    parser.add_argument(
        "--rocm-version",
        required=True,
        help="ROCm release version used when packages were built (e.g. 7.14.0)",
    )
    parser.add_argument(
        "--version-suffix",
        default="",
        help="Build version suffix (e.g. nightly timestamp)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Artifact tree used for kpack gfx-arch detection (directory may be empty for metapackages)",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("/tmp/rocm-pkg-check"),
        help="Placeholder dest dir for PackageConfig (not used for checks)",
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
        help="GPU targets for kpack (space-separated, e.g. gfx1100 gfx942)",
    )
    parser.add_argument(
        "--enable-kpack",
        action="store_true",
        help="Use kpack (multi-arch) dependency rules",
    )
    parser.add_argument(
        "--rpath-pkg",
        action="store_true",
        help="Package was built with --rpath-pkg (skips non-versioned variants)",
    )
    parser.add_argument(
        "--pkg-names",
        nargs="+",
        required=True,
        help="Base package names from package.json (e.g. amdrocm-core-sdk)",
    )
    parser.add_argument(
        "--packages-dir",
        type=Path,
        help="Directory of .deb or .rpm files (required for --mode package-file)",
    )
    parser.add_argument(
        "--verify-installed",
        action="store_true",
        help="Also confirm each expected dependency is installed (installed mode only)",
    )
    parser.add_argument(
        "--transitive",
        action="store_true",
        help="Recursively check amdrocm dependencies from package.json",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Check every variant even if not installed (reports missing package)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose packaging_utils debug prints",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full expected/actual dependency lists (failures always shown)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed inventory of each package variant and its dependencies",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        metavar="PATH",
        help="Write full report to PATH (.json for JSON, otherwise plain text)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the final summary (implies --report for --report-file content)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _suppress_packaging_noise(args.quiet)

    if args.mode == "package-file" and not args.packages_dir:
        print("Error: --packages-dir is required for --mode package-file", file=sys.stderr)
        return 2

    packages_dir = args.packages_dir.resolve() if args.packages_dir else None
    if packages_dir and not packages_dir.is_dir():
        print(f"Error: packages directory not found: {packages_dir}", file=sys.stderr)
        return 2

    # Validate package names exist in package.json early.
    read_package_json_file()
    for name in args.pkg_names:
        get_package_info(name)

    try:
        config = build_checker_config(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    only_installed = not args.all_variants
    reports: list[CheckReport] = []

    if args.transitive:
        reports = check_transitive_closure(
            args.pkg_names,
            config,
            mode=args.mode,
            packages_dir=packages_dir,
        )
    else:
        for pkg_name in args.pkg_names:
            reports.append(
                check_package(
                    pkg_name,
                    config,
                    mode=args.mode,
                    packages_dir=packages_dir,
                    verify_installed=args.verify_installed,
                    only_installed=only_installed,
                )
            )

    summary = build_run_summary(args.pkg_names, reports)
    show_report = args.report or args.report_file is not None

    if not args.summary_only:
        for report in reports:
            print_report(report, verbose=args.verbose, show_report=show_report)

    print_summary(summary)

    if args.report_file is not None:
        write_report_file(summary, args.report_file)
        print(f"\nReport written to: {args.report_file.resolve()}")

    if not summary.passed:
        print(
            f"\n{summary.variants_failed} variant(s) failed dependency verification.",
            file=sys.stderr,
        )
        return 1
    print("\nAll checked packages passed dependency verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
