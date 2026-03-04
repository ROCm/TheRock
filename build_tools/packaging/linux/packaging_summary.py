# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from datetime import datetime, timezone
from packaging_utils import *
from typing import List


@dataclass
class PackageList:
    # All base package names that were attempted
    total: List[str]
    # Packages that were successfully created (versioned + non-versioned)
    built: List[str]
    # Base packages that were skipped
    skipped: List[str]


def write_build_manifest(config: PackageConfig, pkg_list: PackageList):
    """Write manifest files listing built and skipped packages.

    Parameters:
    config: Configuration object containing package metadata
    pkg_list: List of all packages attempted/built/skipped

    Returns: None
    """
    print_function_name()

    # Write successful packages manifest
    manifest_file = Path(config.dest_dir) / "built_packages.txt"

    total_basepkg = len(pkg_list.total) + len(pkg_list.skipped)
    expected = 2 * len(pkg_list.total)
    built = len(pkg_list.built)
    failed = expected - built

    try:
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(f"# Built Packages Manifest\n")
            f.write(f"# Package Type: {config.pkg_type.upper()}\n")
            f.write(f"# ROCm Version: {config.rocm_version}\n")
            f.write(f"# Graphics Architecture: {config.gfx_arch}\n")
            f.write(
                f"# Build Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            )
            f.write(f"# Total base packages: {total_basepkg}\n")
            f.write(f"# Skipped base packages: {len(pkg_list.skipped)}\n")
            f.write(
                f"# Total packages attempted (versioned + non-versioned): {expected}\n"
            )
            f.write(f"# Successfully built: {built}\n")
            f.write(f"# Failed to build: {failed}\n")
            f.write(f"\n")

            if pkg_list.built:
                f.write(f"# Created Packages:\n")
                for pkg in sorted(pkg_list.built):
                    f.write(f"{pkg}\n")

            if pkg_list.skipped:
                f.write(f"\n# Skipped Packages:\n")
                f.write(
                    f"# Note: Package names shown are base names from package.json\n"
                )
                for pkg in sorted(pkg_list.skipped):
                    f.write(f"{pkg}\n")

        print(f"✅ Built packages manifest written to: {manifest_file}")
    except Exception as e:
        print(f"⚠️  WARNING: Failed to write built packages manifest: {e}")


def print_build_status(config: PackageConfig, pkg_list: PackageList):
    """Print a summary of the build process.

    Parameters:
    config: Configuration object containing package metadata
    pkg_list: List of all packages attempted/built/skipped

    Returns: None
    """
    print("\n" + "=" * 80)
    print("BUILD SUMMARY")
    print("=" * 80)

    total_basepkg = len(pkg_list.total) + len(pkg_list.skipped)
    expected = 2 * len(pkg_list.total)
    built = len(pkg_list.built)
    failed = expected - built

    print(f"\nTotal base packages: {total_basepkg} ")
    print(f"⏭️ Skipped base packages: {len(pkg_list.skipped)}")
    print(f"Total packages attempted (versioned + non-versioned): {expected}")
    print(f"✅ Successfully built: {built}")
    print(f"❌ Failed to build: {failed}")

    print(f"\nCreated packages")
    for pkg in sorted(pkg_list.built):
        print(f"   - {pkg}")

    if pkg_list.skipped:
        print(f"\n⏭️   Skipped packages")
        print(f"   (Base package names from package.json)")
        for pkg in sorted(pkg_list.skipped):
            print(f"   - {pkg}")

    print("\n" + "=" * 80)
    print(f"Package type: {config.pkg_type.upper()}")
    print(f"ROCm version: {config.rocm_version}")
    print(f"Output directory: {config.dest_dir}")
    print("=" * 80 + "\n")


def print_build_summary(config: PackageConfig, pkg_list: PackageList):
    write_build_manifest(config, pkg_list)
    print_build_status(config, pkg_list)
