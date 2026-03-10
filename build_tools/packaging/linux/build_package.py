#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""Given ROCm artifacts directories, performs packaging to
create RPM and DEB packages and upload to artifactory server

```
./build_package.py --artifacts-dir ./ARTIFACTS_DIR  \
        --target gfx94X-dcgpu \
        --dest-dir ./OUTPUT_PKGDIR \
        --rocm-version 7.1.0 \
        --pkg-type deb (or rpm) \
        --version-suffix build_type (daily/master/nightly/release)
```
"""

import argparse
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from deb_package import create_deb_package
from packaging_summary import *
from packaging_utils import *
from rpm_package import create_rpm_package


SCRIPT_DIR = Path(__file__).resolve().parent
# Default install prefix
DEFAULT_INSTALL_PREFIX = "/opt/rocm/core"


def parse_input_package_list(pkg_name, artifact_dir):
    """Populate the package list from the provided input arguments.

    Parameters:
    pkg_name : List of packages to be created
    artifact_dir: The path to the Artifactory directory

    Returns: Package list
    """
    print_function_name()
    pkg_list = []
    skipped_list = []
    # If pkg_name is None, include all packages
    if pkg_name is None:
        pkg_list, skipped_list = get_package_list(artifact_dir)
        return pkg_list, skipped_list

    # Proceed if pkg_name is not None
    data = read_package_json_file()

    for entry in data:
        # Skip if packaging is disabled
        if is_packaging_disabled(entry):
            continue

        name = entry.get("Package")

        # Loop through each type in pkg_name
        for pkg in pkg_name:
            if pkg == name:
                pkg_list.append(name)
                break

    print(f"pkg_list:\n  {pkg_list}")
    return pkg_list, skipped_list


def normalize_target_list(targets: list[str]) -> list[str]:
    """Normalize target list by splitting on semicolons, commas, or spaces.

    Accepts targets in multiple formats:
    - Space-separated CLI args: ['gfx94X-dcgpu', 'gfx120X-all']
    - Single comma-separated string: ['gfx94X-dcgpu,gfx120X-all,gfx1151']
    - Single semicolon-separated string: ['gfx94X-dcgpu;gfx120X-all;gfx1151']
    - Mixed: ['gfx94X-dcgpu;gfx120X-all', 'gfx1151']

    Returns a flat list of individual target names.
    """
    normalized = []
    for target in targets:
        # Split by semicolon first, then comma, then whitespace
        if ";" in target:
            normalized.extend(target.split(";"))
        elif "," in target:
            normalized.extend(target.split(","))
        else:
            # Could be space-separated or single value
            normalized.extend(target.split())

    # Remove empty strings and strip whitespace
    return [t.strip() for t in normalized if t.strip()]


def create_package_config(args: argparse.Namespace) -> PackageConfig:
    """Create PackageConfig from command-line arguments.

    Parses and validates input arguments to build the configuration
    object used throughout the packaging process.

    Parameters:
        args: Parsed command-line arguments

    Returns:
        PackageConfig: Fully populated configuration object

    Raises:
        ValueError: If version string is invalid or package type is unsupported
    """
    dest_dir = Path(args.dest_dir).expanduser().resolve()
    normalized_targets = normalize_target_list(args.target)

    # Configure architecture based on multi-arch mode
    if args.enable_kpack:
        # Multi-arch: Build generic package + arch-specific packages for each target
        # Example: amdrocm-runtime (generic) + amdrocm-runtime-gfx94x + amdrocm-runtime-gfx1100
        default_gfx_arch = GFX_GENERIC
        gfxarch_list = normalized_targets
    else:
        # Single-arch: Build only one package for the specified target
        # Example: amdrocm-runtime-gfx94x (no generic, no other variants)
        default_gfx_arch = normalized_targets[0]
        gfxarch_list = []

    # Parse version for install prefix (major.minor)
    parts = args.rocm_version.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Version string '{args.rocm_version}' does not have major.minor versions"
        )
    major = re.match(r"^\d+", parts[0])
    minor = re.match(r"^\d+", parts[1])
    modified_rocm_version = f"{major.group()}.{minor.group()}"

    # Append version to default install prefix
    prefix = args.install_prefix
    if prefix == DEFAULT_INSTALL_PREFIX:
        prefix = f"{prefix}-{modified_rocm_version}"

    # Validate package type
    pkg_type = (args.pkg_type or "").lower()
    valid_types = {"deb", "rpm"}
    if pkg_type not in valid_types:
        raise ValueError(
            f"Invalid package type: {args.pkg_type}. Must be 'deb' or 'rpm'."
        )

    return PackageConfig(
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        dest_dir=dest_dir,
        pkg_type=pkg_type,
        rocm_version=args.rocm_version,
        version_suffix=args.version_suffix,
        install_prefix=prefix,
        gfx_arch=default_gfx_arch,
        enable_rpath=args.rpath_pkg,
        enable_kpack=args.enable_kpack,
        gfxarch_list=tuple(gfxarch_list),
    )


def run(args: argparse.Namespace):
    # Create configuration from arguments
    config = create_package_config(args)

    # Clean the packaging build directories
    clean_package_build_dir(config)

    pkg_list, skipped_list = parse_input_package_list(
        args.pkg_names, config.artifacts_dir
    )

    current_pkg_idx = 0
    try:
        built_pkglist = []
        failed_pkglist = []

        for current_pkg_idx, pkg_name in enumerate(pkg_list):
            print(f"Create {config.pkg_type} package.")

            pkg_info = get_package_info(pkg_name)
            # Check the package is marked as gfxarch package OR meta package
            if is_gfxarch_package(pkg_info, config.enable_kpack) or is_meta_package(
                pkg_info
            ):
                # Use all gfxarch values
                loop_list = list(config.gfxarch_list) + [config.gfx_arch]
            else:
                # Only use default architecture
                loop_list = [config.gfx_arch]

            pkg_built = False
            for gfxarch in loop_list:
                # Create new config with updated gfx_arch (config is immutable)
                build_config = replace(config, gfx_arch=gfxarch)
                if config.pkg_type == "rpm":
                    output_list = create_rpm_package(pkg_name, build_config)
                else:
                    output_list = create_deb_package(pkg_name, build_config)

                if output_list:
                    built_pkglist.extend(output_list)
                    pkg_built = True
                    print(f"Built package List: {built_pkglist}")
                else:
                    # Add failed architecture variant to failed list
                    variant_name = (
                        f"{pkg_name}-{gfxarch}"
                        if gfxarch != config.gfx_arch
                        else pkg_name
                    )
                    failed_pkglist.append(variant_name)

        # Clean the build directories
        clean_package_build_dir(config)

        pkglist_status = PackageList(
            total=pkg_list,
            built=built_pkglist,
            skipped=skipped_list,
            failed=failed_pkglist,
        )

        # Print build summary
        print_build_summary(config, pkglist_status)
    except SystemExit as e:
        # Build aborted somewhere inside create_* functions
        tb = traceback.extract_tb(sys.exc_info()[2])
        if tb:
            filename, line_no, func, text = tb[-1]
            print(f"\n❌ Build aborted due to an error at {filename}:{line_no}: {e}\n")
        else:
            print(f"\n❌ Build aborted due to an error: {e}\n")
        # Record failed package and all pending packages
        failed_pkglist.append(pkg_list[current_pkg_idx])
        pending_pkgs = pkg_list[current_pkg_idx + 1 :]
        failed_pkglist.extend(pending_pkgs)
        pkglist_status = PackageList(
            total=pkg_list,
            built=built_pkglist,
            skipped=skipped_list,
            failed=failed_pkglist,
        )
        print_build_summary(config, pkglist_status)
        # Stop the program
        raise


def main(argv: list[str]):

    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Specify the directory for source artifacts",
    )

    p.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Destination directory where the packages will be materialized",
    )
    p.add_argument(
        "--target",
        type=str,
        nargs="+",
        required=True,
        help="Graphics architecture(s) used for the artifacts (can specify multiple)",
    )

    p.add_argument(
        "--pkg-type",
        type=str,
        required=True,
        help="Choose the package format to be generated: DEB or RPM",
    )

    p.add_argument(
        "--rocm-version", type=str, required=True, help="ROCm Release version"
    )

    p.add_argument(
        "--version-suffix",
        type=str,
        nargs="?",
        help="Version suffix to append to package names",
    )

    p.add_argument(
        "--install-prefix",
        default=f"{DEFAULT_INSTALL_PREFIX}",
        help="Base directory where package will be installed",
    )

    p.add_argument(
        "--rpath-pkg",
        action="store_true",
        help="Enable rpath-pkg mode",
    )

    p.add_argument(
        "--enable-kpack",
        action="store_true",
        help="Enable multi-architecture package generation",
    )

    p.add_argument(
        "--clean-build",
        action="store_true",
        help="Clean the packaging environment",
    )

    p.add_argument(
        "--pkg-names",
        nargs="+",
        help="Specify the packages to be created",
    )

    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
