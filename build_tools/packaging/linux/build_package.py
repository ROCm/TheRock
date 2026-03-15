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
from pathlib import Path

from deb_package import create_deb_package
from packaging_summary import *
from packaging_utils import *
from rpm_package import create_rpm_package


SCRIPT_DIR = Path(__file__).resolve().parent
# Default install prefix
DEFAULT_INSTALL_PREFIX = "/opt/rocm/core"


######################## Begin Packaging Process################################
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


def clean_package_build_dir(config: PackageConfig):
    """Clean the package build directories

    If artifactory directory is provided, clean the same as well

    Parameters:
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    PYCACHE_DIR = Path(SCRIPT_DIR) / "__pycache__"
    remove_dir(PYCACHE_DIR)

    # NOTE: Remove only the build directory
    # Make sure the destination directory is not removed
    remove_dir(Path(config.dest_dir) / config.pkg_type)
    # TBD:
    # Currently RPATH packages are created by modifying the artifacts dir
    # So artifacts dir clean up is required
    # remove_dir(artifacts_dir)


def run(args: argparse.Namespace):
    # Set the global variables
    dest_dir = Path(args.dest_dir).expanduser().resolve()

    # Split version passed to use only major and minor version for prefix folder
    # Split by dot and take first two components
    parts = args.rocm_version.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Version string '{args.rocm_version}' does not have major.minor versions"
        )
    major = re.match(r"^\d+", parts[0])
    minor = re.match(r"^\d+", parts[1])
    modified_rocm_version = f"{major.group()}.{minor.group()}"

    prefix = args.install_prefix

    # Append rocm version to default install prefix
    # TBD: Do we need to append rocm_version to other prefix?
    if prefix == DEFAULT_INSTALL_PREFIX:
        prefix = f"{prefix}-{modified_rocm_version}"

    # Populate package config details from user arguments
    config = PackageConfig(
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        dest_dir=Path(dest_dir),
        pkg_type=args.pkg_type,
        rocm_version=args.rocm_version,
        version_suffix=args.version_suffix,
        install_prefix=prefix,
        gfx_arch=args.target,
        enable_rpath=args.rpath_pkg,
    )

    # Clean the packaging build directories
    clean_package_build_dir(config)

    pkg_list, skipped_list = parse_input_package_list(
        args.pkg_names, config.artifacts_dir
    )
    # Create deb/rpm packages
    valid_types = {"deb", "rpm"}
    pkg_type = (config.pkg_type or "").lower()
    if pkg_type not in valid_types:
        raise ValueError(
            f"Invalid package type: {config.pkg_type}. Must be 'deb' or 'rpm'."
        )

    try:
        built_pkglist = []
        for pkg_name in pkg_list:
            print(f"Create {pkg_type} package.")
            if pkg_type == "rpm":
                output_list = create_rpm_package(pkg_name, config)
            else:
                output_list = create_deb_package(pkg_name, config)

            if output_list:
                built_pkglist.extend(output_list)
                print(f"Built package List: {built_pkglist}")

        # Clean the build directories
        clean_package_build_dir(config)

        pkglist_status = PackageList(
            total=pkg_list,
            built=built_pkglist,
            skipped=skipped_list,
        )

        # Print build summary
        print_build_summary(config, pkglist_status)
    except SystemExit:
        # Build aborted somewhere inside create_* functions
        print("\n❌ Build aborted due to an error.\n")
        pkglist_status = PackageList(
            total=pkg_list,
            built=built_pkglist,
            skipped=skipped_list,
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
        required=True,
        help="Graphics architecture used for the artifacts",
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
