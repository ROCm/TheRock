#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""Given ROCm artifacts directories, performs packaging to
create RPM and DEB packages and upload to artifactory server

```
# Standard release build (auto-detection of targets from artifact directory):
./build_package.py --artifacts-dir ./ARTIFACTS_DIR  \
        --dest-dir ./OUTPUT_PKGDIR \
        --rocm-version 7.15.0 \
        --pkg-type deb (or rpm) \
        --version-suffix 28484694006

# ASAN build — package name gets -asan suffix, install prefix gets -asan-MAJOR.MINOR:
#   Package:        amdrocm-core-asan7.15
#   Install prefix: /opt/rocm/core-asan-7.15
./build_package.py --artifacts-dir ./ARTIFACTS_DIR  \
        --dest-dir ./OUTPUT_PKGDIR \
        --rocm-version 7.15.0 \
        --pkg-type deb (or rpm) \
        --version-suffix 28484694006 \
        --build-variant asan
```

--version-suffix: CI run ID appended to the DEB/RPM version field
  (e.g. '7.15.0-28484694006' for DEB, release='28484694006' for RPM).
  Does not affect the package name or install prefix.

--build-variant: Build type that modifies both the package name and
  install prefix. Currently supports 'asan'.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent

from packaging_summary import *
from packaging_utils import *
from runpath_to_rpath import *

from _therock_utils.artifacts import ArtifactCatalog
from _therock_utils.log_utils import (
    configure_logging,
    TheRockLogger,
    capture_console,
    github_group,
)

from deb_package import *
from rpm_package import *

logger = TheRockLogger(__name__)


# Default install prefix
DEFAULT_INSTALL_PREFIX = "/opt/rocm/core"


class BuildTask(NamedTuple):
    """Task for parallel package building via ProcessPoolExecutor."""

    pkg_name: str
    config: PackageConfig
    logs_dir: Path


def load_kpack_from_manifest(artifacts_dir: Path) -> bool:
    """Detect kpack mode by scanning therock_manifest.json files in artifact directory.

    Returns True if any manifest has KPACK_SPLIT_ARTIFACTS set to True.
    """
    for manifest_path in artifacts_dir.rglob("therock_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("flags", {}).get("KPACK_SPLIT_ARTIFACTS", False):
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


def get_all_target_families(artifact_dir):
    """Extract the list of GFX architectures from artifact directory.

    Auto-detects available GFX architectures by scanning the artifact directory
    for directories matching the pattern {name}_{component}_{target_family}.
    Used for CLI input detection when --target is not explicitly provided.

    Parameters:
        artifact_dir : The path to the Artifactory directory

    Returns:
        list : Sorted list of unique GFX architectures (e.g., ["gfx1100", "gfx942"])

    Raises:
        ValueError: If artifact directory does not exist
    """
    artifact_dir = Path(artifact_dir)

    if not artifact_dir.exists() or not artifact_dir.is_dir():
        raise ValueError(f"Artifact directory does not exist: {artifact_dir}")

    # Use ArtifactCatalog from _therock_utils to get all target families
    catalog = ArtifactCatalog(artifact_dir)
    # Strip xnack suffixes (e.g., gfx942:xnack+ -> gfx942)
    targets = {t.split(":", 1)[0] for t in catalog.all_target_families}
    return sorted(targets)


################### Package Variant Builders #######################
def build_package_variants(pkg_name, config: PackageConfig) -> list:
    """Build all required package variants based on package type and mode.

    This is the main entry point for building packages. It determines which
    variants to build based on:
    - Whether the package is a gfxarch package (GfxArch=True/False)
    - Whether kpack mode is enabled
    - Package type (DEB/RPM)

    Parameters:
        pkg_name: Name of the package to build
        config: Configuration object containing package metadata

    Returns:
        List of built package filenames

    Variants created:
        For GfxArch=False (non-gfxarch packages):
            - Versioned package (e.g., amdrocm-core8.2)
            - Non-versioned package (e.g., amdrocm-core)

        For GfxArch=True (gfxarch packages) in kpack mode (multi-arch):
            - Host package (e.g., amdrocm-fft-host8.2)
            - Device packages (e.g., amdrocm-fft8.2-gfx1100, amdrocm-fft8.2-gfx94x)
            - Meta package (e.g., amdrocm-fft8.2)
            - Non-versioned package (e.g., amdrocm-fft)

        For GfxArch=True (gfxarch packages) in single-arch mode:
            - Versioned package with arch suffix (e.g., amdrocm-fft8.2-gfx1100)
            - Non-versioned package with arch suffix (e.g., amdrocm-fft-gfx1100)
    """
    pkg_info = get_package_info(pkg_name)  # Raises ValueError if not found

    if config.enable_kpack:
        if is_gfxarch_package(pkg_info, config.enable_kpack, config.artifacts_dir):
            # GfxArch=True: host + devices + meta + non-versioned
            return build_gfxarch_package_variants(pkg_name, config)
        else:
            # GfxArch=False: versioned + non-versioned
            return build_simple_package_variants(pkg_name, config)
    else:
        # Single-arch mode
        return build_singlearch_package_variants(pkg_name, config)


def build_gfxarch_package_variants(pkg_name, config: PackageConfig) -> list:
    """Build all variants for a gfxarch package in kpack mode (multi-arch).

    For regular gfxarch packages (Metapackage=False), creates:
    - Host package (e.g., amdrocm-fft-host8.2) - generic artifacts
    - Device packages (e.g., amdrocm-fft8.2-gfx1100, amdrocm-fft8.2-gfx94x) - arch-specific artifacts
    - Meta package (e.g., amdrocm-fft8.2) - depends on host + all devices
    - Non-versioned package (e.g., amdrocm-fft) - user-facing, depends on meta

    For gfxarch metapackages (Metapackage=True + Gfxarch=True), creates:
    - Arch-specific meta packages (e.g., amdrocm-core8.2-gfx1100) - depends on actual packages
    - Generic meta package (e.g., amdrocm-core8.2) - depends on all arch-specific metas
    - Non-versioned package (e.g., amdrocm-core) - user-facing, depends on generic meta
    (No host package - metapackages have no artifacts to split)

    This function builds packages sequentially but is parallel-ready. Each variant
    builder is independent (no shared state, no cleanup during build) and can be
    run in parallel in the future. Cleanup is deferred until all variants complete.

    Parameters:
        pkg_name: Name of the package to build
        config: Configuration object

    Returns:
        List of built package filenames
    """
    built_packages = []
    pkg_info = get_package_info(pkg_name)
    is_meta = is_meta_package(pkg_info)

    # Host package (contains generic artifacts)
    # Skip for metapackages - they have no artifacts, only dependencies
    if not is_meta:
        logger.info(f"Building host variant for {pkg_name}")
        pkg = build_host_package(pkg_name, config)
        if pkg:
            built_packages.extend(pkg)

    # Device packages (one per architecture)
    # For metapackages, these become arch-specific meta packages
    for device_arch in config.gfxarch_list:
        logger.info(f"Building device variant for {pkg_name} ({device_arch})")
        pkg = build_device_package(pkg_name, config, device_arch)
        if pkg:
            built_packages.extend(pkg)

    # Meta package (depends on host + all devices for regular packages,
    # or depends on all arch-specific metas for metapackages)
    logger.info(f"Building meta variant for {pkg_name}")
    pkg = build_meta_package(pkg_name, config)
    if pkg:
        built_packages.extend(pkg)

    # Non-versioned package (user-facing, depends on meta)
    logger.info(f"Building non-versioned variant for {pkg_name}")
    # For gfxarch packages in kpack mode, non-versioned has no arch suffix (e.g., amdrocm-fft)
    # It depends on the meta package which pulls in host + all devices
    meta_config = replace(config, gfx_arch=GFX_META)
    pkg = build_nonversioned_package(pkg_name, meta_config)
    if pkg:
        built_packages.extend(pkg)

    return built_packages


def build_simple_package_variants(pkg_name, config: PackageConfig) -> list:
    """Build variants for a non-gfxarch package in kpack mode.

    Creates:
    - Versioned package (e.g., amdrocm-core8.2)
    - Non-versioned package with no arch suffix (e.g., amdrocm-core)

    Parameters:
        pkg_name: Name of the package to build
        config: Configuration object

    Returns:
        List of built package filenames
    """
    built_packages = []

    # Versioned package
    logger.info(f"Building versioned variant for {pkg_name}")
    pkg = build_versioned_package(pkg_name, config)
    if pkg:
        built_packages.extend(pkg)

    # Non-versioned package
    logger.info(f"Building non-versioned variant for {pkg_name}")
    # For non-gfxarch packages, non-versioned has no arch suffix (e.g., amdrocm-core)
    simple_config = replace(config, gfx_arch="")
    pkg = build_nonversioned_package(pkg_name, simple_config)
    if pkg:
        built_packages.extend(pkg)

    return built_packages


def build_singlearch_package_variants(pkg_name, config: PackageConfig) -> list:
    """Build package variants in single-arch mode (non-kpack).

    Creates:
    - Versioned package (e.g., amdrocm-core8.2 or amdrocm-fft8.2-gfx1100 for gfxarch packages)
    - Non-versioned package (e.g., amdrocm-core or amdrocm-fft-gfx1100 for gfxarch packages)

    In single-arch mode, gfxarch packages include the arch suffix in both variants
    because they're specific to that architecture.

    Parameters:
        pkg_name: Name of the package to build
        config: Configuration object

    Returns:
        List of built package filenames
    """
    built_packages = []

    # Versioned package
    logger.info(f"Building versioned package for {pkg_name} (single-arch mode)")
    try:
        versioned_config = replace(config, versioned_pkg=True)
        if versioned_config.pkg_type == "rpm":
            pkg = create_versioned_rpm_package(pkg_name, versioned_config)
        else:
            pkg = create_versioned_deb_package(pkg_name, versioned_config)
        if pkg:
            built_packages.extend(pkg)
    except Exception as e:
        logger.error(f"Failed to build versioned package for {pkg_name}: {e}")

    # Non-versioned package
    logger.info(f"Building non-versioned package for {pkg_name} (single-arch mode)")
    try:
        # In single-arch mode, non-versioned packages keep the arch suffix
        # to indicate they're specific to that architecture (e.g., amdrocm-fft-gfx1100)
        nonversioned_config = replace(config, versioned_pkg=False)
        if nonversioned_config.pkg_type == "rpm":
            pkg = create_nonversioned_rpm_package(pkg_name, nonversioned_config)
        else:
            pkg = create_nonversioned_deb_package(pkg_name, nonversioned_config)
        if pkg:
            built_packages.extend(pkg)
    except Exception as e:
        logger.error(f"Failed to build non-versioned package for {pkg_name}: {e}")

    return built_packages


def build_host_package(pkg_name, config: PackageConfig) -> list:
    """Build host package variant (contains generic artifacts).

    The host package contains architecture-independent artifacts and is named
    with a -host suffix (e.g., amdrocm-fft-host8.2).

    Parameters:
        pkg_name: Name of the package to build
        config: Configuration object

    Returns:
        List of built package filenames
    """
    host_config = replace(config, gfx_arch=GFX_HOST, versioned_pkg=True)
    try:
        if host_config.pkg_type == "rpm":
            return create_versioned_rpm_package(pkg_name, host_config)
        else:
            return create_versioned_deb_package(pkg_name, host_config)
    except Exception as e:
        logger.error(f"Failed to build host package for {pkg_name}: {e}")
        return []


def build_device_package(pkg_name, config: PackageConfig, device_arch: str) -> list:
    """Build device-specific package variant.

    Device packages contain architecture-specific artifacts and are named
    with an architecture suffix (e.g., amdrocm-fft8.2-gfx1100).

    Parameters:
        pkg_name: Name of the package to build
        config: Configuration object
        device_arch: Device architecture (e.g., "gfx1100", "gfx94x")

    Returns:
        List of built package filenames
    """
    device_config = replace(config, gfx_arch=device_arch, versioned_pkg=True)
    try:
        if device_config.pkg_type == "rpm":
            return create_versioned_rpm_package(pkg_name, device_config)
        else:
            return create_versioned_deb_package(pkg_name, device_config)
    except Exception as e:
        logger.error(
            f"Failed to build device package for {pkg_name} ({device_arch}): {e}"
        )
        return []


def build_meta_package(pkg_name, config: PackageConfig) -> list:
    """Build meta package that depends on host + all devices.

    The meta package is a versioned metapackage with no content, only dependencies.
    It pulls in the host package and all device packages (e.g., amdrocm-fft8.2).

    Parameters:
        pkg_name: Name of the package to build
        config: Configuration object

    Returns:
        List of built package filenames
    """
    meta_config = replace(config, gfx_arch=GFX_META, versioned_pkg=True)
    try:
        if meta_config.pkg_type == "rpm":
            return create_versioned_rpm_package(pkg_name, meta_config)
        else:
            return create_versioned_deb_package(pkg_name, meta_config)
    except Exception as e:
        logger.error(f"Failed to build meta package for {pkg_name}: {e}")
        return []


def build_versioned_package(pkg_name, config: PackageConfig) -> list:
    """Build versioned package (for non-gfxarch packages).

    This creates a versioned package with no architecture suffix,
    used for packages that don't have gfxarch variants (e.g., amdrocm-core8.2).

    Parameters:
        pkg_name: Name of the package to build
        config: Configuration object

    Returns:
        List of built package filenames
    """
    versioned_config = replace(config, gfx_arch="", versioned_pkg=True)
    try:
        if versioned_config.pkg_type == "rpm":
            return create_versioned_rpm_package(pkg_name, versioned_config)
        else:
            return create_versioned_deb_package(pkg_name, versioned_config)
    except Exception as e:
        logger.error(f"Failed to build versioned package for {pkg_name}: {e}")
        return []


def build_nonversioned_package(pkg_name, config: PackageConfig) -> list:
    """Build non-versioned package (user-facing metapackage).

    This creates a non-versioned metapackage that depends on the versioned variant.
    For gfxarch packages, it depends on the meta package. For non-gfxarch packages,
    it depends on the versioned package (e.g., amdrocm-fft -> amdrocm-fft8.2).

    IMPORTANT: Caller must set config.gfx_arch appropriately before calling:
    - GFX_META for gfxarch packages in kpack mode (e.g., "amdrocm-fft")
    - "" (empty string) for non-gfxarch packages in any mode (e.g., "amdrocm-core")
    - Actual arch (e.g., "gfx1100") for gfxarch packages in single-arch mode (e.g., "amdrocm-fft-gfx1100")

    Parameters:
        pkg_name: Name of the package to build
        config: Configuration object with gfx_arch already set by caller

    Returns:
        List of built package filenames
    """
    nonversioned_config = replace(config, versioned_pkg=False)
    try:
        if nonversioned_config.pkg_type == "rpm":
            return create_nonversioned_rpm_package(pkg_name, nonversioned_config)
        else:
            return create_nonversioned_deb_package(pkg_name, nonversioned_config)
    except Exception as e:
        logger.error(f"Failed to build non-versioned package for {pkg_name}: {e}")
        return []


def cleanup_build_directory(config: PackageConfig):
    """Clean up build directory after all package variants are built.

    Parameters:
    config: Configuration object containing dest_dir and pkg_type
    """
    build_dir = Path(config.dest_dir) / config.pkg_type
    if build_dir.exists():
        remove_dir(build_dir)
        logger.debug(f"Cleaned up build directory: {build_dir}")


def cleanup_packaging_environment(config: PackageConfig):
    """Clean the packaging environment (build directories and pycache).

    This is called at the start and end of the packaging run to ensure
    a clean environment. Unlike cleanup_build_directory(), this also
    removes Python cache files.

    Parameters:
    config: Configuration object containing package metadata

    Returns: None
    """
    logger.debug("cleanup_packaging_environment")
    PYCACHE_DIR = Path(SCRIPT_DIR) / "__pycache__"
    remove_dir(PYCACHE_DIR)

    # NOTE: Remove only the build directory
    # Make sure the destination directory is not removed
    remove_dir(Path(config.dest_dir) / config.pkg_type)
    # TBD:
    # Currently RPATH packages are created by modifying the artifacts dir
    # So artifacts dir clean up is required
    # remove_dir(artifacts_dir)


def parse_input_package_list(pkg_name, artifact_dir):
    """Populate the package list from the provided input arguments.

    Parameters:
    pkg_name : List of packages to be created
    artifact_dir: The path to the Artifactory directory

    Returns: Package list
    """
    logger.debug("parse_input_package_list")
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

    logger.debug(f"pkg_list: {pkg_list}")
    return pkg_list, skipped_list


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

    # Determine target architectures
    if args.target:
        # Use explicitly provided targets
        normalized_targets = normalize_target_list(args.target)
    else:
        # Auto-detect from artifact directory
        normalized_targets = get_all_target_families(args.artifacts_dir)
        if not normalized_targets:
            logger.warning(
                f"No GFX architectures found in artifact directory: {args.artifacts_dir}. "
                "Either provide --target explicitly or ensure artifacts are present."
            )
        else:
            logger.info(f"Auto-detected GFX architectures: {normalized_targets}\n")

    # Output packaging architecture list to GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output and normalized_targets:
        with open(github_output, "a", encoding="utf-8") as f:
            targets_str = ",".join(normalized_targets)
            f.write(f"PACKAGING_ARCH_LIST={targets_str}\n")

    # Auto-detect kpack from manifest if not explicitly requested via --enable-kpack
    artifacts_dir = Path(args.artifacts_dir).resolve()
    if not args.enable_kpack:
        args.enable_kpack = load_kpack_from_manifest(artifacts_dir)
        if args.enable_kpack:
            logger.info(
                "Detected KPACK_SPLIT_ARTIFACTS in manifest — producing host + device packages"
            )

    # Configure architecture based on multi-arch mode
    if args.enable_kpack:
        # Multi-arch: Build host + device + meta packages for each target
        # Example: amdrocm-fft-host8.2 + amdrocm-fft8.2-gfx94x + amdrocm-fft8.2 (meta)
        # For non-gfxarch packages: use empty string (no arch variants, just versioned + non-versioned)
        # For gfxarch packages: variant builders will set GFX_HOST, GFX_META, or device arch
        default_gfx_arch = ""  # Default used for non-gfxarch packages
        gfxarch_list = normalized_targets
    else:
        # Single-arch: Build only one package for the specified target
        # Example: amdrocm-fft8.2-gfx94x (no host, no other variants)
        default_gfx_arch = normalized_targets[0] if normalized_targets else ""
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

    # Append version (and build variant for ASan-family builds) to default
    # install prefix. Debug variants (asan-debug, host-asan-debug) collapse to
    # the same "-asan" prefix as their non-debug counterpart.
    # Release:  /opt/rocm/core-7.15
    # ASAN:     /opt/rocm/core-asan-7.15
    prefix = args.install_prefix
    if prefix == DEFAULT_INSTALL_PREFIX:
        if "asan" in args.build_variant:
            prefix = f"{prefix}-asan-{modified_rocm_version}"
        else:
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
        enable_kpack=args.enable_kpack,
        gfxarch_list=tuple(gfxarch_list),
        build_variant=args.build_variant,
    )


def build_pkg(task: BuildTask) -> tuple[str, list[str], str | None]:
    """Build a single package in isolated process. Used by ProcessPoolExecutor."""
    pkg_log_file = task.logs_dir / f"{task.config.pkg_type}-{task.pkg_name}.log"
    with tempfile.TemporaryDirectory(
        prefix=f"pkg_{task.pkg_name}_", ignore_cleanup_errors=True
    ) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        try:
            pkg_config = replace(task.config, dest_dir=temp_dir)
            with capture_console(pkg_log_file):
                output_list = build_package_variants(task.pkg_name, pkg_config)
            # Move packages from temp_dir to final dest_dir.
            # output_list contains plain filenames (e.g., "amdrocm-core-7.1.1.deb"), not paths.
            # This is guaranteed by move_packages_to_destination() which appends file_path.name.
            result: list[str] = []
            task.config.dest_dir.mkdir(parents=True, exist_ok=True)
            for f in output_list:
                src = temp_dir / f
                if src.exists():
                    dst = task.config.dest_dir / f
                    shutil.move(str(src), str(dst))
                    result.append(f)
            return (task.pkg_name, result, None)
        except SystemExit as e:
            return (task.pkg_name, [], f"SystemExit: exited with code {e.code}")
        except Exception as e:
            return (task.pkg_name, [], f"{type(e).__name__}: {e}")


def run(args: argparse.Namespace):
    # Create configuration from arguments
    config = create_package_config(args)

    # Convert RUNPATH to RPATH in artifacts before packaging (use --runpath-pkg to skip)
    if not args.runpath_pkg:
        print("\n=== Converting RUNPATH to RPATH in artifacts ===")
        convert_runpath_to_rpath(config.artifacts_dir)

    # Clean the packaging build directories
    cleanup_packaging_environment(config)

    pkg_list, skipped_list = parse_input_package_list(
        args.pkg_names, config.artifacts_dir
    )

    if not pkg_list:
        logger.error("No packages found to build. Package list is empty.")
        sys.exit(1)

    logs_dir = Path(config.dest_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    built_pkglist = []
    failed_pkglist = []

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(build_pkg, BuildTask(pkg, config, logs_dir)): pkg
            for pkg in pkg_list
        }
        for future in as_completed(futures):
            pkg_name = futures[future]
            try:
                _, output_list, error = future.result()
            except Exception as e:
                logger.error(f"Package {pkg_name} failed: {e}")
                failed_pkglist.append(pkg_name)
                continue

            if error:
                logger.error(f"Package {pkg_name} failed: {error}")
                failed_pkglist.append(pkg_name)
            elif output_list:
                built_pkglist.extend(output_list)
                logger.info(
                    f"Successfully built {len(output_list)} variant(s) for {pkg_name}"
                )
            else:
                failed_pkglist.append(pkg_name)
                logger.error(f"Failed to build any variants for {pkg_name}")

    # Clean the build directories
    cleanup_packaging_environment(config)

    if built_pkglist:
        logger.info(f"Built packages: {built_pkglist}")

    pkglist_status = PackageList(
        total=pkg_list,
        built=built_pkglist,
        skipped=skipped_list,
        failed=failed_pkglist,
    )

    # Print build summary
    print_build_summary(config, pkglist_status)


def main(argv: list[str]):

    p = argparse.ArgumentParser()
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level logging)",
    )
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
        required=False,
        help="Graphics architecture(s) used for the artifacts. "
        "Multiple targets can be specified space-separated, comma-separated, or semicolon-separated "
        "(e.g., 'gfx1100 gfx1101', 'gfx1100,gfx1101', or 'gfx1100;gfx1101'). "
        "If not provided, will auto-detect from artifact directory.",
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
        help=(
            "Release identifier appended to the package version field in DEB/RPM "
            "metadata (e.g. a CI run ID like '28484694006'). "
            "For DEB this becomes the debian revision (e.g. '7.15.0-28484694006'); "
            "for RPM this sets the release field. "
            "Does not affect the package name or install prefix."
        ),
    )

    p.add_argument(
        "--install-prefix",
        default=f"{DEFAULT_INSTALL_PREFIX}",
        help="Base directory where package will be installed",
    )

    p.add_argument(
        "--build-variant",
        default="",
        help=(
            "Build variant (e.g. 'asan', 'asan-debug', 'host-asan', "
            "'host-asan-debug'). When it contains 'asan', the install prefix "
            "becomes DEFAULT_INSTALL_PREFIX-asan-MAJOR.MINOR "
            "(e.g. /opt/rocm/core-asan-7.15)."
        ),
    )

    p.add_argument(
        "--runpath-pkg",
        action="store_true",
        help="Keep RUNPATH in binaries (by default, RUNPATH is converted to RPATH)",
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

    p.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel package builds (default: min(8, cpu_count))",
    )

    args = p.parse_args(argv)

    # Configure logging based on verbose flag
    configure_logging(verbose=args.verbose)

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
