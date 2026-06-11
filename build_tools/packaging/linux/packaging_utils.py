# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import json
import os
import platform
import re
import shutil
import sys

from dataclasses import dataclass, field, replace
from pathlib import Path


# Constants
# Used for creating host package in kpack mode (contains generic content)
GFX_HOST = "gfx_host"
# Used for creating versioned meta package in kpack mode (depends on host + devices)
GFX_META = "gfx_meta"


# User inputs required for packaging
# dest_dir - For saving the rpm/deb packages
# pkg_type - Package type DEB or RPM
# rocm_version - Used along with package name
# version_suffix - Used along with package name
# install_prefix - Install prefix for the package
# gfx_arch - gfxarch used for building package
# enable_rpath - To enable RPATH packages
# versioned_pkg - Used to indicate versioned or non versioned packages
# enable_kpack - To enable multi-architecture support
# gfxarch_list - List of all architectures for multi-arch mode
#
# frozen=True makes this dataclass immutable (hashable and thread-safe).
# Note: gfxarch_list uses tuple instead of list because frozen dataclasses
# require all fields to be immutable types (tuples are immutable, lists are not).
@dataclass(frozen=True)
class PackageConfig:
    artifacts_dir: Path
    dest_dir: Path
    pkg_type: str
    rocm_version: str
    version_suffix: str
    install_prefix: str
    gfx_arch: str
    enable_rpath: bool = False
    versioned_pkg: bool = True
    enable_kpack: bool = False
    gfxarch_list: tuple = field(default_factory=tuple)


SCRIPT_DIR = Path(__file__).resolve().parent
currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name


def print_function_name():
    """Print the name of the calling function.

    Parameters: None

    Returns: None
    """
    print("In function:", currentFuncName(1))


def read_package_json_file():
    """Reads package.json file and return the parsed data.

    Parameters: None

    Returns: Parsed JSON data containing package details
    """
    file_path = SCRIPT_DIR / "package.json"
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def is_key_defined(pkg_info, key):
    """
    Verifies whether a specific key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.
    key : A key to be searched in the dictionary.

    Returns:
    bool: True if key is defined, False otherwise.
    """
    value = ""
    for k in pkg_info:
        if k.lower() == key.lower():
            value = pkg_info[k]

    value = value.strip().lower()
    return value in ("1", "true", "t", "yes", "y", "on", "enable", "enabled", "found")


def is_postinstallscripts_available(pkg_info):
    """
    Verifies whether Postinstall key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if Postinstall key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Postinstall")


def is_meta_package(pkg_info):
    """
    Verifies whether Metapackage key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if Metapackage key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Metapackage")


def is_rpm_stripping_disabled(pkg_info):
    """
    Verifies whether Disable_RPM_STRIP key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if Disable_RPM_STRIP key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Disable_RPM_STRIP")


def is_debug_package_disabled(pkg_info):
    """
    Verifies whether Disable_Debug_Package key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if Disable_Debug_Package key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Disable_Debug_Package")


def is_packaging_disabled(pkg_info):
    """
    Verifies whether 'Disablepackaging' key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if 'Disablepackaging' key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Disablepackaging")


def is_gfxarch_package(pkg_info, enable_kpack=False):
    """Check whether the package is associated with a graphics architecture

    Parameters:
    pkg_info (dict): A dictionary containing package details.
    enable_kpack (bool): Enable multi-architecture support.

    Returns:
    bool : True if Gfxarch is set, else False.
           False if devel package when enable_kpack is True
    """
    if enable_kpack:
        pkgname = pkg_info.get("Package", "")
        # Only non-metapackage -devel should be non-gfxarch
        # Metapackages like amdrocm-core-devel should create arch-specific variants
        if pkgname.endswith("-devel") and not is_meta_package(pkg_info):
            return False

        # Override RCCL Gfxarch behavior in kpack mode
        # When --enable-kpack is used, RCCL should look for architecture-specific artifacts
        # instead of generic artifacts to ensure GPU-specific kernel support (e.g., gfx1201)
        if pkgname in ["amdrocm-rccl", "amdrocm-rccl-test"]:
            return True

    return is_key_defined(pkg_info, "Gfxarch")


def get_package_info(pkgname, raise_if_missing=True):
    """Retrieves package details from a JSON file for the given package name

    Parameters:
    pkgname : Package Name
    raise_if_missing : If True, raise ValueError when package not found.
                       If False, return None (for backward compatibility)

    Returns: Package metadata dictionary, or None if not found and raise_if_missing=False

    Raises:
    ValueError: If package not found and raise_if_missing=True
    """

    # Load JSON data from a file
    data = read_package_json_file()

    for package in data:
        if package.get("Package") == pkgname:
            return package

    # Package not found
    if raise_if_missing:
        raise ValueError(
            f"Package '{pkgname}' not found in package.json. "
            f"Please verify the package name and ensure it's defined in the package configuration."
        )
    return None


def get_package_list(artifact_dir):
    """Read package.json and return a list of package names.

    Packages marked as 'Disablepackaging' are excluded.
    If the entire Artifactory directory is missing, the package is excluded
    unless it is a metapackage.

    Parameters:
        artifact_dir : The path to the Artifactory directory

    Returns:
    pkg_list : list of package names that will be packaged
    skipped_list  : list of package names excluded due to missing artifacts
    """
    pkg_list = []
    skipped_list = []
    artifact_path = Path(artifact_dir)
    data = read_package_json_file()

    try:
        artifact_dirs = {path.name for path in artifact_path.iterdir() if path.is_dir()}
    except FileNotFoundError:
        sys.exit(f"{artifact_dir}: Artifactory directory does not exist, exiting")

    # Create a prefix index for O(1) artifact lookup
    prefix_index = {}
    # Component suffixes from artifact.toml [components.{suffix}.*] definitions
    # See docs/development/artifacts.md for artifact naming conventions
    SUFFIX_MARKERS = ["_dbg_", "_dev_", "_doc_", "_lib_", "_run_", "_test_"]

    for dirname in artifact_dirs:
        for marker in SUFFIX_MARKERS:
            if marker in dirname:
                prefix = dirname.split(marker, 1)[0]
                prefix_index[prefix] = True
                break  # stop after first matching marker

    for pkg_info in data:
        pkg_name = pkg_info["Package"]
        # Skip disabled packages
        if is_packaging_disabled(pkg_info):
            continue

        # Metapackages do not need artifact lookup
        if is_meta_package(pkg_info):
            pkg_list.append(pkg_name)
            continue

        # Check if any artifact matches a known prefix
        artifact_found = any(
            (artifact := art.get("Artifact")) and prefix_index.get(artifact)
            for art in pkg_info.get("Artifactory", [])
        )

        if artifact_found:
            pkg_list.append(pkg_name)
        else:
            skipped_list.append(pkg_name)

    return pkg_list, skipped_list


def remove_dir(dir_name):
    """Remove the directory if it exists

    Parameters:
    dir_name : Path or str
        Directory to be removed

    Returns: None
    """
    dir_path = Path(dir_name)

    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
        print(f"Removed directory: {dir_path}")
    else:
        print(f"Directory does not exist: {dir_path}")


def update_package_name(pkg_name, config: PackageConfig):
    """Update the package name by adding ROCm version and graphics architecture.

    Based on conditions, the function may append:
    - ROCm version
    - '-rpath'
    - Graphics architecture (gfxarch)

    Parameters:
    pkg_name : Package name
    config: Configuration object containing package metadata

    Returns: Updated package name
    """
    print_function_name()

    pkg_suffix = ""
    if config.versioned_pkg:
        # Split version passed to use only major and minor version for package name
        # Split by dot and take first two components
        # Package name will be rocm8.1 and discard all other version part
        parts = config.rocm_version.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Version string '{config.rocm_version}' does not have major.minor versions"
            )
        major = re.match(r"^\d+", parts[0])
        minor = re.match(r"^\d+", parts[1])
        pkg_suffix = f"{major.group()}.{minor.group()}"

    if config.enable_rpath:
        pkg_suffix = f"-rpath{pkg_suffix}"

    pkg_info = get_package_info(pkg_name)
    updated_pkgname = pkg_name
    if config.pkg_type.lower() == "deb":
        updated_pkgname = debian_replace_devel_name(pkg_name)

    # For GFX_HOST in kpack mode, add "-host" before version suffix
    # Result: amdrocm-fft-host8.2 (not amdrocm-fft8.2-host)
    if (
        config.enable_kpack
        and is_gfxarch_package(pkg_info, config.enable_kpack)
        and config.gfx_arch == GFX_HOST
    ):
        updated_pkgname += "-host"

    updated_pkgname += pkg_suffix

    if is_gfxarch_package(pkg_info, config.enable_kpack):
        if config.enable_kpack:
            if config.gfx_arch == GFX_HOST:
                # Host package: "-host" already added before version
                pass
            elif config.gfx_arch == GFX_META:
                # Meta package: no arch suffix (e.g., amdrocm-fft8.2)
                pass
            else:
                # Device package: add gfx arch suffix (e.g., amdrocm-fft8.2-gfx1100)
                gfx_arch = config.gfx_arch.lower().split("-", 1)[0]
                updated_pkgname += "-" + gfx_arch
        else:
            # Single-arch mode: add gfx arch suffix
            gfx_arch = config.gfx_arch.lower().split("-", 1)[0]
            updated_pkgname += "-" + gfx_arch

    return updated_pkgname


def expand_metapackage_to_all_archs(pkg_name, gfxarch_list, config: PackageConfig):
    """Expand a generic metapackage dependency to include all architecture-specific variants.

    For example, if pkg_name is "amdrocm-core" and gfxarch_list is ["gfx94x", "gfx1150"],
    this returns a list: ["amdrocm-core-gfx94x", "amdrocm-core-gfx1150"]

    Parameters:
    pkg_name: Base package name (e.g., "amdrocm-core")
    gfxarch_list: List of architecture targets
    config: Configuration object containing package metadata

    Returns: List of architecture-specific package names
    """
    arch_specific_packages = []

    # Filter archs to only those with artifacts
    filtered_archs = filter_archs_with_artifacts(
        pkg_name, gfxarch_list, config.artifacts_dir
    )

    for gfx_arch in filtered_archs:
        # Create new config for each arch with versioned_pkg=True
        local_config = replace(config, versioned_pkg=True, gfx_arch=gfx_arch)
        # update_package_name will append version and gfx_arch
        arch_pkg = update_package_name(pkg_name, local_config)
        arch_specific_packages.append(arch_pkg)

    return arch_specific_packages


def expand_kpack_meta_dependencies(pkg_name, gfxarch_list, config: PackageConfig):
    """Get dependencies for kpack versioned meta package: host + all device packages.

    For example, if pkg_name is "amdrocm-fft" and gfxarch_list is ["gfx1100", "gfx1101"],
    this returns a list: ["amdrocm-fft-host8.2", "amdrocm-fft8.2-gfx1100", "amdrocm-fft8.2-gfx1101"]

    Parameters:
    pkg_name: Base package name (e.g., "amdrocm-fft")
    gfxarch_list: List of architecture targets
    config: Configuration object containing package metadata

    Returns: List of package names (host + all device packages)
    """
    packages = []

    # Add host package (with -host suffix)
    host_config = replace(config, versioned_pkg=True, gfx_arch=GFX_HOST)
    host_pkg = update_package_name(pkg_name, host_config)
    packages.append(host_pkg)

    # Filter archs to only those with artifacts
    filtered_archs = filter_archs_with_artifacts(
        pkg_name, gfxarch_list, config.artifacts_dir
    )

    # Add arch-specific (device) packages only for available architectures
    for gfx_arch in filtered_archs:
        arch_config = replace(config, versioned_pkg=True, gfx_arch=gfx_arch)
        arch_pkg = update_package_name(pkg_name, arch_config)
        packages.append(arch_pkg)

    return packages


def debian_replace_devel_name(pkg_name):
    """Replace '-devel' with '-dev' in the package name.

    Development package names are defined as -devel in json file
    For Debian packages -dev should be used instead.

    Parameters:
    pkg_name : Package name

    Returns: Updated package name
    """
    print_function_name()
    # Required for debian developement package
    suffix = "-devel"
    if pkg_name.endswith(suffix):
        pkg_name = pkg_name[: -len(suffix)] + "-dev"

    return pkg_name


def process_name_field(
    pkg_info: dict,
    field_key: str,
    transform_fn=None,
) -> str:
    """Process a name field: get -> transform -> join.

    For non-dependency fields: Provides, Replaces, Conflicts, Obsoletes

    Parameters:
    pkg_info: Package details from JSON
    field_key: Key to extract (e.g., "Provides", "Conflicts")
    transform_fn: Optional function to transform each name

    Returns: Comma-separated string of names
    """
    name_list = pkg_info.get(field_key, []) or []
    if transform_fn:
        name_list = [transform_fn(name) for name in name_list]
    return ", ".join(name_list)


def process_main_dependencies(
    pkg_info: dict, field_key: str, config: PackageConfig
) -> str:
    """Process main dependency field (DEBDepends/RPMRequires).

    Delegates to kpack or single-arch handler based on build mode.

    Parameters:
    pkg_info: Package details from JSON
    field_key: Key to extract ("DEBDepends" or "RPMRequires")
    config: Configuration object containing package metadata

    Returns: Comma-separated string of versioned dependencies
    """
    if config.enable_kpack:
        return process_main_dependencies_kpack(pkg_info, field_key, config)
    else:
        return process_main_dependencies_single_arch(pkg_info, field_key, config)


def process_main_dependencies_kpack(
    pkg_info: dict, field_key: str, config: PackageConfig
) -> str:
    """Process main dependencies for kpack (multi-arch) mode.

    Handles:
    - Meta packages: depend on all arch-specific variants
    - Host packages: depend on non-gfxarch packages only
    - Device packages: depend on host + arch-specific gfxarch packages
    - GFX_META packages: depend on host + all device variants

    Parameters:
    pkg_info: Package details from JSON
    field_key: Key to extract ("DEBDepends" or "RPMRequires")
    config: Configuration object containing package metadata

    Returns: Comma-separated string of versioned dependencies
    """
    is_meta = is_meta_package(pkg_info)
    pkg_name = pkg_info.get("Package")

    if is_meta:
        if config.gfx_arch == GFX_META:
            # Meta package: depend on all arch-specific metapackages
            dep_list = expand_metapackage_to_all_archs(
                pkg_name, config.gfxarch_list, config
            )
        else:
            # Arch-specific metapackage: depend on actual runtime packages
            dep_list = pkg_info.get(field_key, [])
            # Filter deps without artifacts
            dep_list = filter_dependencies_by_artifacts(
                dep_list, config.artifacts_dir, config.gfx_arch
            )
    elif config.gfx_arch == GFX_META:
        # GFX_META for non-meta gfxarch packages: depend on host + all device packages
        dep_list = expand_kpack_meta_dependencies(pkg_name, config.gfxarch_list, config)
    elif config.gfx_arch == GFX_HOST:
        # Host package: only include non-gfxarch dependencies
        # Gfxarch deps are pulled via the gfx-specific package
        dep_list = pkg_info.get(field_key, [])
        dep_list = [
            dep
            for dep in dep_list
            if not is_gfxarch_package(
                get_package_info(dep, raise_if_missing=False) or {}, config.enable_kpack
            )
        ]
        # Filter deps without artifacts
        dep_list = filter_dependencies_by_artifacts(
            dep_list, config.artifacts_dir, config.gfx_arch
        )
    elif not is_gfxarch_package(pkg_info, config.enable_kpack):
        # Non-gfxarch versioned package: use all dependencies directly
        # These packages don't have host/device split, so include everything
        dep_list = pkg_info.get(field_key, [])
        # Filter deps without artifacts
        dep_list = filter_dependencies_by_artifacts(
            dep_list, config.artifacts_dir, config.gfx_arch
        )
    else:
        # Device package: depend on host package + gfxarch dependencies with arch suffix
        dep_list = pkg_info.get(field_key, [])
        gfxarch_deps = [
            dep
            for dep in dep_list
            if is_gfxarch_package(
                get_package_info(dep, raise_if_missing=False) or {}, config.enable_kpack
            )
        ]
        # Filter deps without artifacts
        gfxarch_deps = filter_dependencies_by_artifacts(
            gfxarch_deps, config.artifacts_dir, config.gfx_arch
        )
        dep_list = [pkg_name] + gfxarch_deps

    if not dep_list:
        return ""
    return resolve_versioned_dependencies(dep_list, config, is_meta)


def process_main_dependencies_single_arch(
    pkg_info: dict, field_key: str, config: PackageConfig
) -> str:
    """Process main dependencies for single-arch mode.

    Simple case: use full dependency list from package.json and add version suffixes.

    Parameters:
    pkg_info: Package details from JSON
    field_key: Key to extract ("DEBDepends" or "RPMRequires")
    config: Configuration object containing package metadata

    Returns: Comma-separated string of versioned dependencies
    """
    is_meta = is_meta_package(pkg_info)
    dep_list = pkg_info.get(field_key, []) or []

    if not dep_list:
        return ""
    return resolve_versioned_dependencies(dep_list, config, is_meta)


def process_secondary_dependencies(
    pkg_info: dict, field_key: str, config: PackageConfig
) -> str:
    """Process secondary dependency fields (Recommends/Suggests).

    Simple processing: get from JSON and add version suffixes.
    Works the same for both kpack and single-arch modes.

    Parameters:
    pkg_info: Package details from JSON
    field_key: Key to extract (e.g., "DEBRecommends", "RPMSuggests")
    config: Configuration object containing package metadata

    Returns: Comma-separated string of versioned dependencies
    """
    is_meta = is_meta_package(pkg_info)
    dep_list = pkg_info.get(field_key, []) or []

    if not dep_list:
        return ""
    return resolve_versioned_dependencies(dep_list, config, is_meta)


def convert_to_versiondependency(
    dependency_list, config: PackageConfig, preserve_arch=False
):
    """Change ROCm package dependencies to versioned ones.

    If a package depends on any packages listed in `pkg_list`,
    this function appends the dependency name with the specified ROCm version.

    Parameters:
    dependency_list : List of dependent packages
    config: Configuration object containing package metadata
    preserve_arch: If True, preserve the gfx_arch from config instead of forcing generic

    Returns: A string of comma separated versioned packages
    """
    print_function_name()
    # This function is to add Version dependency
    # Make sure the flag is set to True

    # Create config with versioned_pkg=True and conditionally override gfx_arch
    if config.enable_kpack and not preserve_arch:
        if not config.versioned_pkg:
            # Non-versioned package depends on versioned meta package
            # e.g., amdrocm-fft -> amdrocm-fft8.2
            local_config = replace(config, versioned_pkg=True, gfx_arch=GFX_META)
        else:
            # Versioned packages depend on host packages
            # e.g., amdrocm-fft8.2-gfx1100 -> amdrocm-fft-host8.2
            local_config = replace(config, versioned_pkg=True, gfx_arch=GFX_HOST)
    else:
        local_config = replace(config, versioned_pkg=True)

    pkg_list, skipped_list = get_package_list(config.artifacts_dir)

    filtered_deps = []
    # Remove amdrocm* packages that are NOT in pkg_list
    for pkg in dependency_list:
        if not (pkg.startswith("amdrocm") and pkg not in pkg_list):
            filtered_deps.append(pkg)

    updated_depends = [
        f"{update_package_name(pkg,local_config)}" if pkg in pkg_list else pkg
        for pkg in filtered_deps
    ]
    depends = ", ".join(updated_depends)
    return depends


def append_version_suffix(dep_string, config: PackageConfig):
    """Append a ROCm version suffix to dependency names that match known ROCm packages.

    This function takes a comma-separated dependency string,
    identifies which dependencies correspond to packages listed in `pkg_list`,
    and appends the appropriate ROCm version suffix based on the provided configuration.

    Parameters:
    dep_string : A comma-separated list of dependency package names.
    config : Configuration object containing ROCm version, suffix, and packaging type.

    Returns: A comma-separated string where matching dependencies include the version suffix,
    while all others remain unchanged.
    """
    print_function_name()

    pkg_list, skipped_list = get_package_list(config.artifacts_dir)
    updated_depends = []
    dep_list = [d.strip() for d in dep_string.split(",")]

    for dep in dep_list:
        match = None
        # find a matching package prefix
        for pkg in pkg_list:
            if dep.startswith(pkg):
                match = pkg
                break

        # If matched, append version-suffix; otherwise keep original
        if match:
            version = str(config.rocm_version)
            suffix = f"-{config.version_suffix}" if config.version_suffix else ""

            if config.pkg_type.lower() == "deb":
                dep += f"( = {version}{suffix})"
            else:
                dep += f" = {version}{suffix}"

        updated_depends.append(dep)

    depends = ", ".join(updated_depends)
    return depends


def move_packages_to_destination(updated_pkg_name, config: PackageConfig):
    """Move the generated package from the build directory to the destination directory.

    This function is parallel-safe because it uses exact package name matching
    rather than glob patterns that could match multiple variants.

    Parameters:
    updated_pkg_name : Updated package name (e.g., "amdrocm-fft-host8.2", "amdrocm-fft8.2-gfx1100")
                       Should be the result of update_package_name(pkg_name, config)
    config: Configuration object containing package metadata

    Returns:
    output_packages : list of package names moved to the destination folder
    """
    print_function_name()
    output_packages = []
    # Create destination dir to move the packages created
    os.makedirs(config.dest_dir, exist_ok=True)
    print(f"Updated package name: {updated_pkg_name}")
    PKG_DIR = Path(config.dest_dir) / config.pkg_type

    if config.pkg_type.lower() == "deb":
        artifacts = list(PKG_DIR.glob("*.deb"))
        # DEB filename format: {updated_pkg_name}_{version}_{arch}.deb
        # Example: amdrocm-fft-host8.2_8.2.0-12345_amd64.deb
        prefix = f"{updated_pkg_name}_"
    else:
        artifacts = list(PKG_DIR.glob(f"*/RPMS/{platform.machine()}/*.rpm"))
        # RPM filename format: {updated_pkg_name}-{version}-{release}.{arch}.rpm
        # Example: amdrocm-fft-host8.2-8.2.0-12345.x86_64.rpm
        prefix = f"{updated_pkg_name}-"

    # Move deb/rpm files to the destination directory
    for file_path in artifacts:
        file_path = Path(file_path)  # ensure it's a Path object
        file_name = file_path.name  # basename equivalent

        if file_name.startswith(prefix):
            dest_file = Path(config.dest_dir) / file_name

            # if file exists, remove it first
            if dest_file.exists():
                dest_file.unlink()

            shutil.move(str(file_path), str(config.dest_dir))
            output_packages.append(file_name)

    return output_packages


def filter_components_fromartifactory(
    pkg_name, artifacts_dir, gfx_arch, enable_kpack=False
):
    """Get the list of Artifactory directories required for creating the package.

    The `package.json` file defines the required artifactories for each package.

    Parameters:
    pkg_name : package name
    artifacts_dir : Directory where artifacts are saved
    gfx_arch : graphics architecture
    enable_kpack : enable multi-architecture support

    Returns: List of directories
    """
    print_function_name()

    pkg_info = get_package_info(pkg_name)
    sourcedir_list = []

    if enable_kpack:
        # GFX_META is a meta package with no artifacts
        if gfx_arch == GFX_META:
            return sourcedir_list  # Return empty list for meta package
        # GFX_HOST uses "generic" artifacts
        if gfx_arch == GFX_HOST:
            dir_suffix = "generic"
        elif is_gfxarch_package(pkg_info, enable_kpack):
            dir_suffix = gfx_arch
        else:
            dir_suffix = "generic"
    else:
        dir_suffix = (
            gfx_arch if is_gfxarch_package(pkg_info, enable_kpack) else "generic"
        )

    artifactory = pkg_info.get("Artifactory")
    if artifactory is None:
        print(
            f'The "Artifactory" key is missing for {pkg_name}. Is this a meta package?'
        )
        return sourcedir_list

    for artifact in artifactory:
        artifact_prefix = artifact["Artifact"]
        # Package specific key: "Gfxarch"
        # Artifact specific key: "Artifact_Gfxarch"
        # If "Artifact_Gfxarch" key is specified use it for artifact directory suffix
        # Else use the package "Gfxarch" for finding the suffix
        if "Artifact_Gfxarch" in artifact:
            print(f"{pkg_name} : Artifact_Gfxarch key exists for artifacts {artifact}")
            is_gfxarch = str(artifact["Artifact_Gfxarch"]).lower() == "true"

            # In kpack mode, skip non-gfxarch artifacts when building gfx-specific packages
            # This prevents generic artifacts from being included in both base and arch-specific packages
            if enable_kpack and gfx_arch not in (GFX_HOST, GFX_META) and not is_gfxarch:
                print(
                    f"{pkg_name} : Skipping artifact '{artifact_prefix}' for {gfx_arch} package "
                    f"(Artifact_Gfxarch=False, should only be in generic package)"
                )
                continue

            artifact_suffix = gfx_arch if is_gfxarch else "generic"
        else:
            artifact_suffix = dir_suffix

        for subdir in artifact["Artifact_Subdir"]:
            artifact_subdir = subdir["Name"]
            component_list = subdir["Components"]

            for component in component_list:
                source_dir = (
                    Path(artifacts_dir)
                    / f"{artifact_prefix}_{component}_{artifact_suffix}"
                )
                filename = source_dir / "artifact_manifest.txt"
                if not filename.exists():
                    print(f"{pkg_name} : Missing {filename}")
                    continue
                try:
                    with filename.open("r", encoding="utf-8") as file:
                        for line in file:

                            match_found = (
                                isinstance(artifact_subdir, str)
                                and (artifact_subdir.lower() + "/") in line.lower()
                            )

                            if match_found and line.strip():
                                print("Matching line:", line.strip())
                                source_path = source_dir / line.strip()
                                sourcedir_list.append(source_path)
                except OSError as e:
                    print(f"Could not read manifest {filename}: {e}")
                    continue

    return sourcedir_list


def resolve_versioned_dependencies(dep_list, config: PackageConfig, is_meta):
    """Resolve a dependency list into a versioned dependency string.

    Handles three cases based on multi-arch mode and package type:
    - Generic metapackages in multi-arch mode: dependencies are already expanded
      and versioned, so just join and add version suffix.
    - Arch-specific metapackages in multi-arch mode: convert dependencies while
      preserving architecture, then add version suffix.
    - Normal path: convert dependencies and conditionally add version suffix
      for metapackages.

    Parameters:
    dep_list: List of dependency package names
    config: Configuration object containing package metadata
    is_meta: Whether this is a metapackage

    Returns: A comma-separated string of versioned dependencies
    """
    if config.versioned_pkg and config.enable_kpack and config.gfx_arch == GFX_META:
        # GFX_META: versioned meta package depends on host + all devices
        # dep_list already contains versioned arch-specific package names
        # Just add version suffix and join
        deps = append_version_suffix(", ".join(dep_list), config)
    elif (
        config.enable_kpack and is_meta and config.gfx_arch not in (GFX_HOST, GFX_META)
    ):
        # Arch-specific metapackage: preserve architecture for gfxarch dependencies
        # For -devel metapackages: mix arch-specific (gfxarch) and generic (non-gfxarch)
        result_deps = []
        for dep in dep_list:
            dep_info = get_package_info(dep)
            # Only gfxarch dependencies get arch suffix, others stay generic
            preserve = dep_info and is_gfxarch_package(dep_info, config.enable_kpack)
            versioned = convert_to_versiondependency(
                [dep], config, preserve_arch=preserve
            )
            if versioned:  # Filter out empty strings from missing packages
                result_deps.append(versioned)

        deps = ", ".join(result_deps)
        deps = append_version_suffix(deps, config)
    elif (
        config.enable_kpack
        and not is_meta
        and config.gfx_arch not in (GFX_HOST, GFX_META)
    ):
        # Gfx-specific non-meta package:
        # dep_list[0] is the versioned-dependency (resolved as generic)
        # dep_list[1:] are gfxarch dependencies (resolved with arch suffix)
        if not dep_list:
            deps = ""
        else:
            version_deps = convert_to_versiondependency([dep_list[0]], config)
            if len(dep_list) > 1:
                gfx_deps = convert_to_versiondependency(
                    dep_list[1:], config, preserve_arch=True
                )
                deps = f"{version_deps}, {gfx_deps}"
            else:
                deps = version_deps
    else:
        # Normal path: convert dependencies and add version suffix
        deps = convert_to_versiondependency(dep_list, config)
        if is_meta:
            deps = append_version_suffix(deps, config)
    return deps


def has_artifact_for_arch(pkg_name, artifacts_dir, gfx_arch):
    """Check if a package has artifacts available for a specific architecture.

    Parameters:
    pkg_name: Package name to check
    artifacts_dir: Directory where artifacts are stored
    gfx_arch: Graphics architecture to check for

    Returns: True if artifacts exist for the architecture, False otherwise
    """
    pkg_info = get_package_info(pkg_name)
    if pkg_info is None:
        return False

    # Non-gfxarch packages do not need arch-specific artifacts
    if not is_gfxarch_package(pkg_info, enable_kpack=True):
        return True

    # Meta packages do not have their own artifacts
    if is_meta_package(pkg_info):
        return True

    artifactory = pkg_info.get("Artifactory")
    if artifactory is None:
        return False

    # Check if at least one required artifact directory exists for this architecture
    for artifact in artifactory:
        artifact_prefix = artifact["Artifact"]
        # Check for artifact-specific gfxarch override
        if "Artifact_Gfxarch" in artifact:
            is_gfxarch = str(artifact["Artifact_Gfxarch"]).lower() == "true"
            artifact_suffix = gfx_arch if is_gfxarch else "generic"
        else:
            artifact_suffix = gfx_arch

        # When checking for a specific gfx architecture (not host/meta),
        # skip generic-only artifacts - they do not contribute to gfx-specific packages
        if gfx_arch not in (GFX_HOST, GFX_META) and artifact_suffix == "generic":
            continue

        for subdir in artifact["Artifact_Subdir"]:
            artifact_subdir = subdir["Name"]
            component_list = subdir["Components"]
            for component in component_list:
                source_dir = (
                    Path(artifacts_dir)
                    / f"{artifact_prefix}_{component}_{artifact_suffix}"
                )
                if not source_dir.exists():
                    continue

                # Check if the required subdirectory exists in the manifest
                manifest_file = source_dir / "artifact_manifest.txt"
                if not manifest_file.exists():
                    continue

                try:
                    with manifest_file.open("r", encoding="utf-8") as file:
                        for line in file:
                            match_found = (
                                isinstance(artifact_subdir, str)
                                and (artifact_subdir.lower() + "/") in line.lower()
                            )
                            if match_found and line.strip():
                                # Found at least one required subdirectory in the manifest
                                return True
                except OSError:
                    continue

    return False


def filter_archs_with_artifacts(
    pkg_name: str, gfxarch_list, artifacts_dir: Path
) -> list:
    """Filter architecture list to only those with available artifacts.

    This function prevents meta packages from depending on device packages that
    do not exist because their artifacts were not built.

    Parameters:
    pkg_name: Package name to check (e.g., "amdrocm-ck")
    gfxarch_list: Full list of architecture targets to filter
    artifacts_dir: Directory where artifacts are stored

    Returns: Filtered list of architectures that have artifacts available

    Example:
        Input:  ["gfx1100", "gfx1101", "gfx942"], pkg_name="amdrocm-ck"
        Output: ["gfx1100", "gfx942"]  # if gfx1101 has no artifacts for ck
    """
    pkg_info = get_package_info(pkg_name, raise_if_missing=False)
    if pkg_info is None:
        return list(gfxarch_list)

    # Non-gfxarch packages do not have arch-specific variants
    if not is_gfxarch_package(pkg_info, enable_kpack=True):
        return list(gfxarch_list)

    # Meta packages inherit from their dependencies, return all archs
    if is_meta_package(pkg_info):
        return list(gfxarch_list)

    # Filter to only architectures with available artifacts
    available = [
        arch
        for arch in gfxarch_list
        if has_artifact_for_arch(pkg_name, artifacts_dir, arch)
    ]

    if len(available) < len(list(gfxarch_list)):
        missing = set(gfxarch_list) - set(available)
        print(f"WORKAROUND: {pkg_name} missing artifacts for: {sorted(missing)}")

    return available


def filter_dependencies_by_artifacts(
    dep_list: list, artifacts_dir: Path, gfx_arch: str
) -> list:
    """Filter dependency list to exclude packages without artifacts.

    Removes dependencies that do not have artifacts available for the specified
    architecture. This prevents installation failures due to missing packages.

    Parameters:
    dep_list: List of dependency package names
    artifacts_dir: Directory where artifacts are stored
    gfx_arch: Target architecture to check

    Returns: Filtered dependency list
    """
    filtered = []
    for dep in dep_list:
        dep_info = get_package_info(dep, raise_if_missing=False)
        if dep_info is None:
            # Unknown package, keep it (might be system package)
            filtered.append(dep)
            continue

        # Non-gfxarch packages are always available
        if not is_gfxarch_package(dep_info, enable_kpack=True):
            filtered.append(dep)
            continue

        # Check if gfxarch package has artifacts
        if has_artifact_for_arch(dep, artifacts_dir, gfx_arch):
            filtered.append(dep)
        else:
            print(f"WORKAROUND: Excluding {dep} (no artifacts for {gfx_arch})")

    return filtered
