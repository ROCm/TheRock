# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import copy
import glob
import json
import os
import platform
import re
import shutil
import sys

from dataclasses import dataclass, field
from pathlib import Path


# Constants
# Used for creating a generic package in Multi arch mode
GFX_GENERIC = "gfx_generic"


# User inputs required for packaging
# dest_dir - For saving the rpm/deb packages
# pkg_type - Package type DEB or RPM
# rocm_version - Used along with package name
# version_suffix - Used along with package name
# install_prefix - Install prefix for the package
# gfx_arch - gfxarch used for building package
# enable_rpath - To enable RPATH packages
# versioned_pkg - Used to indicate versioned or non versioned packages
# enable_multi_arch - To enable multi-architecture support
# gfxarch_list - List of all architectures for multi-arch mode
@dataclass
class PackageConfig:
    artifacts_dir: Path
    dest_dir: Path
    pkg_type: str
    rocm_version: str
    version_suffix: str
    install_prefix: str
    gfx_arch: str
    enable_rpath: bool = field(default=False)
    versioned_pkg: bool = field(default=True)
    enable_multi_arch: bool = field(default=False)
    gfxarch_list: list = field(default_factory=list)


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
    if value in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "enable",
        "enabled",
        "found",
    ):
        return True
    if value in (
        "",
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "disable",
        "disabled",
        "notfound",
        "none",
        "null",
        "nil",
        "undefined",
        "n/a",
    ):
        return False


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


def is_composite_package(pkg_info):
    """
    Verifies whether composite key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if composite key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "composite")


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


def is_gfxarch_package(pkg_info, enable_multi_arch=False):
    """Check whether the package is associated with a graphics architecture

    Parameters:
    pkg_info (dict): A dictionary containing package details.
    enable_multi_arch (bool): Enable multi-architecture support.

    Returns:
    bool : True if Gfxarch is set, else False.
           False if devel package when enable_multi_arch is True
    """
    if enable_multi_arch:
        pkgname = pkg_info.get("Package", "")
        if pkgname.endswith("-devel"):
            return False

    return is_key_defined(pkg_info, "Gfxarch")


def get_package_info(pkgname):
    """Retrieves package details from a JSON file for the given package name

    Parameters:
    pkgname : Package Name

    Returns: Package metadata
    """

    # Load JSON data from a file
    data = read_package_json_file()

    for package in data:
        if package.get("Package") == pkgname:
            return package

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
    skipped = []
    data = read_package_json_file()

    try:
        dir_entries = os.listdir(artifact_dir)
    except FileNotFoundError:
        sys.exit(f"{artifact_dir}: Artifactory directory does not exist, Exiting")

    for pkg_info in data:
        pkg_name = pkg_info["Package"]
        # Skip disabled packages
        if is_packaging_disabled(pkg_info):
            continue

        # metapackages don't need artifact lookup
        if is_meta_package(pkg_info):
            pkg_list.append(pkg_name)
            continue

        artifactory_list = pkg_info.get("Artifactory", [])
        artifact_found = False

        for artifactory in artifactory_list:
            artifact_name = artifactory.get("Artifact")
            if not artifact_name:
                continue

            # Look for directories starting with the artifact name
            for entry in dir_entries:
                path = Path(artifact_dir) / entry

                if entry.startswith(artifact_name) and path.is_dir():
                    artifact_found = True
                    break

            if artifact_found:
                break

        if artifact_found:
            pkg_list.append(pkg_name)
        else:
            skipped.append(pkg_name)

    return pkg_list, skipped


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


def version_to_str(version_str):
    """Convert a ROCm version string to a numeric representation.

    This function transforms a ROCm version from its dotted format
    (e.g., "7.1.0") into a numeric string (e.g., "70100")
    Ex : 7.10.0 -> 71000
         10.1.0 - > 100100
         7.1 -> 70100
         7.1.1.1 -> 70101

    Parameters:
    version_str: ROCm version separated by dots

    Returns: Numeric string
    """

    parts = version_str.split(".")
    # Ensure we have exactly 3 parts: major, minor, patch
    while len(parts) < 3:
        parts.append("0")  # Default missing parts to "0"
    major, minor, patch = parts[:3]  # Ignore extra parts

    return f"{int(major):01d}{int(minor):02d}{int(patch):02d}"


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

    updated_pkgname += pkg_suffix

    if is_gfxarch_package(pkg_info, config.enable_multi_arch):
        # For multi-arch mode, skip appending gfx_generic
        if config.enable_multi_arch and config.gfx_arch == GFX_GENERIC:
            pass  # Don't append gfx_generic in multi-arch mode
        else:
            # Remove -dcgpu from gfx_arch
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
    local_config = copy.deepcopy(config)
    local_config.versioned_pkg = True

    for gfx_arch in gfxarch_list:
        local_config.gfx_arch = gfx_arch
        # update_package_name will append version and gfx_arch
        arch_pkg = update_package_name(pkg_name, local_config)
        arch_specific_packages.append(arch_pkg)

    return arch_specific_packages


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

    local_config = copy.deepcopy(config)
    local_config.versioned_pkg = True
    # In multi-arch mode, dependencies should always point to generic packages
    # UNLESS preserve_arch is True (for arch-specific metapackages)
    if config.enable_multi_arch and not preserve_arch:
        local_config.gfx_arch = GFX_GENERIC
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

    This function takes a comma‑separated dependency string,
    identifies which dependencies correspond to packages listed in `pkg_list`,
    and appends the appropriate ROCm version suffix based on the provided configuration.

    Parameters:
    dep_string : A comma‑separated list of dependency package names.
    config : Configuration object containing ROCm version, suffix, and packaging type.

    Returns: A comma‑separated string where matching dependencies include the version suffix,
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


def move_packages_to_destination(pkg_name, config: PackageConfig):
    """Move the generated Debian package from the build directory to the destination directory.

    Parameters:
    pkg_name : Package name
    config: Configuration object containing package metadata

    Returns:
    output_packages : list of package names moved to the destination folder
    """
    print_function_name()
    output_packages = []
    # Create destination dir to move the packages created
    os.makedirs(config.dest_dir, exist_ok=True)
    print(f"Package name: {pkg_name}")
    PKG_DIR = Path(config.dest_dir) / config.pkg_type
    if config.pkg_type.lower() == "deb":
        artifacts = list(PKG_DIR.glob("*.deb"))
        # Replace -devel with -dev for debian packages
        pkg_name = debian_replace_devel_name(pkg_name)
    else:
        artifacts = list(PKG_DIR.glob(f"*/RPMS/{platform.machine()}/*.rpm"))

    # Move deb/rpm files to the destination directory
    for file_path in artifacts:
        file_path = Path(file_path)  # ensure it's a Path object
        file_name = file_path.name  # basename equivalent

        if file_name.startswith(pkg_name):
            dest_file = Path(config.dest_dir) / file_name

            # if file exists, remove it first
            if dest_file.exists():
                dest_file.unlink()

            shutil.move(str(file_path), str(config.dest_dir))
            output_packages.append(file_name)

    return output_packages


def filter_components_fromartifactory(
    pkg_name, artifacts_dir, gfx_arch, enable_multi_arch=False
):
    """Get the list of Artifactory directories required for creating the package.

    The `package.json` file defines the required artifactories for each package.

    Parameters:
    pkg_name : package name
    artifacts_dir : Directory where artifacts are saved
    gfx_arch : graphics architecture
    enable_multi_arch : enable multi-architecture support

    Returns: List of directories
    """
    print_function_name()

    pkg_info = get_package_info(pkg_name)
    sourcedir_list = []

    if enable_multi_arch:
        dir_suffix = (
            gfx_arch
            if (
                is_gfxarch_package(pkg_info, enable_multi_arch)
                and gfx_arch != GFX_GENERIC
            )
            else "generic"
        )
    else:
        dir_suffix = (
            gfx_arch if is_gfxarch_package(pkg_info, enable_multi_arch) else "generic"
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
    if (
        config.versioned_pkg
        and config.enable_multi_arch
        and is_meta
        and config.gfx_arch == GFX_GENERIC
    ):
        # dep_list already contains versioned arch-specific package names
        # Just add version suffix and join
        deps = append_version_suffix(", ".join(dep_list), config)
    elif config.enable_multi_arch and is_meta and config.gfx_arch != GFX_GENERIC:
        # Arch-specific metapackage: preserve architecture for gfxarch dependencies
        deps = convert_to_versiondependency(dep_list, config, preserve_arch=True)
        deps = append_version_suffix(deps, config)
    else:
        # Normal path: convert dependencies and add version suffix
        deps = convert_to_versiondependency(dep_list, config)
        if is_meta:
            deps = append_version_suffix(deps, config)
    return deps


def get_dependency_list_for_multiarch(pkg_info, dep_key, config: PackageConfig):
    """Determine the appropriate dependency list for multi-arch mode.

    Parameters:
    pkg_info: Package details from JSON
    dep_key: Dependency key ("DEBDepends" or "RPMRequires")
    config: Configuration object containing package metadata

    Returns: List of dependency package names
    """
    pkg_name = pkg_info.get("Package")
    is_meta = is_meta_package(pkg_info)

    if config.enable_multi_arch and is_meta:
        # For metapackages in multi-arch mode:
        # - Generic variant depends on all arch-specific variants
        # - Arch-specific variants depend on actual runtime packages
        if config.gfx_arch == GFX_GENERIC:
            # Generic metapackage: depend on all arch-specific metapackages
            return expand_metapackage_to_all_archs(
                pkg_name, config.gfxarch_list, config
            )
        else:
            # Arch-specific metapackage: depend on actual runtime packages
            return pkg_info.get(dep_key, [])
    elif config.enable_multi_arch and config.gfx_arch != GFX_GENERIC:
        # For regular (non-meta) packages in multi-arch mode:
        # Non-generic packages only depend on themselves
        return [pkg_name]
    else:
        # Single-arch mode or generic packages: use full dependencies
        return pkg_info.get(dep_key, [])
