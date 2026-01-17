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
from typing import Dict, Set, List, Optional


# User inputs required for packaging
# dest_dir - For saving the rpm/deb packages
# pkg_type - Package type DEB or RPM
# rocm_version - Used along with package name
# version_suffix - Used along with package name
# install_prefix - Install prefix for the package
# gfx_arch - gfxarch used for building artifacts
# enable_rpath - To enable RPATH packages
# versioned_pkg - Used to indicate versioned or non versioned packages
# skip_missing_artifacts - Skip packages with missing artifacts instead of failing
# skipped_packages - List of packages that were skipped (for meta-package filtering)
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
    skip_missing_artifacts: bool = field(default=False)
    skipped_packages: List[str] = field(default_factory=list)


SCRIPT_DIR = Path(__file__).resolve().parent
# SCRIPT_DIR is build_tools/packaging/linux, so go up 3 levels to get to TheRock root
THEROCK_ROOT = SCRIPT_DIR.parent.parent.parent
currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name

# Cache for parsed exclusions from cmake file
_GFX_ARCH_EXCLUSIONS_CACHE: Optional[Dict[str, Set[str]]] = None
# Cache for artifact subdir to package mapping (built from package.json)
_ARTIFACT_SUBDIR_TO_PACKAGES_CACHE: Optional[Dict[str, List[str]]] = None


def build_artifact_subdir_to_packages_map() -> Dict[str, List[str]]:
    """Build a mapping from artifact subdirectory names to package names.

    This dynamically parses package.json to find which packages use which
    artifact subdirectories, eliminating the need for a hardcoded map.

    Returns:
        Dict mapping artifact_subdir names (e.g., "composable_kernel") to list of package names
    """
    global _ARTIFACT_SUBDIR_TO_PACKAGES_CACHE

    if _ARTIFACT_SUBDIR_TO_PACKAGES_CACHE is not None:
        return _ARTIFACT_SUBDIR_TO_PACKAGES_CACHE

    mapping: Dict[str, List[str]] = {}
    data = read_package_json_file()

    for package in data:
        pkg_name = package.get("Package")
        if not pkg_name:
            continue

        artifactory = package.get("Artifactory")
        if not artifactory:
            continue

        for artifact in artifactory:
            artifact_subdirs = artifact.get("Artifact_Subdir", [])
            for subdir in artifact_subdirs:
                subdir_name = subdir.get("Name")
                if subdir_name:
                    if subdir_name not in mapping:
                        mapping[subdir_name] = []
                    if pkg_name not in mapping[subdir_name]:
                        mapping[subdir_name].append(pkg_name)

    _ARTIFACT_SUBDIR_TO_PACKAGES_CACHE = mapping
    return mapping


def parse_gfx_arch_exclusions() -> Dict[str, Set[str]]:
    """Parse the CMake file to extract project exclusions for each GFX architecture.

    This function parses cmake/therock_amdgpu_targets.cmake to build a mapping
    of GFX architectures to excluded projects.

    Returns:
        Dict mapping gfx_arch (e.g., "gfx906", "gfx90X-dcgpu") to set of excluded project names
    """
    global _GFX_ARCH_EXCLUSIONS_CACHE

    if _GFX_ARCH_EXCLUSIONS_CACHE is not None:
        return _GFX_ARCH_EXCLUSIONS_CACHE

    cmake_file = THEROCK_ROOT / "cmake" / "therock_amdgpu_targets.cmake"

    if not cmake_file.exists():
        print(f"Warning: CMake file not found at {cmake_file}")
        _GFX_ARCH_EXCLUSIONS_CACHE = {}
        return _GFX_ARCH_EXCLUSIONS_CACHE

    exclusions: Dict[str, Set[str]] = {}

    with open(cmake_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Simple line-by-line parser to avoid regex catastrophic backtracking
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for therock_add_amdgpu_target calls
        if line.startswith("therock_add_amdgpu_target("):
            # Extract gfx_target from the first line
            match = re.match(r"therock_add_amdgpu_target\((\w+)", line)
            if not match:
                i += 1
                continue

            gfx_target = match.group(1)
            families = []
            excluded_projects = set()

            # Parse the function call (may span multiple lines)
            in_family_section = False
            in_exclude_section = False

            # Continue parsing until we find the closing parenthesis
            j = i
            while j < len(lines):
                current_line = lines[j].strip()

                # Remove comments
                if "#" in current_line:
                    current_line = current_line.split("#")[0].strip()

                # Check for FAMILY keyword
                if "FAMILY" in current_line:
                    in_family_section = True
                    in_exclude_section = False
                    # Extract families from the same line if present
                    parts = current_line.split("FAMILY", 1)
                    if len(parts) > 1:
                        family_parts = parts[1].strip().split()
                        families.extend(
                            [
                                f
                                for f in family_parts
                                if f and not f.startswith("EXCLUDE")
                            ]
                        )

                # Check for EXCLUDE_TARGET_PROJECTS keyword
                elif "EXCLUDE_TARGET_PROJECTS" in current_line:
                    in_family_section = False
                    in_exclude_section = True

                # If we're in family section, collect family names
                elif (
                    in_family_section and not "EXCLUDE_TARGET_PROJECTS" in current_line
                ):
                    tokens = current_line.split()
                    for token in tokens:
                        if token and token not in (")", ""):
                            families.append(token)

                # If we're in exclude section, collect excluded projects
                elif in_exclude_section:
                    tokens = current_line.split()
                    for token in tokens:
                        if (
                            token
                            and token not in (")", "")
                            and not token.startswith("#")
                        ):
                            excluded_projects.add(token)

                # Check if we've reached the end of the function call
                if ")" in current_line:
                    break

                j += 1

            # Store exclusions for the gfx target itself
            if gfx_target not in exclusions:
                exclusions[gfx_target] = set()
            exclusions[gfx_target].update(excluded_projects)

            # Store exclusions for all families this target belongs to
            for family in families:
                if family not in exclusions:
                    exclusions[family] = set()
                exclusions[family].update(excluded_projects)

            # Move to the line after the function call
            i = j + 1
        else:
            i += 1

    _GFX_ARCH_EXCLUSIONS_CACHE = exclusions
    return exclusions


def is_project_excluded_for_gfx_arch(project_name: str, gfx_arch: str) -> bool:
    """Check if a CMake project is excluded for a given GFX architecture.

    Parameters:
        project_name: CMake project name (e.g., "composable_kernel", "hipBLASLt")
        gfx_arch: GFX architecture string (e.g., "gfx906", "gfx90X-dcgpu")

    Returns:
        True if the project is excluded for this architecture, False otherwise
    """
    exclusions = parse_gfx_arch_exclusions()

    # Check exact match first
    if gfx_arch in exclusions and project_name in exclusions[gfx_arch]:
        return True

    # Check family matches (e.g., gfx90X-dcgpu contains gfx906)
    # Extract base family patterns
    for arch_pattern, excluded_projects in exclusions.items():
        if project_name in excluded_projects:
            # Check if gfx_arch matches this pattern
            # Simple heuristic: if gfx_arch contains the pattern or vice versa
            if arch_pattern in gfx_arch or gfx_arch in arch_pattern:
                return True

    return False


def is_package_excluded_for_gfx_arch(pkg_name: str, gfx_arch: str) -> bool:
    """Check if a package should be excluded for a given GFX architecture.

    This checks if ALL artifact subdirectories for this package are excluded.

    Parameters:
        pkg_name: Package name (e.g., "amdrocm-ck", "amdrocm-blas")
        gfx_arch: GFX architecture string (e.g., "gfx906", "gfx90X-dcgpu")

    Returns:
        True if the package should be excluded, False otherwise
    """
    pkg_info = get_package_info(pkg_name)
    if not pkg_info:
        return False

    artifactory = pkg_info.get("Artifactory")
    if not artifactory:
        # Meta packages or packages without artifacts are not excluded based on gfx_arch
        return False

    # Collect all artifact subdirectories for this package
    all_subdirs = []
    for artifact in artifactory:
        for subdir in artifact.get("Artifact_Subdir", []):
            subdir_name = subdir.get("Name")
            if subdir_name:
                all_subdirs.append(subdir_name)

    if not all_subdirs:
        return False

    # Check if ALL subdirectories are excluded
    excluded_count = 0
    for subdir_name in all_subdirs:
        # CMake project names typically match artifact subdir names
        if is_project_excluded_for_gfx_arch(subdir_name, gfx_arch):
            excluded_count += 1

    # Package is excluded only if ALL its artifact subdirs are excluded
    return excluded_count > 0 and excluded_count == len(all_subdirs)


def is_artifact_subdir_excluded_for_gfx_arch(
    artifact_subdir: str, gfx_arch: str
) -> bool:
    """Check if an artifact subdirectory is excluded for a given GFX architecture.

    CMake project names typically match artifact subdirectory names, so we check
    if the artifact_subdir name is excluded as a CMake project.

    Parameters:
        artifact_subdir: Artifact subdirectory name (e.g., "composable_kernel", "hipBLASLt")
        gfx_arch: GFX architecture string (e.g., "gfx906", "gfx90X-dcgpu")

    Returns:
        True if the artifact subdir is excluded, False otherwise
    """
    # CMake project names typically match artifact subdir names directly
    return is_project_excluded_for_gfx_arch(artifact_subdir, gfx_arch)


def filter_dependencies_by_gfx_arch(
    dependency_list: List[str], gfx_arch: str, skipped_packages: List[str] = None
) -> List[str]:
    """Filter a dependency list to remove packages excluded for the given GFX architecture.

    This is used primarily for meta-packages to avoid dependencies on packages
    whose components are completely excluded for the target architecture or were
    skipped due to missing artifacts.

    Parameters:
        dependency_list: List of package dependencies
        gfx_arch: GFX architecture string (e.g., "gfx906", "gfx90X-dcgpu")
        skipped_packages: List of package names that were skipped during build

    Returns:
        Filtered list with excluded/skipped packages removed
    """
    if skipped_packages is None:
        skipped_packages = []

    filtered = []
    for dep in dependency_list:
        # Extract base package name (remove version constraints, etc.)
        base_dep = re.split(r"[(<>=\s]", dep)[0].strip()

        # Check if excluded by architecture
        if is_package_excluded_for_gfx_arch(base_dep, gfx_arch):
            print(f"  Filtering out dependency '{dep}' (excluded for {gfx_arch})")
            continue

        # Check if package was skipped during build
        if base_dep in skipped_packages:
            print(f"  Filtering out dependency '{dep}' (skipped - missing artifacts)")
            continue

        filtered.append(dep)

    return filtered


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


def is_gfxarch_package(pkg_info):
    """Check whether the package is associated with a graphics architecture

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool : True if Gfxarch is set, else False.
           #False if devel package
    """
    #  Disabling this for time being as per the requirements
    #   if pkgname.endswith("-devel"):
    #       return False

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


def get_package_list():
    """Read package.json and return package names.

    Packages marked as 'Disablepackaging' will be excluded from the list

    Parameters: None

    Returns: Package list
    """

    data = read_package_json_file()

    pkg_list = [pkg["Package"] for pkg in data if not is_packaging_disabled(pkg)]
    return pkg_list


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

    if is_gfxarch_package(pkg_info):
        # Remove -dcgpu from gfx_arch
        gfx_arch = config.gfx_arch.lower().split("-", 1)[0]
        updated_pkgname += "-" + gfx_arch

    return updated_pkgname


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
    dependency_list, config: PackageConfig, filter_exclusions: bool = False
):
    """Change ROCm package dependencies to versioned ones.

    If a package depends on any packages listed in `pkg_list`,
    this function appends the dependency name with the specified ROCm version.

    Parameters:
    dependency_list : List of dependent packages
    config: Configuration object containing package metadata
    filter_exclusions: If True, filter out packages excluded for the gfx_arch and skipped packages

    Returns: A string of comma separated versioned packages
    """
    print_function_name()
    # This function is to add Version dependency
    # Make sure the flag is set to True

    # Filter out excluded dependencies if requested
    if filter_exclusions:
        skipped_packages = getattr(config, "skipped_packages", [])
        dependency_list = filter_dependencies_by_gfx_arch(
            dependency_list, config.gfx_arch, skipped_packages
        )

    local_config = copy.deepcopy(config)
    local_config.versioned_pkg = True
    pkg_list = get_package_list()
    updated_depends = [
        f"{update_package_name(pkg,local_config)}" if pkg in pkg_list else pkg
        for pkg in dependency_list
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

    pkg_list = get_package_list()
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

    Returns: None
    """
    print_function_name()

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


def filter_components_fromartifactory(pkg_name, artifacts_dir, gfx_arch):
    """Get the list of Artifactory directories required for creating the package.

    The `package.json` file defines the required artifactories for each package.
    This function now handles missing artifacts gracefully by checking if components
    are excluded for the given GFX architecture.

    Parameters:
    pkg_name : package name
    artifacts_dir : Directory where artifacts are saved
    gfx_arch : graphics architecture

    Returns: List of directories
    """
    print_function_name()

    pkg_info = get_package_info(pkg_name)
    sourcedir_list = []

    dir_suffix = gfx_arch if is_gfxarch_package(pkg_info) else "generic"

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

            # Check if this artifact subdirectory is excluded for this gfx_arch
            if is_artifact_subdir_excluded_for_gfx_arch(artifact_subdir, gfx_arch):
                print(
                    f"  Skipping artifact subdir '{artifact_subdir}' - "
                    f"excluded for {gfx_arch}"
                )
                continue

            for component in component_list:
                source_dir = (
                    Path(artifacts_dir)
                    / f"{artifact_prefix}_{component}_{artifact_suffix}"
                )
                filename = source_dir / "artifact_manifest.txt"

                # Check if the artifact manifest file exists
                if not filename.exists():
                    # Check if this is expected due to exclusion
                    if is_artifact_subdir_excluded_for_gfx_arch(
                        artifact_subdir, gfx_arch
                    ):
                        print(
                            f"  Artifact manifest not found (expected): {filename} - "
                            f"'{artifact_subdir}' excluded for {gfx_arch}"
                        )
                        continue
                    else:
                        print(f"  Warning: Artifact manifest not found: {filename}")
                        print(f"    Package: {pkg_name}")
                        print(f"    Artifact: {artifact_prefix}")
                        print(f"    Component: {component}")
                        print(f"    Subdir: {artifact_subdir}")
                        print(f"    Expected path: {source_dir}")
                        # Don't fail immediately - continue to check other components
                        continue

                try:
                    with open(filename, "r", encoding="utf-8") as file:
                        for line in file:
                            match_found = (
                                isinstance(artifact_subdir, str)
                                and (artifact_subdir.lower() + "/") in line.lower()
                            )

                            if match_found and line.strip():
                                print("Matching line:", line.strip())
                                source_path = source_dir / line.strip()
                                sourcedir_list.append(source_path)
                except FileNotFoundError:
                    # File was deleted between exists() check and open()
                    print(f"  Warning: Could not read artifact manifest: {filename}")
                    continue

    return sourcedir_list
