#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""Given ROCm artifacts directories, performs packaging to
create RPM and DEB packages and upload to artifactory server

```
# With explicit target specification:
./build_package.py --artifacts-dir ./ARTIFACTS_DIR  \
        --target gfx94X-dcgpu \
        --dest-dir ./OUTPUT_PKGDIR \
        --rocm-version 7.1.0 \
        --pkg-type deb (or rpm) \
        --version-suffix build_type (daily/master/nightly/release)

# With auto-detection of targets from artifact directory:
./build_package.py --artifacts-dir ./ARTIFACTS_DIR  \
        --dest-dir ./OUTPUT_PKGDIR \
        --rocm-version 7.1.0 \
        --pkg-type deb (or rpm) \
        --version-suffix build_type (daily/master/nightly/release)
```
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import traceback

from dataclasses import replace
from datetime import datetime, timezone
from email.utils import format_datetime
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_TOOLS_DIR = SCRIPT_DIR.parent.parent

# Add build_tools directory to Python path to import _therock_utils
# This allows the script to be run from anywhere: TheRock root or packaging/linux directory
if str(BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_TOOLS_DIR))

from packaging_summary import *
from packaging_utils import *
from runpath_to_rpath import *

from _therock_utils.artifacts import ArtifactCatalog

# Default install prefix
DEFAULT_INSTALL_PREFIX = "/opt/rocm/core"


def _load_kpack_from_manifest(artifacts_dir: Path) -> bool:
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
    return sorted(catalog.all_target_families)


################### Debian package creation #######################
def create_deb_package(pkg_name, config: PackageConfig):
    """Create a Debian package.

    This function invokes the creation of versioned and non-versioned packages
    and moves the resulting `.deb` files to the destination directory.

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns:
    output_list: List of packages created
    """
    print_function_name()
    print(f"Package Name: {pkg_name}")

    # Non-versioned packages are not required for RPATH packages
    # In multi-arch mode, only create non-versioned packages for generic architecture
    if not config.enable_rpath:
        if not config.enable_kpack or config.gfx_arch == GFX_GENERIC:
            create_nonversioned_deb_package(pkg_name, config)

    create_versioned_deb_package(pkg_name, config)
    output_list = move_packages_to_destination(pkg_name, config)
    # Clean debian build directory
    remove_dir(Path(config.dest_dir) / config.pkg_type)
    return output_list


def create_nonversioned_deb_package(pkg_name, config: PackageConfig):
    """Create a non-versioned Debian meta package (.deb).

    Builds a minimal Debian binary package whose payload is empty and whose primary
    purpose is to express dependencies. The package name does not embed a version

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    # Create immutable config copy with versioned_pkg=False
    build_config = replace(config, versioned_pkg=False)

    package_dir = Path(build_config.dest_dir) / build_config.pkg_type / pkg_name
    deb_dir = package_dir / "debian"
    # Create package directory and debian directory
    os.makedirs(deb_dir, exist_ok=True)

    pkg_info = get_package_info(pkg_name)
    generate_changelog_file(pkg_info, deb_dir, build_config)
    generate_rules_file(pkg_info, deb_dir, build_config)
    generate_control_file(pkg_info, deb_dir, build_config)

    package_with_dpkg_build(package_dir)


def create_versioned_deb_package(pkg_name, config: PackageConfig):
    """Create a versioned Debian package (.deb).

    This function automates the process of building a Debian package by:
    1) Retrieving package metadata and validating required fields.
    2) Generating the `DEBIAN/control` file with appropriate fields (Package,
       Version, Architecture, Maintainer, Description, and dependencies).
    3) Copying the required package contents from an Artifactory repository.
    4) Invoking `dpkg-buildpackage` to assemble the final `.deb` file.

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    # Explicitly ensure versioned_pkg=True
    build_config = replace(config, versioned_pkg=True)
    package_dir = (
        Path(build_config.dest_dir)
        / build_config.pkg_type
        / f"{pkg_name}{build_config.rocm_version}"
    )
    deb_dir = package_dir / "debian"
    # Create package directory and debian directory
    os.makedirs(deb_dir, exist_ok=True)

    pkg_info = get_package_info(pkg_name)
    is_meta = is_meta_package(pkg_info)
    generate_changelog_file(pkg_info, deb_dir, build_config)
    generate_rules_file(pkg_info, deb_dir, build_config)
    generate_control_file(pkg_info, deb_dir, build_config)
    if is_postinstallscripts_available(pkg_info):
        generate_debian_postscripts(pkg_info, deb_dir, build_config)

    sourcedir_list = []
    dir_list = filter_components_fromartifactory(
        pkg_name,
        build_config.artifacts_dir,
        build_config.gfx_arch,
        build_config.enable_kpack,
    )
    sourcedir_list.extend(dir_list)

    print(f"sourcedir_list:\n  {sourcedir_list}")
    if not sourcedir_list and not is_meta:
        if build_config.enable_kpack:
            print(
                f"ERROR: {pkg_name}: Empty sourcedir_list and not a meta package, skipping"
            )
            return []
        else:
            sys.exit(
                f"{pkg_name}: Empty sourcedir_list and not a meta package, exiting"
            )

    if not sourcedir_list:
        print(f"{pkg_name} is a Meta package")
    else:
        # Copy package contents first
        dest_dir = package_dir / Path(build_config.install_prefix).relative_to("/")
        for source_path in sourcedir_list:
            copy_package_contents(source_path, dest_dir)

        if build_config.enable_rpath:
            convert_runpath_to_rpath(package_dir)

        # Generate install file after copying, so we can check for hidden files
        generate_install_file(pkg_info, deb_dir, build_config, dest_dir)

    package_with_dpkg_build(package_dir)


def generate_changelog_file(pkg_info, deb_dir, config: PackageConfig):
    """Generate a Debian changelog entry in `debian/changelog`.

    Parameters:
    pkg_info : Package details from the Json file
    deb_dir: Directory where debian package changelog file is saved
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    changelog = Path(deb_dir) / "changelog"

    pkg_name = update_package_name(pkg_info.get("Package"), config)
    maintainer = pkg_info.get("Maintainer")
    name_part, email_part = maintainer.split("<")
    name = name_part.strip()
    email = email_part.replace(">", "").strip()
    # version is used along with package name
    version = str(config.rocm_version)
    if config.version_suffix:
        version += f"-{str(config.version_suffix)}"

    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/debian_changelog.j2")

    # Prepare context dictionary
    context = {
        "package": pkg_name,
        "version": version,
        "distribution": "UNRELEASED",
        "urgency": "medium",
        "changes": ["Initial release"],  # TODO: Will get from package.json?
        "maintainer_name": name,
        "maintainer_email": email,
        "date": format_datetime(
            datetime.now(timezone.utc)
        ),  # TODO. How to get the date info?
    }

    with changelog.open("w", encoding="utf-8") as f:
        f.write(template.render(context))


def generate_install_file(pkg_info, deb_dir, config: PackageConfig, dest_dir=None):
    """Generate a Debian install entry in `debian/install`.

    Parameters:
    pkg_info : Package details from the Json file
    deb_dir: Directory where debian package control file is saved
    config: Configuration object containing package metadata
    dest_dir: Optional path to check for hidden files

    Returns: None
    """
    print_function_name()
    # Note: pkg_info is not used currently:
    # May be required in future to populate any context
    install_file = Path(deb_dir) / "install"

    # Check if hidden files and regular files exist in the destination directory
    has_hidden_files = False
    has_regular_files = False
    if dest_dir and Path(dest_dir).exists():
        for item in Path(dest_dir).iterdir():
            name = item.name  # get the filename as a string
            # Skip "." and ".."
            if name in [".", ".."]:
                continue

            # Hidden entry
            if name.startswith("."):
                has_hidden_files = True
            else:
                has_regular_files = True

    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/debian_install.j2")
    # Prepare your context dictionary
    context = {
        "path": config.install_prefix,
        "has_hidden_files": has_hidden_files,
        "has_regular_files": has_regular_files,
    }

    with install_file.open("w", encoding="utf-8") as f:
        f.write(template.render(context))


def generate_rules_file(pkg_info, deb_dir, config: PackageConfig):
    """Generate a Debian rules entry in `debian/rules`.

    Parameters:
    pkg_info : Package details from the Json file
    deb_dir: Directory where debian package control file is saved
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    rules_file = Path(deb_dir) / "rules"
    disable_dh_strip = is_key_defined(pkg_info, "Disable_DEB_STRIP")
    disable_dwz = is_key_defined(pkg_info, "Disable_DWZ")
    # Get package name for changelog installation
    pkg_name = update_package_name(pkg_info.get("Package"), config)

    # Disable debian dh_strip for multi-arch builds
    # WORKAROUND: dh_strip's debugedit incorrectly truncates ELF files with
    # unconventional layouts (e.g., program headers at end of file).
    # This causes "program header goes past the end of the file" errors.
    # See: https://github.com/ROCm/TheRock/issues/4047
    if config.enable_kpack:
        disable_dh_strip = True

    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/debian_rules.j2")
    # Prepare  context dictionary
    context = {
        "disable_dwz": disable_dwz,
        "disable_dh_strip": disable_dh_strip,
        "install_prefix": config.install_prefix,
        "pkg_name": pkg_name,
    }

    with rules_file.open("w", encoding="utf-8") as f:
        f.write(template.render(context))
    # set executable permission for rules file
    rules_file.chmod(0o755)


def generate_control_file(pkg_info, deb_dir, config: PackageConfig):
    """Generate a Debian control file entry in `debian/control`.

    Parameters:
    pkg_info: Package details parsed from a JSON file
    deb_dir: Directory where the `debian/control` file will be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    control_file = Path(deb_dir) / "control"
    pkg_name = pkg_info.get("Package")
    is_meta = is_meta_package(pkg_info)

    # Initialize optional fields
    provides = replaces = conflicts = ""
    debrecommends = debsuggests = ""

    if config.versioned_pkg:
        # Get -> Filter -> Transform
        debrecommends = process_dependency_field(pkg_info, "DEBRecommends", config)
        debsuggests = process_dependency_field(pkg_info, "DEBSuggests", config)
        depends = process_dependency_field(
            pkg_info, "DEBDepends", config, use_multiarch=True
        )
    else:
        # Get -> Transform -> Join
        provides = process_name_field(pkg_info, "Provides", debian_replace_devel_name)
        replaces = process_name_field(pkg_info, "Replaces", debian_replace_devel_name)
        conflicts = process_name_field(pkg_info, "Conflicts", debian_replace_devel_name)
        # Non-versioned package depends on versioned package itself
        depends = resolve_versioned_dependencies([pkg_name], config, is_meta)

    pkg_name = update_package_name(pkg_name, config)

    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/debian_control.j2")
    context = {
        "source": pkg_name,
        "depends": depends,
        "pkg_name": pkg_name,
        "arch": pkg_info.get("Architecture"),
        "description_short": pkg_info.get("Description_Short"),
        "description_long": pkg_info.get("Description_Long"),
        "homepage": pkg_info.get("Homepage"),
        "maintainer": pkg_info.get("Maintainer"),
        "priority": pkg_info.get("Priority"),
        "section": pkg_info.get("Section"),
        "version": config.rocm_version,
        "provides": provides,
        "replaces": replaces,
        "conflicts": conflicts,
        "debrecommends": debrecommends,
        "debsuggests": debsuggests,
    }

    with control_file.open("w", encoding="utf-8") as f:
        f.write(template.render(context))
        f.write("\n")  # Adds a blank line. For fixing missing final newline


def generate_debian_postscripts(pkg_info, deb_dir, config: PackageConfig):
    """Generate a Debian postinst/prerm file entry in `debian folder`.

    Parameters:
    pkg_info: Package details parsed from a JSON file
    deb_dir: Directory where the `debian/control` file will be created
    config: Configuration object containing package metadata

    Returns: None
    """
    # Debian maintainer scripts that must be executable
    EXEC_SCRIPTS = {"preinst", "postinst", "prerm", "postrm", "config"}
    pkg_name = pkg_info.get("Package")
    parts = config.rocm_version.split(".")
    if len(parts) < 3:
        raise ValueError(
            f"Version string '{config.rocm_version}' does not have major.minor.patch versions"
        )

    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    # Prepare your context dictionary
    context = {
        "install_prefix": config.install_prefix,
        "version_major": int(re.match(r"^\d+", parts[0]).group()),
        "version_minor": int(re.match(r"^\d+", parts[1]).group()),
        "version_patch": int(re.match(r"^\d+", parts[2]).group()),
        "target": "deb",
    }

    templates_root = Path(SCRIPT_DIR) / "template" / "scripts"
    # Collect all matching files
    for script in EXEC_SCRIPTS:
        pattern = f"{pkg_name}-{script}.j2"
        for file in templates_root.glob(pattern):
            script_file = Path(deb_dir) / script
            template = env.get_template(str(file.relative_to(SCRIPT_DIR)))
            with script_file.open("w", encoding="utf-8") as f:
                f.write(template.render(context))
            os.chmod(script_file, 0o755)


def copy_package_contents(source_dir, destination_dir):
    """Copy package contents from artfactory to package build directory

    Parameters:
    source_dir : Source directory
    destination_dir: Local directory where the package contents should be copied

    Returns: None
    """
    print_function_name()

    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    if not source_dir.is_dir():
        print(f"Directory does not exist: {source_dir}")
        return

    # Ensure destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Copy each item from source to destination
    for item in source_dir.iterdir():
        src = item
        dst = destination_dir / item.name

        if src.is_dir() and not dst.is_symlink():
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
            )
        elif src.is_symlink():
            # Copy the symlink itself (even if dangling)
            link_target = src.readlink()
            dst.symlink_to(link_target)
        else:
            shutil.copy2(src, dst)


def package_with_dpkg_build(pkg_dir):
    """Generate a Debian package using `dpkg-buildpackage`

    Parameters:
    pkg_dir: Path to the directory containing the package contents and the `debian/`
        subdirectory (with `control`, `changelog`, `rules`, etc.).

    Returns: None
    """
    print_function_name()
    # Build the command
    cmd = ["dpkg-buildpackage", "-uc", "-us", "-b"]

    # Execute the command
    try:
        subprocess.run(cmd, check=True, cwd=pkg_dir)
        print(f"Deb Package built successfully: {os.path.basename(pkg_dir)}")
    except subprocess.CalledProcessError as e:
        print(f"Error building deb package: {os.path.basename(pkg_dir)}: {e}")
        sys.exit(e.returncode)


######################## RPM package creation ####################
def create_rpm_package(pkg_name, config: PackageConfig):
    """Create an RPM package.

    This function invokes the creation of versioned and non-versioned packages
    and moves the resulting `.rpm` files to the destination directory.

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns:
    output_list: List of packages created
    """
    print_function_name()
    print(f"Package Name: {pkg_name}")

    # Non-versioned packages are not required for RPATH packages
    # In multi-arch mode, only create non-versioned packages for generic architecture
    if not config.enable_rpath:
        if not config.enable_kpack or config.gfx_arch == GFX_GENERIC:
            create_nonversioned_rpm_package(pkg_name, config)

    create_versioned_rpm_package(pkg_name, config)
    output_list = move_packages_to_destination(pkg_name, config)
    # Clean rpm build directory
    remove_dir(Path(config.dest_dir) / config.pkg_type)
    return output_list


def create_nonversioned_rpm_package(pkg_name, config: PackageConfig):
    """Create a non-versioned RPM meta package (.rpm).

    Builds a minimal RPM binary package whose payload is empty and whose primary
    purpose is to express dependencies. The package name does not embed a version

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    # Create immutable config copy with versioned_pkg=False
    build_config = replace(config, versioned_pkg=False)
    package_dir = Path(build_config.dest_dir) / build_config.pkg_type / pkg_name
    specfile = package_dir / "specfile"
    generate_spec_file(pkg_name, specfile, build_config)
    package_with_rpmbuild(specfile)


def create_versioned_rpm_package(pkg_name, config: PackageConfig):
    """Create a versioned RPM package (.rpm).

    This function automates the process of building a RPM package by:
    1) Generating the spec file with appropriate fields (Package,
       Version, Architecture, Maintainer, Description, and dependencies).
    2) Invoking `rpmbuild` to assemble the final `.rpm` file.

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    # Explicitly ensure versioned_pkg=True
    build_config = replace(config, versioned_pkg=True)
    package_dir = (
        Path(build_config.dest_dir)
        / build_config.pkg_type
        / f"{pkg_name}{build_config.rocm_version}"
    )
    specfile = package_dir / "specfile"
    generate_spec_file(pkg_name, specfile, build_config)
    package_with_rpmbuild(specfile)


def generate_spec_file(pkg_name, specfile, config: PackageConfig):
    """Generate an RPM spec file.

    Parameters:
    pkg_name : Package name
    specfile: Path where the generated spec file should be saved
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    os.makedirs(os.path.dirname(specfile), exist_ok=True)

    pkg_info = get_package_info(pkg_name)
    version = f"{config.rocm_version}"
    is_meta = is_meta_package(pkg_info)

    # Initialize optional fields
    provides = obsoletes = conflicts = ""
    rpmrecommends = rpmsuggests = ""
    sourcedir_list = []
    rpm_scripts = []
    # amdrocm-debugger: Exclude libpython requirements
    # Multiple Python-version-specific binaries are included; the wrapper script
    # automatically selects the binary matching the system's Python version
    exclude_libpython_requires = pkg_name == "amdrocm-debugger"

    if config.versioned_pkg:
        # Get -> Filter -> Transform
        rpmrecommends = process_dependency_field(pkg_info, "RPMRecommends", config)
        rpmsuggests = process_dependency_field(pkg_info, "RPMSuggests", config)
        requires = process_dependency_field(
            pkg_info, "RPMRequires", config, use_multiarch=True
        )

        dir_list = filter_components_fromartifactory(
            pkg_name, config.artifacts_dir, config.gfx_arch, config.enable_kpack
        )
        sourcedir_list.extend(dir_list)

        # Filter out non-existing directories
        sourcedir_list = [path for path in sourcedir_list if os.path.isdir(path)]

        # Warn if we have no artifacts for non-meta packages
        if not sourcedir_list and not is_meta:
            if config.enable_kpack:
                print(
                    f"WARNING: {pkg_name}: Empty sourcedir_list and not a meta package, creating empty RPM"
                )
            else:
                sys.exit(
                    f"{pkg_name}: Empty sourcedir_list and not a meta package, exiting"
                )

        if is_postinstallscripts_available(pkg_info):
            rpm_scripts = generate_rpm_postscripts(pkg_info, config)

        if config.enable_rpath:
            for path in sourcedir_list:
                convert_runpath_to_rpath(path)
    else:
        # Get -> Transform -> Join (no transform needed for RPM)
        provides = process_name_field(pkg_info, "Provides")
        obsoletes = process_name_field(pkg_info, "Obsoletes")
        conflicts = process_name_field(pkg_info, "Conflicts")
        # Non-versioned package requires versioned package itself
        requires = resolve_versioned_dependencies([pkg_name], config, is_meta)

    pkg_name = update_package_name(pkg_name, config)

    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/rpm_specfile.j2")
    context = {
        "pkg_name": pkg_name,
        "version": version,
        "release": config.version_suffix,
        "build_arch": pkg_info.get("BuildArch"),
        "description_short": pkg_info.get("Description_Short"),
        "description_long": pkg_info.get("Description_Long"),
        "group": pkg_info.get("Group"),
        "pkg_license": pkg_info.get("License"),
        "vendor": pkg_info.get("Vendor"),
        "install_prefix": config.install_prefix,
        "requires": requires,
        "provides": provides,
        "obsoletes": obsoletes,
        "conflicts": conflicts,
        "rpmrecommends": rpmrecommends,
        "rpmsuggests": rpmsuggests,
        "disable_rpm_strip": is_rpm_stripping_disabled(pkg_info),
        "disable_debug_package": is_debug_package_disabled(pkg_info),
        "sourcedir_list": sourcedir_list,
        "rpm_scripts": rpm_scripts,
        "exclude_libpython_requires": exclude_libpython_requires,
    }

    with open(specfile, "w", encoding="utf-8") as f:
        f.write(template.render(context))


def generate_rpm_postscripts(pkg_info, config: PackageConfig):
    """Generate RPM postinst/prerm sections.

    Parameters:
    pkg_info: Package details parsed from a JSON file
    config: Configuration object containing package metadata

    Returns: rpm script sections for specfile
    """
    # RPM maintainer scripts
    EXEC_SCRIPTS = {
        "preinst": "%pre",
        "postinst": "%post",
        "prerm": "%preun",
        "postrm": "%postun",
    }
    pkg_name = pkg_info.get("Package")
    parts = config.rocm_version.split(".")
    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    # Prepare your context dictionary
    context = {
        "install_prefix": config.install_prefix,
        "version_major": int(re.match(r"^\d+", parts[0]).group()),
        "version_minor": int(re.match(r"^\d+", parts[1]).group()),
        "version_patch": int(re.match(r"^\d+", parts[2]).group()),
        "target": "rpm",
    }

    templates_root = Path(SCRIPT_DIR) / "template" / "scripts"
    # Collect all matching files
    # This will hold rendered RPM script sections
    rpm_script_sections = {}

    for script, rpm_section in EXEC_SCRIPTS.items():
        pattern = f"{pkg_name}-{script}.j2"

        for file in templates_root.glob(pattern):
            template = env.get_template(str(file.relative_to(SCRIPT_DIR)))
            rendered = template.render(context)

            # Store rendered script under its RPM section name
            rpm_script_sections[rpm_section] = rendered

    return rpm_script_sections


def package_with_rpmbuild(spec_file):
    """Generate a RPM package using `rpmbuild`

    Parameters:
    spec_file: Specfile for RPM package

    Returns: None
    """
    print_function_name()
    package_rpm = os.path.dirname(spec_file)

    try:
        subprocess.run(
            ["rpmbuild", "--define", f"_topdir {package_rpm}", "-ba", spec_file],
            check=True,
        )
        print(f"RPM build completed successfully: {os.path.basename(package_rpm)}")
    except subprocess.CalledProcessError as e:
        print(f"RPM build failed for {os.path.basename(package_rpm)}: {e}")
        sys.exit(e.returncode)


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

    # Determine target architectures
    if args.target:
        # Use explicitly provided targets
        normalized_targets = normalize_target_list(args.target)
    else:
        # Auto-detect from artifact directory
        normalized_targets = get_all_target_families(args.artifacts_dir)
        if not normalized_targets:
            print(
                f"No GFX architectures found in artifact directory: {args.artifacts_dir}. "
                "Either provide --target explicitly or ensure artifacts are present."
            )
        else:
            print(f"Auto-detected GFX architectures: {normalized_targets}")

    # Output packaging architecture list to GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output and normalized_targets:
        with open(github_output, "a", encoding="utf-8") as f:
            targets_str = ",".join(normalized_targets)
            f.write(f"PACKAGING_ARCH_LIST={targets_str}\n")

    # Auto-detect kpack from manifest if not explicitly requested via --enable-kpack
    artifacts_dir = Path(args.artifacts_dir).resolve()
    if not args.enable_kpack:
        args.enable_kpack = _load_kpack_from_manifest(artifacts_dir)
        if args.enable_kpack:
            print(
                "Detected KPACK_SPLIT_ARTIFACTS in manifest — producing host + device packages"
            )

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
