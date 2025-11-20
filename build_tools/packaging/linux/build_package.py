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
import copy
import glob
import os
import platform
import shutil
import subprocess
import sys

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import format_datetime
from jinja2 import Environment, FileSystemLoader, Template
from packaging_utils import *
from pathlib import Path
from runpath_to_rpath import *


# User inputs required for packaging
# dest_dir - For saving the rpm/deb packages
# pkg_type - Package type DEB or RPM
# rocm_version - Used along with package name
# version_suffix - Used along with package name
# install_prefix - Install prefix for the package
# gfx_arch - gfxarch used for building artifacts
# enable_rpath - To enable RPATH packages
# versioned_pkg - Used to indicate versioned or non versioned packages
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


SCRIPT_DIR = Path(__file__).resolve().parent
# Directory for debian and RPM packaging
DEBIAN_CONTENTS_DIR = None
RPM_CONTENTS_DIR = None
# Default install prefix
DEFAULT_INSTALL_PREFIX = "/opt/rocm"

################### Debian package creation #######################
def create_deb_package(pkg_name, config: PackageConfig):
    """Create a Debian package.

    This function invokes the creation of versioned and non-versioned packages
    and moves the resulting `.deb` files to the destination directory.

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    print(f"Package Name: {pkg_name}")

    # Non-versioned packages are not required for RPATH packages
    if not config.enable_rpath:
        create_nonversioned_deb_package(pkg_name, config)

    create_versioned_deb_package(pkg_name, config)
    move_packages_to_destination(pkg_name, config)
    clean_debian_build_dir()


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
    # Set versioned_pkg flag to False
    config.versioned_pkg = False

    package_dir = Path(DEBIAN_CONTENTS_DIR) / f"{pkg_name}"
    deb_dir = package_dir / "debian"
    # Create package directory and debian directory
    os.makedirs(deb_dir, exist_ok=True)

    pkg_info = get_package_info(pkg_name)
    generate_changelog_file(pkg_info, deb_dir, config)
    generate_rules_file(pkg_info, deb_dir, config)
    generate_control_file(pkg_info, deb_dir, config)

    package_with_dpkg_build(package_dir)
    # Set the versioned_pkg flag to True
    config.versioned_pkg = True


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
    config.versioned_pkg = True
    package_dir = Path(DEBIAN_CONTENTS_DIR) / f"{pkg_name}{config.rocm_version}"
    deb_dir = package_dir / "debian"
    # Create package directory and debian directory
    os.makedirs(deb_dir, exist_ok=True)

    pkg_info = get_package_info(pkg_name)

    generate_changelog_file(pkg_info, deb_dir, config)
    generate_rules_file(pkg_info, deb_dir, config)
    generate_install_file(pkg_info, deb_dir, config)
    generate_control_file(pkg_info, deb_dir, config)
    # check the package is group of basic package or not
    pkg_list = pkg_info.get("Includes")

    if pkg_list is None:
        pkg_list = [pkg_info.get("Package")]
    sourcedir_list = []
    for pkg in pkg_list:
        dir_list = filter_components_fromartifactory(
            pkg, config.artifacts_dir, config.gfx_arch
        )
        sourcedir_list.extend(dir_list)

    print(f"sourcedir_list:\n  {sourcedir_list}")
    if not sourcedir_list:
        sys.exit("Empty sourcedir_list, exiting")

    dest_dir = package_dir / Path(config.install_prefix).relative_to("/")
    for source_path in sourcedir_list:
        copy_package_contents(source_path, dest_dir)

    if config.enable_rpath:
        convert_runpath_to_rpath(package_dir)

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


def generate_install_file(pkg_info, deb_dir, config: PackageConfig):
    """Generate a Debian install entry in `debian/install`.

    Parameters:
    pkg_info : Package details from the Json file
    deb_dir: Directory where debian package control file is saved
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    # Note: pkg_info is not used currently:
    # May be required in future to populate any context
    install_file = Path(deb_dir) / "install"

    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/debian_install.j2")
    # Prepare your context dictionary
    context = {
        "path": config.install_prefix,
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
    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/debian_rules.j2")
    # Prepare  context dictionary
    context = {
        "disable_dwz": disable_dwz,
        "disable_dh_strip": disable_dh_strip,
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

    if config.versioned_pkg:
        depends_list = pkg_info.get("DEBDepends", [])
    else:
        depends_list = [pkg_name]

    depends = convert_to_versiondependency(depends_list, config)

    pkg_name = update_package_name(pkg_name, config)
    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/debian_control.j2")
    # Prepare your context dictionary
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
    }

    with control_file.open("w", encoding="utf-8") as f:
        f.write(template.render(context))
        f.write("\n")  # Adds a blank line. For fixing missing final newline


def copy_package_contents(source_dir, destination_dir):
    """Copy package contents from artfactory to package build directory

    Parameters:
    source_dir : Source directory
    destination_dir: Local directory where the package contents should be copied

    Returns: None
    """
    print_function_name()
    if not os.path.isdir(source_dir):
        print(f"Directory does not exist: {source_dir}")
        return

    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Copy each item from source to destination
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(destination_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def package_with_dpkg_build(pkg_dir):
    """Generate a Debian package using `dpkg-buildpackage`

    Parameters:
    pkg_dir: Path to the directory containing the package contents and the `debian/`
        subdirectory (with `control`, `changelog`, `rules`, etc.).

    Returns: None
    """
    print_function_name()
    current_dir = Path.cwd()
    os.chdir(Path(pkg_dir))
    # Build the command
    cmd = ["dpkg-buildpackage", "-uc", "-us", "-b"]

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Deb Package built successfully: {os.path.basename(pkg_dir)}")
    except subprocess.CalledProcessError as e:
        print(f"Error building deb package{os.path.basename(pkg_dir)}: {e}")
        sys.exit(e.returncode)

    os.chdir(current_dir)


######################## RPM package creation ####################
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
    config.versioned_pkg = False
    package_dir = Path(RPM_CONTENTS_DIR) / pkg_name
    specfile = package_dir / "specfile"
    generate_spec_file(pkg_name, specfile, config)
    package_with_rpmbuild(specfile)
    config.versioned_pkg = True


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
    config.versioned_pkg = True
    package_dir = Path(RPM_CONTENTS_DIR) / f"{pkg_name}{config.rocm_version}"
    specfile = package_dir / "specfile"
    generate_spec_file(pkg_name, specfile, config)
    package_with_rpmbuild(specfile)


def create_rpm_package(pkg_name, config: PackageConfig):
    """Create an RPM package.

    This function invokes the creation of versioned and non-versioned packages
    and moves the resulting `.rpm` files to the destination directory.

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print_function_name()
    print(f"Package Name: {pkg_name}")

    if not config.enable_rpath:
        create_nonversioned_rpm_package(pkg_name, config)

    create_versioned_rpm_package(pkg_name, config)
    move_packages_to_destination(pkg_name, config)
    clean_rpm_build_dir()


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

    pkginfo = get_package_info(pkg_name)
    # populate packge version details
    version = f"{config.rocm_version}"
    # TBD: Whether to use component version details?
    #    version = pkginfo.get("Version")

    sourcedir_list = []
    if config.versioned_pkg:
        recommends_list = pkginfo.get("RPMRecommends", [])
        rpmrecommends = convert_to_versiondependency(recommends_list, config)

        requires_list = pkginfo.get("RPMRequires", [])

        # Get the packages included by the composite package
        pkg_list = pkginfo.get("Includes")

        if pkg_list is None:
            pkg_list = [pkg_name]

        for pkg in pkg_list:
            dir_list = filter_components_fromartifactory(
                pkg, config.artifacts_dir, config.gfx_arch
            )
            sourcedir_list.extend(dir_list)

        # Filter out non-existing directories
        sourcedir_list = [path for path in sourcedir_list if os.path.isdir(path)]

        if config.enable_rpath:
            for path in sourcedir_list:
                convert_runpath_to_rpath(path)
    else:
        rpmrecommends = ""
        requires_list = [pkg_name]

    requires = convert_to_versiondependency(requires_list, config)
    # Update package name with version details and gfxarch
    pkg_name = update_package_name(pkg_name, config)

    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/rpm_specfile.j2")

    # Prepare your context dictionary
    context = {
        "pkg_name": pkg_name,
        "version": version,
        "release": config.version_suffix,
        "build_arch": pkginfo.get("BuildArch"),
        "description_short": pkginfo.get("Description_Short"),
        "description_long": pkginfo.get("Description_Long"),
        "group": pkginfo.get("Group"),
        "pkg_license": pkginfo.get("License"),
        "vendor": pkginfo.get("Vendor"),
        "install_prefix": config.install_prefix,
        "requires": requires,
        "rpmrecommends": rpmrecommends,
        "disable_rpm_strip": is_rpm_stripping_disabled(pkginfo),
        "disable_debug_package": is_debug_package_disabled(pkginfo),
        "sourcedir_list": sourcedir_list,
    }

    with open(specfile, "w", encoding="utf-8") as f:
        f.write(template.render(context))


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


############### Common functions for packaging ##################
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
    if config.pkg_type.lower() == "deb":
        artifacts = glob.glob(os.path.join(f"{DEBIAN_CONTENTS_DIR}", "*.deb"))
        # Replace -devel with -dev for debian packages
        pkg_name = debian_replace_devel_name(pkg_name)
    else:
        artifacts = glob.glob(
            os.path.join(
                f"{RPM_CONTENTS_DIR}", "*", f"RPMS/{platform.machine()}", "*.rpm"
            )
        )

    # Move deb/rpm files to the destination directory
    for file_path in artifacts:
        file_name = os.path.basename(file_path)
        if file_name.startswith(pkg_name):
            dest_file = Path(config.dest_dir) / Path(file_path).name
            # if file exists , update it
            if os.path.exists(dest_file):
                os.remove(dest_file)
            shutil.move(file_path, config.dest_dir)


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
    if config.versioned_pkg:
        pkg_suffix = config.rocm_version
    else:
        pkg_suffix = ""

    if config.enable_rpath:
        pkg_suffix = f"-rpath{pkg_suffix}"

    if check_for_gfxarch(pkg_name):
        # Remove -dcgpu from gfx_arch
        gfx_arch = config.gfx_arch.lower().split("-", 1)[0]
        pkg_name = pkg_name + pkg_suffix + "-" + gfx_arch
    else:
        pkg_name = pkg_name + pkg_suffix

    if config.pkg_type.lower() == "deb":
        pkg_name = debian_replace_devel_name(pkg_name)

    return pkg_name


def debian_replace_devel_name(pkg_name):
    """Replace '-devel' with '-dev' in the package name.

    Development package names are defined as -devel in json file
    For Debian packages -dev should be used instead.

    Parameters:
    pkg_name : Package name

    Returns: Updated package name
    """
    print_function_name()
    # Only required for debian developement package
    pkg_name = pkg_name.replace("-devel", "-dev")

    return pkg_name


def convert_to_versiondependency(dependency_list, config: PackageConfig):
    """Change ROCm package dependencies to versioned ones.

    If a package depends on any packages listed in `pkg_list`,
    this function appends the dependency name with the specified ROCm version.

    Parameters:
    dependency_list : List of dependent packages
    config: Configuration object containing package metadata

    Returns: A string of comma separated versioned packages
    """
    print_function_name()
    # This function is to add Version dependency
    # Make sure the flag is set to True

    local_config = copy.deepcopy(config)
    local_config.versioned_pkg = True
    pkg_list = get_package_list()
    updated_depends = [
        f"{update_package_name(pkg,local_config)}" if pkg in pkg_list else pkg
        for pkg in dependency_list
    ]
    depends = ", ".join(updated_depends)
    return depends


def filter_components_fromartifactory(pkg_name, artifacts_dir, gfx_arch):
    """Get the list of Artifactory directories required for creating the package.

    The `package.json` file defines the required artifactories for each package.

    Parameters:
    pkg_name : package name
    artifacts_dir : Directory where artifacts are saved
    gfx_arch : graphics architecture

    Returns: List of directories
    """
    print_function_name()

    pkg_info = get_package_info(pkg_name)
    is_composite = is_composite_package(pkg_info)
    sourcedir_list = []
    component_list = pkg_info.get("Components", [])
    artifact_prefix = pkg_info.get("Artifact")
    artifact_subdir = pkg_info.get("Artifact_Subdir")
    if is_key_defined(pkg_info, "Gfxarch"):
        artifact_suffix = gfx_arch
    else:
        artifact_suffix = "generic"

    for component in component_list:
        source_dir = (
            Path(artifacts_dir) / f"{artifact_prefix}_{component}_{artifact_suffix}"
        )
        filename = source_dir / "artifact_manifest.txt"
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:

                match_found = (
                    isinstance(artifact_subdir, str)
                    and (artifact_subdir.lower() + "/") in line.lower()
                ) or is_composite

                if match_found and line.strip():
                    print("Matching line:", line.strip())
                    source_path = source_dir / line.strip()
                    sourcedir_list.append(source_path)

    return sourcedir_list


def parse_input_package_list(pkg_name):
    """Populate the package list from the provided input arguments.

    Parameters:
    pkg_name : List of packages or type of packages single/composite

    Returns: Package list
    """
    print_function_name()
    pkg_list = []
    # If pkg_name is None, include all packages
    if pkg_name is None:
        pkg_list = get_package_list()
        return pkg_list

    # Proceed if pkg_name is not None
    data = read_package_json_file()

    for entry in data:
        # Skip if packaging is disabled
        if is_packaging_disabled(entry):
            continue

        name = entry.get("Package")
        is_composite = is_composite_package(entry)

        # Loop through each type in pkg_name
        for pkg in pkg_name:
            if pkg == "single" and not is_composite:
                pkg_list.append(name)
                break
            elif pkg == "composite" and is_composite:
                pkg_list.append(name)
                break
            elif pkg == name:
                pkg_list.append(name)
                break

    print(f"pkg_list:\n  {pkg_list}")
    return pkg_list


def clean_rpm_build_dir():
    """Clean the rpm build directory

    Parameters: None
    Returns: None
    """
    if os.path.exists(RPM_CONTENTS_DIR) and os.path.isdir(RPM_CONTENTS_DIR):
        shutil.rmtree(RPM_CONTENTS_DIR)
        print(f"Removed directory: {RPM_CONTENTS_DIR}")


def clean_debian_build_dir():
    """Clean the debian build directory

    Parameters: None
    Returns: None
    """
    if os.path.exists(DEBIAN_CONTENTS_DIR) and os.path.isdir(DEBIAN_CONTENTS_DIR):
        shutil.rmtree(DEBIAN_CONTENTS_DIR)
        print(f"Removed directory: {DEBIAN_CONTENTS_DIR}")


def clean_package_build_dir(artifacts_dir):
    """Clean the package build directories

    If artifactory directory is provided, clean the same as well

    Parameters:
    artifacts_dir : Directory where artifacts are stored

    Returns: None
    """
    print_function_name()

    clean_rpm_build_dir()
    clean_debian_build_dir()

    PYCACHE_DIR = Path(SCRIPT_DIR) / "__pycache__"
    if os.path.exists(PYCACHE_DIR) and os.path.isdir(PYCACHE_DIR):
        shutil.rmtree(PYCACHE_DIR)
        print(f"Removed directory: {PYCACHE_DIR}")

    if os.path.exists(artifacts_dir) and os.path.isdir(artifacts_dir):
        shutil.rmtree(artifacts_dir)
        print(f"Removed directory: {artifacts_dir}")


def run(args: argparse.Namespace):
    # Set the global variables
    dest_dir = Path(args.dest_dir).expanduser().resolve()
    global DEBIAN_CONTENTS_DIR, RPM_CONTENTS_DIR
    DEBIAN_CONTENTS_DIR = dest_dir / "DEB"
    RPM_CONTENTS_DIR = dest_dir / "RPM"

    # Clean the packaging build directories
    clean_package_build_dir("")
    # Append rocm version to default install prefix
    # TBD: Do we need to append rocm_version to other prefix?
    if args.install_prefix == f"{DEFAULT_INSTALL_PREFIX}":
        prefix = args.install_prefix + "-" + args.rocm_version

    # Populate package config details from user arguments
    config = PackageConfig(
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        dest_dir=dest_dir,
        pkg_type=args.pkg_type,
        rocm_version=args.rocm_version,
        version_suffix=args.version_suffix,
        install_prefix=prefix,
        gfx_arch=args.target,
        enable_rpath=args.rpath_pkg,
    )
    pkg_list = parse_input_package_list(args.pkg_names)
    # Create deb/rpm packages
    package_creators = {"deb": create_deb_package, "rpm": create_rpm_package}
    for pkg_name in pkg_list:
        if config.pkg_type and config.pkg_type.lower() in package_creators:
            print(f"Create {config.pkg_type.upper()} package.")
            package_creators[config.pkg_type.lower()](pkg_name, config)
        else:
            print("Create both DEB and RPM packages.")
            for creator in package_creators.values():
                creator(pkg_name, config)
    # TBD:
    # Currently RPATH packages are created by modifying the artifacts dir
    # So artifacts dir clean up is required
    clean_package_build_dir("")
    # clean_package_build_dir(config.artifacts_dir)


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
        help="Specify the packages to be created: single, composite or any specific package name",
    )

    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
