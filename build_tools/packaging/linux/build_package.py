#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""Given ROCm artifacts directories, performs packaging to
create RPM and DEB packages and upload to artifactory server

```
./build_package.py --artifacts-dir ./ARTIFACTS_DIR  \
        --target gfx94X-dcgpu \
        --dest-dir ./OUTPUT_PKGDIR \
        --rocm-version 7.1.0
        --pkg-type deb (or rpm)
```
"""

import argparse
import glob
import inspect
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
DEBIAN_CONTENTS_DIR = Path.cwd() / "DEB"
RPM_CONTENTS_DIR = Path.cwd() / "RPM"
# Default install prefix
DEFAULT_INSTALL_PREFIX = "/opt/rocm"

################### Debian package creation #######################
def create_deb_package(pkg_name, config: PackageConfig):
    """Function to create deb package
    Invoke the creation of versioned and non versioned package
    Move the DEB packages to destination directory

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
    print(f"Package Name: {pkg_name}")

    create_nonversioned_deb_package(pkg_name, config)
    create_versioned_deb_package(pkg_name, config)
    move_packages_to_destination(pkg_name, config)


def create_nonversioned_deb_package(pkg_name, config: PackageConfig):
    """Function to create non versioned deb package
    Get package details and generate control file
    Non versioned Packages are meta packages.
    Create deb package

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
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
    """Function to create deb package
    Get package details and generate control file
    Find the required package contents from artifactory
    Copy the package contents to package creation directory
    Create deb package

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
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
    """Function will generate changelog for debian package

    Parameters:
    pkg_info : Package details from the Json file
    deb_dir: Directory where debian package control file is saved
    config: Configuration object containing package metadata

    Returns: None
    """

    print(inspect.currentframe().f_code.co_name)
    changelog = Path(deb_dir) / "changelog"

    pkg_name = update_package_name(pkg_info.get("Package"), config)
    maintainer = pkg_info.get("Maintainer")
    name_part, email_part = maintainer.split("<")
    name = name_part.strip()
    email = email_part.replace(">", "").strip()
    # version is used along with package name
    version = (
        config.rocm_version
        + "."
        + version_to_str(config.rocm_version)
        + "-"
        + config.version_suffix
    )

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
    """Function will generate install file for debian package

    Parameters:
    pkg_info : Package details from the Json file
    deb_dir: Directory where debian package control file is saved
    config: Configuration object containing package metadata

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
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
    """Function will generate control file for debian package

    Parameters:
    pkg_info : Package details from the Json file
    deb_dir: Directory where debian package control file is saved
    config: Configuration object containing package metadata

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
    rules_file = Path(deb_dir) / "rules"
    disable_dh_strip = is_key_defined(pkg_info, "Disable_DH_STRIP")
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
    """Function will generate control file for debian package

    Parameters:
    pkg_info : Package details from the Json file
    deb_dir: Directory where debian package control file is saved
    config: Configuration object containing package metadata

    Returns: None
    """

    print(inspect.currentframe().f_code.co_name)
    control_file = Path(deb_dir) / "control"

    pkg_name = update_package_name(pkg_info.get("Package"), config)

    if config.versioned_pkg:
        depends_list = pkg_info.get("DEBDepends", [])
        depends = convert_to_versiondependency(depends_list, config)
    else:
        depends = pkg_name + config.rocm_version
    # Note: The dev package name update should be done after version dependency
    # Package.json maintains development package name as devel
    depends = depends.replace("-devel", "-dev")

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
    """Copy package contents from artfactory to package directory

    Parameters:
    source_dir : Source directory
    destination_dir: Directory where package contents are to be copied

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
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
    """Create deb package

    Parameters:
    source_dir : Package directory containing package contents and control file
    output_dir: Directory where package is created
    package_name: Expected package name

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
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
    """Create rpm package by invoking each steps
    Generate spec file
    Create rpm nonversioned package

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
    config.versioned_pkg = False
    package_dir = Path(RPM_CONTENTS_DIR) / pkg_name
    specfile = package_dir / "specfile"
    generate_spec_file(pkg_name, specfile, config)
    package_with_rpmbuild(specfile)
    config.versioned_pkg = True


def create_versioned_rpm_package(pkg_name, config: PackageConfig):
    """Create rpm package by invoking each steps
    Generate spec file
    Create rpm versioned package

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
    config.versioned_pkg = True
    package_dir = Path(RPM_CONTENTS_DIR) / f"{pkg_name}{config.rocm_version}"
    specfile = package_dir / "specfile"
    generate_spec_file(pkg_name, specfile, config)
    package_with_rpmbuild(specfile)


def create_rpm_package(pkg_name, config: PackageConfig):
    """Create rpm package by invoking each steps
    Invoke the creation of versioned and non versioned package
    Move the rpm package to destination directory

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
    print(f"Package Name: {pkg_name}")
    create_nonversioned_rpm_package(pkg_name, config)
    create_versioned_rpm_package(pkg_name, config)
    move_packages_to_destination(pkg_name, config)


def generate_spec_file(pkg_name, specfile, config: PackageConfig):
    """Generate spec file for rpm package

    Parameters:
    pkg_name : Package name
    specfile: Specfile for RPM package
    config: Configuration object containing package metadata

    Returns: None
    """

    print(inspect.currentframe().f_code.co_name)
    os.makedirs(os.path.dirname(specfile), exist_ok=True)

    pkginfo = get_package_info(pkg_name)
    # populate packge version details
    version = f"{config.rocm_version}.{version_to_str(config.rocm_version)}"
    # TBD: Whether to use component version details?
    #    version = pkginfo.get("Version")

    sourcedir_list = []
    if config.versioned_pkg:
        recommends_list = pkginfo.get("RPMRecommends", [])
        rpmrecommends = convert_to_versiondependency(recommends_list, config)

        requires_list = pkginfo.get("RPMRequires", [])
        requires = convert_to_versiondependency(requires_list, config)

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
        requires = pkg_name + config.rocm_version
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
        "sourcedir_list": sourcedir_list,
    }

    with open(specfile, "w", encoding="utf-8") as f:
        f.write(template.render(context))


def package_with_rpmbuild(spec_file):
    """Create rpm package using specfile
    Parameters:
    spec_file: Specfile for RPM package

    Returns: None
    """

    print(inspect.currentframe().f_code.co_name)
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
    """The function moves the package generated in the build folder to the destination folder.

    Parameters:
    pkg_name : Package name
    config: Configuration object containing package metadata

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)

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


def convert_runpath_to_rpath(package_dir):
    """Function will invoke runpath_to_rpath.py script.
    Convert the RUNPATH in binaries and libraries to RPATH

    Parameters:
    package_dir : Package contents directory

    Returns: None
    """

    print(inspect.currentframe().f_code.co_name)
    print("Convert RUNPATH to RPATH")
    try:
        subprocess.run(["python3", "runpath_to_rpath.py", package_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Script failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
        sys.exit(e.returncode)
    except FileNotFoundError as e:
        print(f"Error: Python or script not found. Check your paths. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def update_package_name(pkg_name, config: PackageConfig):
    """Function will update package name by adding suffix.
    rocmversion, -rpath or gfxarch will be added based on conditions
    Note: If package name is updated , make sure to update dependencies as well

    Parameters:
    pkg_name : Package name
    config: Configuration object containing package metadata

    Returns: Updated package name
    """
    print(inspect.currentframe().f_code.co_name)
    if config.versioned_pkg:
        pkg_suffix = config.rocm_version
    else:
        pkg_suffix = ""

    if config.enable_rpath:
        pkg_suffix = f"-rpath{config.rocm_version}"

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
    """Function will replace -devel with -dev string.
    Development package names are defined as devel in json file
    For debian package dev should be used

    Parameters:
    pkg_name : Package name

    Returns: Updated package name
    """
    print(inspect.currentframe().f_code.co_name)
    # Only required for debian developement package
    pkg_name = pkg_name.replace("-devel", "-dev")

    return pkg_name


def convert_to_versiondependency(dependency_list, config: PackageConfig):
    """Change ROCm package dependencies to versioned ones.
    If a package depends on any packages listed in pkg_list,
    the function will append the dependency name with the ROCm version.

    Parameters:
    dependency_list : List of packages
    config: Configuration object containing package metadata

    Returns: String of comma separated packages
    """
    print(inspect.currentframe().f_code.co_name)

    pkg_list = get_package_list()
    updated_depends = [
        f"{update_package_name(pkg,config)}" if pkg in pkg_list else pkg
        for pkg in dependency_list
    ]
    depends = ", ".join(updated_depends)
    return depends


def filter_components_fromartifactory(pkg_name, artifacts_dir, gfx_arch):
    """Get the list of artifactory directories required for creating the package.
    Package.json defines the required artifactories for each package

    Parameters:
    pkg : package name
    gfx_arch : graphics architecture

    Returns: List of directories
    """
    print(inspect.currentframe().f_code.co_name)

    pkg_info = get_package_info(pkg_name)
    is_composite = is_key_defined(pkg_info, "composite")
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
    """Populate the package list based on input arguments
    Exclude disabled packages

    Parameters:
    pkg_name : List of packages or type of packages single/composite

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
    pkg_list = []
    # If pkg_name is None, include all packages
    if pkg_name is None:
        pkg_list = get_package_list()
        return pkg_list

    # Proceed if pkg_name is not None
    data = read_package_json_file()

    for entry in data:
        # Skip if packaging is disabled
        if is_key_defined(entry, "disablepackaging"):
            continue

        name = entry.get("Package")
        is_composite = is_key_defined(entry, "composite")

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


# TODO: The function is not at all required.
# This will be triggered independently
# Wil be removing soon. Keeping it for testing purpose
def download_and_extract_artifacts(run_id, gfxarch):
    """Function will invoke fetch_artifacts.py
    Download the entire artifacts and extract it

    Parameters:
    run_id : Flag to clean artifacts download directory
    gfxarch: Graphics architecture

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
    gfxarch_params = gfxarch + "-dcgpu"
    fetch_script = (SCRIPT_DIR / ".." / ".." / "fetch_artifacts.py").resolve()
    try:
        subprocess.run(
            [
                "python3",
                str(fetch_script),
                "--run-id",
                run_id,
                "--target",
                gfxarch_params,
                "--extract",
                "--output-dir",
                ARTIFACTS_DIR,
            ],
            check=True,
        )
        print("Artifacts fetched successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Artifacts fetch failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
        sys.exit(e.returncode)
    except FileNotFoundError as e:
        print("Error: Python or script not found. Check your paths. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def clean_artifacts_dir(artifacts_dir):
    """Clean the artifacts directory

    Parameters:
    artifacts_dir : Directory where artifacts are stored

    Returns: None
    """
    print(inspect.currentframe().f_code.co_name)
    if os.path.exists(artifacts_dir) and os.path.isdir(artifacts_dir):
        shutil.rmtree(artifacts_dir)
        print(f"Removed directory: {artifacts_dir}")

    if os.path.exists(DEBIAN_CONTENTS_DIR) and os.path.isdir(DEBIAN_CONTENTS_DIR):
        shutil.rmtree(DEBIAN_CONTENTS_DIR)
        print(f"Removed directory: {DEBIAN_CONTENTS_DIR}")
    if os.path.exists(RPM_CONTENTS_DIR) and os.path.isdir(RPM_CONTENTS_DIR):
        shutil.rmtree(RPM_CONTENTS_DIR)
        print(f"Removed directory: {RPM_CONTENTS_DIR}")

    PYCACHE_DIR = "__pycache__"
    if os.path.exists(PYCACHE_DIR) and os.path.isdir(PYCACHE_DIR):
        shutil.rmtree(PYCACHE_DIR)
        print(f"Removed directory: {PYCACHE_DIR}")


def run(args: argparse.Namespace):
    # Clean the packaging artifacts
    clean_artifacts_dir("")
    # Append rocm version to default install prefix
    # TBD: Do we need to append rocm_version to other prefix?
    if args.install_prefix == f"{DEFAULT_INSTALL_PREFIX}":
        prefix = args.install_prefix + "-" + args.rocm_version

    # Populate package config details from user arguments
    config = PackageConfig(
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        dest_dir=Path(args.dest_dir).resolve(),
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
    clean_artifacts_dir("")
    # clean_artifacts_dir(config.artifacts_dir)


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
        "--run-id",
        type=str,
        help="Specify the artifacts run-id",
    )

    p.add_argument(
        "--rocm-version", type=str, default="9.9.9", help="ROCm Release version"
    )

    p.add_argument(
        "--version-suffix",
        default="crdnnh",
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
