#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""Given ROCm artifacts directories, performs packaging to
create RPM and DEB packages and upload to artifactory server

```
./build_package.py --artifact-url https://therock-artifacts.s3.amazonaws.com/16418185899-linux/index-gfx94X-dcgpu.html \
        --dest-dir ./OUTPUT_PKGDIR \
        --rocm-version 7.1.0
```
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader, Template
from packaging_utils import *
from pathlib import Path

# User inputs required for packaging
# pkg_dir - For saving the rpm/deb packages
# rocm_version - Used along with package name
# version_suffix - Used along with package name
# install_prefix - Install prefix for the package
# gfx_arch - gfxarch used for building artifacts
@dataclass
class PackageConfig:
    pkg_dir: str
    rocm_version: str
    version_suffix: str
    install_prefix: str
    gfx_arch: str
    enable_rpath: bool


ARTIFACTS_DIR = Path.cwd() / "artifacts_tar"
# Directory for debian and RPM packaging
DEBIAN_CONTENTS_DIR = Path.cwd() / "DEB"
RPM_CONTENTS_DIR = Path.cwd() / "RPM"
# Default install prefix
DEFAULT_INSTALL_PREFIX = "/opt/rocm"

################### Debian package creation #######################
def create_deb_package(pkg_name, config: PackageConfig):
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

    # Create package contents in DEB/pkg_name/install_prefix folder
    package_dir = f"{DEBIAN_CONTENTS_DIR}/{pkg_name}"
    dest_dir = f"{package_dir}/{config.install_prefix}"
    pkg_info = get_package_info(pkg_name)
    generate_control_file(pkg_info, package_dir, config)
    # check the package is group of basic package or not
    pkg_list = pkg_info.get("Includes")

    if pkg_list is None:
        pkg_list = [pkg_info.get("Package")]

    sourcedir_list = []
    for pkg in pkg_list:
        dir_list = filter_components_fromartifactory(pkg, config.gfx_arch)
        sourcedir_list.extend(dir_list)

    for source_path in sourcedir_list:
        print(source_path)
        copy_package_contents(source_path, dest_dir)

    pkg_name = update_package_name(pkg_name, config)
    pkg_name = pkg_name.replace("-devel", "-dev")
    arch = pkg_info.get("Architecture")
    version_str = version_to_str(config.rocm_version)
    debpkg_name = (
        f"{pkg_name}_{config.rocm_version}.{version_str}-{config.version_suffix}_{arch}"
    )
    if config.enable_rpath:
        print("ENABLE RPATH")
        subprocess.run(["python3", "runpath_to_rpath.py", package_dir])

    package_with_dpkg_deb(package_dir, config.pkg_dir, debpkg_name)


def generate_control_file(pkg_info, pkg_dir, config: PackageConfig):
    """Function will generate control file for debian package

    Parameters:
    pkg_info : Package details from the Json file
    pkg_dir: Directory where package contents and control file is saved
    config: Configuration object containing package metadata

    Returns: None
    """

    print("Generate control file")
    control_dir = f"{pkg_dir}/DEBIAN"
    os.makedirs(control_dir, exist_ok=True)
    controlfile = f"{control_dir}/control"

    pkg_name = update_package_name(pkg_info.get("Package"), config)
    # Only required for debian developement package
    pkg_name = pkg_name.replace("-devel", "-dev")
    arch = pkg_info.get("Architecture")
    description = pkg_info.get("Description")
    # version = pkg_info.get("Version")
    version = config.rocm_version
    section = pkg_info.get("Section")
    priority = pkg_info.get("Priority")
    maintainer = pkg_info.get("Maintainer")
    homepage = pkg_info.get("Homepage")
    depends_list = pkg_info.get("DEBDepends", [])
    depends = convert_to_versiondependency(depends_list, config)
    # Note: The dev package name update should be done after version dependency
    # Package.json maintains development package name as devel
    depends = depends.replace("-devel", "-dev")

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("controlfile_template.j2")
    # Prepare your context dictionary
    context = {
        "arch": arch,
        "depends": depends,
        "description": description,
        "homepage": homepage,
        "maintainer": maintainer,
        "pkg_name": pkg_name,
        "priority": priority,
        "section": section,
        "version": version,
    }

    with open(controlfile, "w") as f:
        f.write(template.render(context))
        f.write("\n")  # Adds a blank line. For fixing missing final newline


def copy_package_contents(source_dir, destination_dir):
    """Copy package contents from artfactory to package directory

    Parameters:
    source_dir : Source directory
    destination_dir: Directory where package contents are to be copied

    Returns: None
    """
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


def package_with_dpkg_deb(source_dir, output_dir, package_name):
    """Create deb package

    Parameters:
    source_dir : Package directory containing package contents and control file
    output_dir: Directory where package is created
    package_name: Expected package name

    Returns: None
    """
    # Construct paths
    output_deb = f"{output_dir}/{package_name}.deb"

    # Build the command
    cmd = ["fakeroot", "dpkg-deb", "-Zgzip", "--build", source_dir, output_deb]

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print("Package built successfully.")
    except subprocess.CalledProcessError as e:
        print("Error building package:", e)


######################## RPM package creation ####################
def create_rpm_package(pkg_name, config: PackageConfig):
    """Create rpm package by invoking each steps
    Get package details and generate spec file
    Create rpm package
    Move the rpm package to destination directory

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns: None
    """

    package_dir = f"{RPM_CONTENTS_DIR}/{pkg_name}"
    specfile = f"{package_dir}/specfile"
    pkg_info = get_package_info(pkg_name)
    generate_spec_file(pkg_info, specfile, config)

    package_with_rpmbuild(specfile)
    rpm_files = glob.glob(os.path.join(f"{package_dir}/RPMS/x86_64", "*.rpm"))
    # Move each file to the target directory
    for file_path in rpm_files:
        dest_file = f"{config.pkg_dir}/{os.path.basename(file_path)}"
        if os.path.exists(dest_file):
            os.remove(dest_file)
        shutil.move(file_path, config.pkg_dir)


def generate_spec_file(pkginfo, specfile, config: PackageConfig):
    """Generate spec file for rpm package

    Parameters:
    pkginfo : Package details from the Json file
    specfile: Specfile for RPM package
    config: Configuration object containing package metadata

    Returns: None
    """

    print("Generate Specfile")
    os.makedirs(os.path.dirname(specfile), exist_ok=True)

    # Update package name with version details and gfxarch
    pkg_name = update_package_name(pkginfo.get("Package"), config)
    # populate packge config details
    install_prefix = config.install_prefix
    version = f"{config.rocm_version}.{version_to_str(config.rocm_version)}"
    # TBD: Whether to use component version details?
    #    version = pkginfo.get("Version")
    release = config.version_suffix
    # Populate package details from Json
    description = pkginfo.get("Description")
    arch = pkginfo.get("Architecture")
    build_arch = pkginfo.get("BuildArch")
    section = pkginfo.get("Section")
    priority = pkginfo.get("Priority")
    maintainer = pkginfo.get("Maintainer")
    group = pkginfo.get("Group")
    vendor = pkginfo.get("Vendor")
    pkg_license = pkginfo.get("License")
    homepage = pkginfo.get("Homepage")
    recommends_list = pkginfo.get("RPMRecommends", [])
    rpmrecommends = convert_to_versiondependency(recommends_list, config)

    requires_list = pkginfo.get("RPMRequires", [])
    requires = convert_to_versiondependency(requires_list, config)

    # check the package is group of basic package or not
    pkg_list = pkginfo.get("Includes")

    if pkg_list is None:
        pkg_list = [pkginfo.get("Package")]

    sourcedir_list = []
    for pkg in pkg_list:
        dir_list = filter_components_fromartifactory(pkg, config.gfx_arch)
        sourcedir_list.extend(dir_list)

    # Filter out non-existing directories
    sourcedir_list = [path for path in sourcedir_list if os.path.isdir(path)]

    if config.enable_rpath:
        print("ENABLE RPATH")
        for path in sourcedir_list:
            subprocess.run(["python3", "runpath_to_rpath.py", path])

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("specfile_template.j2")

    # Prepare your context dictionary
    context = {
        "pkg_name": pkg_name,
        "version": version,
        "release": release,
        "build_arch": build_arch,
        "description": description,
        "group": group,
        "pkg_license": pkg_license,
        "vendor": vendor,
        "install_prefix": install_prefix,
        "requires": requires,
        "rpmrecommends": rpmrecommends,
        "sourcedir_list": sourcedir_list,
    }

    with open(specfile, "w") as f:
        f.write(template.render(context))


def package_with_rpmbuild(spec_file):
    """Create rpm package using specfile
    Parameters:
    spec_file: Specfile for RPM package

    Returns: None
    """

    package_rpm = os.path.dirname(spec_file)

    try:
        subprocess.run(
            ["rpmbuild", "--define", f"_topdir {package_rpm}", "-ba", spec_file],
            check=True,
        )
        print("RPM build completed successfully.")
    except subprocess.CalledProcessError as e:
        print("RPM build failed:", e)


############### Common functions for packaging ##################
def update_package_name(pkg_name, config: PackageConfig):
    """Function will update package name by adding suffix.
    rocmversion, -rpath or gfxarch will be added based on conditions
    Note: If package name is updated , make sure to update dependencies as well

    Parameters:
    pkg_name : Package name
    config: Configuration object containing package metadata

    Returns: Updated package name
    """

    pkg_suffix = config.rocm_version
    if config.enable_rpath:
        pkg_suffix = f"-rpath{config.rocm_version}"

    if check_for_gfxarch(pkg_name):
        pkg_name = pkg_name + pkg_suffix + "-" + config.gfx_arch
        # pkg_name = pkg_name + "-" + config.gfx_arch + pkg_suffix
    else:
        pkg_name = pkg_name + pkg_suffix
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

    pkg_list = get_package_list()
    updated_depends = [
        f"{update_package_name(pkg,config)}" if pkg in pkg_list else pkg
        for pkg in dependency_list
    ]
    depends = ", ".join(updated_depends)
    return depends


def filter_components_fromartifactory(pkg, gfx_arch):
    """Get the list of artifactory directories required for creating the package.
    Package.json defines the required artifactories for each package

    Parameters:
    pkg : package name

    Returns: List of directories
    """

    pkg_info = get_package_info(pkg)
    sourcedir_list = []
    component_list = pkg_info.get("Components", [])
    artifact_prefix = pkg_info.get("Artifact")
    if str(pkg_info.get("Gfxarch", "false")).strip().lower() == "true":
        artifact_suffix = gfx_arch + "-dcgpu"
    else:
        artifact_suffix = "generic"

    for component in component_list:
        source_dir = f"{ARTIFACTS_DIR}/{artifact_prefix}_{component}_{artifact_suffix}"
        filename = f"{source_dir}/artifact_manifest.txt"
        with open(filename, "r") as file:
            for line in file:
                if (
                    pkg in line
                    or pkg.replace("-", "_") in line
                    or pkg.replace("-devel", "") in line
                    or pkg.replace("-dev", "") in line
                ):
                    print("Matching line:", line.strip())
                    source_path = f"{source_dir}/{line.strip()}"
                    sourcedir_list.append(source_path)

    print(sourcedir_list)
    return sourcedir_list


def extract_build_id(url):
    """Extract the buildid from the input URL
    Parameters:
    artifact_url : Artifacts directory URL

    Returns: build id
    """

    match = re.search(r"/(\d+)-linux/", url)
    if match:
        return match.group(1)
    else:
        return None


def get_gfxarch_from_url(artifact_url):
    """Extract the gfxarch from the input URL
    Parameters:
    artifact_url : Artifacts directory URL

    Returns: None
    """

    # https://therock-artifacts.s3.amazonaws.com/16418185899-linux/index-gfx94X-dcgpu.html
    url_index = artifact_url.rstrip("/").split("/")[-1]
    split_strings = url_index.split("-")
    # Find the part containing 'gfx'
    gfx_arch = next((part for part in split_strings if "gfx" in part), None)
    return gfx_arch


def parse_input_package_list(pkg_name):
    """Populate the package list based on input arguments
    Exclude disabled packages

    Parameters:
    pkg_name : List of packages or type of packages single/composite

    Returns: None
    """

    pkg_list = []
    # If pkg_type is None, include all packages
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
        is_composite = any(key.lower() == "composite" for key in entry)

        # Loop through each type in pkg_type
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

    print(pkg_list)
    return pkg_list


def clean_artifacts_dir(clean_all):
    """Clean the artifacts directory

    Parameters:
    clean_all : Flag to clean artifacts download directory

    Returns: None
    """
    if clean_all:
        if os.path.exists(ARTIFACTS_DIR) and os.path.isdir(ARTIFACTS_DIR):
            shutil.rmtree(ARTIFACTS_DIR)
            print(f"Removed directory: {ARTIFACTS_DIR}")

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
    clean_artifacts_dir(args.clean_build)
    # Create destination dir to save the created packages
    os.makedirs(args.dest_dir, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    gfxarch = get_gfxarch_from_url(args.artifact_url)
    # TBD: Whether to parse from url or get it user arguments
    #    artifact_url = "/".join(args.artifact_url.rstrip("/").split("/")[:-1])
    #    gfxarch = args.gfx_arch
    #    run_id = args.run_id
    pkg_type = args.pkg_type
    # Append rocm version to default install prefix
    if args.install_prefix == f"{DEFAULT_INSTALL_PREFIX}":
        prefix = args.install_prefix + "-" + args.rocm_version
    # Populate package config details from user arguments
    config = PackageConfig(
        pkg_dir=args.dest_dir,
        rocm_version=args.rocm_version,
        version_suffix=args.version_suffix,
        install_prefix=prefix,
        gfx_arch=gfxarch,
        enable_rpath=args.rpath_pkg,
    )
    pkg_list = parse_input_package_list(args.pkg_names)
    # Download and extract the required artifacts
    run_id = extract_build_id(args.artifact_url)
    gfxarch_params = gfxarch + "-dcgpu"

    subprocess.run(
        [
            "python3",
            "../../fetch_artifacts.py",
            "--run-id",
            run_id,
            "--target",
            gfxarch_params,
            "--extract",
            "--all",
            "--output-dir",
            ARTIFACTS_DIR,
        ]
    )
    # Using fetch_artiifacts.py from current folder.
    # Commented for time being and use the one provided by build_tools
    #    for pkg_name in pkg_list:
    #        download_and_extract_artifacts(str(artifact_url), pkg_name, gfxarch)

    # Create deb/rpm packages
    package_creators = {"deb": create_deb_package, "rpm": create_rpm_package}
    for pkg_name in pkg_list:
        if pkg_type and pkg_type.lower() in package_creators:
            print(f"Create pkg_type.upper() package.")
            package_creators[pkg_type.lower()](pkg_name, config)
        else:
            print("Create both DEB and RPM packages.")
            for creator in package_creators.values():
                creator(pkg_name, config)
    # The artifacts directory should be cleaned
    clean_artifacts_dir("True")


def main(argv: list[str]):

    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifact-url",
        type=str,
        required=True,
        help="Source artifacts/ dir from a build",
    )
    p.add_argument(
        "--run-id",
        type=str,
        help="Source artifacts/ dir from a build",
    )

    p.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Destination directory in which to materialize packages",
    )
    p.add_argument(
        "--gfx-arch",
        help="Graphix architecture used for building",
    )

    p.add_argument(
        "--pkg-type",
        help="Choose the package format to be generated: DEB or RPM",
    )
    p.add_argument("--rocm-version", default="9.9.9", help="ROCm Release version")

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
        help="Specify the packages to be created: single composite or any specific package name",
    )

    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
