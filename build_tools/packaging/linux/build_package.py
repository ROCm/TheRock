#!/usr/bin/env python3
"""Given ROCm artifacts directories, performs packaging to
create RPM and DEB packages and upload to artifactory server

```
./build_package.py --artifact-url https://therock-artifacts.s3.amazonaws.com/16418185899-linux/index-gfx94X-dcgpu.html \
        --dest-dir ./OUTPUT_PKGDIR \
        --rocm-version 7.1.0
```
"""

import subprocess
import shutil
import glob
import argparse
from pathlib import Path
import sys
import os

from packaging_utils import *
from fetch_artifacts import *
from dataclasses import dataclass

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

# Directory for debian and RPM packaging 
DEBIAN_CONTENTS_DIR = f"{os.getcwd()}/DEB" 
RPM_CONTENTS_DIR = f"{os.getcwd()}/RPM" 
# Default install prefix
DEFAULT_INSTALL_PREFIX = "/opt/rocm"

################### Debian package creation #######################
def create_deb_package(pkg_name, config: PackageConfig):
    ''' Function to create deb package 
    Get package details and generate control file
    Find the required package contents from artifactory
    Copy the package contents to package creation directory
    Create deb package'''

    # Create package contents in DEB/pkg_name/install_prefix folder
    package_dir = f"{DEBIAN_CONTENTS_DIR}/{pkg_name}"
    dest_dir = f"{package_dir}/{config.install_prefix}"
    pkg_info = get_package_info(pkg_name)
    generate_contol_file(pkg_info, package_dir, config)
    # check the package is group of basic package or not
    pkg_list = pkg_info.get("Includes")
    
    if pkg_list is None:
        pkg_list = [pkg_info.get("Package")] 
        
    sourcedir_list = []
    for pkg in pkg_list:
        dir_list = filter_components_fromartifactory(pkg ,dest_dir)
        sourcedir_list.extend(dir_list)


    for source_path in sourcedir_list:
        print(source_path)
        copy_package_contents(source_path, dest_dir )

    pkg_name = update_package_name(pkg_name, config )
    pkg_name = pkg_name.replace("-devel", "-dev")

    debpkg_name = f"{pkg_name}_{config.rocm_version}.{version_to_str(config.rocm_version)}-{config.version_suffix}_{pkg_info.get("Architecture")}" 
    if config.enable_rpath:
        print("ENABLE RPATH")
        subprocess.run(["python3", "runpath_to_rpath.py", package_dir])

    package_with_dpkg_deb(package_dir, config.pkg_dir, debpkg_name)

def generate_contol_file(pkginfo, package_dir, config: PackageConfig):
    '''Function will generate control file for debian package'''
    print("Generate control file")
    control_dir = f"{package_dir}/DEBIAN"
    os.makedirs(control_dir, exist_ok=True)
    controlfile = f"{control_dir}/control"

    pkg_name = update_package_name(pkginfo.get("Package"), config )
    # Only required for debian developement package
    pkg_name = pkg_name.replace("-devel", "-dev")
    arch = pkginfo.get("Architecture")
    description = pkginfo.get("Description")
    version = pkginfo.get("Version")
    section = pkginfo.get("Section")
    priority = pkginfo.get("Priority")
    maintainer = pkginfo.get("Maintainer")
    homepage = pkginfo.get("Homepage")
    depends_list = pkginfo.get("DEBDepends", [])
    depends = convert_to_versiondependency(depends_list, config)
    # Note: The dev package name update should be done after version dependency
    # Package.json maintains development package name as devel
    depends = depends.replace("-devel", "-dev")

    with open(controlfile, 'w') as f:
        f.write(f"Architecture: {arch}\n")
        f.write(f"Depends: {depends}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Homepage: {homepage}\n")
        f.write(f"Maintainer: {maintainer}\n")
        f.write(f"Package: {pkg_name}\n")
        f.write(f"Priority: {priority}\n")
        f.write(f"Section: {section}\n")
        f.write(f"Version: {version}\n")
        f.close()

def copy_package_contents(source_dir, destination_dir):
    ''' Copy package contents from artfactory to package directory'''
   
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
    ''' Create deb package '''
    # Construct paths
    output_deb = f"{output_dir}/{package_name}.deb"

    # Build the command
    cmd = [
        "fakeroot",
        "dpkg-deb",
        "-Zgzip",
        "--build",
        source_dir,
        output_deb
    ]

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print("Package built successfully.")
    except subprocess.CalledProcessError as e:
        print("Error building package:", e)

######################## RPM package creation #################### 
def create_rpm_package(pkg_name, config: PackageConfig):
    ''' Create rpm package by invoking each steps 
    Get package details and generate spec file
    Create rpm package 
    Move the rpm package to destination directory'''

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
    ''' Generate spec file for rpm package'''
    print("Generate Specfile")
    os.makedirs(os.path.dirname(specfile), exist_ok=True)
     
    # Update package name with version details and gfxarch
    pkg_name = update_package_name(pkginfo.get("Package"), config )
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
    
    with open(specfile, 'w') as f:
        f.write(f"Name: {pkg_name}\n")
        f.write(f"Version: {version}\n")
        f.write(f"Release: {release}\n")
        f.write(f"BuildArch: {build_arch}\n")
        f.write(f"Summary: {description}\n")
        f.write(f"Group: {group}\n")
        f.write(f"License: {pkg_license}\n")
        f.write(f"Vendor: {vendor}\n")
        f.write(f"Prefix: {install_prefix}\n")
        # use if check for tags that are optional/can be empty
        if requires:
            f.write(f"Requires: {requires}\n")
        if rpmrecommends:
            f.write(f"Recommends: {rpmrecommends}\n")
        f.write(f"%description\n")
        f.write(f"{description}\n")
        f.write(f"%prep\n")
        f.write(f"%setup -T -D -c -n {pkg_name}\n")
        f.write(f"%build\n")
        f.write(f"%install\n")
        f.write(f"mkdir -p  $RPM_BUILD_ROOT{install_prefix}\n")
    
        # check the package is group of basic package or not
        pkg_list = pkginfo.get("Includes")
    
        if pkg_list is None:
            pkg_list = [pkginfo.get("Package")] 
        
        sourcedir_list = []
        for pkg in pkg_list:
            dir_list = filter_components_fromartifactory(pkg, install_prefix)
            sourcedir_list.extend(dir_list)

        for path in  sourcedir_list:
            if not os.path.isdir(path):
                print(f"Directory does not exist: {path}")
                continue
            if config.enable_rpath:
                print("ENABLE RPATH")
                subprocess.run(["python3", "runpath_to_rpath.py", path])
            f.write(f"cp -R  {path}/* $RPM_BUILD_ROOT{install_prefix}\n")
 
        f.write(f"%files\n")
        f.write(f"{install_prefix}\n")
        f.write(f"%clean\n")
        f.write(f"rm -rf $RPM_BUILD_ROOT\n")
        f.close()

def package_with_rpmbuild(spec_file):
    '''Create rpm package using specfile'''

    package_rpm = os.path.dirname(spec_file) 

    try:
        subprocess.run(
            ["rpmbuild", "--define", f"_topdir {package_rpm}", "-ba", spec_file],
            check=True
        )
        print("RPM build completed successfully.")
    except subprocess.CalledProcessError as e:
        print("RPM build failed:", e)

############### Common functions for packaging ##################
def update_package_name(pkg_name, config: PackageConfig):
    '''Function will update package name by adding suffix.
       rocmversion, -rpath or gfxarch will be added based on conditions
       Note: If package name is updated , make sure to update dependencies as well'''
    pkg_suffix = config.rocm_version
    if config.enable_rpath:
        pkg_suffix = f"-rpath{config.rocm_version}"

    if check_for_gfxarch(pkg_name):
        pkg_name = pkg_name + pkg_suffix + "-" + config.gfx_arch
        #pkg_name = pkg_name + "-" + config.gfx_arch + pkg_suffix
    else:
        pkg_name = pkg_name + pkg_suffix
    return pkg_name

def convert_to_versiondependency(dependency_list, config: PackageConfig):
    '''Change ROCm package dependencies to versioned ones.
    If a package depends on any packages listed in pkg_list,
    the function will append the dependency name with the ROCm version.'''
    pkg_list = get_package_list()
    updated_depends = [
        f"{update_package_name(pkg,config)}" if pkg in pkg_list else pkg
        for pkg in dependency_list
    ]
    depends = ", ".join(updated_depends)
    return depends

def filter_components_fromartifactory(pkg, dest_dir):
    '''Get the list of artifactory directories required for creating the package.
    Package.json defines the required artifactories for each package'''

    pkg_info = get_package_info(pkg)
    sourcedir_list = []
    component_list = pkg_info.get("Components", [])
    artifact_prefix = pkg_info.get("Artifact")
    for component in component_list:
        source_dir = f"{ARTIFACTS_EXTRACT_DIR}/{artifact_prefix}_{component}"
        filename = f"{source_dir}/artifact_manifest.txt"
        with open(filename, 'r') as file:
            for line in file:
                if pkg in line or pkg.replace("-", "_") in line or pkg.replace("-devel", "") in line or pkg.replace("-dev", "") in line:
                    print("Matching line:", line.strip())
                    source_path = f"{source_dir}/{line.strip()}"
                    sourcedir_list.append(source_path)

    print(sourcedir_list)
    return sourcedir_list

def get_gfxarch_from_url(artifact_url):
    '''Extract the gfxarch from the input URL  '''
    # https://therock-artifacts.s3.amazonaws.com/16418185899-linux/index-gfx94X-dcgpu.html
    url_index = artifact_url.rstrip('/').split('/')[-1]
    split_strings = url_index.split('-')
    # Find the part containing 'gfx'
    gfx_arch = next((part for part in split_strings if 'gfx' in part), None)
    return gfx_arch

def parse_input_package_list(pkg_name):
    ''' Populate the package list based on input arguments
        Exclude disabled packages '''
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

def  clean_artifacts_dir(clean_all):
    ''' Clean the artifacts directory'''
    if clean_all:
        clean_artifacts_download_dir()

    clean_artifacts_extract_dir()
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
    #Clean the packaging artifacts
    clean_artifacts_dir(args.clean_build)
    #Create destination dir to save the created packages
    os.makedirs(args.dest_dir, exist_ok=True)

    gfxarch = get_gfxarch_from_url(args.artifact_url)
    # TBD: Full URL will be passed or just Build-ID
    artifact_url = '/'.join(args.artifact_url.rstrip('/').split('/')[:-1])
    # TBD: Whether to parse from url or get it user arguments
    # gfxarch = args.gfx_arch

    pkg_type = args.pkg_type

    # Append rocm version to default install prefix
    if args.install_prefix == f"{DEFAULT_INSTALL_PREFIX}":
        prefix = args.install_prefix +"-"+ args.rocm_version
    # Populate package config details from user arguments 
    config = PackageConfig(
        pkg_dir=args.dest_dir,
        rocm_version=args.rocm_version,
        version_suffix=args.version_suffix,
        install_prefix=prefix,
        gfx_arch=gfxarch,
        enable_rpath=args.rpath_pkg
    )
    pkg_list = parse_input_package_list(args.pkg_names)
    #Download and extract the required artifacts
    for pkg_name in pkg_list:
        download_and_extract_artifacts(str(artifact_url) ,pkg_name, gfxarch)

    # Create deb/rpm packages
    package_creators = {
        "deb": create_deb_package,
        "rpm": create_rpm_package
    }
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
    p.add_argument("--rocm-version", 
        default="9.9.9", 
        help="ROCm Release version")

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
        nargs='+',
        help="Specify the packages to be created: single composite or any specific package name",
    )

    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
