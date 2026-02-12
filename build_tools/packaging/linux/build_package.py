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
import glob
import os
import shutil
import subprocess
import sys

from datetime import datetime, timezone
from email.utils import format_datetime
from jinja2 import Environment, FileSystemLoader, Template
from packaging_summary import *
from packaging_utils import *
from pathlib import Path
from runpath_to_rpath import *


SCRIPT_DIR = Path(__file__).resolve().parent
# Default install prefix
DEFAULT_INSTALL_PREFIX = "/opt/rocm/core"


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
    if not config.enable_rpath:
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
    # Set versioned_pkg flag to False
    config.versioned_pkg = False

    package_dir = Path(config.dest_dir) / config.pkg_type / pkg_name
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
    package_dir = (
        Path(config.dest_dir) / config.pkg_type / f"{pkg_name}{config.rocm_version}"
    )
    deb_dir = package_dir / "debian"
    # Create package directory and debian directory
    os.makedirs(deb_dir, exist_ok=True)

    pkg_info = get_package_info(pkg_name)
    is_meta = is_meta_package(pkg_info)
    generate_changelog_file(pkg_info, deb_dir, config)
    generate_rules_file(pkg_info, deb_dir, config)
    generate_control_file(pkg_info, deb_dir, config)
    if is_postinstallscripts_available(pkg_info):
        generate_debian_postscripts(pkg_info, deb_dir, config)

    sourcedir_list = []
    dir_list = filter_components_fromartifactory(
        pkg_name, config.artifacts_dir, config.gfx_arch
    )
    sourcedir_list.extend(dir_list)

    print(f"sourcedir_list:\n  {sourcedir_list}")
    if not sourcedir_list and not is_meta:
        sys.exit(f"{pkg_name}: Empty sourcedir_list and not a meta package, exiting")

    if not sourcedir_list:
        print(f"{pkg_name} is a Meta package")
    else:
        # Copy package contents first
        dest_dir = package_dir / Path(config.install_prefix).relative_to("/")
        for source_path in sourcedir_list:
            copy_package_contents(source_path, dest_dir)

        if config.enable_rpath:
            convert_runpath_to_rpath(package_dir)

        # Generate install file after copying, so we can check for hidden files
        generate_install_file(pkg_info, deb_dir, config, dest_dir)

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
    provides = ""
    replaces = ""
    conflicts = ""
    debrecommends = ""
    debsuggests = ""

    if config.versioned_pkg:
        recommends_list = pkg_info.get("DEBRecommends", [])
        debrecommends = convert_to_versiondependency(recommends_list, config)
        suggests_list = pkg_info.get("DEBSuggests", [])
        debsuggests = convert_to_versiondependency(suggests_list, config)

        depends_list = pkg_info.get("DEBDepends", [])
    else:
        depends_list = [pkg_name]
        provides_list = [
            debian_replace_devel_name(pkg)
            for pkg in (pkg_info.get("Provides", []) or [])
        ]
        provides = ", ".join(provides_list)
        replaces_list = [
            debian_replace_devel_name(pkg)
            for pkg in (pkg_info.get("Replaces", []) or [])
        ]
        replaces = ", ".join(replaces_list)
        conflicts_list = [
            debian_replace_devel_name(pkg)
            for pkg in (pkg_info.get("Conflicts", []) or [])
        ]
        conflicts = ", ".join(conflicts_list)

    depends = convert_to_versiondependency(depends_list, config)
    if is_meta_package(pkg_info):
        depends = append_version_suffix(depends, config)

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
            f"Version string '{args.rocm_version}' does not have major.minor.patch versions"
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

    if not config.enable_rpath:
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
    config.versioned_pkg = False
    package_dir = Path(config.dest_dir) / config.pkg_type / pkg_name
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
    package_dir = (
        Path(config.dest_dir) / config.pkg_type / f"{pkg_name}{config.rocm_version}"
    )
    specfile = package_dir / "specfile"
    generate_spec_file(pkg_name, specfile, config)
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
    # populate packge version details
    version = f"{config.rocm_version}"
    # TBD: Whether to use component version details?
    #    version = pkg_info.get("Version")
    provides = ""
    obsoletes = ""
    conflicts = ""
    rpmrecommends = ""
    rpmsuggests = ""
    sourcedir_list = []
    rpm_scripts = []
    if config.versioned_pkg:
        recommends_list = pkg_info.get("RPMRecommends", [])
        rpmrecommends = convert_to_versiondependency(recommends_list, config)
        suggests_list = pkg_info.get("RPMSuggests", [])
        rpmsuggests = convert_to_versiondependency(suggests_list, config)

        requires_list = pkg_info.get("RPMRequires", [])

        dir_list = filter_components_fromartifactory(
            pkg_name, config.artifacts_dir, config.gfx_arch
        )
        sourcedir_list.extend(dir_list)

        # Filter out non-existing directories
        sourcedir_list = [path for path in sourcedir_list if os.path.isdir(path)]

        if is_postinstallscripts_available(pkg_info):
            rpm_scripts = generate_rpm_postscripts(pkg_info, config)

        if config.enable_rpath:
            for path in sourcedir_list:
                convert_runpath_to_rpath(path)
    else:
        # Provides, Obsoletes and Conflicts field is only valid
        # for non-versioned packages
        provides = ", ".join(pkg_info.get("Provides", []) or [])
        obsoletes = ", ".join(pkg_info.get("Obsoletes", []) or [])
        conflicts = ", ".join(pkg_info.get("Conflicts", []) or [])
        requires_list = [pkg_name]

    requires = convert_to_versiondependency(requires_list, config)
    if is_meta_package(pkg_info):
        requires = append_version_suffix(requires, config)

    # Update package name with version details and gfxarch
    pkg_name = update_package_name(pkg_name, config)

    env = Environment(loader=FileSystemLoader(str(SCRIPT_DIR)))
    template = env.get_template("template/rpm_specfile.j2")

    # Prepare your context dictionary
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

    # Append rocm version to default install prefix
    # TBD: Do we need to append rocm_version to other prefix?
    if args.install_prefix == f"{DEFAULT_INSTALL_PREFIX}":
        prefix = args.install_prefix + "-" + modified_rocm_version

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
