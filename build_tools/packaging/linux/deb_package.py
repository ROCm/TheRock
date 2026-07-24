#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Debian package creation functions for ROCm packaging."""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import format_datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

from packaging_utils import *

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent


def create_nonversioned_deb_package(pkg_name, config: PackageConfig):
    """Create a non-versioned Debian meta package (.deb).

    Builds a minimal Debian binary package whose payload is empty and whose primary
    purpose is to express dependencies. The package name does not embed a version

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns:
    output_list: List of packages created
    """
    print_function_name()
    # Create immutable config copy with versioned_pkg=False
    build_config = replace(config, versioned_pkg=False)

    # Use updated package name for build directory to avoid collisions between variants
    pkg_info = get_package_info(pkg_name)  # Needed for update_package_name
    updated_pkg_name = update_package_name(pkg_info.get("Package"), build_config)
    package_dir = Path(build_config.dest_dir) / build_config.pkg_type / updated_pkg_name
    deb_dir = package_dir / "debian"
    # Create package directory and debian directory
    os.makedirs(deb_dir, exist_ok=True)
    generate_changelog_file(pkg_info, deb_dir, build_config)
    generate_rules_file(pkg_info, deb_dir, build_config)
    generate_control_file(pkg_info, deb_dir, build_config)

    package_with_dpkg_build(package_dir)

    # Move packages to destination
    updated_pkg_name = update_package_name(pkg_name, build_config)
    output_list = move_packages_to_destination(updated_pkg_name, build_config)
    return output_list


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

    Returns:
    output_list: List of packages created
    """
    print_function_name()
    # Explicitly ensure versioned_pkg=True
    build_config = replace(config, versioned_pkg=True)

    # Use updated package name for build directory to avoid collisions between variants
    # Each variant (host, device, meta) gets its own build directory
    updated_pkg_name = update_package_name(pkg_name, build_config)
    package_dir = Path(build_config.dest_dir) / build_config.pkg_type / updated_pkg_name
    deb_dir = package_dir / "debian"
    # Create package directory and debian directory
    os.makedirs(deb_dir, exist_ok=True)

    pkg_info = get_package_info(pkg_name)  # Raises ValueError if not found
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
    # GFX_META is a versioned meta package (empty content, just dependencies)
    is_gfx_meta = build_config.enable_kpack and build_config.gfx_arch == GFX_META
    if not sourcedir_list and not is_meta and not is_gfx_meta:
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

        has_ldconfig_file = generate_ldconfig_file(pkg_info, package_dir, build_config)

        # Generate install file after copying, so we can check for hidden files
        generate_install_file(
            pkg_info,
            deb_dir,
            build_config,
            dest_dir,
            has_ldconfig_file=has_ldconfig_file,
        )

    package_with_dpkg_build(package_dir)

    # Move packages to destination
    updated_pkg_name = update_package_name(pkg_name, build_config)
    output_list = move_packages_to_destination(updated_pkg_name, build_config)
    return output_list


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

    env = Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=True,
            default=False,
        ),
    )
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


def generate_install_file(
    pkg_info,
    deb_dir,
    config: PackageConfig,
    dest_dir=None,
    *,
    has_ldconfig_file=False,
):
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

    env = Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=True,
            default=False,
        ),
    )
    template = env.get_template("template/debian_install.j2")
    # Prepare your context dictionary
    context = {
        "path": config.install_prefix,
        "has_hidden_files": has_hidden_files,
        "has_regular_files": has_regular_files,
        "has_ldconfig_file": has_ldconfig_file,
    }

    with install_file.open("w", encoding="utf-8") as f:
        f.write(template.render(context))


def generate_ldconfig_file(pkg_info, package_dir, config: PackageConfig):
    """Generate a versioned dynamic-loader configuration owned by the package."""
    paths = pkg_info.get("LdconfigPaths", [])
    if not paths:
        return False

    major_minor = ".".join(config.rocm_version.split(".")[:2])
    filename = f"10-{pkg_info['Package']}{major_minor}.conf"
    config_dir = Path(package_dir) / "etc" / "ld.so.conf.d"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / filename).write_text(
        "".join(f"{path}\n" for path in paths), encoding="utf-8"
    )
    return True


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

    env = Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=True,
            default=False,
        ),
    )
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
        debrecommends = process_secondary_dependencies(
            pkg_info, "DEBRecommends", config
        )
        debsuggests = process_secondary_dependencies(pkg_info, "DEBSuggests", config)
        depends = process_main_dependencies(pkg_info, "DEBDepends", config)
    else:
        # Get -> Transform -> Join
        provides = process_name_field(pkg_info, "Provides", debian_replace_devel_name)
        replaces = process_name_field(pkg_info, "Replaces", debian_replace_devel_name)
        conflicts = process_name_field(pkg_info, "Conflicts", debian_replace_devel_name)
        # Non-versioned package depends on versioned package itself
        depends = resolve_versioned_dependencies([pkg_name], config, is_meta)

    pkg_name = update_package_name(pkg_name, config)

    env = Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=True,
            default=False,
        ),
    )
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

    env = Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=True,
            default=False,
        ),
    )
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
