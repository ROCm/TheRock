#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RPM package creation functions for ROCm packaging."""

import os
import re
import subprocess
import sys
from dataclasses import replace
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

from packaging_utils import *
from _therock_utils.log_utils import get_logger

logger = get_logger(__name__)

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent


def create_nonversioned_rpm_package(pkg_name, config: PackageConfig):
    """Create a non-versioned RPM meta package (.rpm).

    Builds a minimal RPM binary package whose payload is empty and whose primary
    purpose is to express dependencies. The package name does not embed a version

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns:
    output_list: List of packages created
    """
    logger.debug("create_nonversioned_rpm_package")
    # Create immutable config copy with versioned_pkg=False
    build_config = replace(config, versioned_pkg=False)

    # Use updated package name for build directory to avoid collisions between variants
    updated_pkg_name = update_package_name(pkg_name, build_config)
    package_dir = Path(build_config.dest_dir) / build_config.pkg_type / updated_pkg_name
    specfile = package_dir / "specfile"
    generate_spec_file(pkg_name, specfile, build_config)
    package_with_rpmbuild(specfile)

    # Move packages to destination
    output_list = move_packages_to_destination(updated_pkg_name, build_config)
    return output_list


def create_versioned_rpm_package(pkg_name, config: PackageConfig):
    """Create a versioned RPM package (.rpm).

    This function automates the process of building a RPM package by:
    1) Generating the spec file with appropriate fields (Package,
       Version, Architecture, Maintainer, Description, and dependencies).
    2) Invoking `rpmbuild` to assemble the final `.rpm` file.

    Parameters:
    pkg_name : Name of the package to be created
    config: Configuration object containing package metadata

    Returns:
    output_list: List of packages created
    """
    logger.debug("create_versioned_rpm_package")
    # Explicitly ensure versioned_pkg=True
    build_config = replace(config, versioned_pkg=True)

    # Use updated package name for build directory to avoid collisions between variants
    # Each variant (host, device, meta) gets its own build directory
    updated_pkg_name = update_package_name(pkg_name, build_config)
    package_dir = Path(build_config.dest_dir) / build_config.pkg_type / updated_pkg_name
    specfile = package_dir / "specfile"
    generate_spec_file(pkg_name, specfile, build_config)
    package_with_rpmbuild(specfile)

    # Move packages to destination
    output_list = move_packages_to_destination(updated_pkg_name, build_config)
    return output_list


def generate_spec_file(pkg_name, specfile, config: PackageConfig):
    """Generate an RPM spec file.

    Parameters:
    pkg_name : Package name
    specfile: Path where the generated spec file should be saved
    config: Configuration object containing package metadata

    Returns: None
    """
    logger.debug("generate_spec_file")
    os.makedirs(os.path.dirname(specfile), exist_ok=True)

    pkg_info = get_package_info(pkg_name)  # Raises ValueError if not found
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
        rpmrecommends = process_secondary_dependencies(
            pkg_info, "RPMRecommends", config
        )
        rpmsuggests = process_secondary_dependencies(pkg_info, "RPMSuggests", config)
        requires = process_main_dependencies(pkg_info, "RPMRequires", config)

        dir_list = filter_components_fromartifactory(
            pkg_name, config.artifacts_dir, config.gfx_arch, config.enable_kpack
        )
        sourcedir_list.extend(dir_list)

        # Filter out non-existing directories
        sourcedir_list = [path for path in sourcedir_list if os.path.isdir(path)]

        # GFX_META is a versioned meta package (empty content, just dependencies)
        is_gfx_meta = config.enable_kpack and config.gfx_arch == GFX_META

        # Warn if we have no artifacts for non-meta packages
        if not sourcedir_list and not is_meta and not is_gfx_meta:
            if config.enable_kpack:
                logger.warning(
                    f"{pkg_name}: Empty sourcedir_list and not a meta package, creating empty RPM"
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

    env = Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=True,
            default=False,
        ),
    )
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
        "disable_rpm_strip": True,
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
    spec_file: Path to the RPM spec file

    Returns: None
    """
    logger.debug("package_with_rpmbuild")
    # Build the command
    cmd = [
        "rpmbuild",
        "-bb",
        spec_file,
        "--define",
        f"_topdir {spec_file.parent}",
    ]

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"RPM Package built successfully: {spec_file.name}\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building RPM package: {spec_file.name}: {e}")
        sys.exit(e.returncode)
