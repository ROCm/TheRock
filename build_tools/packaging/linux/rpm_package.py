#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RPM package creation functions for ROCm packaging."""

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

from packaging_utils import *
from _therock_utils.log_utils import TheRockLogger

logger = TheRockLogger(__name__)

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent


def load_alternatives_binaries() -> dict[str, list[str]]:
    """Load package-specific binary alternatives from JSON."""

    alternatives_file = (
        SCRIPT_DIR / "template" / "scripts" / "amdrocm-alternatives.json"
    )

    with alternatives_file.open(encoding="utf-8") as file:
        alternatives_binaries = json.load(file)

    if not isinstance(alternatives_binaries, dict):
        raise ValueError(f"{alternatives_file} must contain a JSON object")

    for package_name, binaries in alternatives_binaries.items():
        if not isinstance(package_name, str):
            raise ValueError(
                f"Invalid package name in {alternatives_file}: " f"{package_name!r}"
            )

        if not isinstance(binaries, list) or not all(
            isinstance(binary, str) for binary in binaries
        ):
            raise ValueError(
                f"Binary list for {package_name!r} must be a list " "of strings"
            )

    return alternatives_binaries


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
    alternatives_binaries = load_alternatives_binaries()

    # amdrocm-debugger: Exclude libpython requires only (provides are not an issue).
    # Multiple Python-version-specific binaries are included; the wrapper script
    # automatically selects the binary matching the system's Python version.
    exclude_libpython_requires = pkg_name == "amdrocm-debugger"
    # amdrocm-profiler: Exclude vendored TBB from both Requires AND Provides metadata.
    # rocprofiler-systems bundles TBB for Dyninst; suppress public Provides so
    # dyninst/tbb do not block dnf autoremove (ROCM-28385), and suppress
    # auto-Requires so profiler does not couple to distro tbb at install time.
    exclude_vendored_tbb_metadata = pkg_name == "amdrocm-profiler"

    if config.versioned_pkg:
        # Get -> Filter -> Transform
        rpmrecommends = process_secondary_dependencies(
            pkg_info, "RPMRecommends", config
        )
        rpmsuggests = process_secondary_dependencies(pkg_info, "RPMSuggests", config)
        requires = process_versioned_dependencies(pkg_info, "RPMRequires", config)

        dir_list = filter_components_fromartifactory(
            pkg_name,
            config.artifacts_dir,
            config.gfx_arch,
            config.enable_kpack,
            target_members=package_target_members(config),
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

        # Packages listed in amdrocm-alternatives.json use the shared
        # amdrocm-postinst.j2 and amdrocm-prerm.j2 templates.
        if pkg_name in alternatives_binaries:
            rpm_scripts = generate_rpm_postscripts(pkg_info, config)

    else:
        # Get -> Transform -> Join (no transform needed for RPM)
        provides = process_name_field(pkg_info, "Provides")
        obsoletes = process_name_field(pkg_info, "Obsoletes")
        conflicts = process_name_field(pkg_info, "Conflicts")
        # Non-versioned package requires versioned package itself
        requires = process_nonversioned_dependencies(pkg_info, config)

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
        "exclude_vendored_tbb_metadata": exclude_vendored_tbb_metadata,
    }

    with open(specfile, "w", encoding="utf-8") as f:
        f.write(template.render(context))


def generate_rpm_postscripts(pkg_info, config: PackageConfig):
    """Generate RPM %post and %preun sections.

    Parameters:
    pkg_info: Package details parsed from a JSON file
    config: Configuration object containing package metadata

    Returns: Dictionary containing rendered RPM script sections.
    """

    pkg_name = pkg_info.get("Package")
    parts = config.rocm_version.split(".")

    if len(parts) < 3:
        raise ValueError(
            f"Version string '{config.rocm_version}' does not have "
            "major.minor.patch versions"
        )

    version_major_match = re.match(r"^\d+", parts[0])
    version_minor_match = re.match(r"^\d+", parts[1])
    version_patch_match = re.match(r"^\d+", parts[2])

    if not all(
        (
            version_major_match,
            version_minor_match,
            version_patch_match,
        )
    ):
        raise ValueError(f"Unable to parse version string '{config.rocm_version}'")

    env = Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=True,
            default=False,
        ),
    )

    alternatives_binaries = load_alternatives_binaries()

    if pkg_name not in alternatives_binaries:
        raise ValueError(
            f"No alternatives configuration found for RPM package " f"{pkg_name!r}"
        )

    context = {
        "install_prefix": config.install_prefix,
        "version_major": int(version_major_match.group()),
        "version_minor": int(version_minor_match.group()),
        "version_patch": int(version_patch_match.group()),
        "target": "rpm",
        "package_name": pkg_name,
        "binaries": alternatives_binaries[pkg_name],
    }

    shared_scripts = {
        "postinst": "%post",
        "prerm": "%preun",
    }

    rpm_script_sections = {}

    for script, rpm_section in shared_scripts.items():
        template_name = f"template/scripts/amdrocm-{script}.j2"
        template = env.get_template(template_name)
        rpm_script_sections[rpm_section] = template.render(context)

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
        logger.info(f"RPM Package built successfully: {spec_file.parent.name}\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building RPM package: {spec_file.parent.name}: {e}")
        sys.exit(e.returncode)
