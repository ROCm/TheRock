# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RPM package creation functions."""

import os
import subprocess
import sys
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

from packaging_utils import *
from runpath_to_rpath import *


SCRIPT_DIR = Path(__file__).resolve().parent


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
    # amdrocm-debugger: Exclude libpython requirements
    # Multiple Python-version-specific binaries are included; the wrapper script
    # automatically selects the binary matching the system's Python version
    exclude_libpython_requires = pkg_name == "amdrocm-debugger"

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
