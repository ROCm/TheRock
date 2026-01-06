# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import json
import os
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name


def print_function_name():
    """Print the name of the calling function.

    Parameters: None

    Returns: None
    """
    print("In function:", currentFuncName(1))


def read_package_json_file():
    """Reads package.json file and return the parsed data.

    Parameters: None

    Returns: Parsed JSON data containing package details
    """
    file_path = SCRIPT_DIR / "package.json"
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def is_key_defined(pkg_info, key):
    """
    Verifies whether a specific key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.
    key : A key to be searched in the dictionary.

    Returns:
    bool: True if key is defined, False otherwise.
    """
    value = ""
    for k in pkg_info:
        if k.lower() == key.lower():
            value = pkg_info[k]

    value = value.strip().lower()
    if value in (
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "enable",
        "enabled",
        "found",
    ):
        return True
    if value in (
        "",
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "disable",
        "disabled",
        "notfound",
        "none",
        "null",
        "nil",
        "undefined",
        "n/a",
    ):
        return False


def is_postinstallscripts_available(pkg_info):
    """
    Verifies whether Postinstall key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if Postinstall key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Postinstall")


def is_meta_package(pkg_info):
    """
    Verifies whether Metapackage key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if Metapackage key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Metapackage")


def is_composite_package(pkg_info):
    """
    Verifies whether composite key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if composite key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "composite")


def is_rpm_stripping_disabled(pkg_info):
    """
    Verifies whether Disable_RPM_STRIP key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if Disable_RPM_STRIP key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Disable_RPM_STRIP")


def is_debug_package_disabled(pkg_info):
    """
    Verifies whether Disable_Debug_Package key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if Disable_Debug_Package key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Disable_Debug_Package")


def is_packaging_disabled(pkg_info):
    """
    Verifies whether 'Disablepackaging' key is enabled for a package.

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool: True if 'Disablepackaging' key is defined, False otherwise.
    """

    return is_key_defined(pkg_info, "Disablepackaging")


def is_gfxarch_package(pkg_info):
    """Check whether the package is associated with a graphics architecture

    Parameters:
    pkg_info (dict): A dictionary containing package details.

    Returns:
    bool : True if Gfxarch is set, else False.
           #False if devel package
    """
    #  Disabling this for time being as per the requirements
    #   if pkgname.endswith("-devel"):
    #       return False

    return is_key_defined(pkg_info, "Gfxarch")


def get_package_info(pkgname):
    """Retrieves package details from a JSON file for the given package name

    Parameters:
    pkgname : Package Name

    Returns: Package metadata
    """

    # Load JSON data from a file
    data = read_package_json_file()

    for package in data:
        if package.get("Package") == pkgname:
            return package

    return None


def get_package_list():
    """Read package.json and return package names.

    Packages marked as 'Disablepackaging' will be excluded from the list

    Parameters: None

    Returns: Package list
    """

    data = read_package_json_file()

    pkg_list = [pkg["Package"] for pkg in data if not is_packaging_disabled(pkg)]
    return pkg_list


def remove_dir(dir_name):
    """Remove the directory if it exists

    Parameters:
    dir_name : Path or str
        Directory to be removed

    Returns: None
    """
    dir_path = Path(dir_name)

    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
        print(f"Removed directory: {dir_path}")
    else:
        print(f"Directory does not exist: {dir_path}")


def version_to_str(version_str):
    """Convert a ROCm version string to a numeric representation.

    This function transforms a ROCm version from its dotted format
    (e.g., "7.1.0") into a numeric string (e.g., "70100")
    Ex : 7.10.0 -> 71000
         10.1.0 - > 100100
         7.1 -> 70100
         7.1.1.1 -> 70101

    Parameters:
    version_str: ROCm version separated by dots

    Returns: Numeric string
    """

    parts = version_str.split(".")
    # Ensure we have exactly 3 parts: major, minor, patch
    while len(parts) < 3:
        parts.append("0")  # Default missing parts to "0"
    major, minor, patch = parts[:3]  # Ignore extra parts

    return f"{int(major):01d}{int(minor):02d}{int(patch):02d}"
