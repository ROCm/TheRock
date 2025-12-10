# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import json
import os
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name

import re
import logging
import yaml
import copy

# Create a common logger
logger = logging.getLogger("rocm_installer")
logger.setLevel(logging.INFO)  # default level

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add handler if not already added
if not logger.hasHandlers():
    logger.addHandler(ch)


def print_dict_summary(failure_dict):
    """
    Prints a clean summary of failures without table formatting.
    
    Parameters:
    failure_dict : dict
        Dictionary mapping package names to failure reasons
        
    Raises:
        None - catches all exceptions internally
    """
    try:
        if not failure_dict:
            logger.info("All packages installed successfully.")
            return

        if not isinstance(failure_dict, dict):
            logger.error(f"Expected dict for failure summary, got {type(failure_dict).__name__}")
            return

        lines = []
        lines.append("====== Installation Failure Summary ======\n")

        for pkg, reason in failure_dict.items():
            try:
                clean_reason = str(reason).strip() if reason else "Unknown error"
                lines.append(f" Package: {pkg}")
                lines.append(f"  Reason : {clean_reason}\n")
            except Exception as e:
                logger.error(f"Error formatting failure entry for {pkg}: {e}")
                lines.append(f" Package: {pkg}")
                lines.append(f"  Reason : <formatting error>\n")

        summary_output = "\n".join(lines)
        logger.info("\n" + summary_output)
        
    except Exception as e:
        logger.exception(f"Unexpected error in print_dict_summary: {e}")

def load_yaml_config(yaml_path: str, variables: dict = None) -> dict:
    """
    Load a YAML configuration file and replace placeholders dynamically.

    :param yaml_path: Path to the YAML file.
    :param variables: Dictionary of dynamic variables to substitute (e.g., artifact_group, run_id)
    :return: Dictionary with all placeholders substituted.
    
    :raises FileNotFoundError: If YAML file doesn't exist
    :raises yaml.YAMLError: If YAML syntax is invalid
    :raises ValueError: If yaml_path is empty or invalid
    """
    try:
        if not yaml_path:
            raise ValueError("YAML path cannot be empty")
        
        if variables is None:
            variables = {}
        
        if not isinstance(variables, dict):
            raise TypeError(f"Variables must be a dict, got {type(variables).__name__}")

        # Check if file exists
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
        
        if not os.path.isfile(yaml_path):
            raise ValueError(f"Path is not a file: {yaml_path}")

        # Load YAML file
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML syntax in {yaml_path}: {e}")
            raise
        except PermissionError as e:
            logger.error(f"Permission denied reading {yaml_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading YAML file {yaml_path}: {e}")
            raise

        if raw_config is None:
            logger.warning(f"YAML file {yaml_path} is empty")
            return {}

        # Compile regex pattern for placeholder replacement
        try:
            pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            raise ValueError(f"Regex compilation failed: {e}") from e

        def replace_placeholders(obj):
            """Recursively replace placeholders in nested structures."""
            try:
                if isinstance(obj, dict):
                    return {k: replace_placeholders(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_placeholders(v) for v in obj]
                elif isinstance(obj, str):
                    return pattern.sub(lambda m: str(variables.get(m.group(1), m.group(0))), obj)
                else:
                    return obj
            except Exception as e:
                logger.error(f"Error replacing placeholders in {obj}: {e}")
                return obj

        try:
            result = replace_placeholders(copy.deepcopy(raw_config))
        except Exception as e:
            logger.error(f"Error during placeholder replacement: {e}")
            raise ValueError(f"Placeholder replacement failed: {e}") from e

        return result
        
    except (FileNotFoundError, ValueError, TypeError, yaml.YAMLError):
        raise
    except Exception as e:
        logger.exception(f"Unexpected error loading YAML config from {yaml_path}: {e}")
        raise RuntimeError(f"Failed to load YAML config: {e}") from e


# -------------------------------
# Global OS identification lists
# -------------------------------
DEBIAN_OS_IDS = {"ubuntu", "debian"}
RPM_OS_IDS = {"rhel", "centos", "sles", "almalinux", "fedora", "rocky", "redhat"}


def get_os_id(os_release_path="/etc/os-release"):
    """
    Detect the OS family of the current system.

    Reads the OS release information from `/etc/os-release` to determine
    whether the system belongs to Debian, RedHat, or SUSE family.
    Falls back to generic Linux detection if `/etc/os-release` is not found.

    Parameters:
    os_release_path : str, optional
        Path to the OS release file (default is "/etc/os-release").

    Returns:
    tuple : (os_id, os_family) where:
        - os_id: specific OS identifier (e.g., "ubuntu", "rhel")
        - os_family: OS family as one of: "debian", "rpm", "suse", "linux", or "unknown"
        
    Raises:
        RuntimeError: If OS detection fails critically
    """
    os_release = {}
    
    try:
        # Try to read the OS release file
        try:
            with open(os_release_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        
                        if "=" in line:
                            k, v = line.split("=", 1)
                            os_release[k] = v.strip('"').strip("'")
                    except ValueError as e:
                        logger.warning(f"Skipping malformed line {line_num} in {os_release_path}: {line}")
                        continue
                        
        except FileNotFoundError:
            logger.warning(f"OS release file not found: {os_release_path}")
            # Fallback to platform detection
            try:
                import platform
                system_name = platform.system().lower()
                if "linux" in system_name:
                    logger.info("Detected generic Linux system")
                    return "linux", "linux"
                else:
                    logger.warning(f"Unknown system: {system_name}")
                    return "unknown", "unknown"
            except Exception as e:
                logger.error(f"Failed to detect system using platform module: {e}")
                return "unknown", "unknown"
                
        except PermissionError as e:
            logger.error(f"Permission denied reading {os_release_path}: {e}")
            raise RuntimeError(f"Cannot read OS release file: permission denied") from e
        except Exception as e:
            logger.error(f"Error reading {os_release_path}: {e}")
            raise RuntimeError(f"Failed to read OS release file: {e}") from e

        # Extract OS ID and ID_LIKE
        os_id = os_release.get("ID", "").lower().strip()
        os_like = os_release.get("ID_LIKE", "").lower().strip()
        
        if not os_id:
            logger.warning("OS ID not found in /etc/os-release")
            return "unknown", "unknown"

        logger.debug(f"Detected OS ID: {os_id}, ID_LIKE: {os_like}")

        # Check for Debian-based systems
        if os_id in DEBIAN_OS_IDS or any(x in os_like for x in DEBIAN_OS_IDS):
            logger.debug(f"Identified as Debian-based system")
            return os_id, "debian"

        # Check for RPM-based systems (RedHat, CentOS, etc.)
        if os_id in RPM_OS_IDS or any(x in os_like for x in RPM_OS_IDS):
            logger.debug(f"Identified as RPM-based system")
            return os_id, "rpm"

        # Check for SUSE systems
        if "suse" in os_id or "suse" in os_like:
            logger.debug(f"Identified as SUSE-based system")
            return os_id, "suse"

        # Unknown OS
        logger.warning(f"Unknown OS family for ID: {os_id}")
        return os_id, "unknown"
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error detecting OS: {e}")
        raise RuntimeError(f"OS detection failed: {e}") from e


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


def check_for_gfxarch(pkgname):
    """Check whether the package is associated with a graphics architecture

    Parameters:
    pkgname : Package Name

    Returns:
    bool : True if Gfxarch is set else False.
           False if devel package
    """

    if pkgname.endswith("-devel"):
        return False

    pkg_info = get_package_info(pkgname)
    if str(pkg_info.get("Gfxarch", "false")).strip().lower() == "true":
        return True
    return False


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
    dir_name : Directory to be removed

    Returns: None
    """
    if os.path.exists(dir_name) and os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
        print(f"Removed directory: {dir_name}")


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