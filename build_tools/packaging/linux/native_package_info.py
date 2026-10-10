#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

""" Provides utilities to load and manage ROCm package metadata from JSON files. 
This file is imported by other scripts (installer, uninstaller) and is not executed directly.

Load all packages from a JSON file:

"""


import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from packaging_utils import *


class PackageInfo:
    """
    Represents a single ROCm package definition loaded from JSON.
    Encapsulates metadata and dependency relationships.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        rocm_version: str = "",
        artifact_group: str = "",
        os_family: str = "",
    ):
        """
        Initialize a PackageInfo object with data from JSON and context.

        Parameters:
        data : dict
            JSON dictionary representing the package metadata.
        rocm_version : str
            ROCm version string to be associated with the package.
        artifact_group : str
            Artifact group / GPU family (e.g., gfx94X-dcgpu).
        os_family : str
            Operating system family (e.g., debian, redhat, suse).

        Returns: None
        """

        self.package = data.get("Package")
        self.version = data.get("Version", "")
        self.architecture = data.get("Architecture", "amd64")
        self.build_arch = data.get("BuildArch", "x86_64")
        self.deb_depends = data.get("DEBDepends", [])
        self.rpm_requires = data.get("RPMRequires", [])
        self.maintainer = data.get("Maintainer", "")
        self.description_short = data.get("Description_Short", "")
        self.description_long = data.get("Description_Long", "")
        self.section = data.get("Section", "")
        self.priority = data.get("Priority", "")
        self.group = data.get("Group", "")
        self.license = data.get("License", "")
        self.vendor = data.get("Vendor", "")
        self.homepage = data.get("Homepage", "")
        self.components = data.get("Components", [])
        self.artifact = data.get("Artifact", "")
        self.artifact_subdir = data.get("Artifact_Subdir", "")
        self.gfxarch = str(data.get("Gfxarch", "False")).lower() == "true"
        self.composite = data.get("Composite", "no")  # default to "no" if field missing

        # Added new contextual fields
        self.rocm_version = rocm_version
        self.artifact_group = artifact_group
        self.gfx_suffix = self._derive_gfx_suffix(artifact_group)
        self.os_family = os_family

    def _derive_gfx_suffix(self, artifact_group: str) -> str:
        """
        Extract the GPU architecture suffix from the artifact group string.

        Example:
        'gfx94X-dcgpu' -> 'gfx94x'

        Parameters:
        artifact_group : str
            Artifact group string to parse.

        Returns:
        str : Lowercase GPU suffix or empty string if not present.
        """
        if not artifact_group:
            return ""
        return artifact_group.split("-")[0].lower()

    def is_composite(self) -> bool:
        """
        Check if the package is composite (i.e., bundles multiple artifacts).

        Returns:
        bool : True if composite, False otherwise.
        """
        return str(self.composite).strip().lower() == "yes"

    def summary(self) -> str:
        """
        Return a human-readable summary of the package.

        Returns:
        str : Format "<package> (<version>) - <short description>"
        """
        return f"{self.package} ({self.version}) - {self.description_short}"


class PackageLoader:
    """
    Handles loading, classification, and name derivation of ROCm packages from JSON.
    """

    def __init__(
        self, json_path: str, rocm_version: str = "", artifact_group: str = ""
    ):
        """
        Initialize a PackageLoader for a given JSON file.

        Parameters:
        json_path : str
            Path to the JSON file containing package definitions.
        rocm_version : str, optional
            ROCm version to associate with packages.
        artifact_group : str, optional
            Artifact group / GPU family (e.g., gfx94X-dcgpu).

        Raises:
        FileNotFoundError : if the JSON file does not exist.
        """
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Package JSON file not found: {json_path}")
        self.rocm_version = rocm_version
        self.artifact_group = artifact_group
        self._data = self._load_json()
        self.os_family = get_os_id()

    def _load_json(self) -> List[Dict[str, Any]]:
        """
        Internal method to read JSON data from file.

        Returns:
        list of dict : List of package definitions.
        """

        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_packages(self) -> List[PackageInfo]:
        """
        Load all package definitions from the JSON file.

        Returns:
        list of PackageInfo : All packages with context applied.
        """
        return [
            PackageInfo(entry, self.rocm_version, self.artifact_group, self.os_family)
            for entry in self._data
        ]

    def load_composite_packages(self) -> List[PackageInfo]:
        """
        Filter and return only composite packages.

        Returns:
        list of PackageInfo : Packages where composite=True.
        """
        return [pkg for pkg in self.load_all_packages() if pkg.is_composite()]

    def load_non_composite_packages(self) -> List[PackageInfo]:
        """
        Filter and return only non-composite (base) packages.

        Returns:
        list of PackageInfo : Packages where composite=False.
        """
        return [pkg for pkg in self.load_all_packages() if not pkg.is_composite()]

    def get_package_by_name(self, name: str) -> Optional[PackageInfo]:
        """
        Find a package by its name.

        Parameters:
        name : str
            Package name to look up.

        Returns:
        PackageInfo or None : The matching package object, or None if not found.
        """
        for pkg in self.load_all_packages():
            if pkg.package == name:
                return pkg
        return None

    def get_all_package_names(self) -> set[str]:
        """
        Return the set of all package names defined in the JSON.

        Returns:
        set of str : Package names.
        """
        return {entry.get("Name") for entry in self._data if "Name" in entry}

    def derive_package_names(self, pkg: PackageInfo, version_flag: bool) -> str:

        """
        Compute derived package names for a given package, including valid dependencies.

        The derived names may include:
        - ROCm version suffix
        - GPU architecture suffix
        - Conversion from '-devel' to '-dev' on Debian

        Parameters:
        pkg : PackageInfo
            Base package for which to derive names.
        version_flag : bool
            Include ROCm version in the derived package names if True.

        Returns:
        list of str : Flattened list of derived package names.
        """
        derived_packages = []

        # Get valid dependencies only
        all_pkg_names = self.get_all_package_names()
        deps = pkg.deb_depends if self.os_family == "debian" else pkg.rpm_requires
        valid_deps = [dep for dep in deps if dep in all_pkg_names]

        # Combine current package + valid deps
        pkgs_to_process = valid_deps + [pkg.package]

        sorted_pacakges = []
        derived_packages = []

        for base in pkgs_to_process:
            # Find PackageInfo for this base
            base_pkg = self.get_package_by_name(base)
            if not base_pkg:
                continue

            if base_pkg.os_family == "debian":
                base = re.sub("-devel$", "-dev", base)
            # Determine name with version / gfx suffix
            if (
                base_pkg.gfxarch
                and "devel" not in base.lower()
                and "dev" not in base.lower()
            ):
                if version_flag:
                    derived_packages.append(
                        f"{base}{base_pkg.rocm_version}-{base_pkg.gfx_suffix}"
                    )
                else:
                    derived_packages.append(f"{base}-{base_pkg.gfx_suffix}")
            elif version_flag:
                derived_packages.append(f"{base}{base_pkg.rocm_version}")
            else:
                derived_packages.append(base)

        import itertools

        # Normalize each item to list
        flattened = list(
            itertools.chain.from_iterable(
                sublist if isinstance(sublist, list) else [sublist]
                for sublist in derived_packages
            )
        )
        return flattened
