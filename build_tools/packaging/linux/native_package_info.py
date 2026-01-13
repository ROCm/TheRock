#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Provides utilities to load and manage ROCm package metadata from JSON files.
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
        os_id: str = ""
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
        self.metapackage = data.get("Metapackage", "no")  # default to "no" if field missing
        self.disable_packaging = data.get("DisablePackaging", "no")  # default to "no" if field missing

        # Added new contextual fields
        self.rocm_version = rocm_version
        self.artifact_group = artifact_group
        self.gfx_suffix = self._derive_gfx_suffix(artifact_group)
        self.os_family = os_family
        self.os_id = os_id

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

    def is_metapackage(self) -> bool:
        """
        Check if the package is metapackage (i.e., bundles multiple artifacts).

        Returns:
        bool : True if metapackage, False otherwise.
        """
        value = str(self.metapackage).strip().lower()
        return value in ("yes", "true", "1", "t", "y")

    def is_packaging_disabled(self) -> bool:
        """
        Check if packaging is disabled for this package.

        Returns:
        bool : True if DisablePackaging is set, False otherwise.
        """
        value = str(self.disable_packaging).strip().lower()
        return value in ("yes", "true", "1", "t", "y")

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
        ValueError : if json_path is empty or invalid
        RuntimeError : if OS detection fails or JSON loading fails
        """
        try:
            if not json_path:
                raise ValueError("JSON path cannot be empty")
            
            self.json_path = Path(json_path)
            
            if not self.json_path.exists():
                raise FileNotFoundError(f"Package JSON file not found: {json_path}")
            
            if not self.json_path.is_file():
                raise ValueError(f"Path is not a file: {json_path}")
            
            self.rocm_version = rocm_version
            self.artifact_group = artifact_group
            
            # Load JSON data
            try:
                self._data = self._load_json()
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in {json_path}: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to load JSON from {json_path}: {e}") from e
            
            # Detect OS
            try:
                self.os_id, self.os_family = get_os_id()
            except Exception as e:
                raise RuntimeError(f"Failed to detect operating system: {e}") from e
                
        except (FileNotFoundError, ValueError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error initializing PackageLoader: {type(e).__name__} - {str(e)}") from e

    def _load_json(self) -> List[Dict[str, Any]]:
        """
        Internal method to read JSON data from file.

        Returns:
        list of dict : List of package definitions.
        
        Raises:
        json.JSONDecodeError : if JSON is malformed
        FileNotFoundError : if file doesn't exist
        PermissionError : if file cannot be read
        """
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Validate that data is a list
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array, got {type(data).__name__}")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON syntax in {self.json_path}: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"File not found: {self.json_path}")
            raise
        except PermissionError as e:
            logger.error(f"Permission denied reading file: {self.json_path}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error loading JSON from {self.json_path}: {e}")
            raise

    def load_all_packages(self) -> List[PackageInfo]:
        """
        Load all package definitions from the JSON file.

        Returns:
        list of PackageInfo : All packages with context applied.
        
        Raises:
        ValueError : if package data is malformed
        """
        try:
            packages = []
            for idx, entry in enumerate(self._data):
                try:
                    if not isinstance(entry, dict):
                        logger.warning(f"Skipping entry {idx}: expected dict, got {type(entry).__name__}")
                        continue
                    
                    pkg = PackageInfo(
                        entry, 
                        self.rocm_version, 
                        self.artifact_group, 
                        self.os_family,
                        self.os_id
                    )
                    packages.append(pkg)
                    
                except KeyError as e:
                    logger.error(f"Package entry {idx} missing required field: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing package entry {idx}: {type(e).__name__} - {str(e)}")
                    continue
            
            return packages
            
        except Exception as e:
            logger.exception(f"Unexpected error loading all packages: {e}")
            raise

    def load_metapackage_packages(self) -> List[PackageInfo]:
        """
        Filter and return only metapackages.

        Returns:
        list of PackageInfo : Packages where metapackage=True and DisablePackaging is not set.
        """
        all_packages = self.load_all_packages()
        metapackages = [pkg for pkg in all_packages if pkg.is_metapackage() and not pkg.is_packaging_disabled()]
        logger.info(f"Loaded {len(metapackages)} metapackage(s) from {len(all_packages)} total packages in {self.json_path}")
        return metapackages

    def load_non_metapackage_packages(self) -> List[PackageInfo]:
        """
        Filter and return only non-metapackage (base) packages.

        Returns:
        list of PackageInfo : Packages where metapackage=False and DisablePackaging is not set.
        """
        all_packages = self.load_all_packages()
        non_metapackages = [pkg for pkg in all_packages if not pkg.is_metapackage() and not pkg.is_packaging_disabled()]
        logger.info(f"Loaded {len(non_metapackages)} non-metapackage(s) from {len(all_packages)} total packages in {self.json_path}")
        return non_metapackages

    def get_package_by_name(self, name: str) -> Optional[PackageInfo]:
        """
        Find a package by its name.

        Parameters:
        name : str
            Package name to look up.

        Returns:
        PackageInfo or None : The matching package object, or None if not found.
        """
        try:
            if not name:
                logger.warning("Empty package name provided")
                return None
            
            for pkg in self.load_all_packages():
                if pkg.package == name:
                    return pkg
            
            logger.debug(f"Package '{name}' not found")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for package '{name}': {e}")
            return None

    def get_all_package_names(self) -> set:
        """
        Return the set of all package names defined in the JSON.

        Returns:
        set of str : Package names.
        """
        try:
            # Use "Package" field, not "Name" - that's what's in the JSON
            names = set()
            for entry in self._data:
                if not isinstance(entry, dict):
                    continue
                
                # Try "Package" field first (correct field name)
                pkg_name = entry.get("Package")
                if pkg_name:
                    names.add(pkg_name)
                # Fallback to "Name" for backwards compatibility
                elif "Name" in entry:
                    names.add(entry.get("Name"))
            
            return names
            
        except Exception as e:
            logger.error(f"Error getting package names: {e}")
            return set()

    def derive_package_names(self, pkg: PackageInfo, version_flag: bool) -> List[str]:
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
        
        Raises:
        AttributeError : if package object is missing required attributes
        ValueError : if package data is invalid
        """
        try:
            if not pkg:
                raise ValueError("Package object is None")
            
            if not hasattr(pkg, 'package') or not pkg.package:
                raise AttributeError("Package object missing 'package' attribute")
            
            derived_packages = []

            # Get valid dependencies only
            try:
                all_pkg_names = self.get_all_package_names()
            except Exception as e:
                logger.error(f"Error getting all package names: {e}")
                all_pkg_names = set()
            
            # Get dependencies based on OS family
            # For metapackages, skip dependencies as they will be pulled in automatically
            if pkg.is_metapackage():
                valid_deps = []
            else:
                try:
                    deps = pkg.deb_depends if self.os_family == "debian" else pkg.rpm_requires
                    if not isinstance(deps, list):
                        logger.warning(f"Dependencies for {pkg.package} is not a list, using empty list")
                        deps = []
                except AttributeError as e:
                    logger.warning(f"Package {pkg.package} missing dependency attributes: {e}")
                    deps = []
                
                valid_deps = [dep for dep in deps if dep in all_pkg_names]

            # Combine current package + valid deps
            pkgs_to_process = valid_deps + [pkg.package]

            for base in pkgs_to_process:
                try:
                    # Find PackageInfo for this base
                    base_pkg = self.get_package_by_name(base)
                    if not base_pkg:
                        logger.debug(f"Package '{base}' not found, skipping")
                        continue

                    # Convert -devel to -dev on Debian
                    if base_pkg.os_family == "debian":
                        try:
                            base = re.sub("-devel$", "-dev", base)
                        except re.error as e:
                            logger.error(f"Regex error processing package name '{base}': {e}")
                            continue
                    
                    # Determine name with version / gfx suffix
                    try:
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
                    except AttributeError as e:
                        logger.error(f"Package {base} missing required attribute for name derivation: {e}")
                        # Fallback to base name
                        derived_packages.append(base)
                        
                except Exception as e:
                    logger.error(f"Error processing package '{base}': {type(e).__name__} - {str(e)}")
                    continue

            # Flatten the list
            try:
                import itertools
                flattened = list(
                    itertools.chain.from_iterable(
                        sublist if isinstance(sublist, list) else [sublist]
                        for sublist in derived_packages
                    )
                )
            except Exception as e:
                logger.error(f"Error flattening package list: {e}")
                # Fallback: assume all are strings
                flattened = [p for p in derived_packages if isinstance(p, str)]
            
            if not flattened:
                logger.warning(f"No derived package names generated for {pkg.package}")
            
            return flattened
            
        except (ValueError, AttributeError):
            raise
        except Exception as e:
            logger.exception(f"Unexpected error deriving package names: {e}")
            raise
