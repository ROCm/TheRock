# build_tools/packaging/linux/package_info.py

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
        """Extract gfx suffix like 'gfx94x' from 'gfx94X-dcgpu'."""
        if not artifact_group:
            return ""
        return artifact_group.split("-")[0].lower()

    def is_composite(self) -> bool:
        """Check whether the package is composite (bundles multiple artifacts)."""
        return str(self.composite).strip().lower() == "yes"

    def summary(self) -> str:
        return f"{self.package} ({self.version}) - {self.description_short}"


class PackageLoader:
    """
    Handles loading and classifying packages from JSON files.
    """

    def __init__(
        self, json_path: str, rocm_version: str = "", artifact_group: str = ""
    ):
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Package JSON file not found: {json_path}")
        self.rocm_version = rocm_version
        self.artifact_group = artifact_group
        self._data = self._load_json()
        self.os_family = get_os_id()

    def _load_json(self) -> List[Dict[str, Any]]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_packages(self) -> List[PackageInfo]:
        """Load all package definitions from the JSON."""
        return [
            PackageInfo(entry, self.rocm_version, self.artifact_group, self.os_family)
            for entry in self._data
        ]

    def load_composite_packages(self) -> List[PackageInfo]:
        """Return only composite packages."""
        return [pkg for pkg in self.load_all_packages() if pkg.is_composite()]

    def load_non_composite_packages(self) -> List[PackageInfo]:
        """Return only non-composite (base) packages."""
        return [pkg for pkg in self.load_all_packages() if not pkg.is_composite()]

    def get_package_by_name(self, name: str) -> Optional[PackageInfo]:
        """Find a package by its name."""
        for pkg in self.load_all_packages():
            if pkg.package == name:
                return pkg
        return None

    def get_all_package_names(self) -> set[str]:
        """Return a set of all package names defined in the JSON."""
        return {entry.get("Name") for entry in self._data if "Name" in entry}

    def derive_package_names(self, pkg: PackageInfo, version_flag: bool) -> str:

        """
        Returns a list of derived package names including valid dependencies.
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


if __name__ == "__main__":
    # Example usage from command line
    import argparse

    parser = argparse.ArgumentParser(description="ROCm Package Info Loader")
    parser.add_argument(
        "--package_json", required=True, help="Path to package JSON file"
    )
    parser.add_argument("--rocm-version", default="", help="ROCm version string")
    parser.add_argument(
        "--artifact-group", default="", help="AMD GPU family (e.g. gfx94X-dcgpu)"
    )
    parser.add_argument(
        "--composite", default="false", help="Load composite packages only (true/false)"
    )
    parser.add_argument(
        "--version",
        default="false",
        help="Include ROCm version in derived name (true/false)",
    )
    args = parser.parse_args()

    loader = PackageLoader(
        args.package_json,
        rocm_version=args.rocm_version,
        artifact_group=args.artifact_group,
    )

    version_flag = args.version.lower() == "true"

    if args.composite.lower() == "true":
        packages = loader.load_composite_packages()
        logger.info(f"Loaded {len(packages)} composite packages:")
    else:
        packages = loader.load_non_composite_packages()
        logger.info(f"Loaded {len(packages)} non-composite packages:")
