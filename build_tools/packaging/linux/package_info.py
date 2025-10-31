# build_tools/packaging/linux/package_info.py

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class PackageInfo:
    """
    Represents a single ROCm package definition loaded from JSON.
    Encapsulates metadata and dependency relationships.
    """

    def __init__(self, data: Dict[str, Any]):
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
        self.gfxarch = data.get("Gfxarch", "False")

    def is_composite(self) -> bool:
        """
        Check whether the package is composite (i.e., bundles multiple artifacts).
        ROCm convention: Artifact == 'composite' or contains sub-packages.
        """
        return self.artifact.lower() == "composite" or len(self.components) > 1

    def summary(self) -> str:
        return f"{self.package} ({self.version}) - {self.description_short}"

    def __repr__(self):
        kind = "Composite" if self.is_composite() else "Base"
        return f"<{kind} PackageInfo {self.package}>"


class PackageLoader:
    """
    Handles loading and classifying packages from JSON files.
    """

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Package JSON file not found: {json_path}")
        self._data = self._load_json()

    def _load_json(self) -> List[Dict[str, Any]]:
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_packages(self) -> List[PackageInfo]:
        """Load all package definitions from the JSON."""
        return [PackageInfo(entry) for entry in self._data]

    def load_composite_packages(self) -> List[PackageInfo]:
        """Return only composite packages."""
        all_packages = self.load_all_packages()
        return [pkg for pkg in all_packages if pkg.is_composite()]

    def load_non_composite_packages(self) -> List[PackageInfo]:
        """Return only non-composite (base) packages."""
        all_packages = self.load_all_packages()
        return [pkg for pkg in all_packages if not pkg.is_composite()]

    def get_package_by_name(self, name: str) -> Optional[PackageInfo]:
        """Find a package by its name."""
        for pkg in self.load_all_packages():
            if pkg.package == name:
                return pkg
        return None


if __name__ == "__main__":
    # Example usage from command line
    import argparse

    parser = argparse.ArgumentParser(description="ROCm Package Info Loader")
    parser.add_argument("--package_json", required=True, help="Path to package JSON file")
    parser.add_argument("--composite", default="false", help="Load composite packages only (true/false)")
    args = parser.parse_args()

    loader = PackageLoader(args.package_json)

    if args.composite.lower() == "true":
        packages = loader.load_composite_packages()
        print(f"Loaded {len(packages)} composite packages:")
    else:
        packages = loader.load_non_composite_packages()
        print(f"Loaded {len(packages)} non-composite packages:")

    for pkg in packages:
        print("  -", pkg.summary())

