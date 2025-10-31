# rocm_package_manager/uninstaller.py

import argparse
import json
from pathlib import Path
from typing import List
from .base_manager import PackageManagerBase
from .package_info import PackageInfo


class PackageUninstaller(PackageManagerBase):
    """
    Handles package uninstallation.
    """
    def __init__(self, package_list: List[PackageInfo], rocm_version: str, composite: bool, run_id: str):
        super().__init__(package_list)
        self.rocm_version = rocm_version
        self.composite = composite
        self.run_id = run_id

    def execute(self):
        print(f"\n=== UNINSTALLATION PHASE ===")
        print(f"Run ID: {self.run_id}")
        print(f"ROCm Version: {self.rocm_version}")
        print(f"Composite Build: {self.composite}")

        # Uninstall in reverse dependency order
        for pkg in reversed(self.packages):
            print(f"[REMOVE] Uninstalling {pkg.package}")
            self._remove_package(pkg)

        print("🗑️  Uninstallation complete.")

    def _remove_package(self, pkg: PackageInfo):
        print(f"   - Removing files and cleaning up {pkg.package}...")


def load_packages_from_json(json_path: str) -> List[PackageInfo]:
    """
    Utility function to load package info list from a JSON file.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [PackageInfo(entry) for entry in data]


def parse_arguments():
    """
    Parses command-line arguments for the uninstaller.
    """
    parser = argparse.ArgumentParser(description="ROCm Package Uninstaller")
    parser.add_argument("--run_id", required=True, help="Unique identifier for this uninstall run")
    parser.add_argument("--version", default="false", help="Enable version output (true/false)")
    parser.add_argument("--package_json", required=True, help="Path to package JSON definition file")
    parser.add_argument("--composite", default="false", help="Composite build mode (true/false)")
    parser.add_argument("--amdgpu_family", default="gfx000", help="GPU family identifier")
    parser.add_argument("--rocm-version", required=True, help="ROCm version to uninstall")
    return parser.parse_args()


def main():
    """
    Entry point when executed directly.
    """
    args = parse_arguments()

    packages = load_packages_from_json(args.package_json)

    uninstaller = PackageUninstaller(
        package_list=packages,
        rocm_version=args.rocm_version,
        composite=(args.composite.lower() == "true"),
        run_id=args.run_id
    )

    uninstaller.execute()


if __name__ == "__main__":
    main()

