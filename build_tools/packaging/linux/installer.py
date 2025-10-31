# rocm_package_manager/installer.py

import argparse
import json
from pathlib import Path
from typing import List
from base_manager import PackageManagerBase
from package_info import PackageInfo
from package_info import PackageLoader



class PackageInstaller(PackageManagerBase):
    """
    Handles package installation.
    """
    def __init__(self, package_list: List[PackageInfo], dest_dir: str, rocm_version: str, composite: bool):
        super().__init__(package_list)
        self.dest_dir = dest_dir
        self.rocm_version = rocm_version
        self.composite = composite

    def execute(self):
        print(f"\n=== INSTALLATION PHASE ===")
        print(f"Destination Directory: {self.dest_dir}")
        print(f"ROCm Version: {self.rocm_version}")
        print(f"Composite Build: {self.composite}")

        for pkg in self.packages:
            print(f"[INSTALL] Installing {pkg.package} ({pkg.architecture})")
            self._install_dependencies(pkg)
            self._install_package(pkg)

        print("Installation complete.")

    def _install_dependencies(self, pkg: PackageInfo):
        deps = pkg.deb_depends or pkg.rpm_requires
        if deps:
            print(f"- Installing dependencies for {pkg.package}: {', '.join(deps)}")

    def _install_package(self, pkg: PackageInfo):
        package_dir = Path(self.dest_dir) / pkg.artifact_subdir
        print(f"   - Creating install directory: {package_dir}")
        package_dir.mkdir(parents=True, exist_ok=True)
        print(f"   - Installing package files for {pkg.package} into {package_dir}...")


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
    Parses command-line arguments for the installer.
    """
    parser = argparse.ArgumentParser(description="ROCm Package Installer")
    parser.add_argument("--dest-dir", required=True, help="Destination directory for installation")
    parser.add_argument("--version", default="false", help="Enable versioning output (true/false)")
    parser.add_argument("--package_json", required=True, help="Path to package JSON definition file")
    parser.add_argument("--composite", default="false", help="Enable composite build mode (true/false)")
    parser.add_argument("--amdgpu_family", default="gfx000", help="GPU family identifier")
    parser.add_argument("--upload", default="none", help="Upload mode (pre/post/none)")
    parser.add_argument("--rocm-version", required=True, help="ROCm version to install")
    return parser.parse_args()


def main():
    """
    Entry point when called directly.
    """
    args = parse_arguments()

    loader = PackageLoader(args.package_json)
    #packages = load_packages_from_json(args.package_json)
    packages = loader.load_composite_packages() if args.composite.lower() == "true" else loader.load_non_composite_packages()

    installer = PackageInstaller(
        package_list=packages,
        dest_dir=args.dest_dir,
        rocm_version=args.rocm_version,
        composite=(args.composite.lower() == "true")
    )

    installer.execute()


if __name__ == "__main__":
    main()

