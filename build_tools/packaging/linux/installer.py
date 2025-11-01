# rocm_package_manager/installer.py

import argparse
import json
from pathlib import Path
from typing import List
from base_manager import PackageManagerBase
from package_info import PackageInfo
from package_info import PackageLoader
import re
import os
import logging
import subprocess
from utils import get_os_id
from utils import logger


class PackageInstaller(PackageManagerBase):
    """
    Handles package installation.
    """

    def __init__(self, package_list: List[PackageInfo], dest_dir: str, rocm_version: str, version_flag: bool, upload: str, amdgpu_family: str, composite: bool, loader):
        super().__init__(package_list)
        self.dest_dir = dest_dir
        self.rocm_version = rocm_version
        self.composite = composite
        self.version_flag = version_flag
        self.os_family = get_os_id()
        self.amdgpu_family = amdgpu_family
        self.upload = upload
        self.loader = loader


    def execute(self):
        logger.info(f"\n=== INSTALLATION PHASE ===")
        logger.info(f"Destination Directory: {self.dest_dir}")
        logger.info(f"ROCm Version: {self.rocm_version}")
        logger.info(f"Composite Build: {self.composite}")


        if self.upload == "post":
            self.populate_repo_file(self.dest_dir)

        for pkg in self.packages:
            logger.info(f"[INSTALL] Installing {pkg.package} ({pkg.architecture})")
            self._install_package(pkg)

        logger.info("Installation complete.")


    def _run_install_command(self, pkg_name, use_repo, pkg_path=None):
        """
        Build and run OS-specific install command for a package.

        :param pkg_name: Name of the package (base name or full path)
        :param pkg_path: Full path for local install (required for local)
        :param use_repo: True if installing from repository, else False
        """
        try:
            if not pkg_name:
                logger.error("Package name is None — cannot install.")
                return

            os_family = self.detect_os_family()
            cmd = None

            # Determine command based on source type and OS
            if not use_repo:
                if not pkg_path:
                    logger.error(f"Local pkg_path must be provided for {pkg_name}")
                    return

                if os_family == "debian":
                    cmd = ["sudo", "dpkg", "-i", pkg_path]
                elif os_family == "redhat":
                    cmd = ["sudo", "rpm", "-ivh", "--replacepkgs", pkg_path]
                elif os_family == "suse":
                    cmd = [
                        "sudo",
                        "zypper",
                        "--non-interactive",
                        "install",
                        "--replacepkgs",
                        pkg_path,
                    ]
                else:
                    logger.error(f"Unsupported OS for local install: {pkg_name}")
                    return
            else:
                if os_family == "debian":
                    cmd = ["sudo", "apt-get", "install", "-y", pkg_name]
                elif os_family == "redhat":
                    cmd = ["sudo", "yum", "install", "-y", pkg_name]
                elif os_family == "suse":
                    cmd = ["sudo", "zypper", "--non-interactive", "install", pkg_name]
                else:
                    logger.error(f"Unsupported OS for repo install: {pkg_name}")
                    return

            # Double-check cmd was built correctly
            if not cmd:
                logger.error(f"No install command generated for {pkg_name}")
                return

            logger.info(f"Running install command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            if result.returncode != 0:
                logger.error(f"Failed to install {pkg_name}:\n{result.stdout}")
            else:
                logger.info(f"Installed {pkg_name} successfully")

        except TypeError as e:
            logger.exception(f"TypeError while installing {pkg_name}: {e}")
        except FileNotFoundError as e:
            logger.exception(f"Command not found: {e}")
        except Exception as e:
            logger.exception(f"Unexpected exception installing {pkg_name}: {e}")

    # ---------------------------------------------------------------------
    # Repo Population
    # ---------------------------------------------------------------------
    def populate_repo_file(self, dest_dir: str):
        """
        Populate a repo file for post-upload installation.
        - Debian: creates /etc/apt/sources.list.d/rocm.list
        - RPM-based: placeholder (to be implemented)
        """
        logger.info(f"Populating repo file for OS: {self.os_family}")

        try:
            base_url = f"https://therock-deb-rpm-test.s3.us-east-2.amazonaws.com/{self.amdgpu_family}_{dest_dir}"

            if self.os_family == "debian":
                repo_file_path = "/etc/apt/sources.list.d/rocm.list"
                repo_entry = f"deb [trusted=yes] {base_url}/deb stable main\n"

                logger.info(f"Writing Debian repo entry to {repo_file_path}")

                cmd = f'echo "{repo_entry.strip()}" | sudo tee {repo_file_path} > /dev/null'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(
                        f"Failed to populate repo file: {result.stderr.strip()}"
                    )
                    raise RuntimeError(
                        f"Error populating repo file: {result.stderr.strip()}"
                    )

                logger.info("Running apt-get update...")
                subprocess.run(["sudo", "apt-get", "update"], check=False)

            elif os_family == "redhat":
                logger.info("Detected RPM-based system. Placeholder for repo setup.")
                repo_file_path = "/etc/yum.repos.d/rocm.repo"
                repo_entry = (
                    f"[rocm]\nname=ROCm Repo\nbaseurl={base_url}/rpm\n"
                    "enabled=1\ngpgcheck=0\n"
                )
                with open(repo_file_path, "w") as f:
                    f.write(repo_entry)
                subprocess.run(["sudo", "yum", "clean", "all"], check=False)
                subprocess.run(["sudo", "yum", "makecache"], check=False)
            else:
                logger.warning(
                    f"Unsupported OS family for repo population: {os_family}"
                )

        except Exception as e:
            logger.error(f"Error populating repo file: {e}")
            raise

    def _run_install_command(self, pkg_name, use_repo, pkg_path=None):
        """
        Build and run OS-specific install command for a package.

        :param pkg_name: Name of the package (base name or full path)
        :param pkg_path: Full path for local install (required for local)
        :param use_repo: True if installing from repository, else False
        """
        try:
            if not pkg_name:
                logger.error("Package name is None — cannot install.")
                return

            cmd = None

            # Determine command based on source type and OS
            if not use_repo:
                if not pkg_path:
                    logger.error(f"Local pkg_path must be provided for {pkg_name}")
                    return

                if self.os_family == "debian":
                    cmd = ["sudo", "dpkg", "-i", pkg_path]
                elif self.os_family == "redhat":
                    cmd = ["sudo", "rpm", "-ivh", "--replacepkgs", pkg_path]
                elif self.os_family == "suse":
                    cmd = [
                        "sudo",
                        "zypper",
                        "--non-interactive",
                        "install",
                        "--replacepkgs",
                        pkg_path,
                    ]
                else:
                    logger.error(f"Unsupported OS for local install: {pkg_name}")
                    return
            else:
                if self.os_family == "debian":
                    cmd = ["sudo", "apt-get", "install", "-y", pkg_name]
                elif self.os_family == "redhat":
                    cmd = ["sudo", "yum", "install", "-y", pkg_name]
                elif self.os_family == "suse":
                    cmd = ["sudo", "zypper", "--non-interactive", "install", pkg_name]
                else:
                    logger.error(f"Unsupported OS for repo install: {pkg_name}")
                    return

            # Double-check cmd was built correctly
            if not cmd:
                logger.error(f"No install command generated for {pkg_name}")
                return

            logger.info(f"Running install command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            if result.returncode != 0:
                logger.error(f"Failed to install {pkg_name}:\n{result.stdout}")
            else:
                logger.info(f"Installed {pkg_name} successfully")

        except TypeError as e:
            logger.exception(f"TypeError while installing {pkg_name}: {e}")
        except FileNotFoundError as e:
            logger.exception(f"Command not found: {e}")
        except Exception as e:
            logger.exception(f"Unexpected exception installing {pkg_name}: {e}")

    def find_packages_for_base(self, dest_dir, derived_name, use_repo):
        """
        Look up packages in local directory or return derived name for repo installation.
        """

        if use_repo:
            return derived_name
        else:
            # If local directory has .deb/.rpm files return matches
            all_files = [
                f for f in os.listdir(dest_dir) if f.endswith((".deb", ".rpm"))
            ]
            matched = [
                os.path.join(dest_dir, f)
                for f in all_files
                if f.startswith(derived_name)
            ]
            if matched:
                return matched
            else:
                logger.error(f"No matching package found for: {derived_name}")

    def _install_package(self, pkg: PackageInfo):
        derived_pkgs = []
        # Handle dependencies

        if self.version_flag:
            derived_pkgs.extend(self.loader.derive_package_names(pkg,True))
        else:
            derived_pkgs.extend(self.loader.derive_package_names(pkg,True))
            derived_pkgs.extend(self.loader.derive_package_names(pkg,False))
        for pkg_name in derived_pkgs:
            if self.upload == "pre":
                derived_name = self.find_packages_for_base(self.dest_dir,pkg,True)
                logger.info(f" - Installing derived package files for {derived_name}...")
                self._run_install_command(derived_name, False, self.dest_dir)
            elif self.upload == "post":
                logger.info(f" - Installing derived package files for {pkg_name} from repo...")
                self._run_install_command(pkg_name, True, self.dest_dir)

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
    parser.add_argument("--dest_dir", required=True, help="Destination directory for installation")
    parser.add_argument("--version", default="false", help="Enable versioning output (true/false)")
    parser.add_argument("--package_json", required=True, help="Path to package JSON definition file")
    parser.add_argument("--composite", default="false", help="Enable composite build mode (true/false)")
    parser.add_argument("--amdgpu_family", default="gfx000", help="GPU family identifier")
    parser.add_argument("--upload", default="none", help="Upload mode (pre/post/none)")
    parser.add_argument("--rocm_version", required=True, help="ROCm version to install")
    return parser.parse_args()


def main():
    """
    Entry point when called directly.
    """
    args = parse_arguments()

    loader = PackageLoader(args.package_json, args.rocm_version, args.amdgpu_family)
    #packages = load_packages_from_json(args.package_json)
    packages = loader.load_composite_packages() if args.composite.lower() == "true" else loader.load_non_composite_packages()


    installer = PackageInstaller(
        package_list=packages,
        dest_dir=args.dest_dir,
        rocm_version=args.rocm_version,
        version_flag=args.version.lower() == "true",
        upload=args.upload,
        amdgpu_family=args.amdgpu_family,
        composite=(args.composite.lower() == "true"),
        loader = loader

    )

    installer.execute()


if __name__ == "__main__":
    main()

