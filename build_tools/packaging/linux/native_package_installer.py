# rocm_package_manager/installer.py

import argparse
import json
from pathlib import Path
from typing import List
from packaging_base_manager import PackageManagerBase
from native_package_info import PackageInfo
from native_package_info import PackageLoader
import re
import os
import logging
import subprocess
from packaging_utils import *


class PackageInstaller(PackageManagerBase):
    """
    Handles package installation.
    """

    def __init__(
        self,
        package_list: List[PackageInfo],
        dest_dir: str,
        run_id: str,
        rocm_version: str,
        version_flag: bool,
        upload: str,
        artifact_group: str,
        composite: bool,
        loader,
    ):
        super().__init__(package_list)
        self.dest_dir = dest_dir
        self.run_id = run_id
        self.rocm_version = rocm_version
        self.composite = composite
        self.version_flag = version_flag
        self.os_family = get_os_id()
        self.artifact_group = artifact_group
        self.upload = upload
        self.loader = loader

    def execute(self):
        logger.info(f"\n=== INSTALLATION PHASE ===")
        logger.info(f"Destination Directory: {self.dest_dir}")
        logger.info(f"ROCm Version: {self.rocm_version}")
        logger.info(f"Composite Build: {self.composite}")

        if self.upload == "post":
            self.populate_repo_file(self.run_id)

        for pkg in self.packages:
            logger.info(f"[INSTALL] Installing {pkg.package} ({pkg.architecture})")
            self._install_package(pkg)

        logger.info("Installation complete.")

    def _run_install_command(self, pkg_name, use_repo):
        """
        Build and run OS-specific install command for a package.

        :param pkg_name: Name of the package (base name or full path)
        :param pkg_path: Full path for local install (required for local)
        :param use_repo: True if installing from repository, else False
        """

        try:
            if not pkg_name:
                logger.error("Package name is None cannot install.")
                return

            cmd = None

            # Determine command based on source type and OS
            if self.upload == "pre":

                if self.os_family == "debian":
                    cmd = ["sudo", "dpkg", "-i", pkg_name]
                elif self.os_family == "redhat":
                    cmd = ["sudo", "rpm", "-ivh", "--replacepkgs", pkg_name]
                elif self.os_family == "suse":
                    cmd = [
                        "sudo",
                        "zypper",
                        "--non-interactive",
                        "install",
                        "--replacepkgs",
                        pkg_name,
                    ]
                else:
                    logger.error(f"Unsupported OS for local install: {pkg_name}")
                    return
            elif self.upload == "post":
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

    # ---------------------------------------------------------------------
    # Repo Population
    # ---------------------------------------------------------------------
    def populate_repo_file(self, run_id: str):
        """
        Populate a repo file for post-upload installation.
        - Debian: creates /etc/apt/sources.list.d/rocm.list
        - RPM-based: placeholder (to be implemented)
        """
        logger.info(f"Populating repo file for OS: {self.os_family}")

        try:
            base_url = f"https://therock-deb-rpm-test.s3.us-east-2.amazonaws.com/{self.artifact_group}_{run_id}"

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

    def find_packages_for_base(self, dest_dir, derived_name):
        """
        Look up packages in local directory or return derived name for repo installation.
        """

        if self.upload == "post":
            return derived_name
        else:
            # If local directory has .deb/.rpm files return matches
            all_files = [
                f for f in os.listdir(dest_dir) if f.endswith((".deb", ".rpm"))
            ]

            pattern = rf"^{re.escape(derived_name)}[_-]{re.escape(self.rocm_version)}[^\s]*\.(deb|rpm)$"

            matched = [
                os.path.join(dest_dir, f) for f in all_files if re.match(pattern, f)
            ]
            if matched:
                return matched
            else:
                logger.error(f"No matching package found for: {derived_name}")

    def _install_package(self, pkg: PackageInfo):
        derived_pkgs = []
        # Handle dependencies

        if self.version_flag:
            derived_name = self.loader.derive_package_names(pkg, True)
            derived_pkgs.extend(derived_name)
        else:
            derived_name = self.loader.derive_package_names(pkg, True)
            derived_pkgs.extend(derived_name)
            derived_name = self.loader.derive_package_names(pkg, False)
            derived_pkgs.extend(derived_name)

        for pkg_name in derived_pkgs:
            if self.upload == "pre":
                derived_name = self.find_packages_for_base(self.dest_dir, pkg_name)
                if derived_name:
                    for derived_pkg in derived_name:
                        self._run_install_command(derived_pkg, True)
            elif self.upload == "post":
                self._run_install_command(pkg_name, True)


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
    # parser.add_argument("--dest-dir", required=True, help="Destination directory for installation")
    parser.add_argument(
        "--version", default="false", help="Enable versioning output (true/false)"
    )
    parser.add_argument(
        "--package-json", required=True, help="Path to package JSON definition file"
    )
    parser.add_argument(
        "--composite", default="false", help="Enable composite build mode (true/false)"
    )
    parser.add_argument(
        "--artifact-group", default="gfx000", help="GPU family identifier"
    )
    parser.add_argument("--rocm-version", required=True, help="ROCm version to install")

    # Add both as optional
    parser.add_argument(
        "--dest-dir", help="Destination directory for installation (optional)"
    )
    parser.add_argument(
        "--run-id", help="Unique identifier for this installation run (optional)"
    )

    return parser.parse_args()


def main():
    """
    Entry point when called directly.
    """
    args = parse_arguments()

    #  Validation: Ensure at least one of them is provided
    if not args.dest_dir and not args.run_id:
        parser.error("You must specify at least one of --dest-dir or --run-id")

    loader = PackageLoader(args.package_json, args.rocm_version, args.artifact_group)
    # packages = load_packages_from_json(args.package_json)
    packages = (
        loader.load_composite_packages()
        if args.composite.lower() == "true"
        else loader.load_non_composite_packages()
    )

    upload = "pre"
    # You can also normalize or auto-assign dest_dir if run_id is given
    if args.run_id and not args.dest_dir:
        upload = "post"

    print("upload=", upload)
    installer = PackageInstaller(
        package_list=packages,
        dest_dir=args.dest_dir,
        run_id=args.run_id,
        rocm_version=args.rocm_version,
        version_flag=args.version.lower() == "true",
        upload=upload,
        artifact_group=args.artifact_group,
        composite=(args.composite.lower() == "true"),
        loader=loader,
    )

    installer.execute()


if __name__ == "__main__":
    main()
