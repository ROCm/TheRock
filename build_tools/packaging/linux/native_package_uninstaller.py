#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Uninstalls ROCm packages from the system using OS package managers.
Composite uninstall (removes all composite packages in reverse order):

```
./uninstall_package.py --run-id 123456 \
    --package-json ./packages.json \
    --rocm-version 6.2.0 \
    --artifact-group gfx94X-dcgpu \
    --composite true 
```
Non-composite uninstall (removes only rocm-core and its versioned package):

```
./uninstall_package.py --run-id 123456 \
    --package-json ./packages.json \
    --rocm-version 6.2.0 \
    --artifact-group gfx94X-dcgpu \
    --composite false 
```
 
"""

import argparse
import json
from pathlib import Path
from typing import List
from packaging_base_manager import PackageManagerBase
from native_package_info import PackageInfo
from native_package_info import PackageLoader
from packaging_utils import *
import subprocess


class PackageUninstaller(PackageManagerBase):
    """
    Handles ROCm package uninstallation on the local system.

    Depending on the mode, either removes all composite packages
    in reverse order or just the core package.
    """

    def __init__(
        self,
        package_list: List[PackageInfo],
        rocm_version: str,
        composite: bool,
        run_id: str,
        loader,
    ):
        super().__init__(package_list)
        self.rocm_version = rocm_version
        self.composite = composite
        self.run_id = run_id
        self.loader = loader
        self.os_family = get_os_id()

    def execute(self):
        """
        Perform the uninstallation.

        Composite mode:
            - Uninstall all packages in reverse dependency order.
        Non-composite mode:
            - Only uninstall 'rocm-core' and its derived packages.

        Logs the progress and errors.
        """
        logger.info(f"\n=== UNINSTALLATION PHASE ===")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"ROCm Version: {self.rocm_version}")
        logger.info(f"Composite Build: {self.composite}")

        # Uninstall in reverse dependency order
        if self.composite:
            for pkg in reversed(self.packages):
                logger.info(f"[REMOVE] Uninstalling {pkg.package}")
                if pkg:
                    derived_name = self.loader.derive_package_names(pkg, True)
                if derived_name:
                    for derived_pkg in derived_name:
                        self._run_uninstall_command(derived_pkg)
        else:
            pkg = self.loader.get_package_by_name("rocm-core")
            if pkg:
                derived_name = self.loader.derive_package_names(pkg, True)
                if derived_name:
                    for derived_pkg in derived_name:
                        self._run_uninstall_command(derived_pkg)
        logger.info(" Uninstallation complete.")

    def _run_uninstall_command(self, pkg_name):
        """
        Execute OS-specific uninstall command for a single package.

        Parameters:
        pkg_name : str
            The base name of the package to uninstall.

        Notes:
        - Debian uses 'apt-get autoremove'
        - RedHat uses 'yum remove'
        - SUSE uses 'zypper remove'
        - Unsupported OS will log an error
        """
        cmd = None

        if self.os_family == "debian":
            cmd = ["sudo", "apt-get", "autoremove", "-y", pkg_name]
        elif self.os_family == "redhat":
            cmd = ["sudo", "yum", "remove", "-y", pkg_name]
        elif self.os_family == "suse":
            cmd = ["sudo", "zypper", "remove", pkg_name]
        else:
            logger.error(f"Unsupported OS for repo uninstall: {pkg_name}")
            return

        # Execute command
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            if result.returncode != 0:
                logger.error(f"Failed to uninstall {pkg_name}:\n{result.stdout}")
            else:
                logger.info(f"Uninstalled {pkg_name}")
        except Exception as e:
            logger.exception(f"Exception uninstalling {pkg_name}: {e}")


def parse_arguments():
    """
    Parses command-line arguments for the uninstaller.
    """
    parser = argparse.ArgumentParser(description="ROCm Package Uninstaller")
    parser.add_argument(
        "--run-id", required=True, help="Unique identifier for this uninstall run"
    )
    parser.add_argument(
        "--package-json", required=True, help="Path to package JSON definition file"
    )
    parser.add_argument(
        "--composite", default="false", help="Composite build mode (true/false)"
    )
    parser.add_argument(
        "--artifact-group", default="gfx000", help="GPU family identifier"
    )
    parser.add_argument(
        "--rocm-version", required=True, help="ROCm version to uninstall"
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the uninstaller script.

    - Parses command-line arguments
    - Loads packages using PackageLoader
    - Initializes the PackageUninstaller
    - Executes uninstallation
    """
    args = parse_arguments()

    loader = PackageLoader(args.package_json, args.rocm_version, args.artifact_group)
    # packages = load_packages_from_json(args.package_json)
    packages = (
        loader.load_composite_packages()
        if args.composite.lower() == "true"
        else loader.load_non_composite_packages()
    )

    uninstaller = PackageUninstaller(
        package_list=packages,
        rocm_version=args.rocm_version,
        composite=(args.composite.lower() == "true"),
        run_id=args.run_id,
        loader=loader,
    )

    uninstaller.execute()


if __name__ == "__main__":
    main()
