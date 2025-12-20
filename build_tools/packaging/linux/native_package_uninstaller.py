#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Uninstalls ROCm packages from the system using OS package managers.
Metapackage uninstall (removes all metapackages in reverse order):

```
./uninstall_package.py \
    --package-json ./packages.json \
    --rocm-version 6.2.0 \
    --artifact-group gfx94X-dcgpu \
    --metapackage true
```
Non-metapackage uninstall (removes all non-metapackages in reverse order):

```
./uninstall_package.py \
    --package-json ./packages.json \
    --rocm-version 6.2.0 \
    --artifact-group gfx94X-dcgpu \
    --metapackage false
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

    Depending on the mode, either removes all metapackages
    in reverse order or all non-metapackages in reverse order.
    """

    def __init__(
        self,
        package_list: List[PackageInfo],
        rocm_version: str,
        metapackage: bool,
        loader,
    ):
        """
        Initialize PackageUninstaller with configuration and package list.
        
        Raises:
            ValueError: If required parameters are invalid
            RuntimeError: If OS detection fails
        """
        try:
            super().__init__(package_list)
            
            # Validate required parameters
            if not rocm_version:
                raise ValueError("ROCm version is required")
            
            self.rocm_version = rocm_version
            self.metapackage = metapackage
            self.loader = loader
            self.failed_packages = {}
            
            # Detect OS
            try:
                self.os_id, self.os_family = get_os_id()
                logger.info(f"Detected OS: {self.os_id} (family: {self.os_family})")
            except Exception as e:
                error_msg = f"Failed to detect operating system: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
                
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            error_msg = f"Unexpected error initializing PackageUninstaller: {type(e).__name__} - {str(e)}"
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e

    def execute(self):
        """
        Perform the uninstallation.

        Metapackage mode:
            - Uninstall all metapackages in reverse dependency order.
        Non-metapackage mode:
            - Uninstall all non-metapackages in reverse dependency order.

        Logs the progress and errors.
        """
        try:
            logger.info(f"\n=== UNINSTALLATION PHASE ===")
            logger.info(f"ROCm Version: {self.rocm_version}")
            logger.info(f"Metapackage Build: {self.metapackage}")

            # Uninstall all packages in reverse dependency order
            for pkg in reversed(self.packages):
                try:
                    logger.info(f"[REMOVE] Uninstalling {pkg.package}")
                    if not pkg:
                        logger.warning("Encountered None package object, skipping")
                        continue
                        
                    try:
                        derived_name = self.loader.derive_package_names(pkg, True)
                    except AttributeError as e:
                        error_msg = f"Package object missing required attribute: {e}"
                        logger.error(error_msg)
                        self.failed_packages[pkg.package] = error_msg
                        continue
                    except Exception as e:
                        error_msg = f"Error deriving package names: {type(e).__name__} - {str(e)}"
                        logger.error(error_msg)
                        self.failed_packages[pkg.package] = error_msg
                        continue
                    
                    if derived_name:
                        for derived_pkg in derived_name:
                            self._run_uninstall_command(derived_pkg)
                    else:
                        logger.warning(f"No derived package names found for {pkg.package}")
                        
                except (OSError, IOError) as e:
                    error_msg = f"I/O error while uninstalling {pkg.package}: {e}"
                    logger.error(error_msg)
                    self.failed_packages[pkg.package] = error_msg
                except subprocess.CalledProcessError as e:
                    error_msg = f"Command failed for {pkg.package}: {e.stderr if e.stderr else e.output}"
                    logger.error(error_msg)
                    self.failed_packages[pkg.package] = error_msg
                except Exception as e:
                    error_msg = f"Unexpected error uninstalling {pkg.package}: {str(e)}"
                    logger.exception(error_msg)
                    self.failed_packages[pkg.package] = error_msg
                    
            logger.info("Uninstallation complete.")
            
            # Print summary of failures
            self.print_summary()
            
        except KeyboardInterrupt:
            logger.warning("Uninstallation interrupted by user")
            self.print_summary()
            raise
        except Exception as e:
            logger.exception(f"Critical error during uninstallation execution: {e}")
            self.print_summary()
            raise

    def print_summary(self):
        """
        Print a summary of package uninstallation results.
        
        Displays success message if all packages uninstalled successfully,
        or a detailed list of failures if any occurred.
        """
        try:
            if not self.failed_packages:
                logger.info("All packages uninstalled successfully.")
                return

            logger.info("\n=== SUMMARY OF FAILURES ===")
            logger.info(f"Total failed packages: {len(self.failed_packages)}")
            
            try:
                print_dict_summary(self.failed_packages)
            except Exception as e:
                logger.error(f"Error printing detailed summary: {e}")
                # Fallback to simple listing
                for pkg_name, error_msg in self.failed_packages.items():
                    logger.error(f"  - {pkg_name}: {error_msg}")
                    
        except Exception as e:
            logger.exception(f"Unexpected error in print_summary: {e}")

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
        try:
            if not pkg_name:
                error_msg = "Package name is None, cannot uninstall."
                logger.error(error_msg)
                self.failed_packages["Unknown"] = error_msg
                return

            cmd = None

            # Determine command based on OS family
            if self.os_family == "debian":
                cmd = ["sudo", "apt-get", "autoremove", "-y", pkg_name]
            elif self.os_family == "redhat" or self.os_family == "rpm":
                cmd = ["sudo", "yum", "remove", "-y", pkg_name]
            elif self.os_family == "suse":
                cmd = ["sudo", "zypper", "remove", "-y", pkg_name]
            else:
                error_msg = f"Unsupported OS family '{self.os_family}' for package uninstall"
                logger.error(error_msg)
                self.failed_packages[pkg_name] = error_msg
                return

            if not cmd:
                error_msg = "No uninstall command generated"
                logger.error(error_msg)
                self.failed_packages[pkg_name] = error_msg
                return

            logger.info(f"Running uninstall command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                timeout=300  # 5 minute timeout for uninstallation
            )
            
            if result.returncode != 0:
                failure_reason = result.stdout.strip() if result.stdout else "Unknown error"
                logger.error(f"Failed to uninstall {pkg_name} (exit code {result.returncode}):\n{failure_reason}")
                self.failed_packages[pkg_name] = f"Exit code {result.returncode}: {failure_reason}"
            else:
                logger.info(f"Successfully uninstalled {pkg_name}")
                
        except subprocess.TimeoutExpired as e:
            error_msg = f"Uninstallation timed out after {e.timeout} seconds"
            logger.error(f"Timeout while uninstalling {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg
        except FileNotFoundError as e:
            error_msg = f"Command or file not found: {e.filename if hasattr(e, 'filename') else str(e)}"
            logger.error(f"FileNotFoundError while uninstalling {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg
        except PermissionError as e:
            error_msg = f"Permission denied: {str(e)}"
            logger.error(f"PermissionError while uninstalling {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg
        except OSError as e:
            error_msg = f"OS error: {e.strerror if hasattr(e, 'strerror') else str(e)}"
            logger.error(f"OSError while uninstalling {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__} - {str(e)}"
            logger.exception(f"Unexpected exception while uninstalling {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg


def parse_arguments():
    """
    Parses command-line arguments for the uninstaller.
    """
    parser = argparse.ArgumentParser(description="ROCm Package Uninstaller")
    parser.add_argument(
        "--package-json", help="Path to package JSON definition file (optional, tightly coupled with --metapackage)"
    )
    parser.add_argument(
        "--metapackage", help="Metapackage build mode (true/false, tightly coupled with --package-json)"
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
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        args = parse_arguments()

        # Validate tight coupling: --package-json and --metapackage must both be provided or both omitted
        has_package_json = args.package_json is not None
        has_metapackage = args.metapackage is not None
        
        if has_package_json != has_metapackage:
            logger.error("--package-json and --metapackage are tightly coupled. Both must be provided or both must be omitted.")
            return 1

        # Load packages
        loader = None
        if args.package_json and args.metapackage:
            # Load packages from JSON file
            if not os.path.exists(args.package_json):
                logger.error(f"Package JSON file not found: {args.package_json}")
                return 1
            
            try:
                loader = PackageLoader(args.package_json, args.rocm_version, args.artifact_group)
                packages = (
                    loader.load_metapackage_packages()
                    if args.metapackage.lower() == "true"
                    else loader.load_non_metapackage_packages()
                )
            except FileNotFoundError as e:
                logger.error(f"Failed to load package definitions: {e}")
                return 1
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in package file: {e}")
                return 1
            except Exception as e:
                logger.exception(f"Error loading packages: {e}")
                return 1
        else:
            # Default: uninstall "amdrocm" package only
            logger.info("No package JSON provided. Defaulting to uninstall 'amdrocm' metapackage.")
            
            # Create a temporary package JSON with just amdrocm
            import tempfile
            amdrocm_json = [
                {
                    "Package": "amdrocm",
                    "Version": "",
                    "Architecture": "amd64",
                    "BuildArch": "x86_64",
                    "DEBDepends": [],
                    "RPMRequires": [],
                    "Metapackage": "True",
                    "Gfxarch": "True"
                }
            ]
            
            # Write temporary JSON file
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                    json.dump(amdrocm_json, temp_file)
                    temp_json_path = temp_file.name
                
                # Create loader with temporary JSON
                loader = PackageLoader(temp_json_path, args.rocm_version, args.artifact_group)
                packages = loader.load_metapackage_packages()
                args.metapackage = "true"  # Set metapackage flag for consistency
                
                # Clean up temp file
                os.remove(temp_json_path)
                
            except Exception as e:
                logger.exception(f"Error creating default package loader: {e}")
                return 1

        if not packages:
            logger.warning("No packages to uninstall")
            return 0

        logger.info(f"Uninstalling in {'metapackage' if args.metapackage and args.metapackage.lower() == 'true' else 'non-metapackage'} mode")

        # Initialize uninstaller
        try:
            metapackage_flag = args.metapackage and args.metapackage.lower() == "true"
            uninstaller = PackageUninstaller(
                package_list=packages,
                rocm_version=args.rocm_version,
                metapackage=metapackage_flag,
                loader=loader,
            )
        except Exception as e:
            logger.exception(f"Failed to initialize uninstaller: {e}")
            return 1

        # Execute uninstallation
        try:
            uninstaller.execute()
            
            # Check if there were any failures
            if uninstaller.failed_packages:
                logger.error(f"Uninstallation completed with {len(uninstaller.failed_packages)} failure(s)")
                return 1
            else:
                logger.info("All packages uninstalled successfully")
                return 0
                
        except KeyboardInterrupt:
            logger.warning("\nUninstallation interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            logger.exception(f"Uninstallation failed with error: {e}")
            return 1

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")
        return 1


if __name__ == "__main__":
    import sys
    import os
    main()
