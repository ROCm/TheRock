#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

""" Installs ROCm packages either from local package files or from a remote repository.

Local installation (uses .deb/.rpm files from a directory):
```
./native_package_installer.py --dest-dir ./PKG_DIR \
    --package-json ./packages.json \
    --rocm-version 6.2.0 \
    --artifact-group gfx94X-dcgpu \
    --release-type test \
    --version true/false \
    --metapackage true/false \
    [--package-suffix asan] \
    [--bucket therock-deb-rpm-test]
```

Repository installation (uses run-id to fetch from remote repo):
```
./native_package_installer.py --run-id 123456 \
    --package-json ./packages.json \
    --rocm-version 6.2.0 \
    --artifact-group gfx94X-dcgpu \
    --release-type test \
    --bucket therock-deb-rpm-test \
    --version true/false \
    --metapackage true/false \
    [--package-suffix asan]
```

Note: --bucket is REQUIRED when using --run-id, but OPTIONAL when using --dest-dir
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
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
    Handles installation of ROCm packages.

    Depending on mode:
    - Pre-upload: installs local .deb/.rpm packages from a directory
    - Post-upload: installs from a repository (using run-id)
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
        release_type: str,
        bucket: Optional[str] = None,
        package_suffix: Optional[str] = None,
        metapackage: bool = False,
        loader = None,
    ):
        """
        Initialize PackageInstaller with configuration and package list.
        
        Raises:
            ValueError: If required parameters are invalid
            FileNotFoundError: If configuration file cannot be found
            RuntimeError: If OS detection fails
        """
        try:
            super().__init__(package_list)
            
            # Validate required parameters
            if not rocm_version:
                raise ValueError("ROCm version is required")
            
            if not release_type:
                raise ValueError("Release type is required")
            
            if upload not in ["pre", "post"]:
                raise ValueError(f"Invalid upload mode: {upload}. Must be 'pre' or 'post'")
            
            if upload == "pre" and not dest_dir:
                raise ValueError("dest_dir is required for pre-upload mode")
                
            if upload == "post":
                if not run_id:
                    raise ValueError("run_id is required for post-upload mode")
                if not bucket:
                    raise ValueError("bucket is required for post-upload mode (when --run-id is provided)")
            
            self.dest_dir = dest_dir
            self.run_id = run_id
            self.rocm_version = rocm_version
            self.metapackage = metapackage
            self.version_flag = version_flag
            self.artifact_group = artifact_group
            self.release_type = release_type
            self.bucket = bucket or ""
            self.package_suffix = package_suffix or ""
            self.upload = upload
            self.failed_packages = {}
            self.loader = loader
            
            # Detect OS
            try:
                self.os_id, self.os_family = get_os_id()
                logger.info(f"Detected OS: {self.os_id} (family: {self.os_family})")
            except Exception as e:
                error_msg = f"Failed to detect operating system: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
            # Load S3 configuration
            try:
                config_path = "build_tools/packaging/linux/packaging_install.yaml"
                self.s3_config = load_yaml_config(
                    config_path,
                    variables={
                        "artifact_group": self.artifact_group,
                        "run_id": self.run_id,
                        "bucket": self.bucket,
                    },
                )
                logger.info("S3 configuration loaded successfully")
            except FileNotFoundError as e:
                error_msg = f"Configuration file not found: {config_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg) from e
            except Exception as e:
                error_msg = f"Failed to load configuration: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
                
        except (ValueError, FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            error_msg = f"Unexpected error initializing PackageInstaller: {type(e).__name__} - {str(e)}"
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e

    def execute(self):
        """
        Perform installation of all packages.

        Logs the installation phase and iterates through each package,
        calling _install_package().
        """
        try:
            logger.info(f"\n=== INSTALLATION PHASE ===")
            logger.info(f"Destination Directory: {self.dest_dir}")
            logger.info(f"ROCm Version: {self.rocm_version}")
            logger.info(f"Metapackage Build: {self.metapackage}")

            if self.upload == "post":
                self.populate_repo_file(self.run_id)

            for pkg in self.packages:
                try:
                    logger.info(f"[INSTALL] Installing {pkg.package} ({pkg.architecture})")
                    self._install_package(pkg)
                except (OSError, IOError) as e:
                    error_msg = f"I/O error while installing {pkg.package}: {e}"
                    logger.error(error_msg)
                    self.failed_packages[pkg.package] = error_msg
                except subprocess.CalledProcessError as e:
                    error_msg = f"Command failed for {pkg.package}: {e.stderr if e.stderr else e.output}"
                    logger.error(error_msg)
                    self.failed_packages[pkg.package] = error_msg
                except Exception as e:
                    error_msg = f"Unexpected error installing {pkg.package}: {str(e)}"
                    logger.exception(error_msg)
                    self.failed_packages[pkg.package] = error_msg

            logger.info("Installation complete.")

            # Print summary of failures
            self.print_summary()
            
        except KeyboardInterrupt:
            logger.warning("Installation interrupted by user")
            self.print_summary()
            raise
        except Exception as e:
            logger.exception(f"Critical error during installation execution: {e}")
            self.print_summary()
            raise

    def _run_install_command(self, pkg_name, use_repo):
        """
        Execute OS-specific installation command for a package.

        Parameters:
        pkg_name : str
            Package name or full path
        use_repo : bool
            True if installing from repository, False if local files
        """

        try:
            if not pkg_name or pkg_name == "rocmamd-smi":
                error_msg = "Package name is None/rocmamd-smi, cannot install."
                logger.error(error_msg)
                self.failed_packages["Unknown"] = error_msg
                return

            cmd = None

            # Determine command based on source type and OS
            if self.upload == "pre":
                if self.os_family == "debian":
                    cmd = ["sudo", "dpkg", "-i", pkg_name]
                elif self.os_family == "rpm":
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
                    error_msg = f"Unsupported OS family '{self.os_family}' for local install"
                    logger.error(error_msg)
                    self.failed_packages[pkg_name] = error_msg
                    return
                    
            elif self.upload == "post":
                if self.os_family == "debian":
                    cmd = ["sudo", "apt-get", "install", "-y", pkg_name]
                elif self.os_family == "rpm":
                    cmd = ["sudo", "yum", "install", "-y", pkg_name]
                elif self.os_family == "suse":
                    cmd = ["sudo", "zypper", "--non-interactive", "install", pkg_name]
                else:
                    error_msg = f"Unsupported OS family '{self.os_family}' for repo install"
                    logger.error(error_msg)
                    self.failed_packages[pkg_name] = error_msg
                    return

            # Double-check cmd was built correctly
            if not cmd:
                error_msg = f"No install command generated for package"
                logger.error(error_msg)
                self.failed_packages[pkg_name] = error_msg
                return

            logger.info(f"Running install command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                timeout=600  # 10 minute timeout for package installation
            )

            if result.returncode != 0:
                failure_reason = result.stdout.strip() if result.stdout else "Unknown error"
                logger.error(f"Failed to install {pkg_name} (exit code {result.returncode}):\n{failure_reason}")
                self.failed_packages[pkg_name] = f"Exit code {result.returncode}: {failure_reason}"
            else:
                logger.info(f"Successfully installed {pkg_name}")

        except subprocess.TimeoutExpired as e:
            error_msg = f"Installation timed out after {e.timeout} seconds"
            logger.error(f"Timeout while installing {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg
        except FileNotFoundError as e:
            error_msg = f"Command or file not found: {e.filename if hasattr(e, 'filename') else str(e)}"
            logger.error(f"FileNotFoundError while installing {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg
        except PermissionError as e:
            error_msg = f"Permission denied: {str(e)}"
            logger.error(f"PermissionError while installing {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg
        except OSError as e:
            error_msg = f"OS error: {e.strerror if hasattr(e, 'strerror') else str(e)}"
            logger.error(f"OSError while installing {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg
        except TypeError as e:
            error_msg = f"Type error in command arguments: {str(e)}"
            logger.exception(f"TypeError while installing {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__} - {str(e)}"
            logger.exception(f"Unexpected exception while installing {pkg_name}: {error_msg}")
            self.failed_packages[pkg_name] = error_msg

    # ---------------------------------------------------------------------
    # Repo Population
    # ---------------------------------------------------------------------
    def populate_repo_file(self, run_id: str):
        """
        Create a repository file for post-upload installation.

        - For Debian: writes /etc/apt/sources.list.d/rocm.list
        - For RPM: writes /etc/yum.repos.d/rocm.repo (placeholder)
        
        Raises:
            RuntimeError: If repo file cannot be created or updated
            KeyError: If required configuration is missing
        """
        logger.info(f"Populating repo file for OS: {self.os_family}")

        try:
            # Validate configuration exists
            if not self.s3_config:
                raise ValueError("S3 configuration is not loaded")
                
            base_url = (
                self.s3_config.get(self.os_family, {})
                .get(self.release_type, {})
                .get("s3")
            )
            
            if not base_url:
                error_msg = (
                    f"S3 URL not found for OS_Family: '{self.os_family}', "
                    f"release_type: '{self.release_type}'. Configuration may be incomplete."
                )
                logger.error(error_msg)
                raise KeyError(error_msg)
            
            logger.info(f"Using S3 URL: {base_url}")
            
            if self.os_family == "debian":
                try:
                    repo_file_path = (
                        self.s3_config.get("repos", {})
                        .get(self.os_id, {})
                        .get("rocm_repo_file")
                    )
                    
                    if not repo_file_path:
                        raise KeyError(f"Repository file path not found for OS: {self.os_id}")
                    
                    repo_entry = f"deb [trusted=yes] {base_url}/deb stable main\n"
                    logger.info(f"Writing Debian repo entry to {repo_file_path}")

                    cmd = f'echo "{repo_entry.strip()}" | sudo tee {repo_file_path} > /dev/null'
                    result = subprocess.run(
                        cmd, 
                        shell=True, 
                        capture_output=True, 
                        text=True,
                        timeout=30
                    )

                    if result.returncode != 0:
                        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                        logger.error(f"Failed to populate repo file: {error_msg}")
                        raise RuntimeError(f"Error populating repo file: {error_msg}")

                    logger.info("Running apt-get update...")
                    update_result = subprocess.run(
                        ["sudo", "apt-get", "update"], 
                        capture_output=True, 
                        text=True,
                        timeout=300
                    )
                    
                    if update_result.returncode != 0:
                        logger.warning(f"apt-get update had issues: {update_result.stderr}")
                        
                except subprocess.TimeoutExpired as e:
                    error_msg = f"Command timed out after {e.timeout} seconds"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

            elif self.os_family == "rpm":
                try:
                    logger.info("Detected RPM-based system. Setting up repo file.")
                    repo_file_path = "/etc/yum.repos.d/rocm.repo"
                    repo_entry = (
                        f"[rocm]\nname=ROCm Repo\nbaseurl={base_url}/rpm\n"
                        "enabled=1\ngpgcheck=0\n"
                    )
                    
                    with open(repo_file_path, "w") as f:
                        f.write(repo_entry)
                    
                    logger.info(f"Created repo file: {repo_file_path}")
                    
                    # Clean and rebuild cache
                    subprocess.run(
                        ["sudo", "yum", "clean", "all"], 
                        capture_output=True, 
                        timeout=60
                    )
                    subprocess.run(
                        ["sudo", "yum", "makecache"], 
                        capture_output=True, 
                        timeout=300
                    )
                    
                except (IOError, OSError) as e:
                    error_msg = f"Failed to write repo file: {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                    
            elif self.os_family == "suse":
                logger.warning("SUSE repository setup not yet implemented")
                raise NotImplementedError(f"Repo population not implemented for {self.os_family}")
            else:
                logger.warning(f"Unsupported OS family for repo population: {self.os_family}")
                raise ValueError(f"Unsupported OS family: {self.os_family}")

        except KeyError as e:
            logger.error(f"Configuration key error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Value error in repo population: {e}")
            raise
        except PermissionError as e:
            error_msg = f"Permission denied while creating repo file: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except subprocess.TimeoutExpired as e:
            error_msg = f"Subprocess timeout during repo setup: {e.cmd}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error populating repo file: {type(e).__name__} - {str(e)}"
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e

    def find_packages_for_base(self, dest_dir, derived_name):
        """
        Locate package files in local directory matching the derived name.

        Parameters:
        dest_dir : str
            Directory containing package files
        derived_name : str
            Derived package name of Base package

        Returns:
        list of matched files or derived name (for repo)
        
        Raises:
            FileNotFoundError: If destination directory doesn't exist
            PermissionError: If directory cannot be accessed
        """

        try:
            if self.upload == "post":
                return derived_name
            
            # Validate destination directory exists
            if not dest_dir:
                logger.error("Destination directory is None or empty")
                return None
                
            if not os.path.exists(dest_dir):
                error_msg = f"Destination directory does not exist: {dest_dir}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            if not os.path.isdir(dest_dir):
                error_msg = f"Path is not a directory: {dest_dir}"
                logger.error(error_msg)
                raise NotADirectoryError(error_msg)
            
            # List all package files in directory
            try:
                all_files = [
                    f for f in os.listdir(dest_dir) if f.endswith((".deb", ".rpm"))
                ]
            except PermissionError as e:
                error_msg = f"Permission denied accessing directory: {dest_dir}"
                logger.error(error_msg)
                raise PermissionError(error_msg) from e

            if not all_files:
                logger.warning(f"No .deb or .rpm files found in {dest_dir}")
                return None

            # Build regex pattern for matching
            try:
                pattern = rf"^{re.escape(derived_name)}[_-]{re.escape(self.rocm_version)}[^\s]*\.(deb|rpm)$"
                compiled_pattern = re.compile(pattern)
            except re.error as e:
                error_msg = f"Invalid regex pattern for package matching: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

            # Find matching packages
            matched = [
                os.path.join(dest_dir, f) for f in all_files if compiled_pattern.match(f)
            ]
            
            if matched:
                logger.info(f"Found {len(matched)} package(s) matching '{derived_name}'")
                return matched
            else:
                logger.error(f"No matching package found for: {derived_name} (version: {self.rocm_version})")
                return None
                
        except FileNotFoundError:
            raise
        except PermissionError:
            raise
        except NotADirectoryError as e:
            logger.error(str(e))
            return None
        except Exception as e:
            error_msg = f"Unexpected error finding packages: {type(e).__name__} - {str(e)}"
            logger.exception(error_msg)
            return None

    def _install_package(self, pkg: PackageInfo):
        """
        Install a single package including dependencies.

        Parameters:
        pkg : PackageInfo
            Package metadata object
            
        Raises:
            AttributeError: If package object is missing required attributes
            FileNotFoundError: If package files cannot be found
        """
        try:
            if not pkg:
                error_msg = "Package object is None"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            derived_pkgs = []
            
            # Derive package names based on version flag
            try:
                if self.version_flag:
                    derived_name = self.loader.derive_package_names(pkg, True)
                    derived_pkgs.extend(derived_name)
                else:
                    derived_name = self.loader.derive_package_names(pkg, True)
                    derived_pkgs.extend(derived_name)
                    derived_name = self.loader.derive_package_names(pkg, False)
                    derived_pkgs.extend(derived_name)
            except AttributeError as e:
                error_msg = f"Package object missing required attribute: {e}"
                logger.error(error_msg)
                raise AttributeError(error_msg) from e
            except Exception as e:
                error_msg = f"Error deriving package names: {type(e).__name__} - {str(e)}"
                logger.error(error_msg)
                raise

            if not derived_pkgs:
                logger.warning(f"No derived packages found for {pkg.package}")
                return

            # Install each derived package
            for pkg_name in derived_pkgs:
                try:
                    if self.upload == "pre":
                        derived_name = self.find_packages_for_base(self.dest_dir, pkg_name)
                        if derived_name:
                            if isinstance(derived_name, list):
                                for derived_pkg in derived_name:
                                    self._run_install_command(derived_pkg, True)
                            else:
                                logger.warning(f"Expected list of packages, got {type(derived_name)}")
                        else:
                            logger.warning(f"No package files found for {pkg_name}")
                    elif self.upload == "post":
                        self._run_install_command(pkg_name, True)
                    else:
                        logger.error(f"Invalid upload mode: {self.upload}")
                        
                except FileNotFoundError as e:
                    error_msg = f"Package file not found: {str(e)}"
                    logger.error(error_msg)
                    self.failed_packages[pkg_name] = error_msg
                except Exception as e:
                    error_msg = f"Error installing {pkg_name}: {type(e).__name__} - {str(e)}"
                    logger.error(error_msg)
                    self.failed_packages[pkg_name] = error_msg
                    
        except ValueError:
            raise
        except AttributeError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error in _install_package: {type(e).__name__} - {str(e)}"
            logger.exception(error_msg)
            if hasattr(pkg, 'package'):
                self.failed_packages[pkg.package] = error_msg
            raise

    def print_summary(self):
        """
        Print a summary of package installation results.
        
        Displays success message if all packages installed successfully,
        or a detailed list of failures if any occurred.
        """
        try:
            if not self.failed_packages:
                logger.info("All packages installed successfully.")
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
        "--package-json", help="Path to package JSON definition file (optional, tightly coupled with --metapackage)"
    )
    parser.add_argument(
        "--metapackage", help="Enable metapackage build mode (true/false, tightly coupled with --package-json)"
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
    parser.add_argument(
        "--package-suffix",
        default=None,nargs="?",
        help="Package suffix for custom builds (optional, e.g., 'asan' for AddressSanitizer builds)",
    )
    parser.add_argument(
        "--release-type",
        required=True,
        help="Release type identifier (required, e.g., 'test', 'release')"
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket name for package repository (required when --run-id is provided, optional when --dest-dir is provided)"
    )

    return parser.parse_args()


def main():
    """
    Main entry point for installer script.

    - Parses command-line arguments
    - Loads packages from JSON
    - Initializes PackageInstaller
    - Executes installation
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        args = parse_arguments()

        # Validation: Ensure at least one of them is provided
        if not args.dest_dir and not args.run_id:
            logger.error("You must specify at least one of --dest-dir or --run-id")
            return 1

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
            # Default: install "amdrocm" package only
            logger.info("No package JSON provided. Defaulting to install 'amdrocm' metapackage.")
            
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
            logger.warning("No packages to install")
            return 0

        # Determine upload mode
        upload = "pre"
        if args.run_id and not args.dest_dir:
            upload = "post"

        logger.info(f"Installation mode: {upload}")

        # Initialize installer
        try:
            metapackage_flag = args.metapackage and args.metapackage.lower() == "true"
            installer = PackageInstaller(
                package_list=packages,
                dest_dir=args.dest_dir,
                run_id=args.run_id,
                rocm_version=args.rocm_version,
                version_flag=args.version.lower() == "true",
                upload=upload,
                artifact_group=args.artifact_group,
                release_type=args.release_type,
                bucket=args.bucket,
                package_suffix=args.package_suffix,
                metapackage=metapackage_flag,
                loader=loader,
            )
        except Exception as e:
            logger.exception(f"Failed to initialize installer: {e}")
            return 1

        # Execute installation
        try:
            installer.execute()
            
            # Check if there were any failures
            if installer.failed_packages:
                logger.error(f"Installation completed with {len(installer.failed_packages)} failure(s)")
                return 1
            else:
                logger.info("All packages installed successfully")
                return 0
                
        except KeyboardInterrupt:
            logger.warning("\nInstallation interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            logger.exception(f"Installation failed with error: {e}")
            return 1

    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")
        return 1


if __name__ == "__main__":
    import sys
    main()
