#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Script to install and test native packages (DEB/RPM).

This script installs packages using apt (for DEB) or dnf (for RPM) and verifies
that the installation was successful. If all packages are installed (no specific
package name provided), it validates that the installed packages match the
built_packages.txt manifest.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Optional


class PackageInstaller:
    """Handles installation and testing of native packages."""

    def __init__(
        self,
        package_type: str,
        package_folder: str,
        package_names: Optional[List[str]] = None,
        rpm_package_prefix: Optional[str] = None,
        docker_image: Optional[str] = None,
        uninstall: Optional[List[str]] = None,
        skip_package: Optional[str] = None,
    ):
        """Initialize the package installer.

        Args:
            package_type: Type of package ('deb' or 'rpm')
            package_folder: Folder containing the packages
            package_names: Optional list of specific package names to install
            rpm_package_prefix: Optional prefix for RPM installation
            docker_image: Optional Docker image to run installation inside
            uninstall: Optional list of package names to uninstall (if empty list, uninstall all)
        """
        self.package_type = package_type.lower()
        self.package_folder = Path(package_folder)
        self.package_names = package_names
        self.rpm_package_prefix = rpm_package_prefix
        self.docker_image = docker_image
        self.uninstall = uninstall
        self.skip_package = skip_package

        # Validate inputs
        if self.package_type not in ["deb", "rpm"]:
            raise ValueError(f"Invalid package type: {package_type}. Must be 'deb' or 'rpm'")

        if not self.package_folder.exists():
            raise ValueError(f"Package folder does not exist: {package_folder}")

        if not self.package_folder.is_dir():
            raise ValueError(f"Package folder is not a directory: {package_folder}")

    def read_built_packages(self) -> Set[str]:
        """Read the built_packages.txt manifest file.

        Returns:
            Set of package filenames that were built

        Raises:
            FileNotFoundError: If built_packages.txt doesn't exist
        """
        manifest_file = self.package_folder / "built_packages.txt"

        if not manifest_file.exists():
            raise FileNotFoundError(
                f"built_packages.txt not found in {self.package_folder}"
            )

        built_packages = set()
        in_created_section = False

        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Start of created packages section
                if line == "# Created Packages:":
                    in_created_section = True
                    continue

                # End of created packages section (start of skipped section)
                if line.startswith("# Skipped Packages:"):
                    break

                # Add packages from created section (skip comments and empty lines)
                if in_created_section and line and not line.startswith("#"):
                    built_packages.add(line)

        return built_packages

    def get_packages_to_install(self) -> List[Path]:
        """Get the list of package files to install.

        Returns:
            List of package file paths

        Raises:
            ValueError: If no packages found or specified packages don't exist
        """
        extension = f".{self.package_type}"

        if self.package_names:
            # Install specific packages
            packages = []
            for pkg_name in self.package_names:
                # Support both full filename and base name
                if not pkg_name.endswith(extension):
                    pkg_pattern = f"{pkg_name}*{extension}"
                    matching = list(self.package_folder.glob(pkg_pattern))
                    if not matching:
                        raise ValueError(
                            f"Package not found: {pkg_name} in {self.package_folder}"
                        )
                    packages.extend(matching)
                else:
                    pkg_path = self.package_folder / pkg_name
                    if not pkg_path.exists():
                        raise ValueError(f"Package not found: {pkg_path}")
                    packages.append(pkg_path)
        else:
            # Install all packages in the folder
            packages = list(self.package_folder.glob(f"*{extension}"))

        if not packages:
            raise ValueError(f"No {extension} packages found in {self.package_folder}")

        # convert to absolute path
        packages = sorted([str(pkg.absolute()) for pkg in packages])

        # skip packages
        if self.skip_package:
            print("\n" + "=" * 80)
            print("SKIPPING PACKAGES")
            for skip in self.skip_package:
                 print(f"   - {Path(skip).name}")
            print("=" * 80)

            for skip in self.skip_package:
                # iterate over a copy remove from original packages
                for pkg in packages[:]:
                    if skip in pkg:
                        packages.remove(pkg)

        return packages

    def install_deb_packages(self, package_paths: List[Path]) -> bool:
        """Install DEB packages using apt.

        Args:
            packages: List of package file paths to install

        Returns:
            True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING DEB PACKAGES")
        print("=" * 80)

        print(f"\nPackages to install ({len(package_paths)}):")
        for pkg in package_paths:
            print(f"   - {Path(pkg).name}")

        # Install using apt
        cmd = ["sudo", "apt", "install", "-y"] + package_paths

        print(f"\nRunning: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(result.stdout)
            print("\n✅ DEB packages installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to install DEB packages")
            print(f"Error output:\n{e.stdout}")
            return False

    def install_rpm_packages(self, package_paths: List[Path]) -> bool:
        """Install RPM packages using dnf.

        Args:
            packages: List of package file paths to install

        Returns:
            True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING RPM PACKAGES")
        print("=" * 80)

        print(f"\nPackages to install ({len(package_paths)}):")
        for pkg in package_paths:
            print(f"   - {Path(pkg).name}")

        # TODO: Add support for --rpm-package-prefix
        # This will use: rpm -i --prefix <prefix> <packages>
        # For now, use dnf install

        # Install using dnf
        cmd = ["sudo", "dnf", "install", "-y"] + package_paths

        print(f"\nRunning: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(result.stdout)
            print("\n✅ RPM packages installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to install RPM packages")
            print(f"Error output:\n{e.stdout}")
            return False

    def get_package_base_names(self, packages: List[Path]) -> List[str]:
        """Extract base package names from package files.

        Args:
            packages: List of package file paths

        Returns:
            List of base package names (without version/extension)
        """
        base_names = []
        for pkg in packages:
            # For RPM: extract name using rpm query
            if self.package_type == "rpm":
                try:
                    result = subprocess.run(
                        ["rpm", "-qp", "--queryformat", "%{NAME}", str(pkg.absolute())],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    base_names.append(result.stdout.strip())
                except subprocess.CalledProcessError:
                    # Fallback: extract from filename
                    name = pkg.name.rsplit("-", 2)[0] if "-" in pkg.name else pkg.stem
                    base_names.append(name)
            else:  # deb
                # For DEB: extract name using dpkg-deb
                try:
                    result = subprocess.run(
                        ["dpkg-deb", "-f", str(pkg.absolute()), "Package"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    base_names.append(result.stdout.strip())
                except subprocess.CalledProcessError:
                    # Fallback: extract from filename
                    name = pkg.name.split("_")[0] if "_" in pkg.name else pkg.stem
                    base_names.append(name)
        
        return base_names

    def uninstall_deb_packages(self, package_names: List[str]) -> bool:
        """Uninstall DEB packages using apt.

        Args:
            package_names: List of package names to uninstall

        Returns:
            True if uninstallation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("UNINSTALLING DEB PACKAGES")
        print("=" * 80)

        print(f"\nPackages to uninstall ({len(package_names)}):")
        for pkg in package_names:
            print(f"   - {pkg}")

        # Uninstall using apt
        cmd = ["sudo", "apt", "remove", "-y"] + package_names

        print(f"\nRunning: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(result.stdout)
            print("\n✅ DEB packages uninstalled successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to uninstall DEB packages")
            print(f"Error output:\n{e.stdout}")
            return False

    def uninstall_rpm_packages(self, package_names: List[str]) -> bool:
        """Uninstall RPM packages using dnf.

        Args:
            package_names: List of package names to uninstall

        Returns:
            True if uninstallation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("UNINSTALLING RPM PACKAGES")
        print("=" * 80)

        print(f"\nPackages to uninstall ({len(package_names)}):")
        for pkg in package_names:
            print(f"   - {pkg}")

        # Uninstall using dnf
        cmd = ["sudo", "dnf", "remove", "-y"] + package_names

        print(f"\nRunning: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(result.stdout)
            print("\n✅ RPM packages uninstalled successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to uninstall RPM packages")
            print(f"Error output:\n{e.stdout}")
            return False

    def install_in_docker(self, packages: List[Path]) -> bool:
        """Install packages inside a Docker container.

        Args:
            packages: List of package file paths to install

        Returns:
            True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print(f"INSTALLING {self.package_type.upper()} PACKAGES IN DOCKER")
        print("=" * 80)
        print(f"Docker Image: {self.docker_image}")

        # Get package paths relative to package folder
        package_names = [pkg.name for pkg in packages]

        print(f"\nPackages to install ({len(packages)}):")
        for pkg in package_names:
            print(f"   - {pkg}")

        # Absolute path to package folder
        abs_package_folder = str(self.package_folder.absolute())

        # Build package paths inside container
        container_pkg_dir = "/packages"
        container_packages = [f"{container_pkg_dir}/{name}" for name in package_names]

        # Build installation command based on package type
        if self.package_type == "deb":
            install_cmd = f"apt update && apt install -y {' '.join(container_packages)}"
        else:  # rpm
            # TODO: Add support for --rpm-package-prefix when implemented
            install_cmd = f"dnf install -y {' '.join(container_packages)}"

        # Build docker run command
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after exit
            "-v", f"{abs_package_folder}:{container_pkg_dir}:ro",  # Mount package folder as read-only
            self.docker_image,
            "bash", "-c", install_cmd
        ]

        print(f"\nRunning Docker command:")
        print(f"  docker run --rm \\")
        print(f"    -v {abs_package_folder}:{container_pkg_dir}:ro \\")
        print(f"    {self.docker_image} \\")
        print(f"    bash -c '{install_cmd}'")
        print()

        try:
            result = subprocess.run(
                docker_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(result.stdout)
            print(f"\n✅ {self.package_type.upper()} packages installed successfully in Docker")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to install {self.package_type.upper()} packages in Docker")
            print(f"Error output:\n{e.stdout}")
            return False
        except FileNotFoundError:
            print("\n❌ Docker command not found. Please ensure Docker is installed and in PATH")
            return False

    def verify_installation(self, installed_packages: List[Path]) -> bool:
        """Verify that installation matches the manifest.

        Args:
            installed_packages: List of package files that were installed

        Returns:
            True if verification successful, False otherwise
        """
        # Only verify if installing all packages (no specific package names)
        if self.package_names:
            print("\n⏭️  Skipping manifest verification (specific packages selected)")
            return True

        print("\n" + "=" * 80)
        print("VERIFYING INSTALLATION")
        print("=" * 80)

        try:
            built_packages = self.read_built_packages()
        except FileNotFoundError as e:
            print(f"\n⚠️  WARNING: {e}")
            print("Skipping manifest verification")
            return True

        # Get installed package filenames
        installed_names = {pkg.name for pkg in installed_packages}

        # Compare with built packages
        if installed_names == built_packages:
            print(f"\n✅ Verification passed: All {len(built_packages)} packages from manifest were installed")
            return True
        else:
            print(f"\n❌ Verification failed: Installed packages don't match manifest")

            missing = built_packages - installed_names
            extra = installed_names - built_packages

            if missing:
                print(f"\nPackages in manifest but not installed ({len(missing)}):")
                for pkg in sorted(missing):
                    print(f"   - {pkg}")

            if extra:
                print(f"\nPackages installed but not in manifest ({len(extra)}):")
                for pkg in sorted(extra):
                    print(f"   - {pkg}")

            return False

    def run(self) -> bool:
        """Execute the installation/uninstallation and testing process.

        Returns:
            True if all operations successful, False otherwise
        """
        # Handle uninstall mode
        if self.uninstall is not None:
            return self.run_uninstall()
        
        # Handle install mode (original behavior)
        print("\n" + "=" * 80)
        print("PACKAGE INSTALLATION AND TESTING")
        print("=" * 80)
        print(f"\nPackage Type: {self.package_type.upper()}")
        print(f"Package Folder: {self.package_folder}")
        if self.package_names:
            print(f"Specific Packages: {', '.join(self.package_names)}")
        else:
            print("Installing: All packages in folder")
        if self.rpm_package_prefix:
            print(f"RPM Prefix: {self.rpm_package_prefix}")
        if self.docker_image:
            print(f"Docker Image: {self.docker_image}")

        try:
            # Get packages to install
            packages = self.get_packages_to_install()

            # Install packages
            if self.docker_image:
                # Install inside Docker container
                success = self.install_in_docker(packages)
                # Skip verification for Docker installations
                # (container is removed after installation)
                verification_passed = True
                if success:
                    print("\n⏭️  Skipping manifest verification (Docker installation)")
            else:
                # Install on host system
                if self.package_type == "deb":
                    success = self.install_deb_packages(packages)
                else:  # rpm
                    success = self.install_rpm_packages(packages)

                if not success:
                    return False

                # Verify installation
                verification_passed = self.verify_installation(packages)

            # Print final status
            print("\n" + "=" * 80)
            if success and verification_passed:
                print("✅ INSTALLATION TEST PASSED")
            else:
                print("❌ INSTALLATION TEST FAILED")
            print("=" * 80 + "\n")

            return success and verification_passed

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_uninstall(self) -> bool:
        """Execute the uninstallation process.

        Returns:
            True if uninstallation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("PACKAGE UNINSTALLATION")
        print("=" * 80)
        print(f"\nPackage Type: {self.package_type.upper()}")
        print(f"Package Folder: {self.package_folder}")

        try:
            # Determine which packages to uninstall
            if self.uninstall:
                # Uninstall specific packages passed with --uninstall
                package_names = self.uninstall
                print(f"Uninstalling specific packages: {', '.join(package_names)}")
            else:
                # Uninstall all packages in folder
                print("Uninstalling: All packages in folder")
                packages = self.get_packages_to_install()
                package_names = self.get_package_base_names(packages)

            if not package_names:
                print("\n⚠️  No packages to uninstall")
                return True

            # Uninstall packages (only on host system, not in Docker)
            if self.docker_image:
                print("\n⚠️  WARNING: Uninstall in Docker is not supported (containers are ephemeral)")
                print("Docker containers are removed after use, so uninstall is not needed")
                return True

            if self.package_type == "deb":
                success = self.uninstall_deb_packages(package_names)
            else:  # rpm
                success = self.uninstall_rpm_packages(package_names)

            # Print final status
            print("\n" + "=" * 80)
            if success:
                print("✅ UNINSTALLATION COMPLETED SUCCESSFULLY")
            else:
                print("❌ UNINSTALLATION FAILED")
            print("=" * 80 + "\n")

            return success

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Install and test native packages (DEB/RPM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install all DEB packages in a folder
  python install_test_native_package.py --package-type deb --package-folder /path/to/packages

  # Install all RPM packages
  python install_test_native_package.py --package-type rpm --package-folder /path/to/packages

  # Install specific DEB packages
  python install_test_native_package.py --package-type deb --package-folder /path/to/packages \\
      --package-name amdrocm-core amdrocm-hip

  # Install DEB packages inside a Docker container
  python install_test_native_package.py --package-type deb --package-folder /path/to/packages \\
      --docker-image ubuntu:22.04

  # Install RPM packages inside a Docker container
  python install_test_native_package.py --package-type rpm --package-folder /path/to/packages \\
      --docker-image almalinux:9

  # Install RPM packages inside a manylinux Docker container
  python install_test_native_package.py --package-type rpm --package-folder /path/to/packages \\
      --docker-image ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:6e8242d347af7e0c43c82d5031a3ac67b669f24898ea8dc2f

  # Uninstall all DEB packages in a folder
  python install_test_native_package.py --package-type deb --package-folder /path/to/packages --uninstall

  # Uninstall specific packages
  python install_test_native_package.py --package-type rpm --package-folder /path/to/packages \\
      --uninstall amdrocm-core amdrocm-hip

  # Install RPM package with prefix (placeholder - not yet implemented)
  python install_test_native_package.py --package-type rpm --package-folder /path/to/packages \\
      --rpm-package-prefix /opt/rocm
        """,
    )

    parser.add_argument(
        "--package-type",
        type=str,
        required=True,
        choices=["deb", "rpm"],
        help="Type of package to install (deb or rpm)",
    )

    parser.add_argument(
        "--package-folder",
        type=str,
        required=True,
        help="Folder containing the packages to install",
    )

    parser.add_argument(
        "--package-name",
        type=str,
        nargs="+",
        help="Optional: Specific package name(s) to install. If not provided, all packages in folder will be installed.",
    )

    parser.add_argument(
        "--rpm-package-prefix",
        type=str,
        help="Optional: Prefix for RPM installation (only valid for rpm package type). Placeholder for future implementation.",
    )

    parser.add_argument(
        "--docker-image",
        type=str,
        help="Optional: Docker image to use for installation. If provided, installation will be performed inside a Docker container.",
    )

    parser.add_argument(
        "--uninstall",
        nargs="*",
        help="Uninstall packages instead of installing. If no package names provided, uninstalls all packages in folder. If package names provided, uninstalls only those packages.",
    )

    parser.add_argument(
        "--skip-package",
        type=str,
        nargs="+",
        help="Skip installing packages provided",
    )

    args = parser.parse_args()

    # Validate RPM prefix is only used with RPM packages
    if args.rpm_package_prefix and args.package_type != "rpm":
        parser.error("--rpm-package-prefix can only be used with --package-type rpm")
    
    # Validate mutually exclusive options
    if args.uninstall is not None and args.package_name:
        parser.error("--uninstall and --package-name cannot be used together. Use --uninstall with package names to uninstall specific packages.")
    
    if args.uninstall is not None and args.docker_image:
        parser.error("--uninstall cannot be used with --docker-image. Docker containers are ephemeral and don't need uninstallation.")

    # Create installer and run
    installer = PackageInstaller(
        package_type=args.package_type,
        package_folder=args.package_folder,
        package_names=args.package_name,
        rpm_package_prefix=args.rpm_package_prefix,
        docker_image=args.docker_image,
        uninstall=args.uninstall,
        skip_package = args.skip_package,
    )

    success = installer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

