#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Full installation test script for ROCm native packages.

This script downloads packages from S3, installs them on the system,
and verifies that the installation was successful.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import json


class PackageFullTester:
    """Full installation tester for ROCm packages."""

    def __init__(
        self,
        package_type: str,
        s3_bucket: str,
        artifact_group: str,
        artifact_id: str,
        rocm_version: str,
        download_dir: str,
        install_prefix: str = "/opt/rocm",
        s3_path: Optional[str] = None,
    ):
        """Initialize the package full tester.

        Args:
            package_type: Type of package ('deb' or 'rpm')
            s3_bucket: S3 bucket name containing packages
            artifact_group: GPU architecture group (e.g., gfx94X-dcgpu)
            artifact_id: Artifact run ID
            rocm_version: ROCm version
            download_dir: Directory to download packages to
            install_prefix: Installation prefix (default: /opt/rocm)
            s3_path: Optional custom S3 path
        """
        self.package_type = package_type.lower()
        self.s3_bucket = s3_bucket
        self.artifact_group = artifact_group
        self.artifact_id = artifact_id
        self.rocm_version = rocm_version
        self.download_dir = Path(download_dir)
        self.install_prefix = install_prefix
        self.s3_path = s3_path

        # Validate inputs
        if self.package_type not in ["deb", "rpm"]:
            raise ValueError(f"Invalid package type: {package_type}. Must be 'deb' or 'rpm'")

        # Create download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def construct_s3_path(self) -> str:
        """Construct the S3 path for package download.

        Returns:
            S3 path string
        """
        if self.s3_path:
            return self.s3_path

        # Standard path structure: s3://bucket/artifact_group/artifact_id/
        return f"s3://{self.s3_bucket}/{self.artifact_group}/{self.artifact_id}/"

    def download_packages_from_s3(self) -> List[Path]:
        """Download packages from S3 bucket.

        Returns:
            List of downloaded package file paths

        Raises:
            RuntimeError: If download fails
        """
        print("\n" + "=" * 80)
        print("DOWNLOADING PACKAGES FROM S3")
        print("=" * 80)

        s3_path = self.construct_s3_path()
        print(f"\nS3 Path: {s3_path}")
        print(f"Download Directory: {self.download_dir}")

        # Download packages using AWS CLI
        extension = "deb" if self.package_type == "deb" else "rpm"
        
        # Try to sync the entire directory
        cmd = [
            "aws", "s3", "sync",
            s3_path,
            str(self.download_dir),
            "--exclude", "*",
            "--include", f"*.{extension}"
        ]

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
        except subprocess.CalledProcessError as e:
            print(f"Failed to download packages from S3")
            print(f"Error output:\n{e.stdout}")
            raise RuntimeError(f"S3 download failed: {e}")

        # List downloaded packages
        packages = list(self.download_dir.glob(f"*.{extension}"))
        
        if not packages:
            raise RuntimeError(f"No {extension} packages found after download")

        print(f"\nDownloaded {len(packages)} packages:")
        for pkg in sorted(packages):
            file_size = pkg.stat().st_size / (1024 * 1024)  # MB
            print(f"   - {pkg.name} ({file_size:.2f} MB)")

        return sorted(packages)

    def install_deb_packages(self, packages: List[Path]) -> bool:
        """Install DEB packages using apt.

        Args:
            packages: List of package file paths to install

        Returns:
            True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING DEB PACKAGES")
        print("=" * 80)

        # Convert to absolute paths
        package_paths = [str(pkg.absolute()) for pkg in packages]

        print(f"\nPackages to install ({len(package_paths)}):")
        for pkg in package_paths:
            print(f"   - {Path(pkg).name}")

        # Update apt cache first
        print("\nUpdating apt cache...")
        try:
            subprocess.run(
                ["apt", "update"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: apt update failed (may be expected in some environments)")

        # Install using apt
        cmd = ["apt", "install", "-y"] + package_paths

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
            print("\nDEB packages installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\nFailed to install DEB packages")
            print(f"Error output:\n{e.stdout}")
            return False

    def install_rpm_packages(self, packages: List[Path]) -> bool:
        """Install RPM packages using dnf.

        Args:
            packages: List of package file paths to install

        Returns:
            True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING RPM PACKAGES")
        print("=" * 80)

        # Convert to absolute paths
        package_paths = [str(pkg.absolute()) for pkg in packages]

        print(f"\nPackages to install ({len(package_paths)}):")
        for pkg in package_paths:
            print(f"   - {Path(pkg).name}")

        # Install using dnf
        cmd = ["dnf", "install", "-y"] + package_paths

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
            print("\nRPM packages installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\nFailed to install RPM packages")
            print(f"Error output:\n{e.stdout}")
            return False

    def verify_rocm_installation(self) -> bool:
        """Verify that ROCm is properly installed.

        Returns:
            True if verification successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("VERIFYING ROCM INSTALLATION")
        print("=" * 80)

        # Check if installation prefix exists
        install_path = Path(self.install_prefix)
        if not install_path.exists():
            print(f"\nInstallation directory not found: {self.install_prefix}")
            return False

        print(f"\nInstallation directory exists: {self.install_prefix}")

        # List of key components to check
        key_components = [
            "bin/rocminfo",
            "bin/hipcc",
            "include/hip/hip_runtime.h",
            "lib/libamdhip64.so",
        ]

        print("\nChecking for key ROCm components:")
        all_found = True
        found_count = 0

        for component in key_components:
            component_path = install_path / component
            if component_path.exists():
                print(f"   ✅ {component}")
                found_count += 1
            else:
                print(f"   ⚠️  {component} (not found)")
                all_found = False

        print(f"\nComponents found: {found_count}/{len(key_components)}")

        # Check installed packages
        print("\nChecking installed packages:")
        try:
            if self.package_type == "deb":
                cmd = ["dpkg", "-l"]
                grep_pattern = "rocm"
            else:
                cmd = ["rpm", "-qa"]
                grep_pattern = "rocm"

            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            rocm_packages = [line for line in result.stdout.split('\n') if grep_pattern.lower() in line.lower()]
            print(f"   Found {len(rocm_packages)} ROCm packages installed")
            
            if rocm_packages:
                print("\n   Sample packages:")
                for pkg in rocm_packages[:5]:  # Show first 5
                    print(f"      {pkg.strip()}")
                if len(rocm_packages) > 5:
                    print(f"      ... and {len(rocm_packages) - 5} more")

        except subprocess.CalledProcessError as e:
            print(f"Could not query installed packages")

        # Try to run rocminfo if available
        rocminfo_path = install_path / "bin" / "rocminfo"
        if rocminfo_path.exists():
            print("\nTrying to run rocminfo...")
            try:
                result = subprocess.run(
                    [str(rocminfo_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                )
                print("rocminfo executed successfully")
                # Print first few lines of output
                lines = result.stdout.split('\n')[:10]
                print("\n   First few lines of rocminfo output:")
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
            except subprocess.TimeoutExpired:
                print(f"rocminfo timed out (may require GPU hardware)")
            except subprocess.CalledProcessError as e:
                print(f"rocminfo failed (may require GPU hardware)")
            except Exception as e:
                print(f"Could not run rocminfo: {e}")

        # Return success if at least some components were found
        if found_count >= 2:  # Require at least 2 key components
            print("\nROCm installation verification PASSED")
            return True
        else:
            print("\nROCm installation verification FAILED")
            return False

    def run(self) -> bool:
        """Execute the full installation test process.

        Returns:
            True if all operations successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("FULL INSTALLATION TEST - NATIVE LINUX PACKAGES")
        print("=" * 80)
        print(f"\nPackage Type: {self.package_type.upper()}")
        print(f"S3 Bucket: {self.s3_bucket}")
        print(f"Artifact Group: {self.artifact_group}")
        print(f"Artifact ID: {self.artifact_id}")
        print(f"ROCm Version: {self.rocm_version}")
        print(f"Download Directory: {self.download_dir}")
        print(f"Install Prefix: {self.install_prefix}")

        try:
            # Step 1: Download packages from S3
            packages = self.download_packages_from_s3()

            # Step 2: Install packages
            if self.package_type == "deb":
                install_success = self.install_deb_packages(packages)
            else:  # rpm
                install_success = self.install_rpm_packages(packages)

            if not install_success:
                return False

            # Step 3: Verify installation
            verification_success = self.verify_rocm_installation()

            # Print final status
            print("\n" + "=" * 80)
            if install_success and verification_success:
                print("FULL INSTALLATION TEST PASSED")
                print("\nROCm has been successfully installed and verified!")
            else:
                print("FULL INSTALLATION TEST FAILED")
            print("=" * 80 + "\n")

            return install_success and verification_success

        except Exception as e:
            print(f"\nError during full installation test: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Full installation test for ROCm native packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and install DEB packages from S3
  python package_full_test.py \\
      --package-type deb \\
      --s3-bucket therock-dev-packages \\
      --artifact-group gfx94X-dcgpu \\
      --artifact-id 12345 \\
      --rocm-version 8.0.0 \\
      --download-dir /tmp/rocm_packages

  # Download and install RPM packages with custom S3 path
  python package_full_test.py \\
      --package-type rpm \\
      --s3-bucket therock-nightly-packages \\
      --artifact-group gfx110X-all \\
      --artifact-id 67890 \\
      --rocm-version 8.0.1 \\
      --download-dir /tmp/rocm_packages \\
      --install-prefix /opt/rocm \\
      --s3-path s3://custom-bucket/custom/path/
        """,
    )

    parser.add_argument(
        "--package-type",
        type=str,
        required=True,
        choices=["deb", "rpm"],
        help="Type of package to test (deb or rpm)",
    )

    parser.add_argument(
        "--s3-bucket",
        type=str,
        required=True,
        help="S3 bucket name containing packages",
    )

    parser.add_argument(
        "--artifact-group",
        type=str,
        required=True,
        help="GPU architecture group (e.g., gfx94X-dcgpu, gfx110X-all)",
    )

    parser.add_argument(
        "--artifact-id",
        type=str,
        required=True,
        help="Artifact run ID",
    )

    parser.add_argument(
        "--rocm-version",
        type=str,
        required=True,
        help="ROCm version (e.g., 8.0.0, 8.0.1rc1)",
    )

    parser.add_argument(
        "--download-dir",
        type=str,
        required=True,
        help="Directory to download packages to",
    )

    parser.add_argument(
        "--install-prefix",
        type=str,
        default="/opt/rocm/core",
        help="Installation prefix (default: /opt/rocm/core)",
    )

    parser.add_argument(
        "--s3-path",
        type=str,
        help="Optional custom S3 path (overrides standard path construction)",
    )

    args = parser.parse_args()

    # Create tester and run
    tester = PackageFullTester(
        package_type=args.package_type,
        s3_bucket=args.s3_bucket,
        artifact_group=args.artifact_group,
        artifact_id=args.artifact_id,
        rocm_version=args.rocm_version,
        download_dir=args.download_dir,
        install_prefix=args.install_prefix,
        s3_path=args.s3_path,
    )

    success = tester.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


