#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Full installation test script for ROCm native packages.

This version installs packages directly from ROCm nightly repositories using
native package managers (apt/dnf) instead of downloading from S3 first.

Supports nightly builds only:
- Nightly builds: https://rocm.nightlies.amd.com/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


class PackageFullTester:
    """Full installation tester for ROCm packages."""

    def __init__(
        self,
        package_type: str,
        repo_base_url: str,
        artifact_id: str,
        rocm_version: str,
        os_profile: str,
        date: str,
        install_prefix: str = "/opt/rocm",
        gfx_arch: Optional[str] = None,
    ):
        """Initialize the package full tester.

        Args:
            package_type: Type of package ('deb' or 'rpm')
            repo_base_url: Base URL for nightly repository (e.g., https://rocm.nightlies.amd.com)
            artifact_id: Artifact run ID
            rocm_version: ROCm version
            os_profile: OS profile (e.g., ubuntu2404, rhel8)
            date: Build date in YYYYMMDD format (required for nightly builds)
            install_prefix: Installation prefix (default: /opt/rocm)
            gfx_arch: GPU architecture (default: gfx94x)
        """
        self.package_type = package_type.lower()
        self.repo_base_url = repo_base_url.rstrip("/")
        self.artifact_id = artifact_id
        self.rocm_version = rocm_version
        self.os_profile = os_profile
        self.date = date
        self.install_prefix = install_prefix
        self.gfx_arch = gfx_arch.lower()

        # Validate inputs
        if self.package_type not in ["deb", "rpm"]:
            raise ValueError(
                f"Invalid package type: {package_type}. Must be 'deb' or 'rpm'"
            )

        if not self.date or len(self.date) != 8:
            raise ValueError(
                f"Invalid date format: {date}. Must be YYYYMMDD (e.g., 20260204)"
            )

        # Construct repository URL for nightly builds: base_url/{deb|rpm}/YYYYMMDD-RUNID/
        self.repo_url = (
            f"{self.repo_base_url}/{self.package_type}/{self.date}-{self.artifact_id}/"
        )

    def construct_repo_url_with_os(self) -> str:
        """Construct the full repository URL including OS profile for nightly builds.

        Returns:
            Full repository URL
        """
        if self.package_type == "deb":
            # Nightly DEB repository structure: base_url/deb/YYYYMMDD-RUNID/pool/main/
            return self.repo_url
        else:  # rpm
            # Nightly RPM repository structure: base_url/rpm/YYYYMMDD-RUNID/os_profile/x86_64/
            return f"{self.repo_url}x86_64/"

    def setup_deb_repository(self) -> bool:
        """Setup DEB repository on the system.

        Returns:
            True if setup successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("SETTING UP DEB REPOSITORY")
        print("=" * 80)

        repo_url = self.construct_repo_url_with_os()
        print(f"\nRepository URL: {repo_url}")
        print(f"OS Profile: {self.os_profile}")

        # Add repository to sources list
        print("\nAdding ROCm repository...")
        sources_list = f"/etc/apt/sources.list.d/rocm-test.list"

        repo_entry = f"deb [arch=amd64 trusted=yes] {repo_url} stable main\n"

        try:
            with open(sources_list, "w") as f:
                f.write(repo_entry)
            print(f"[PASS] Repository added to {sources_list}")
            print(f"       {repo_entry.strip()}")
        except Exception as e:
            print(f"[FAIL] Failed to add repository: {e}")
            return False

        # Update package lists
        print("\nUpdating package lists...")
        print("=" * 80)
        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                ["apt", "update"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Stream output line by line
            for line in process.stdout:
                line = line.rstrip()
                print(line)  # Print immediately
                sys.stdout.flush()  # Ensure immediate display

            # Wait for process to complete
            return_code = process.wait(timeout=120)

            if return_code == 0:
                print("\n[PASS] Package lists updated")
                return True
            else:
                print(
                    f"\n[FAIL] Failed to update package lists (exit code: {return_code})"
                )
                return False
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"\n[FAIL] apt update timed out")
            return False
        except Exception as e:
            print(f"[FAIL] Error updating package lists: {e}")
            return False

    def setup_rpm_repository(self) -> bool:
        """Setup RPM repository on the system.

        Returns:
            True if setup successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("SETTING UP RPM REPOSITORY")
        print("=" * 80)

        repo_url = self.construct_repo_url_with_os()
        print(f"\nRepository URL: {repo_url}")
        print(f"OS Profile: {self.os_profile}")

        # Create repository file
        print("\nCreating ROCm repository file...")
        repo_file = "/etc/yum.repos.d/rocm-test.repo"

        repo_content = f"""[rocm-test]
name=ROCm Test Repository
baseurl={repo_url}
enabled=1
gpgcheck=0
"""

        try:
            with open(repo_file, "w") as f:
                f.write(repo_content)
            print(f"[PASS] Repository file created: {repo_file}")
            print(f"\nRepository configuration:")
            print(repo_content)
        except Exception as e:
            print(f"[FAIL] Failed to create repository file: {e}")
            return False

        # Clean dnf cache
        print("\nCleaning dnf cache...")
        try:
            result = subprocess.run(
                ["dnf", "clean", "all"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
            )
            print("[PASS] dnf cache cleaned")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] Failed to clean dnf cache")
            print(f"Error: {e.stdout}")
            return False
        except subprocess.TimeoutExpired:
            print(f"[FAIL] dnf clean timed out")
            return False

    def install_deb_packages(self) -> bool:
        """Install ROCm DEB packages from repository.

        Returns:
            True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING DEB PACKAGES FROM REPOSITORY")
        print("=" * 80)

        # Construct package name
        package_name = f"amdrocm-{self.gfx_arch}"
        print(f"\nPackage to install: {package_name}")

        # Install using apt
        cmd = ["apt", "install", "-y", package_name]
        print(f"\nRunning: {' '.join(cmd)}")
        print("=" * 80)
        print("Installation progress (streaming output):\n")

        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Stream output line by line
            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                print(line)  # Print immediately
                output_lines.append(line)
                sys.stdout.flush()  # Ensure immediate display

            # Wait for process to complete
            return_code = process.wait(timeout=1800)  # 30 minute timeout

            if return_code == 0:
                print("\n" + "=" * 80)
                print("[PASS] DEB packages installed successfully from repository")
                return True
            else:
                print("\n" + "=" * 80)
                print(
                    f"[FAIL] Failed to install DEB packages (exit code: {return_code})"
                )
                return False

        except subprocess.TimeoutExpired:
            process.kill()
            print("\n" + "=" * 80)
            print("[FAIL] Installation timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"\n[FAIL] Error during installation: {e}")
            return False

    def install_rpm_packages(self) -> bool:
        """Install ROCm RPM packages from repository.

        Returns:
            True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING RPM PACKAGES FROM REPOSITORY")
        print("=" * 80)

        # Construct package name
        package_name = f"amdrocm-{self.gfx_arch}"
        print(f"\nPackage to install: {package_name}")

        # Install using dnf
        cmd = ["dnf", "install", "-y", package_name]
        print(f"\nRunning: {' '.join(cmd)}")
        print("=" * 80)
        print("Installation progress (streaming output):\n")

        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Stream output line by line
            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                print(line)  # Print immediately
                output_lines.append(line)
                sys.stdout.flush()  # Ensure immediate display

            # Wait for process to complete
            return_code = process.wait(timeout=1800)  # 30 minute timeout

            if return_code == 0:
                print("\n" + "=" * 80)
                print("[PASS] RPM packages installed successfully from repository")
                return True
            else:
                print("\n" + "=" * 80)
                print(
                    f"[FAIL] Failed to install RPM packages (exit code: {return_code})"
                )
                return False

        except subprocess.TimeoutExpired:
            process.kill()
            print("\n" + "=" * 80)
            print("[FAIL] Installation timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"\n[FAIL] Error during installation: {e}")
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
            print(f"\n[FAIL] Installation directory not found: {self.install_prefix}")
            return False

        print(f"\n[PASS] Installation directory exists: {self.install_prefix}")

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
                print(f"   [PASS] {component}")
                found_count += 1
            else:
                print(f"   [WARN] {component} (not found)")
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

            rocm_packages = [
                line
                for line in result.stdout.split("\n")
                if grep_pattern.lower() in line.lower()
            ]
            print(f"   Found {len(rocm_packages)} ROCm packages installed")

            if rocm_packages:
                print("\n   Sample packages:")
                for pkg in rocm_packages[:5]:  # Show first 5
                    print(f"      {pkg.strip()}")
                if len(rocm_packages) > 5:
                    print(f"      ... and {len(rocm_packages) - 5} more")

        except subprocess.CalledProcessError as e:
            print(f"   [WARN] Could not query installed packages")

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
                print("   [PASS] rocminfo executed successfully")
                # Print first few lines of output
                lines = result.stdout.split("\n")[:10]
                print("\n   First few lines of rocminfo output:")
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
            except subprocess.TimeoutExpired:
                print("   [WARN] rocminfo timed out (may require GPU hardware)")
            except subprocess.CalledProcessError as e:
                print(f"   [WARN] rocminfo failed (may require GPU hardware)")
            except Exception as e:
                print(f"   [WARN] Could not run rocminfo: {e}")

        # Test rdhc.py if available
        self.test_rdhc()

        # Return success if at least some components were found
        if found_count >= 2:  # Require at least 2 key components
            print("\n[PASS] ROCm installation verification PASSED")
            return True
        else:
            print("\n[FAIL] ROCm installation verification FAILED")
            return False

    def test_rdhc(self) -> bool:
        """Test rdhc.py binary in libexec/rocm-core/.

        Returns:
            True if test successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("TESTING RDHC.PY")
        print("=" * 80)

        install_path = Path(self.install_prefix)
        rdhc_script = install_path / "libexec" / "rocm-core" / "rdhc.py"

        # Check if script exists
        if not rdhc_script.exists():
            print(f"\n[WARN] rdhc.py not found at: {rdhc_script}")
            print("       This is expected if rocm-core package is not installed")
            return False

        print(f"\n[PASS] rdhc.py found at: {rdhc_script}")

        # Check if script is executable or can be run with python
        if os.access(rdhc_script, os.X_OK):
            cmd = [str(rdhc_script)]
        else:
            cmd = [sys.executable, str(rdhc_script)]

        # Try to run with --help first, then without arguments
        test_args = ["--all"]
        print(f"\nTrying to run rdhc.py with --all...")
        print(f"Command: {' '.join(cmd + test_args)}")

        try:
            result = subprocess.run(
                cmd + test_args,
                cwd=str(install_path),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            print("   [PASS] rdhc.py executed successfully with --all")
            if result.stdout:
                # Print first few lines of output
                lines = result.stdout.split("\n")[:5]
                print("\n   First few lines of output:")
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
            return True
        except subprocess.TimeoutExpired:
            print("   [WARN] rdhc.py --alltimed out")
            # Try without arguments
            return self._try_rdhc_without_args(cmd, install_path)
        except subprocess.CalledProcessError:
            print("   [WARN] rdhc.py --all failed, trying without arguments...")
            # Try without arguments
            return self._try_rdhc_without_args(cmd, install_path)
        except Exception as e:
            print(f"   [WARN] Could not run rdhc.py: {e}")
            return False

    def _try_rdhc_without_args(self, cmd: list, install_path: Path) -> bool:
        """Try running rdhc.py without arguments.

        Args:
            cmd: Command to run (without arguments)
            install_path: Installation prefix path

        Returns:
            True if successful, False otherwise
        """
        print(f"\nTrying to run rdhc.py without arguments...")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(install_path),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
            print("   [PASS] rdhc.py executed successfully")
            if result.stdout:
                # Print first few lines of output
                lines = result.stdout.split("\n")[:5]
                print("\n   First few lines of output:")
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
            return True
        except subprocess.TimeoutExpired:
            print("   [WARN] rdhc.py timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"   [WARN] rdhc.py failed (return code: {e.returncode})")
            if e.stdout:
                # Print first few lines of error output
                lines = e.stdout.split("\n")[:3]
                print("\n   Error output:")
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
            return False
        except Exception as e:
            print(f"   [WARN] Could not run rdhc.py: {e}")
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
        print(f"Repository Base URL: {self.repo_base_url}")
        print(f"Artifact ID: {self.artifact_id}")
        print(f"Build Date: {self.date}")
        print(f"ROCm Version: {self.rocm_version}")
        print(f"OS Profile: {self.os_profile}")
        print(f"GPU Architecture: {self.gfx_arch}")
        print(f"Install Prefix: {self.install_prefix}")
        print(f"\nRepository URL: {self.construct_repo_url_with_os()}")

        try:
            # Step 1: Setup repository
            if self.package_type == "deb":
                setup_success = self.setup_deb_repository()
            else:  # rpm
                setup_success = self.setup_rpm_repository()

            if not setup_success:
                return False

            # Step 2: Install packages
            if self.package_type == "deb":
                install_success = self.install_deb_packages()
            else:  # rpm
                install_success = self.install_rpm_packages()

            if not install_success:
                return False

            # Step 3: Verify installation
            verification_success = self.verify_rocm_installation()

            # Print final status
            print("\n" + "=" * 80)
            if install_success and verification_success:
                print("[PASS] FULL INSTALLATION TEST PASSED")
                print(
                    "\nROCm has been successfully installed from repository and verified!"
                )
            else:
                print("[FAIL] FULL INSTALLATION TEST FAILED")
            print("=" * 80 + "\n")

            return install_success and verification_success

        except Exception as e:
            print(f"\n[FAIL] Error during full installation test: {e}")
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
  # Install from nightly DEB repository (Ubuntu 24.04)
  python package_full_test_v2.py \\
      --package-type deb \\
      --repo-base-url https://rocm.nightlies.amd.com \\
      --artifact-id 21658678136 \\
      --date 20260204 \\
      --rocm-version 8.0.0 \\
      --os-profile ubuntu2404 \\
      --gfx-arch gfx94x

  # Install from nightly RPM repository (RHEL 8)
  python package_full_test_v2.py \\
      --package-type rpm \\
      --repo-base-url https://rocm.nightlies.amd.com \\
      --artifact-id 21658678136 \\
      --date 20260204 \\
      --rocm-version 8.0.0 \\
      --os-profile rhel8 \\
      --gfx-arch gfx94x

  # Install from nightly for different GPU (Strix Halo)
  python package_full_test_v2.py \\
      --package-type deb \\
      --repo-base-url https://rocm.nightlies.amd.com \\
      --artifact-id 21658678136 \\
      --date 20260204 \\
      --rocm-version 8.0.0 \\
      --os-profile ubuntu2404 \\
      --gfx-arch gfx1151
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
        "--repo-base-url",
        type=str,
        required=True,
        help="Base URL for nightly repository (e.g., https://rocm.nightlies.amd.com)",
    )

    parser.add_argument(
        "--artifact-id",
        type=str,
        required=True,
        help="Artifact run ID (e.g., 21658678136)",
    )

    parser.add_argument(
        "--rocm-version",
        type=str,
        required=True,
        help="ROCm version (e.g., 8.0.0, 7.9.0rc1)",
    )

    parser.add_argument(
        "--os-profile",
        type=str,
        required=True,
        help="OS profile (e.g., ubuntu2404, rhel8, debian12, sles16)",
    )

    parser.add_argument(
        "--gfx-arch",
        type=str,
        default="gfx94x",
        help="GPU architecture (default: gfx94x). Examples: gfx94x, gfx110x, gfx1151",
    )

    parser.add_argument(
        "--install-prefix",
        type=str,
        default="/opt/rocm/core",
        help="Installation prefix (default: /opt/rocm/core)",
    )

    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Build date in YYYYMMDD format (required for nightly builds, e.g., 20260204)",
    )

    args = parser.parse_args()

    # Validate and normalize parameters
    if not args.artifact_id or not args.artifact_id.strip():
        parser.error("Artifact ID cannot be empty")

    if not args.rocm_version or not args.rocm_version.strip():
        parser.error("ROCm version cannot be empty")

    if not args.os_profile or not args.os_profile.strip():
        parser.error("OS profile cannot be empty")

    if not args.date or not args.date.strip():
        parser.error("Build date cannot be empty")

    if len(args.date) != 8 or not args.date.isdigit():
        parser.error(
            f"Invalid date format: {args.date}. Must be YYYYMMDD (e.g., 20260204)"
        )
    # Print configuration
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Package Type: {args.package_type}")
    print(f"Repository Base URL: {args.repo_base_url}")
    print(f"Artifact ID: {args.artifact_id}")
    print(f"Build Date: {args.date}")
    print(f"ROCm Version: {args.rocm_version}")
    print(f"OS Profile: {args.os_profile}")
    print(f"GPU Architecture: {args.gfx_arch}")
    print(f"Install Prefix: {args.install_prefix}")
    print("=" * 80)

    # Create installer and run
    tester = PackageFullTester(
        package_type=args.package_type,
        repo_base_url=args.repo_base_url,
        artifact_id=args.artifact_id,
        rocm_version=args.rocm_version,
        os_profile=args.os_profile,
        date=args.date,
        install_prefix=args.install_prefix,
        gfx_arch=args.gfx_arch,
    )

    success = tester.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
