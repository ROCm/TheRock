#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Full installation test script for ROCm native packages (V2 - Optimized).

This optimized version reduces the number of parameters by moving URL generation
and package name construction logic to the YAML workflow file.

Supports both nightly and prerelease builds:
- Nightly builds: https://rocm.nightlies.amd.com/
- Prerelease builds: https://rocm.prereleases.amd.com/packages/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


class NativeLinuxPackagesTester:
    """Full installation tester for ROCm native Linux packages with minimal parameters."""

    @staticmethod
    def _derive_package_type(os_profile: str) -> str:
        """Derive package type from OS profile.

        Args:
            os_profile: OS profile (e.g., ubuntu2404, rhel8, debian12, sles16, azl3)

        Returns:
            Package type ('deb' or 'rpm')
        """
        os_profile_lower = os_profile.lower()
        if os_profile_lower.startswith(("ubuntu", "debian")):
            return "deb"
        elif os_profile_lower.startswith(
            ("rhel", "sles", "almalinux", "centos", "azl")
        ):
            return "rpm"
        else:
            raise ValueError(
                f"Unable to derive package type from OS profile: {os_profile}. "
                "Supported profiles: ubuntu*, debian*, rhel*, sles*, almalinux*, centos*, azl*"
            )

    def _is_sles(self) -> bool:
        """Check if the OS profile is SLES (SUSE Linux Enterprise Server).

        Returns:
            True if SLES, False otherwise
        """
        return self.os_profile.lower().startswith("sles")

    def __init__(
        self,
        repo_url: str,
        os_profile: str,
        release_type: str = "nightly",
        install_prefix: str = "/opt/rocm/core",
        gfx_arch: Optional[str] = None,
        gpg_key_url: Optional[str] = None,
    ):
        """Initialize the package full tester.

        Args:
            repo_url: Full repository URL (constructed in YAML)
            os_profile: OS profile (e.g., ubuntu2404, rhel8, debian12, sles16, azl3)
            release_type: Type of release ('nightly' or 'prerelease')
            install_prefix: Installation prefix (default: /opt/rocm/core)
            gfx_arch: GPU architecture (default: gfx94x)
            gpg_key_url: GPG key URL (only needed for prerelease)
        """
        self.os_profile = os_profile.lower()
        self.package_type = self._derive_package_type(os_profile)
        self.repo_url = repo_url.rstrip("/")
        self.release_type = release_type.lower()
        self.install_prefix = install_prefix
        self.gfx_arch = gfx_arch.lower() if gfx_arch else "gfx94x"
        self.gpg_key_url = gpg_key_url

        # Construct package name from gfx_arch
        self.package_name = f"amdrocm-{self.gfx_arch}"

        # Validate inputs
        if self.release_type not in ["nightly", "prerelease"]:
            raise ValueError(
                f"Invalid release type: {release_type}. Must be 'nightly' or 'prerelease'"
            )

    def setup_gpg_key(self) -> bool:
        """Setup GPG key for repositories that require GPG verification.

        Returns:
            True if setup successful, False otherwise
        """
        if not self.gpg_key_url:
            return True  # Not needed if no GPG key URL provided

        print("\n" + "=" * 80)
        print("SETTING UP GPG KEY")
        print("=" * 80)

        print(f"\nGPG Key URL: {self.gpg_key_url}")

        if self.package_type == "deb":
            # For DEB, import GPG key using pipeline approach
            keyring_dir = "/etc/apt/keyrings"
            keyring_file = f"{keyring_dir}/rocm.gpg"

            try:
                # Create keyring directory
                print(f"\nCreating keyring directory: {keyring_dir}...")
                result = subprocess.run(
                    ["mkdir", "--parents", "--mode=0755", keyring_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                )
                print(f"[PASS] Created keyring directory: {keyring_dir}")

                # Download, dearmor, and write GPG key using pipeline
                # wget URL -O - | gpg --dearmor | tee keyring_file > /dev/null
                print(f"\nDownloading and importing GPG key from {self.gpg_key_url}...")
                pipeline_cmd = (
                    f"wget -q -O - {self.gpg_key_url} | "
                    f"gpg --dearmor | "
                    f"tee {keyring_file} > /dev/null"
                )

                result = subprocess.run(
                    pipeline_cmd,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )

                # Set proper permissions on the keyring file
                os.chmod(keyring_file, 0o644)
                print(f"[PASS] GPG key imported to {keyring_file}")
                return True

            except subprocess.CalledProcessError as e:
                print(f"[FAIL] Failed to setup GPG key: {e}")
                if e.stderr:
                    print(f"Error output: {e.stderr.decode()}")
                return False
            except Exception as e:
                print(f"[FAIL] Error setting up GPG key: {e}")
                return False
        else:  # rpm
            # For RPM (including SLES), GPG key URL is specified in repo file
            # zypper will automatically fetch and use the GPG key from the URL
            # No need to download or import separately (following official ROCm documentation)
            return True

    def setup_deb_repository(self) -> bool:
        """Setup DEB repository on the system.

        Returns:
            True if setup successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("SETTING UP DEB REPOSITORY")
        print("=" * 80)

        print(f"\nRepository URL: {self.repo_url}")
        print(f"Release Type: {self.release_type}")

        # Setup GPG key if GPG key URL is provided
        if self.gpg_key_url:
            if not self.setup_gpg_key():
                return False

        # Add repository to sources list
        print("\nAdding ROCm repository...")
        sources_list = f"/etc/apt/sources.list.d/rocm-test.list"

        if self.gpg_key_url:
            # Use GPG key verification
            keyring_file = "/etc/apt/keyrings/rocm.gpg"
            repo_entry = f"deb [arch=amd64 signed-by={keyring_file}] {self.repo_url} stable main\n"
        else:
            # No GPG check (trusted=yes)
            repo_entry = f"deb [arch=amd64 trusted=yes] {self.repo_url} stable main\n"

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

        print(f"\nRepository URL: {self.repo_url}")
        print(f"Release Type: {self.release_type}")
        print(f"OS Profile: {self.os_profile}")

        # Setup GPG key if GPG key URL is provided (only needed for non-SLES systems)
        # SLES uses --gpg-auto-import-keys flag which handles it automatically
        if self.gpg_key_url and not self._is_sles():
            if not self.setup_gpg_key():
                return False

        # SLES uses zypper, others use dnf/yum
        if self._is_sles():
            return self._setup_sles_repository()
        else:
            return self._setup_dnf_repository()

    def _setup_sles_repository(self) -> bool:
        """Setup repository for SLES using zypper.

        Returns:
            True if setup successful, False otherwise
        """
        repo_name = "rocm-test"
        repo_file = f"/etc/zypp/repos.d/{repo_name}.repo"

        # Remove existing repository if it exists
        print(f"\nRemoving existing repository '{repo_name}' if it exists...")
        subprocess.run(
            ["zypper", "--non-interactive", "removerepo", repo_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )  # Ignore errors if repo doesn't exist

        # Create repository file following official ROCm documentation format
        # Reference: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-sles.html
        print(f"\nCreating ROCm repository file at {repo_file}...")
        if self.gpg_key_url:
            # Use GPG key verification (gpgcheck=1)
            repo_content = f"""[{repo_name}]
name=ROCm {self.release_type} repository
baseurl={self.repo_url}
enabled=1
gpgcheck=1
gpgkey={self.gpg_key_url}
"""
        else:
            # No GPG check (gpgcheck=0)
            repo_content = f"""[{repo_name}]
name=ROCm {self.release_type} repository
baseurl={self.repo_url}
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

        # Refresh repository metadata
        print("\nRefreshing repository metadata...")
        try:
            # Use --non-interactive to avoid prompts
            # If GPG key URL is provided, use --gpg-auto-import-keys to automatically import and trust GPG keys
            refresh_cmd = ["zypper", "--non-interactive"]
            if self.gpg_key_url:
                refresh_cmd.append("--gpg-auto-import-keys")
            refresh_cmd.append("refresh")
            refresh_cmd.append(repo_name)
            process = subprocess.Popen(
                refresh_cmd,
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
                print("\n[PASS] Repository metadata refreshed")
                return True
            else:
                print(
                    f"\n[FAIL] Failed to refresh repository metadata (exit code: {return_code})"
                )
                return False
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"\n[FAIL] zypper refresh timed out")
            return False
        except Exception as e:
            print(f"[FAIL] Error refreshing repository metadata: {e}")
            return False

    def _setup_dnf_repository(self) -> bool:
        """Setup repository for RHEL/AlmaLinux/CentOS using dnf/yum.

        Returns:
            True if setup successful, False otherwise
        """
        print("\nUsing dnf/yum for repository setup...")

        # Create repository file
        print("\nCreating ROCm repository file...")
        repo_file = "/etc/yum.repos.d/rocm-test.repo"

        if self.gpg_key_url:
            # Use GPG key verification
            repo_content = f"""[rocm-test]
name=ROCm Repository
baseurl={self.repo_url}
enabled=1
gpgcheck=1
gpgkey={self.gpg_key_url}
"""
        else:
            # No GPG check
            repo_content = f"""[rocm-test]
name=Native Linux Package Test Repository
baseurl={self.repo_url}
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
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Failed to clean dnf cache (may not be critical)")
            print(f"Error: {e.stdout}")
        except subprocess.TimeoutExpired:
            print(f"[WARN] dnf clean timed out (may not be critical)")

        # Refresh repository metadata
        print("\nRefreshing repository metadata...")
        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                ["dnf", "makecache"],
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
                print("\n[PASS] Repository metadata refreshed")
                return True
            else:
                print(
                    f"\n[FAIL] Failed to refresh repository metadata (exit code: {return_code})"
                )
                return False
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"\n[FAIL] dnf makecache timed out")
            return False
        except Exception as e:
            print(f"[FAIL] Error refreshing repository metadata: {e}")
            return False

    def install_deb_packages(self) -> bool:
        """Install ROCm DEB packages from repository.

        Returns:
            True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING DEB PACKAGES FROM REPOSITORY")
        print("=" * 80)

        print(f"\nPackage to install: {self.package_name}")

        # Install using apt
        cmd = ["apt", "install", "-y", self.package_name]
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

        print(f"\nPackage to install: {self.package_name}")

        # Use zypper for SLES, dnf for others
        if self._is_sles():
            # If no GPG key URL, skip GPG checks during installation
            if not self.gpg_key_url:
                cmd = [
                    "zypper",
                    "--non-interactive",
                    "--no-gpg-checks",
                    "install",
                    "-y",
                    self.package_name,
                ]
            else:
                # If GPG key URL is provided, use --gpg-auto-import-keys to automatically import and trust GPG keys
                cmd = [
                    "zypper",
                    "--non-interactive",
                    "--gpg-auto-import-keys",
                    "install",
                    "-y",
                    self.package_name,
                ]
            print("[INFO] Using zypper for SLES package installation")
        else:
            cmd = ["dnf", "install", "-y", self.package_name]
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
            elif self._is_sles():
                # Use zypper for SLES to list installed packages
                cmd = ["zypper", "--non-interactive", "search", "-i", "rocm"]
                grep_pattern = "rocm"
            else:
                # Use rpm for other RPM-based systems (RHEL, AlmaLinux, CentOS)
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
            print("   [WARN] rdhc.py --all timed out")
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
        print("FULL INSTALLATION TEST - NATIVE LINUX PACKAGES (V2 - OPTIMIZED)")
        print("=" * 80)
        print(f"\nOS Profile: {self.os_profile}")
        print(f"Package Type (derived): {self.package_type.upper()}")
        print(f"Release Type: {self.release_type.upper()}")
        print(f"Repository URL: {self.repo_url}")
        print(f"GPU Architecture: {self.gfx_arch}")
        print(f"Package Name: {self.package_name}")
        print(f"Install Prefix: {self.install_prefix}")

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
        description="Full installation test for ROCm native packages (V2 - Optimized with minimal parameters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install from nightly DEB repository (Ubuntu 24.04)
  python native_linux_packages_test.py \\
      --os-profile ubuntu2404 \\
      --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
      --gfx-arch gfx94x \\
      --release-type nightly \\
      --install-prefix /opt/rocm/core

  # Install from prerelease DEB repository (Ubuntu 24.04)
  python native_linux_packages_test.py \\
      --os-profile ubuntu2404 \\
      --repo-url https://rocm.prereleases.amd.com/packages/ubuntu2404 \\
      --gfx-arch gfx94x \\
      --release-type prerelease \\
      --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg \\
      --install-prefix /opt/rocm/core

  # Install from nightly RPM repository (RHEL 8)
  python native_linux_packages_test.py \\
      --os-profile rhel8 \\
      --repo-url https://rocm.nightlies.amd.com/rpm/20260204-21658678136/rhel8/x86_64/ \\
      --gfx-arch gfx94x \\
      --release-type nightly \\
      --install-prefix /opt/rocm/core

  # Install from prerelease RPM repository (RHEL 8)
  python native_linux_packages_test.py \\
      --os-profile rhel8 \\
      --repo-url https://rocm.prereleases.amd.com/packages/rhel8/x86_64/ \\
      --gfx-arch gfx94x \\
      --release-type prerelease \\
      --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg \\
      --install-prefix /opt/rocm/core
        """,
    )

    parser.add_argument(
        "--os-profile",
        type=str,
        required=True,
        help="OS profile (e.g., ubuntu2404, rhel8, debian12, sles16, azl3). Package type is derived from this.",
    )

    parser.add_argument(
        "--repo-url",
        type=str,
        required=True,
        help="Full repository URL (constructed in YAML workflow)",
    )

    parser.add_argument(
        "--gfx-arch",
        type=str,
        default="gfx94x",
        help="GPU architecture (default: gfx94x). Examples: gfx94x, gfx110x, gfx1151",
    )

    parser.add_argument(
        "--release-type",
        type=str,
        required=True,
        choices=["nightly", "prerelease"],
        help="Type of release: 'nightly' or 'prerelease'",
    )

    parser.add_argument(
        "--install-prefix",
        type=str,
        default="/opt/rocm/core",
        help="Installation prefix (default: /opt/rocm/core)",
    )

    parser.add_argument(
        "--gpg-key-url",
        type=str,
        help="GPG key URL (only needed for prerelease)",
    )

    args = parser.parse_args()

    # Validate and normalize parameters
    if not args.release_type or not args.release_type.strip():
        parser.error("Release Type cannot be empty")

    if not args.os_profile or not args.os_profile.strip():
        parser.error("OS profile cannot be empty")

    if not args.repo_url or not args.repo_url.strip():
        parser.error("Repository URL cannot be empty")

    if not args.gfx_arch or not args.gfx_arch.strip():
        parser.error("GPU architecture cannot be empty")

    # Derive package type from OS profile
    try:
        derived_package_type = NativeLinuxPackagesTester._derive_package_type(
            args.os_profile
        )
    except ValueError as e:
        parser.error(str(e))

    # Print configuration
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"OS Profile: {args.os_profile}")
    print(f"Package Type (derived): {derived_package_type}")
    print(f"Release Type: {args.release_type}")
    print(f"Repository URL: {args.repo_url}")
    print(f"GPU Architecture: {args.gfx_arch}")
    print(f"Install Prefix: {args.install_prefix}")
    if args.gpg_key_url:
        print(f"GPG Key URL: {args.gpg_key_url}")
    print("=" * 80)

    # Create installer and run
    tester = NativeLinuxPackagesTester(
        os_profile=args.os_profile,
        repo_url=args.repo_url,
        release_type=args.release_type,
        install_prefix=args.install_prefix,
        gfx_arch=args.gfx_arch,
        gpg_key_url=args.gpg_key_url,
    )

    success = tester.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
