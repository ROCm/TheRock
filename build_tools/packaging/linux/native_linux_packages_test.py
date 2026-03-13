#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Full installation test script for ROCm native packages.

The test runs in three steps (invoked one by one from main):
  1. Repo setup and install: set up package-manager repository and install
     ROCm packages (amdrocm-{gfx_arch}, amdrocm-core-sdk-{gfx_arch}).
  2. Basic verification: sanity check that install prefix exists and minimum
     key components are present.
  3. Full verification: detailed checks (components, installed packages,
     rocminfo, rdhc.py).



Path and repo name are overridable via environment variables: ROCM_REPO_NAME (repo id used for
APT list, Zypper/Yum repo file and section), ROCM_APT_KEYRING_DIR, ROCM_APT_SOURCES_LIST,
ROCM_APT_KEYRING_FILE, ROCM_ZYPP_REPOS_DIR, ROCM_YUM_REPOS_DIR,
ROCM_RDHC_REL_PATH (relative path from install prefix to rdhc binary).

Prerequisites:
- This script does NOT start Docker or a VM. You must run it inside an existing
  container or VM that matches the target OS (e.g., Ubuntu for deb, AlmaLinux/RHEL
  for rpm, SLES container for sles). Start the appropriate Docker image or VM
  first, then invoke this script from inside that environment.
- Root or sudo is required (repository setup, package install, keyring writes).
- System packages: python3, pip, wget, curl; pip packages: pyelftools, requests,
  prettytable, PyYAML.

Example invocations:

  # Nightly DEB (Ubuntu 24.04) - run inside ubuntu:24.04 container or VM
  python3 native_linux_packages_test.py \\
    --os-profile ubuntu2404 \\
    --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
    --gfx-arch gfx94x \\
    --release-type nightly

  # Prerelease DEB with GPG verification
  python3 native_linux_packages_test.py \\
    --os-profile ubuntu2404 \\
    --repo-url https://rocm.prereleases.amd.com/packages/ubuntu2404 \\
    --release-type prerelease \\
    --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg

  # Nightly RPM (RHEL 8) - run inside rhel8/almalinux container or VM
  python3 native_linux_packages_test.py \\
    --os-profile rhel8 \\
    --repo-url https://rocm.nightlies.amd.com/rpm/20260204-21658678136/x86_64/ \\
    --gfx-arch gfx94x \\
    --release-type nightly

  # Prerelease RPM (SLES 16)
  python3 native_linux_packages_test.py \\
    --os-profile sles16 \\
    --repo-url https://rocm.prereleases.amd.com/packages/sles16/x86_64/ \\
    --release-type prerelease \\
    --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg
"""

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional, Union


def _env(key: str, default: str) -> str:
    """Return os.environ[key] if set and non-empty, else default."""
    v = os.environ.get(key, "").strip()
    return v if v else default


# --- Config: paths overridable via environment variables ---
# ROCM_REPO_NAME: logical repo name used for APT list, Zypper/Yum repo file and section id
# ROCM_APT_*, ROCM_ZYPP_*, ROCM_YUM_*, ROCM_RDHC_REL_PATH
REPO_NAME = _env("ROCM_REPO_NAME", "rocm-test")
APT_KEYRING_DIR = _env("ROCM_APT_KEYRING_DIR", "/etc/apt/keyrings")
APT_SOURCES_LIST = _env(
    "ROCM_APT_SOURCES_LIST", f"/etc/apt/sources.list.d/{REPO_NAME}.list"
)
APT_KEYRING_FILE = _env("ROCM_APT_KEYRING_FILE", "/etc/apt/keyrings/rocm.gpg")
ZYPP_REPOS_DIR = _env("ROCM_ZYPP_REPOS_DIR", "/etc/zypp/repos.d")
YUM_REPOS_DIR = _env("ROCM_YUM_REPOS_DIR", "/etc/yum.repos.d")
VERIFY_KEY_COMPONENTS = [
    "bin/rocminfo",
    "bin/hipcc",
    "bin/clinfo",
    "include/hip/hip_runtime.h",
    "lib/libamdhip64.so",
]
# Relative path from install prefix to rdhc binary (script); overridable via ROCM_RDHC_REL_PATH
RDHC_REL_PATH = _env("ROCM_RDHC_REL_PATH", "libexec/rocm-core/rdhc.py")

# Timeouts (seconds) and verification threshold
GPG_MKDIR_TIMEOUT_SEC = 10
GPG_KEY_TIMEOUT_SEC = 60
APT_UPDATE_TIMEOUT_SEC = 120
ZYPP_CLEAN_TIMEOUT_SEC = 60
ZYPP_REFRESH_TIMEOUT_SEC = 120
DNF_CLEAN_TIMEOUT_SEC = 60
INSTALL_TIMEOUT_SEC = 1800  # 30 minutes
ROCMINFO_TIMEOUT_SEC = 30
RDHC_TIMEOUT_SEC = 30
VERIFY_MIN_COMPONENTS = 2


def _run_streaming(cmd: list[str], timeout_sec: int) -> int:
    """Run a command with streaming stdout/stderr and return its exit code.

    Lines are printed as they are produced. Raises subprocess.TimeoutExpired
    (after killing the process) or OSError on failure.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line.rstrip())
            sys.stdout.flush()
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        raise


class NativeLinuxPackagesTester:
    """Full installation tester for ROCm native Linux packages."""

    @staticmethod
    def _derive_package_type(os_profile: str) -> str:
        """Derive package type from OS profile.

        Args:
            os_profile: OS profile (e.g., ubuntu2404, rhel8, debian12, sles16, almalinux9, centos7, azl3)

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
        install_prefix: Optional[str] = None,
        gfx_arch: Optional[Union[str, list[str]]] = None,
        gpg_key_url: Optional[str] = None,
    ):
        """Initialize the package Install tester.

        Args:
            repo_url: Full repository URL (constructed in YAML)
            os_profile: OS profile (e.g., ubuntu2404, rhel8, debian12, sles15, sles16, almalinux9, centos7, azl3)
            release_type: Type of release ('nightly' or 'prerelease')
            install_prefix: Installation prefix (default: /opt/rocm/core)
            gfx_arch: GPU architecture(s) as a single value or list (default: gfx94x).
                Only the first element is used for package name and installation.
            gpg_key_url: GPG key URL
        """
        self.os_profile = os_profile.lower()
        self.package_type = self._derive_package_type(os_profile)
        self.repo_url = repo_url.rstrip("/")
        self.release_type = release_type.lower()
        self.install_prefix = install_prefix
        # Normalize to list; only the first element is used for now
        if gfx_arch is None:
            self.gfx_arch_list: list[str] = ["gfx94x"]
        elif isinstance(gfx_arch, str):
            self.gfx_arch_list = [gfx_arch] if gfx_arch.strip() else ["gfx94x"]
        else:
            self.gfx_arch_list = [a for a in gfx_arch if a and str(a).strip()] or [
                "gfx94x"
            ]
        self.gfx_arch = self.gfx_arch_list[0].lower()
        self.gpg_key_url = gpg_key_url

        # Packages to install, in order
        self.package_names = [
            f"amdrocm-{self.gfx_arch}",
            f"amdrocm-core-sdk-{self.gfx_arch}",
        ]

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
            keyring_dir = APT_KEYRING_DIR
            keyring_file = f"{keyring_dir}/rocm.gpg"

            try:
                # Create keyring directory
                print(f"\nCreating keyring directory: {keyring_dir}...")
                result = subprocess.run(
                    ["mkdir", "--parents", "--mode=0755", keyring_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=GPG_MKDIR_TIMEOUT_SEC,
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
                    timeout=GPG_KEY_TIMEOUT_SEC,
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
            except OSError as e:
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
        sources_list = APT_SOURCES_LIST

        if self.gpg_key_url:
            # Use GPG key verification
            keyring_file = APT_KEYRING_FILE
            repo_entry = f"deb [arch=amd64 signed-by={keyring_file}] {self.repo_url} stable main\n"
        else:
            # No GPG check (trusted=yes)
            repo_entry = f"deb [arch=amd64 trusted=yes] {self.repo_url} stable main\n"

        try:
            with open(sources_list, "w") as f:
                f.write(repo_entry)
            print(f"[PASS] Repository added to {sources_list}")
            print(f"       {repo_entry.strip()}")
        except OSError as e:
            print(f"[FAIL] Failed to add repository: {e}")
            return False

        # Update package lists
        print("\nUpdating package lists...")
        print("=" * 80)
        try:
            return_code = _run_streaming(["apt", "update"], APT_UPDATE_TIMEOUT_SEC)
            if return_code == 0:
                print("\n[PASS] Package lists updated")
                return True
            print(f"\n[FAIL] Failed to update package lists (exit code: {return_code})")
            return False
        except subprocess.TimeoutExpired:
            print("\n[FAIL] apt update timed out")
            return False
        except OSError as e:
            print(f"[FAIL] Error updating package lists: {e}")
            return False

    def _setup_sles_repository(self) -> bool:
        """Setup repository for SLES using zypper.

        Returns:
            True if setup successful, False otherwise
        """
        repo_name = REPO_NAME
        repo_file = Path(ZYPP_REPOS_DIR) / f"{repo_name}.repo"

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
        except OSError as e:
            print(f"[FAIL] Failed to create repository file: {e}")
            return False

        # Clean zypper cache
        print("\nCleaning zypper cache...")
        try:
            result = subprocess.run(
                ["zypper", "--non-interactive", "clean", "--all"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=ZYPP_CLEAN_TIMEOUT_SEC,
            )
            if result.returncode == 0:
                print("[PASS] zypper cache cleaned")
            else:
                print(
                    f"[WARN] zypper clean returned {result.returncode} (may not be critical)"
                )
        except subprocess.TimeoutExpired:
            print(f"[WARN] zypper clean timed out (may not be critical)")
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"[WARN] zypper clean failed: {e} (may not be critical)")

        # Refresh repository metadata
        print("\nRefreshing repository metadata...")
        try:
            # Use --non-interactive to avoid prompts
            # If GPG key URL is provided, use --gpg-auto-import-keys to automatically import and trust GPG keys
            refresh_cmd = ["zypper", "--non-interactive"]
            if self.gpg_key_url:
                refresh_cmd.append("--gpg-auto-import-keys")
            refresh_cmd.extend(["refresh", repo_name])
            return_code = _run_streaming(refresh_cmd, ZYPP_REFRESH_TIMEOUT_SEC)
            if return_code == 0:
                print("\n[PASS] Repository metadata refreshed")
                return True
            print(
                f"\n[FAIL] Failed to refresh repository metadata (exit code: {return_code})"
            )
            return False
        except subprocess.TimeoutExpired:
            print("\n[FAIL] zypper refresh timed out")
            return False
        except OSError as e:
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
        repo_name = REPO_NAME
        repo_file = Path(YUM_REPOS_DIR) / f"{repo_name}.repo"

        if self.gpg_key_url:
            # Use GPG key verification
            repo_content = f"""[rocm_name]
name=ROCm Repository
baseurl={self.repo_url}
enabled=1
gpgcheck=1
gpgkey={self.gpg_key_url}
"""
        else:
            # No GPG check
            repo_content = f"""[rocm_name]
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
        except OSError as e:
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
                timeout=DNF_CLEAN_TIMEOUT_SEC,
            )
            print("[PASS] dnf cache cleaned")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Failed to clean dnf cache (may not be critical)")
            print(f"Error: {e.stdout}")
        except subprocess.TimeoutExpired:
            print(f"[WARN] dnf clean timed out (may not be critical)")

        print("\n[PASS] DNF repository setup complete")
        return True

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

    def install_deb_packages(self) -> bool:
        """Install ROCm DEB packages from repository.

        Returns:
            True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING DEB PACKAGES FROM REPOSITORY")
        print("=" * 80)

        print(f"\nPackages to install (in order): {self.package_names}")

        # Install using apt (packages in list order)
        cmd = ["apt", "install", "-y"] + self.package_names
        print(f"\nRunning: {' '.join(cmd)}")
        print("=" * 80)
        print("Installation progress (streaming output):\n")

        try:
            return_code = _run_streaming(cmd, INSTALL_TIMEOUT_SEC)
            if return_code == 0:
                print("\n" + "=" * 80)
                print("[PASS] DEB packages installed successfully from repository")
                return True
            print("\n" + "=" * 80)
            print(f"[FAIL] Failed to install DEB packages (exit code: {return_code})")
            return False
        except subprocess.TimeoutExpired:
            print("\n" + "=" * 80)
            print(f"[FAIL] Installation timed out after {INSTALL_TIMEOUT_SEC} minutes")
            return False
        except OSError as e:
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

        print(f"\nPackages to install (in order): {self.package_names}")

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
                ] + self.package_names
            else:
                # If GPG key URL is provided, use --gpg-auto-import-keys to automatically import and trust GPG keys
                cmd = [
                    "zypper",
                    "--non-interactive",
                    "--gpg-auto-import-keys",
                    "install",
                    "-y",
                ] + self.package_names
            print("[INFO] Using zypper for SLES package installation")
        else:
            cmd = ["dnf", "install", "-y"] + self.package_names
        print(f"\nRunning: {' '.join(cmd)}")
        print("=" * 80)
        print("Installation progress (streaming output):\n")

        try:
            return_code = _run_streaming(cmd, INSTALL_TIMEOUT_SEC)
            if return_code == 0:
                print("\n" + "=" * 80)
                print("[PASS] RPM packages installed successfully from repository")
                return True
            print("\n" + "=" * 80)
            print(f"[FAIL] Failed to install RPM packages (exit code: {return_code})")
            return False
        except subprocess.TimeoutExpired:
            print("\n" + "=" * 80)
            print(f"[FAIL] Installation timed out after {INSTALL_TIMEOUT_SEC} minutes")
            return False
        except OSError as e:
            print(f"\n[FAIL] Error during installation: {e}")
            return False

    def run_repo_setup_and_install(self) -> bool:
        """Step 1: Set up package-manager repository and install ROCm packages.

        Returns:
            True if repository setup and package installation both succeeded.
        """
        print("\n" + "=" * 80)
        print("STEP 1: REPOSITORY SETUP AND PACKAGE INSTALLATION")
        print("=" * 80)
        print(f"\nOS Profile: {self.os_profile}")
        print(f"Package Type (derived): {self.package_type.upper()}")
        print(f"Repository URL: {self.repo_url}")
        print(f"Packages (in order): {self.package_names}")

        if self.package_type == "deb":
            if not self.setup_deb_repository():
                return False
            return self.install_deb_packages()
        else:
            if not self.setup_rpm_repository():
                return False
            return self.install_rpm_packages()

    def run_basic_verification(self) -> bool:
        """Step 2: Basic install verification (components, packages, rocminfo).

        Checks install prefix, key components, lists installed packages, runs rocminfo
        if available. Does not run test_rdhc (that is in run_full_verification).

        Returns:
            True if basic verification passed (enough components found).
        """
        print("\n" + "=" * 80)
        print("STEP 2: BASIC INSTALL VERIFICATION")
        print("=" * 80)

        install_path = Path(self.install_prefix)
        if not install_path.exists():
            print(f"\n[FAIL] Installation directory not found: {self.install_prefix}")
            return False

        print(f"\n[PASS] Installation directory exists: {self.install_prefix}")

        key_components = VERIFY_KEY_COMPONENTS
        print("\nChecking for key ROCm components:")
        found_count = 0
        for component in key_components:
            component_path = install_path / component
            if component_path.exists():
                print(f"   [PASS] {component}")
                found_count += 1
            else:
                print(f"   [WARN] {component} (not found)")

        print(f"\nComponents found: {found_count}/{len(key_components)}")

        # Check installed packages
        print("\nChecking installed packages:")
        try:
            if self.package_type == "deb":
                cmd = ["dpkg", "-l"]
                grep_pattern = "rocm"
            elif self._is_sles():
                cmd = ["zypper", "--non-interactive", "search", "-i", "rocm"]
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
                print("\n   Sample packages (Show first 5):")
                for pkg in rocm_packages[:5]:
                    print(f"      {pkg.strip()}")
                if len(rocm_packages) > 5:
                    print(f"      ... and {len(rocm_packages) - 5} more")
        except subprocess.CalledProcessError:
            print("   [WARN] Could not query installed packages")

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
                    timeout=ROCMINFO_TIMEOUT_SEC,
                )
                print("   [PASS] rocminfo executed successfully")
                lines = result.stdout.split("\n")[:10]
                print("\n   First few lines of rocminfo output:")
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
            except subprocess.TimeoutExpired:
                print("   [WARN] rocminfo timed out (may require GPU hardware)")
            except subprocess.CalledProcessError:
                print("   [WARN] rocminfo failed (may require GPU hardware)")
            except OSError as e:
                print(f"   [WARN] Could not run rocminfo: {e}")

        if found_count >= VERIFY_MIN_COMPONENTS:
            print("\n[PASS] Basic verification PASSED")
            return True
        print("\n[FAIL] Basic verification FAILED (insufficient components)")
        return False

    def run_full_verification(self) -> bool:
        """Step 3: Full verification — runs test_rdhc only."""
        print("\n" + "=" * 80)
        print("STEP 3: FULL VERIFICATION (RDHC)")
        print("=" * 80)
        return self.test_rdhc()

    def test_rdhc(self) -> bool:
        """Test rdhc.py binary in libexec/rocm-core/.

        Returns:
            True if test successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("TESTING RDHC.PY")
        print("=" * 80)

        install_path = Path(self.install_prefix)
        rdhc_script = (install_path / RDHC_REL_PATH).resolve()

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

        # Set RDHC arguments for full test
        test_args = ["--rocm-install-prefix", "/opt/rocm/core", "--all"]
        print(f"\nRun rdhc.py with --rocm-install-prefix /opt/rocm/core --all...")
        print(f"Command: {' '.join(cmd + test_args)}")

        try:
            result = subprocess.run(
                cmd + test_args,
                cwd=str(install_path),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=RDHC_TIMEOUT_SEC,
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
            print("   [WARN] rdhc.py --all timed out")
            return False
        except subprocess.CalledProcessError:
            print("   [WARN] rdhc.py --all failed")
            return False
        except OSError as e:
            print(f"   [WARN] Could not run rdhc.py: {e}")
            return False


def main():
    """Main entry point for the Native Linux Package Installation Test script."""
    epilog = """
Examples:
  # Nightly DEB (Ubuntu 24.04) - run inside matching container/VM
  python native_linux_packages_test.py --os-profile ubuntu2404 \\
      --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
      --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

  # Prerelease DEB with GPG verification
  python native_linux_packages_test.py --os-profile ubuntu2404 \\
      --repo-url https://rocm.prereleases.amd.com/packages/ubuntu2404 \\
      --gfx-arch gfx94x --release-type prerelease --install-prefix /opt/rocm/core \\
      --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg

  # Nightly RPM (RHEL 8)
  python native_linux_packages_test.py --os-profile rhel8 \\
      --repo-url https://rocm.nightlies.amd.com/rpm/20260204-21658678136/rhel8/x86_64/ \\
      --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

  # Prerelease RPM (RHEL 8)
  python native_linux_packages_test.py --os-profile rhel8 \\
      --repo-url https://rocm.prereleases.amd.com/packages/rhel8/x86_64/ \\
      --gfx-arch gfx94x --release-type prerelease --install-prefix /opt/rocm/core \\
      --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg
"""

    parser = argparse.ArgumentParser(
        description="Full installation test for ROCm native packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    parser.add_argument(
        "--os-profile",
        type=str,
        required=True,
        help="OS profile (e.g., ubuntu2404, rhel8, debian12, sles15, sles16, almalinux9, centos7, azl3). Package type is derived from this.",
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
        nargs="+",
        required=True,
        metavar="ARCH",
        help="GPU architecture(s) as a list. Only the first is used for now. Examples: gfx94x, gfx110x gfx1151",
    )

    parser.add_argument(
        "--release-type",
        type=str,
        choices=["nightly", "prerelease"],
        help="Type of release: 'nightly' or 'prerelease'",
    )

    parser.add_argument(
        "--install-prefix",
        type=str,
        help="Installation prefix (e.g. /opt/rocm/core)",
    )

    parser.add_argument(
        "--gpg-key-url",
        type=str,
        help="GPG key URL",
    )

    parser.add_argument(
        "--test-type",
        type=str,
        choices=["sanity", "full"],
        default="sanity",
        help="Test type: 'sanity' runs only basic verification; 'full' runs basic + full verification.",
    )

    args = parser.parse_args()

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
    print(f"GPU Architecture(s): {args.gfx_arch} (using first: {args.gfx_arch[0]})")
    print(f"Install Prefix: {args.install_prefix}")
    print(f"Test Type: {args.test_type}")
    if args.gpg_key_url:
        print(f"GPG Key URL: {args.gpg_key_url}")
    print("=" * 80)

    # Create tester and run the three operations in order
    tester = NativeLinuxPackagesTester(
        os_profile=args.os_profile,
        repo_url=args.repo_url,
        release_type=args.release_type,
        install_prefix=args.install_prefix,
        gfx_arch=args.gfx_arch,
        gpg_key_url=args.gpg_key_url,
    )

    print("\n" + "=" * 80)
    print("INSTALLATION TEST - NATIVE LINUX PACKAGES")
    print("=" * 80)
    print(f"Release Type: {tester.release_type.upper()}")
    print(f"Install Prefix: {tester.install_prefix}")
    print(f"Test Type: {args.test_type}")
    print("=" * 80)

    try:
        # 1. Repo setup and install (always)
        if not tester.run_repo_setup_and_install():
            print("\n[FAIL] Step 1 (repo setup and install) failed.")
            sys.exit(1)

        # 2. Basic sanity / basic install verification (always)
        if not tester.run_basic_verification():
            print("\n[FAIL] Step 2 (basic verification) failed.")
            sys.exit(1)

        # 3. Full verification only when test_type is "full"
        if args.test_type == "full":
            if not tester.run_full_verification():
                print("\n[FAIL] Step 3 (full verification) failed.")
                sys.exit(1)

        print("\n" + "=" * 80)
        print("[PASS] INSTALLATION TEST PASSED")
        if args.test_type == "sanity":
            print("(sanity: basic verification completed)")
        else:
            print("ROCm has been successfully installed from repository and verified!")
        print("=" * 80 + "\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAIL] Error during installation test: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
