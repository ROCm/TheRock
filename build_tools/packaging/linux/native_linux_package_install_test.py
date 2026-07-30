#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Full installation and simulate-install test script for ROCm native packages.

Test modes (--test-type):
- sanity: Basic test. Repo-based install plus basic verification only
  (steps 1 and 2).
- quick / standard: CI aliases for sanity.
- full: Full test. Repo-based install plus basic verification plus full
  verification (steps 1, 2, and 3).
  Steps (invoked one by one from main):
  1. Repo setup and install: set up package-manager repository and install
     ROCm packages. Arch suffixes in metapackage names are used only when both
     ``--gfx-arch`` and ``--rocm-version`` are set (e.g. ``amdrocm7.13-gfx1100``).
     With ``--rocm-version`` only: ``amdrocm7.13`` / ``amdrocm-core-sdk7.13``.
     With neither or ``--gfx-arch`` alone: ``amdrocm`` / ``amdrocm-core-sdk``.
  2. Basic verification: install prefix, key components, installed packages
     list, rocminfo. (Run for both sanity and full.)
  3. Full verification: rdhc.py / RDHC test. (Run only for full.)
  4. Uninstall (optional, ``--with-uninstall`` / ``RUN_UNINSTALL=1``): remove
     installed metapackages in reverse order, autoremove (deb), and verify no
     ROCm packages remain. Run only after steps 1–2 (sanity) or 1–3 (full)
     succeed. Not run for ``simulate`` or ``install`` test types.
- comprehensive: CI alias for full.
- install: Repo-based install only (step 1). No rocminfo or component checks.
  Used by release workflows that dispatch install tests off the critical path.
- simulate: Dry-run only. Simulated install of local .deb or .rpm files
  (apt install --simulate or rpm -Uvh --test --nodeps). No repo setup or
  actual install. Requires --packages-dir.

Path and repo name are overridable via environment variables: ROCM_REPO_NAME (repo id used for
APT list, Zypper/Yum repo file and section), ROCM_APT_SOURCES_LIST,
ROCM_APT_KEYRING_FILE, ROCM_ZYPP_REPOS_DIR, ROCM_YUM_REPOS_DIR,
ROCM_RDHC_REL_PATH (relative path from install prefix to rdhc binary).

Prerequisites:
- This script does NOT start Docker or a VM. You must run it inside an existing
 container or VM that matches the target OS (e.g., Ubuntu for deb, AlmaLinux/RHEL
 for rpm, SLES container for sles). Start the appropriate Docker image or VM
 first, then invoke this script from inside that environment.
- Root or sudo permissions may be required (repository setup, package install, keyring writes).
- System packages (install with the OS package manager):
  Debian/Ubuntu: apt install -y python3 python3-pip wget curl
  RHEL/Alma/CentOS/AZL: dnf install -y python3 python3-pip wget curl
  SLES: zypper install -y python3 python3-pip wget curl
- Python packages: listed in build_tools/packaging/linux/tests/requirements.txt.
  Install with: pip install -r build_tools/packaging/linux/tests/requirements.txt
  (or from build_tools/packaging/linux/tests: pip install -r requirements.txt).
  Equivalent one-liner: pip install requests prettytable PyYAML

CI typically runs this module under pytest (same file; reporting handled by pytest)::

    pytest build_tools/packaging/linux/native_linux_package_install_test.py -vv --tb=short

Workflow/container ``env`` maps to CLI flags via :func:`_argv_from_ci_env` + ``test_native_linux_package_install``.
For versioned metapackage names only, set ``NATIVE_LINUX_INSTALL_ROCM_VERSION`` and omit ``--rocm-version`` when unversioned packages are desired.
For multiple arches from CI, set ``GFX_ARCH`` to whitespace-separated tokens (e.g. ``gfx94x gfx1100``), semicolon-separated (e.g. ``gfx94x;gfx1100``), or a single comma-separated value (e.g. ``gfx94x,gfx1100``); optional ``NATIVE_LINUX_INSTALL_ROCM_VERSION`` pairs with ``GFX_ARCH`` like ``--rocm-version`` with ``--gfx-arch`` on the CLI.
You can still invoke this file as a script for ad-hoc runs (no pytest required).

Example invocations:

 # Nightly DEB (Ubuntu 24.04) - run inside ubuntu:24.04 container or VM
 python3 native_linux_package_install_test.py \\
         --os-profile ubuntu2404 \\
         --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
         --gfx-arch gfx94x \\
         --release-type nightly

 # Prerelease DEB with GPG verification
 python3 native_linux_package_install_test.py \\
         --os-profile ubuntu2404 \\
         --repo-url https://rc.repo.amd.com/rocm/core/packages/ubuntu2404 \\
         --release-type prerelease \\
         --gpg-key-url https://rc.repo.amd.com/rocm/core/packages/gpg/rocm.gpg

 # Nightly RPM (RHEL 8) - run inside a rhel8/UBI 8 container or VM
 python3 native_linux_package_install_test.py \\
         --os-profile rhel8 \\
         --repo-url https://nightly.repo.amd.com/rocm/core/packages/rpm/20260204-21658678136/x86_64/ \\
         --gfx-arch gfx94x \\
         --release-type nightly

 # Prerelease RPM (SLES 16)
 python3 native_linux_package_install_test.py \\
         --os-profile sles16 \\
         --repo-url https://rc.repo.amd.com/rocm/core/packages/sles16/x86_64/ \\
         --release-type prerelease \\
         --gpg-key-url https://rc.repo.amd.com/rocm/core/packages/gpg/rocm.gpg

 # --test-type sanity (default): repo install + basic verification only (steps 1-2)
 python3 native_linux_package_install_test.py --test-type sanity \\
         --os-profile ubuntu2404 \\
         --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
         --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # --test-type full: same as sanity plus rdhc full verification (steps 1-3)
 python3 native_linux_package_install_test.py --test-type full \\
         --os-profile ubuntu2404 \\
         --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
         --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # --test-type install: repo install only (no verification)
 python3 native_linux_package_install_test.py --test-type install \\
         --os-profile ubuntu2404 \\
         --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
         --gfx-arch gfx94x --release-type nightly

 # Versioned metapackages with multiple GPU architectures (requires --rocm-version for arch in names).
 # Installs e.g. amdrocm7.13-gfx94x, amdrocm-core-sdk7.13-gfx94x, amdrocm7.13-gfx1100, ...
 python3 native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
 --rocm-version 7.13.1 --gfx-arch gfx94x gfx1100 --release-type nightly \\
 --install-prefix /opt/rocm/core

 # Same semantics: comma- or semicolon-separated arches in one --gfx-arch argument (or repeat --gfx-arch).
 python3 native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
 --rocm-version 7.13 --gfx-arch gfx94x,gfx1100 --release-type nightly \\
 --install-prefix /opt/rocm/core
 python3 native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
 --rocm-version 7.13 --gfx-arch 'gfx94x;gfx1100' --release-type nightly \\
 --install-prefix /opt/rocm/core

 # Versioned generic metapackages (no arch suffix) when --rocm-version is set without --gfx-arch.
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://therock-dev-artifacts.s3.amazonaws.com/25137154844-linux/packages/deb \\
 --rocm-version 7.13 --release-type dev --install-prefix /opt/rocm/core

 # Simulate install (dry-run) from local .deb or .rpm directory
 python3 native_linux_package_install_test.py --test-type simulate --packages-dir /path/to/pkgs --os-profile ubuntu2404
 python3 native_linux_package_install_test.py --test-type simulate --packages-dir /path/to/rpms --pkg-type rpm
"""

import argparse
import os
import re
import stat
import subprocess
import sys
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path, PurePosixPath

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from packaging_utils import normalize_target_list


def _env(key: str, default: str) -> str:
    """Return os.environ[key] if set and non-empty, else default."""
    v = os.environ.get(key, "").strip()
    return v if v else default


# --- Config: paths overridable via environment variables ---
# ROCM_REPO_NAME: logical repo name used for APT list, Zypper/Yum repo file and section id
# ROCM_APT_*, ROCM_ZYPP_*, ROCM_YUM_*, ROCM_RDHC_REL_PATH
REPO_NAME = _env("ROCM_REPO_NAME", "rocm-test")
APT_SOURCES_LIST = _env(
    "ROCM_APT_SOURCES_LIST", f"/etc/apt/sources.list.d/{REPO_NAME}.list"
)
# Derived from REPO_NAME like APT_SOURCES_LIST above, and deliberately not
# /etc/apt/keyrings/rocm.gpg. No package owns that path, but the current
# documented amdgpu package manager steps set the signing key up there
# ("wget .../rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg")
# and point signed-by= at it. This harness writes its keyring the same way, so
# sharing the path would overwrite the key a host is already using.
APT_KEYRING_FILE = _env("ROCM_APT_KEYRING_FILE", f"/etc/apt/keyrings/{REPO_NAME}.gpg")
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

# Pytest/CI only: becomes ``--rocm-version``.
ENV_NATIVE_LINUX_INSTALL_ROCM_VERSION = "NATIVE_LINUX_INSTALL_ROCM_VERSION"
# Pytest/CI only: becomes ``--with-uninstall`` when set to 1/true/yes.
ENV_RUN_UNINSTALL = "RUN_UNINSTALL"

# Timeouts (seconds) and verification threshold
GPG_MKDIR_TIMEOUT_SEC = 10
GPG_KEY_TIMEOUT_SEC = 60
APT_UPDATE_TIMEOUT_SEC = 120
ZYPP_CLEAN_TIMEOUT_SEC = 60
ZYPP_REFRESH_TIMEOUT_SEC = 120
DNF_CLEAN_TIMEOUT_SEC = 60
INSTALL_TIMEOUT_SEC = 1800  # 30 minutes
UNINSTALL_TIMEOUT_SEC = 600  # 10 minutes
ROCMINFO_TIMEOUT_SEC = 30
# rdhc.py ``--all`` runs the full ROCm deployment health check suite; 30s was too
# short in container CI (timeouts under load). Optional cluster checks are skipped
# separately via ``--skip-optional-cluster-checks`` in ``test_rdhc``.
RDHC_TIMEOUT_SEC = 600  # 10 minutes
VERIFY_MIN_COMPONENTS = 2
_TEST_TYPE_MAP = {
    "": "sanity",
    "quick": "sanity",
    "standard": "sanity",
    "comprehensive": "full",
    "full": "full",
    "install": "install",
    "sanity": "sanity",
    "simulate": "simulate",
}


def _normalize_test_type(test_type: str | None) -> str:
    """Map shared CI test types to native package install test modes.

    quick/standard/empty -> sanity, comprehensive/full -> full.
    Native modes (install/sanity/full/simulate) are also accepted directly.
    """
    normalized = (test_type or "").strip().lower()
    try:
        return _TEST_TYPE_MAP[normalized]
    except KeyError as e:
        valid = ", ".join(sorted(k or "<empty>" for k in _TEST_TYPE_MAP))
        raise ValueError(
            f"Unsupported test_type {test_type!r}. Expected one of: {valid}."
        ) from e


def run_simulate_install_test(pkg_type: str, packages_dir: str) -> bool:
    """Run simulated package install test (dry-run only, no actual install).

    Equivalent to the GitHub Actions 'Simulated install Test' step:
    - deb: apt install --simulate *.deb
    - rpm: rpm -Uvh --test --nodeps *.rpm

    Returns:
    True if simulate succeeded, False otherwise.
    """
    path = Path(packages_dir).resolve()
    if not path.is_dir():
        print(f"[FAIL] Not a directory: {packages_dir}", file=sys.stderr)
        return False

    if pkg_type == "deb":
        debs = [str(p.resolve()) for p in path.glob("*.deb")]
        if not debs:
            print(f"[FAIL] No .deb files found in {packages_dir}", file=sys.stderr)
            return False
        print("Simulate installing DEB packages on host system for testing")
        # Use absolute paths so apt treats them as local files, not package names
        cmd = ["apt", "install", "--simulate"] + debs
    elif pkg_type == "rpm":
        rpms = [str(p.resolve()) for p in path.glob("*.rpm")]
        if not rpms:
            print(f"[FAIL] No .rpm files found in {packages_dir}", file=sys.stderr)
            return False
        print("Simulate installing RPM packages for testing")
        # Use absolute paths for consistency
        cmd = ["rpm", "-Uvh", "--test", "--nodeps"] + rpms
    else:
        print(
            f"[FAIL] Unsupported pkg_type: {pkg_type}. Use 'deb' or 'rpm'.",
            file=sys.stderr,
        )
        return False

    try:
        subprocess.run(cmd, check=True)
        print("[PASS] Simulated install test completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"[FAIL] Simulated install failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        return False
    except FileNotFoundError as e:
        print(f"[FAIL] Command not found: {e}", file=sys.stderr)
        return False


def _is_rocm_related_package_name(name: str) -> bool:
    """Return True if a package name looks ROCm-related (rocm or amdrocm)."""
    lower = name.lower()
    return "rocm" in lower or "amdrocm" in lower


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


class NativeLinuxPackageInstallTest:
    """Runner for the native Linux package install test (repo setup, install, verification)."""

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

    @staticmethod
    def _major_minor_rocm_version_from_input(rocm_version: str | None) -> str | None:
        """Parse ROCm version for arch-specific package names: major.minor only.

        Examples: ``7.13.1`` → ``7.13``, ``v7.13`` → ``7.13``,
        ``7.14.0~20260520`` / ``7.14.0~rc1-123456`` → ``7.14``. Used when forming
        names like ``amdrocm7.13-gfx1100``. Returns ``None`` if input is absent
        or blank.

        Raises:
        ValueError: Non-empty input that does not start with a major.minor pattern.
        """
        if rocm_version is None:
            return None
        s = str(rocm_version).strip()
        if not s:
            return None
        if s.lower().startswith("v"):
            s = s[1:].lstrip()
        m = re.match(r"^(\d+)\.(\d+)", s)
        if not m:
            raise ValueError(
                "Invalid ROCm version "
                f"{rocm_version!r}: expected major.minor (e.g. 7.13 or 7.13.1)."
            )
        return f"{int(m.group(1))}.{int(m.group(2))}"

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
        install_prefix: str | None = None,
        gfx_arch: str | list[str] | None = None,
        rocm_version: str | None = None,
        gpg_key_url: str | None = None,
        build_variant: str = "",
    ):
        """Initialize the native Linux package install test runner.

        Args:
        repo_url: Full repository URL (constructed in YAML)
        os_profile: OS profile (e.g., ubuntu2404, rhel8, debian12, sles15, sles16, almalinux9, centos7, azl3)
        release_type: Type of release ('nightly' or 'prerelease')
        install_prefix: Installation prefix (default: /opt/rocm/core)
        gfx_arch: Optional GPU architecture(s). Used in package names only
        together with ``rocm_version``; otherwise ignored for install targets.
        rocm_version: Optional ROCm release (e.g. ``7.13`` or ``7.13.1``).
        Major.minor only is used in package names. With ``gfx_arch`` and
        ``rocm_version``: ``amdrocm{version}-{arch}`` per arch. With
        ``rocm_version`` only: ``amdrocm{version}`` / ``amdrocm-core-sdk{version}``.
        If unset: unversioned ``amdrocm`` / ``amdrocm-core-sdk`` (``gfx_arch`` alone
        does not add arch suffixes).
        gpg_key_url: GPG key URL
        build_variant: Build variant (e.g. 'asan'). When set, '-{variant}' is
        inserted before the version suffix in package names so that variant
        packages are tested (e.g. ``amdrocm-asan7.15-gfx942`` instead of
        ``amdrocm7.15-gfx942``).
        """
        self.os_profile = os_profile.lower()
        self.package_type = self._derive_package_type(os_profile)
        self.repo_url = repo_url.rstrip("/")
        self.release_type = release_type.lower()
        self.install_prefix = install_prefix
        self.gfx_arch_list = normalize_target_list(
            gfx_arch, lowercase=True, dedupe=True
        )
        self.rocm_version_major_minor = self._major_minor_rocm_version_from_input(
            rocm_version
        )
        # Primary arch (compat / display): first listed after normalization, else unset
        self.gfx_arch: str | None = (
            self.gfx_arch_list[0] if self.gfx_arch_list else None
        )
        self.gpg_key_url = gpg_key_url
        self.build_variant = build_variant.strip().lower()

        # Build variants that produce distinct package names with a '-{variant}'
        # suffix inserted before the version. 'release' is the default build and
        # does NOT alter the package name (amdrocm7.15, not amdrocm-release7.15).
        _NAMED_VARIANTS = {"asan"}

        # Metapackage install targets (four combinations of optional inputs).
        # When build_variant is a named variant (e.g. 'asan'), '-{variant}' is
        # inserted before the version suffix:
        #   gfx_arch + rocm_version -> amdrocm-asan{major.minor}-{arch} per arch
        #   gfx_arch only           -> amdrocm-asan / amdrocm-core-sdk-asan
        #   rocm_version only       -> amdrocm-asan{major.minor} / amdrocm-core-sdk-asan{major.minor}
        #   neither                 -> amdrocm-asan / amdrocm-core-sdk-asan
        ver = self.rocm_version_major_minor
        variant_sep = (
            f"-{self.build_variant}" if self.build_variant in _NAMED_VARIANTS else ""
        )
        if self.gfx_arch_list and ver:
            self.package_names = []
            for arch in self.gfx_arch_list:
                self.package_names.extend(
                    [
                        f"amdrocm{variant_sep}{ver}-{arch}",
                        f"amdrocm-core-sdk{variant_sep}{ver}-{arch}",
                    ]
                )
        elif ver:
            self.package_names = [
                f"amdrocm{variant_sep}{ver}",
                f"amdrocm-core-sdk{variant_sep}{ver}",
            ]
        else:
            self.package_names = [
                f"amdrocm{variant_sep}",
                f"amdrocm-core-sdk{variant_sep}",
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
            # Write to the same path the sources entry pins with signed-by=.
            # These were separate expressions that happened to agree, so any
            # change to one silently broke apt's ability to find the keyring.
            # PurePosixPath, not Path: these are paths on the target Linux
            # filesystem handed to mkdir(1), tee(1) and chmod(1). Path follows
            # the local flavour, so on Windows it renders "\etc\apt\keyrings".
            keyring_file = PurePosixPath(APT_KEYRING_FILE)
            keyring_dir = keyring_file.parent

            try:
                # Create keyring directory
                print(f"\nCreating keyring directory: {keyring_dir}...")
                subprocess.run(
                    ["sudo", "mkdir", "--parents", "--mode=0755", str(keyring_dir)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=GPG_MKDIR_TIMEOUT_SEC,
                )
                print(f"[PASS] Created keyring directory: {keyring_dir}")

                # Download, dearmor and write the key as three list-form calls
                # rather than one shell pipeline: the URL and the keyring path
                # are both configurable, and interpolating them into a shell
                # string makes them command injection vectors.
                print(f"\nDownloading and importing GPG key from {self.gpg_key_url}...")
                armored = subprocess.run(
                    ["wget", "-q", "-O", "-", self.gpg_key_url],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=GPG_KEY_TIMEOUT_SEC,
                ).stdout
                dearmored = subprocess.run(
                    ["gpg", "--dearmor"],
                    input=armored,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=GPG_KEY_TIMEOUT_SEC,
                ).stdout
                subprocess.run(
                    ["sudo", "tee", str(keyring_file)],
                    input=dearmored,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=GPG_KEY_TIMEOUT_SEC,
                )

                # Set proper permissions on the keyring file
                subprocess.run(
                    ["sudo", "chmod", "0644", str(keyring_file)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print(f"[PASS] GPG key imported to {keyring_file}")
                return True

            except subprocess.CalledProcessError as e:
                print(f"[FAIL] Failed to setup GPG key: {e}")
                if e.stderr:
                    print(f"Error output: {e.stderr.decode()}")
                return False
            except subprocess.TimeoutExpired as e:
                # TimeoutExpired is a SubprocessError, not an OSError or a
                # CalledProcessError, so neither handler around it catches one.
                # Without this the whole run dies on a slow mirror instead of
                # reporting a failed key import like every other failure here.
                print(f"[FAIL] Timed out setting up GPG key: {e}")
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
        sources_list = Path(APT_SOURCES_LIST)

        if self.gpg_key_url:
            # Use GPG key verification (arch=amd64 matches ROCm Ubuntu install docs)
            # PurePosixPath for the same reason: this lands in signed-by= inside
            # a sources file on the target, not on the machine running the test.
            apt_keyring = PurePosixPath(APT_KEYRING_FILE)
            repo_entry = f"deb [arch=amd64 signed-by={apt_keyring}] {self.repo_url} stable main\n"
        else:
            # No GPG check (trusted=yes; arch=amd64 matches install_rocm_packages.sh)
            repo_entry = f"deb [arch=amd64 trusted=yes] {self.repo_url} stable main\n"

        try:
            subprocess.run(
                ["sudo", "tee", str(sources_list)],
                input=repo_entry.encode(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            print(f"[PASS] Repository added to {sources_list}")
            print(f" {repo_entry.strip()}")
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"[FAIL] Failed to add repository: {e}")
            return False

        # Update package lists
        print("\nUpdating package lists...")
        print("=" * 80)
        try:
            return_code = _run_streaming(
                ["sudo", "apt", "update"], APT_UPDATE_TIMEOUT_SEC
            )
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
            repo_file.write_text(repo_content, encoding="utf-8")
            print(f"[PASS] Repository file created: {repo_file}")
            print("\nRepository configuration:")
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
            print("[WARN] zypper clean timed out (may not be critical)")
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
            repo_content = f"""[{repo_name}]
name=ROCm Repository
baseurl={self.repo_url}
enabled=1
gpgcheck=1
gpgkey={self.gpg_key_url}
"""
        else:
            # No GPG check
            repo_content = f"""[{repo_name}]
name=Native Linux Package Test Repository
baseurl={self.repo_url}
enabled=1
gpgcheck=0
"""

        try:
            repo_file.write_text(repo_content, encoding="utf-8")
            print(f"[PASS] Repository file created: {repo_file}")
            print("\nRepository configuration:")
            print(repo_content)
        except OSError as e:
            print(f"[FAIL] Failed to create repository file: {e}")
            return False

        # Clean dnf cache
        print("\nCleaning dnf cache...")
        try:
            subprocess.run(
                ["dnf", "clean", "all"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=DNF_CLEAN_TIMEOUT_SEC,
            )
            print("[PASS] dnf cache cleaned")
        except subprocess.CalledProcessError as e:
            print("[WARN] Failed to clean dnf cache (may not be critical)")
            print(f"Error: {e.stdout}")
        except subprocess.TimeoutExpired:
            print("[WARN] dnf clean timed out (may not be critical)")

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
        cmd = ["sudo", "apt", "install", "-y"] + self.package_names
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
        """Step 1: Repo setup and install. Run for both sanity (basic) and full test.

        Returns:
        True if repository setup and package installation both succeeded.
        """
        print("\n" + "=" * 80)
        print("STEP 1: REPOSITORY SETUP AND PACKAGE INSTALLATION")
        print("=" * 80)
        print(f"\nOS Profile: {self.os_profile}")
        print(f"Package Type (derived): {self.package_type.upper()}")
        # Log how GPU arch relates to package_names (same four cases as __init__).
        if self.gfx_arch_list and self.rocm_version_major_minor:
            print(f"GPU Architecture(s): {self.gfx_arch_list} (used in package names)")
        elif self.gfx_arch_list:
            print(
                f"GPU Architecture(s): {self.gfx_arch_list} "
                "(not used in package names without --rocm-version / "
                f"{ENV_NATIVE_LINUX_INSTALL_ROCM_VERSION})"
            )
        else:
            ver = self.rocm_version_major_minor
            if ver:
                print(
                    "GPU Architecture(s): (none — generic versioned packages "
                    f"amdrocm{ver}, amdrocm-core-sdk{ver})"
                )
            else:
                print(
                    "GPU Architecture(s): (none — generic packages amdrocm, amdrocm-core-sdk)"
                )
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
        """Step 2: Basic test — install prefix, key components, packages list, rocminfo.

        Used by both --test-type sanity and full. Does not run test_rdhc
        (that is Step 3 / run_full_verification, full test only).

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
                print(f" [PASS] {component}")
                found_count += 1
            else:
                print(f" [WARN] {component} (not found)")

        print(f"\nComponents found: {found_count}/{len(key_components)}")

        # Verify installed files are owned by root:root and have safe
        # permissions (no group/other-writable paths or setuid/setgid bits),
        # which guards against local PATH-hijack privilege escalation.
        security_ok = self.verify_installed_file_security()

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
            print(f" Found {len(rocm_packages)} ROCm packages installed")
            if rocm_packages:
                print("\n Sample packages (Show first 5):")
                for pkg in rocm_packages[:5]:
                    print(f" {pkg.strip()}")
                if len(rocm_packages) > 5:
                    print(f" ... and {len(rocm_packages) - 5} more")
        except subprocess.CalledProcessError:
            print(" [WARN] Could not query installed packages")

        # Verify installed ELF files use RPATH and not RUNPATH.
        if not self.verify_no_runpath():
            print("\n[FAIL] Basic verification FAILED (ELF files still using RUNPATH)")
            return False

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
                print(" [PASS] rocminfo executed successfully")
                lines = result.stdout.split("\n")[:10]
                print("\n First few lines of rocminfo output:")
                for line in lines:
                    if line.strip():
                        print(f" {line}")
            except subprocess.TimeoutExpired:
                print(" [WARN] rocminfo timed out (may require GPU hardware)")
            except subprocess.CalledProcessError:
                print(" [WARN] rocminfo failed (may require GPU hardware)")
            except OSError as e:
                print(f" [WARN] Could not run rocminfo: {e}")

        if found_count < VERIFY_MIN_COMPONENTS:
            print("\n[FAIL] Basic verification FAILED (insufficient components)")
            return False
        if not security_ok:
            print(
                "\n[FAIL] Basic verification FAILED "
                "(files not owned by root or with insecure permissions)"
            )
            return False
        print("\n[PASS] Basic verification PASSED")
        return True

    @staticmethod
    def _format_flagged_reason(st: os.stat_result) -> str:
        """Format an owner/group/mode + 'why flagged' annotation from a stat result.

        Renders which rule(s) a path violated (non-root owner, group/other
        writable, setuid/setgid on a regular file) so the CI log shows *why*
        each path was flagged rather than just its name. Mirrors the flagging
        logic: symlink mode bits are ignored (only ownership is meaningful for
        links) and the setuid/setgid rule applies to regular files only, since
        setgid on a directory is a normal, benign group-inheritance pattern.

        Returns the bare annotation body (no surrounding parentheses) so callers
        can wrap it as needed.
        """
        mode = st.st_mode
        reasons = []
        if st.st_uid != 0 or st.st_gid != 0:
            reasons.append("non-root-owner")
        if not stat.S_ISLNK(mode):
            if mode & 0o002:
                reasons.append("other-writable")
            if mode & 0o020:
                reasons.append("group-writable")
            if stat.S_ISREG(mode) and mode & (stat.S_ISUID | stat.S_ISGID):
                reasons.append("setuid/setgid")
        why = ",".join(reasons) if reasons else "?"
        return f"uid={st.st_uid} gid={st.st_gid} mode={stat.S_IMODE(mode):04o} -> {why}"

    @classmethod
    def _describe_flagged_path(cls, path: str) -> str:
        """Best-effort ``(owner/group/mode -> why)`` annotation for a ``find`` hit.

        ``find`` prints only names, so re-``lstat`` the path to explain why it
        was flagged. The prefix itself is often a symlink that ``find -H``
        follows and flags via its *target*; ``lstat`` would only see the link's
        meaningless ``0777`` bits, so for a symlink we additionally ``stat`` the
        target and report that. Never raises: on any error it returns an
        annotation noting the failure so reporting is unaffected.
        """
        try:
            st = os.lstat(path)
        except OSError as e:
            return f"(stat failed: {e.strerror or e})"
        if stat.S_ISLNK(st.st_mode):
            try:
                target = os.stat(path)
            except OSError as e:
                return f"(symlink; target stat failed: {e.strerror or e})"
            return f"(symlink target: {cls._format_flagged_reason(target)})"
        return f"({cls._format_flagged_reason(st)})"

    def verify_installed_file_security(self) -> bool:
        """Verify installed files are owned by root:root with safe permissions.

        Combines two related install-tree security checks into a single
        traversal. A path under the prefix is flagged when any of the following
        holds:

        - Ownership: its owner uid or group gid is not 0 (not ``root:root``);
          a non-root-owned path can be tampered with by that owner.
        - Writability: it is writable by group or other (mode bits ``0o022``).
          This would let an unprivileged user drop or replace a binary in a
          directory on ``PATH`` and have another user (or root) execute it.
          ROCm ships no group/other-writable paths (including sticky
          directories), so any such path is flagged.
        - setuid/setgid on a **regular file**: it carries mode bits ``0o6000``,
          a direct privilege-escalation surface. This rule applies to regular
          files only: setgid on a *directory* is a normal, benign
          group-inheritance pattern (``drwxr-sr-x``) and is not flagged.

        Symbolic links are exempt from the permission rules: on Linux a
        symlink's own mode bits are always ``lrwxrwxrwx`` and are ignored by
        the kernel (the target's permissions govern access), so checking them
        would produce false positives. The install prefix itself is commonly a
        symlink (e.g. ``/opt/rocm/core`` -> ``/opt/rocm/core-X.Y``), so
        ``find -H`` follows that top-level link to scan the real tree while not
        following links found *inside* the tree. ``-xdev`` keeps the scan on the
        prefix's own filesystem so bind mounts inside the tree are not crossed.

        Uses ``find`` (C-level traversal) for speed on large install trees and
        falls back to a pure-Python ``os.walk`` scan if ``find`` is unavailable
        so the check is not silently skipped.

        Returns:
        True if no offending path is found (or the check could not be run),
        False if any offending path is found.
        """
        print("\nVerifying installed files are owned by root with safe permissions...")
        # PurePosixPath, not Path: this is a path on the target Linux filesystem
        # handed to find(1). Path follows the local flavour, so on Windows it
        # renders "\opt\rocm\core" and the scan silently targets the wrong path.
        install_prefix = str(PurePosixPath(self.install_prefix))
        try:
            result = subprocess.run(
                [
                    "find",
                    # follow the prefix if it is a symlink, but not links inside
                    "-H",
                    install_prefix,
                    # do not cross into other filesystems mounted under prefix
                    "-xdev",
                    "(",
                    # not owned by root:root
                    "(",
                    "!",
                    "-uid",
                    "0",
                    "-o",
                    "!",
                    "-gid",
                    "0",
                    ")",
                    "-o",
                    # insecure permissions
                    "(",
                    # group/other-writable on any non-symlink (a symlink's own
                    # mode bits are meaningless; the target is checked on its own)
                    "(",
                    "!",
                    "-type",
                    "l",
                    "-perm",
                    "/022",
                    ")",
                    "-o",
                    # setuid/setgid on a regular file only; setgid on a
                    # directory (drwxr-sr-x) is a benign group-inheritance
                    # pattern and must not be flagged.
                    "(",
                    "-type",
                    "f",
                    "-perm",
                    "/6000",
                    ")",
                    ")",
                    ")",
                    "-print",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as e:
            # find not available on this host; fall back to a Python walk so the
            # check still runs rather than being silently skipped.
            print(f" [WARN] 'find' unavailable ({e}); falling back to Python scan")
            return self._verify_installed_file_security_python(Path(install_prefix))

        # find can exit non-zero (e.g. permission denied on a subtree) while
        # still printing partial results; surface that rather than passing
        # silently on an incomplete scan.
        if result.returncode != 0:
            print(f" [WARN] find exited {result.returncode}: {result.stderr[:200]}")
        elif result.stderr.strip():
            print(f" [WARN] find reported: {result.stderr[:200]}")

        bad = [line for line in result.stdout.splitlines() if line]
        if bad:
            print(
                f" [FAIL] {len(bad)} path(s) not owned by root "
                "or with insecure permissions:"
            )
            for line in bad[:10]:
                print(f"   {line} {self._describe_flagged_path(line)}")
            if len(bad) > 10:
                print(f"   ... and {len(bad) - 10} more")
            return False
        print(" [PASS] All installed files owned by root with safe permissions")
        return True

    def _verify_installed_file_security_python(self, install_path: Path) -> bool:
        """Pure-Python fallback for the install-tree security check.

        Walks the install tree with ``os.walk`` (does not follow symlinked
        directories found inside the tree) and ``os.lstat`` each entry, applying
        the same rules as :meth:`verify_installed_file_security`: flag entries
        not owned by ``root:root``, group/other-writable entries, and
        setuid/setgid *regular files*. Symlinks are exempt from the permission
        rules since their mode bits are meaningless on Linux, and setgid
        directories are not flagged (a benign group-inheritance pattern).
        Slower than ``find`` on large trees but portable.

        Returns:
        True if no offending path is found, False if any offending path is found.
        """
        # Store (path, stat) so printing can annotate why each path was flagged
        # without re-stat'ing (the stat result here is authoritative).
        bad: list[tuple[Path, os.stat_result]] = []
        for root, dirs, files in os.walk(install_path):
            for name in dirs + files:
                entry = Path(root) / name
                try:
                    st = os.lstat(entry)
                except OSError:
                    continue
                mode = st.st_mode
                not_root = st.st_uid != 0 or st.st_gid != 0
                if stat.S_ISLNK(mode):
                    # A symlink's own mode bits are always 0o777 and ignored by
                    # the kernel; only ownership is meaningful for links.
                    if not_root:
                        bad.append((entry, st))
                    continue
                group_other_writable = bool(mode & 0o022)
                # setgid on a directory is benign; only flag setuid/setgid on
                # regular files (a real privilege-escalation surface).
                setid_file = stat.S_ISREG(mode) and bool(
                    mode & (stat.S_ISUID | stat.S_ISGID)
                )
                if not_root or group_other_writable or setid_file:
                    bad.append((entry, st))
        if bad:
            print(
                f" [FAIL] {len(bad)} path(s) not owned by root "
                "or with insecure permissions:"
            )
            for entry, st in bad[:10]:
                print(f"   {entry} ({self._format_flagged_reason(st)})")
            if len(bad) > 10:
                print(f"   ... and {len(bad) - 10} more")
            return False
        print(" [PASS] All installed files owned by root with safe permissions")
        return True

    def run_full_verification(self) -> bool:
        """Step 3: Full test — runs test_rdhc (rdhc.py) only. Used when --test-type is full."""
        print("\n" + "=" * 80)
        print("STEP 3: FULL VERIFICATION (RDHC)")
        print("=" * 80)
        return self.test_rdhc()

    def list_installed_rocm_packages(self) -> list[str]:
        """Query the system package manager for installed rocm/amdrocm packages.

        Returns:
            Sorted list of package names matching rocm/amdrocm. Empty on query failure.
        """
        try:
            if self.package_type == "deb":
                result = subprocess.run(
                    ["dpkg", "-l"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                names: list[str] = []
                for line in result.stdout.splitlines():
                    if not line.startswith("ii"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2 and _is_rocm_related_package_name(parts[1]):
                        names.append(parts[1])
                return sorted(names)

            result = subprocess.run(
                ["rpm", "-qa"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return sorted(
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip() and _is_rocm_related_package_name(line.strip())
            )
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Could not query installed packages: {e}")
            return []
        except OSError as e:
            print(f"[WARN] Could not query installed packages: {e}")
            return []

    def uninstall_packages(self) -> bool:
        """Remove installed metapackages in reverse install order.

        deb: ``sudo apt remove -y`` then ``sudo apt autoremove -y``.
        rpm: ``dnf remove -y`` or ``zypper remove -y`` on SLES.

        Returns:
            True if uninstall commands succeeded, False otherwise.
        """
        print("\n" + "=" * 80)
        print("STEP 4a: UNINSTALL PACKAGES")
        print("=" * 80)

        packages_to_remove = list(reversed(self.package_names))
        if not packages_to_remove:
            print("[WARN] No package names configured for uninstall")
            return True

        print(f"\nPackages to remove (reverse install order): {packages_to_remove}")

        if self.package_type == "deb":
            remove_cmd = ["sudo", "apt", "remove", "-y"] + packages_to_remove
            autoremove_cmd = ["sudo", "apt", "autoremove", "-y"]
        elif self._is_sles():
            remove_cmd = [
                "zypper",
                "--non-interactive",
                "remove",
                "-y",
            ] + packages_to_remove
            autoremove_cmd = None
        else:
            remove_cmd = ["dnf", "remove", "-y"] + packages_to_remove
            autoremove_cmd = None

        print(f"\nRunning: {' '.join(remove_cmd)}")
        print("=" * 80)
        print("Uninstall progress (streaming output):\n")

        try:
            return_code = _run_streaming(remove_cmd, UNINSTALL_TIMEOUT_SEC)
            if return_code != 0:
                print("\n" + "=" * 80)
                print(f"[FAIL] Failed to remove packages (exit code: {return_code})")
                return False

            if autoremove_cmd:
                print(f"\nRunning: {' '.join(autoremove_cmd)}")
                print("=" * 80)
                print("Autoremove progress (streaming output):\n")
                return_code = _run_streaming(autoremove_cmd, UNINSTALL_TIMEOUT_SEC)
                if return_code != 0:
                    print("\n" + "=" * 80)
                    print(f"[FAIL] apt autoremove failed (exit code: {return_code})")
                    return False

            print("\n" + "=" * 80)
            print("[PASS] Package uninstall completed successfully")
            return True
        except subprocess.TimeoutExpired:
            print("\n" + "=" * 80)
            print(
                f"[FAIL] Uninstall timed out after {UNINSTALL_TIMEOUT_SEC // 60} minutes"
            )
            return False
        except OSError as e:
            print(f"\n[FAIL] Error during uninstall: {e}")
            return False

    def run_uninstall_verification(self) -> bool:
        """Step 4b: Verify no ROCm packages remain after uninstall.

        Optionally logs install-prefix cleanup status (non-fatal).

        Returns:
            True if zero ROCm packages remain installed.
        """
        print("\n" + "=" * 80)
        print("STEP 4b: UNINSTALL VERIFICATION")
        print("=" * 80)

        remaining = self.list_installed_rocm_packages()
        if remaining:
            print(f"\n[FAIL] {len(remaining)} ROCm package(s) still installed:")
            for pkg in remaining[:10]:
                print(f"  {pkg}")
            if len(remaining) > 10:
                print(f"  ... and {len(remaining) - 10} more")
            return False

        print("\n[PASS] No ROCm packages remain installed")

        install_path = Path(self.install_prefix)
        if not install_path.exists():
            print(f"[PASS] Install prefix removed: {self.install_prefix}")
        else:
            leftover = [
                component
                for component in VERIFY_KEY_COMPONENTS
                if (install_path / component).exists()
            ]
            if leftover:
                print(
                    f"[WARN] Install prefix still contains key components: {leftover}"
                )
            else:
                print(
                    f"[INFO] Install prefix exists but key ROCm components are gone: "
                    f"{self.install_prefix}"
                )

        print("\n[PASS] Uninstall verification PASSED")
        return True

    def run_uninstall_and_verify(self) -> bool:
        """Step 4: Uninstall metapackages and verify clean teardown.

        Returns:
            True if uninstall and verification both succeeded.
        """
        print("\n" + "=" * 80)
        print("STEP 4: UNINSTALL AND VERIFY")
        print("=" * 80)

        before = self.list_installed_rocm_packages()
        print(f"\nROCm packages before uninstall: {len(before)}")
        if before:
            print(" Sample packages (first 5):")
            for pkg in before[:5]:
                print(f"  {pkg}")

        if not self.uninstall_packages():
            return False
        return self.run_uninstall_verification()

    def test_rdhc(self) -> bool:
        """Test rdhc.py binary in libexec/rocm-core/.

        Returns:
        True if test successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("TESTING RDHC.PY")
        print("=" * 80)

        install_path = Path(self.install_prefix).resolve()
        rdhc_script = (install_path / RDHC_REL_PATH).resolve()
        rocm_install_prefix_arg = str(install_path)

        # Check if script exists
        if not rdhc_script.exists():
            print(f"\n[WARN] rdhc.py not found at: {rdhc_script}")
            print(" This is expected if rocm-core package is not installed")
            return False

        print(f"\n[PASS] rdhc.py found at: {rdhc_script}")

        # Always run rdhc with this process's interpreter.
        cmd = [sys.executable, str(rdhc_script)]

        # Set RDHC arguments for full test
        test_args = [
            "--rocm-install-prefix",
            rocm_install_prefix_arg,
            "--all",
            "--skip-optional-cluster-checks",
        ]
        print(
            f"\nRun rdhc.py with --rocm-install-prefix {rocm_install_prefix_arg} --all..."
        )
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
            print(" [PASS] rdhc.py executed successfully")
            if result.stdout:
                # Print first few lines of output
                lines = result.stdout.split("\n")[:5]
                print("\n First few lines of output:")
                for line in lines:
                    if line.strip():
                        print(f" {line}")
            return True
        except subprocess.TimeoutExpired:
            print(" [WARN] rdhc.py --all timed out")
            return False
        except subprocess.CalledProcessError:
            print(" [WARN] rdhc.py --all failed")
            return False
        except OSError as e:
            print(f" [WARN] Could not run rdhc.py: {e}")
            return False

    def verify_no_runpath(self) -> bool:
        """Verify installed ELF files use DT_RPATH (not DT_RUNPATH).

        During packaging, RUNPATH is converted to RPATH (see runpath_to_rpath.py).
        Each installed ELF's dynamic section is read with ``readelf -d`` and
        classified by looking for DT_RPATH first, falling back to DT_RUNPATH:

        * has DT_RPATH -> converted correctly (expected)
        * no DT_RPATH but has DT_RUNPATH -> conversion was missed (failure)
        * neither tag -> nothing to convert; counted and reported for
          visibility only, not treated as a failure

        For files that do have DT_RPATH, the rpath value is additionally
        inspected for entries that are not ``$ORIGIN``-relative (i.e. fixed or
        absolute paths). A relocatable package should only use
        ``$ORIGIN``-relative rpaths, so fixed paths are reported for visibility
        but are not treated as a failure here.

        If ``readelf`` is unavailable the check is skipped (non-fatal), matching
        the tolerant behaviour used elsewhere in basic verification.

        Returns:
        True if no installed ELF uses DT_RUNPATH (or the check could not run),
        False if any offending file is found.
        """
        print("\nVerifying installed ELF files use RPATH (not RUNPATH)...")
        install_path = Path(self.install_prefix)
        runpath_only: list[str] = []
        no_path: list[str] = []
        fixed_rpath: list[tuple[str, list[str]]] = []
        rpath_count = 0
        for root, _dirs, files in os.walk(install_path):
            for name in files:
                filepath = Path(root) / name
                if filepath.is_symlink():
                    continue
                try:
                    # Cheap ELF magic check before invoking readelf.
                    with open(filepath, "rb") as f:
                        if f.read(4) != b"\x7fELF":
                            continue
                except OSError:
                    continue
                try:
                    result = subprocess.run(
                        ["readelf", "-d", str(filepath)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                except OSError as e:
                    print(
                        f" [WARN] 'readelf' unavailable ({e}); skipping RUNPATH check"
                    )
                    return True
                if result.returncode != 0:
                    print(
                        f" [WARN] readelf failed for {filepath}: {result.stderr.strip()}"
                    )
                    continue
                # Confirm DT_RPATH is present; only fall back to DT_RUNPATH if
                # it is not. Note "(RPATH)" is not a substring of "(RUNPATH)".
                if "(RPATH)" in result.stdout:
                    rpath_count += 1
                    fixed = self._fixed_rpath_entries(result.stdout)
                    if fixed:
                        fixed_rpath.append((str(filepath), fixed))
                elif "(RUNPATH)" in result.stdout:
                    runpath_only.append(str(filepath))
                else:
                    no_path.append(str(filepath))
        print(
            f" Scanned ELF files: {rpath_count} with DT_RPATH, "
            f"{len(runpath_only)} with DT_RUNPATH only, "
            f"{len(no_path)} with neither"
        )
        if no_path:
            # Count only; the individual files are not interesting to list.
            print(
                f" [INFO] {len(no_path)} ELF file(s) have neither DT_RPATH nor DT_RUNPATH"
            )
        if fixed_rpath:
            # List every offending file and its non-$ORIGIN entries in full.
            print(
                f" [WARN] {len(fixed_rpath)} ELF file(s) have DT_RPATH entries that are "
                "not $ORIGIN-relative (fixed/absolute paths):"
            )
            for path, entries in fixed_rpath:
                print(f"   {path}: {':'.join(entries)}")
        if runpath_only:
            print(f" [FAIL] {len(runpath_only)} ELF file(s) still using DT_RUNPATH:")
            for entry in runpath_only[:10]:
                print(f"   {entry}")
            if len(runpath_only) > 10:
                print(f"   ... and {len(runpath_only) - 10} more")
            return False
        print(" [PASS] No installed ELF files use DT_RUNPATH")
        return True

    @staticmethod
    def _fixed_rpath_entries(readelf_output: str) -> list[str]:
        """Return DT_RPATH entries that are not ``$ORIGIN``-relative.

        Parses the ``Library rpath: [...]`` value from ``readelf -d`` output and
        returns each colon-separated entry that does not reference ``$ORIGIN``
        (i.e. fixed or absolute paths). Returns an empty list if there is no
        rpath value or all entries are ``$ORIGIN``-relative.
        """
        match = re.search(r"\(RPATH\)\s+Library rpath: \[([^\]]*)\]", readelf_output)
        if not match:
            return []
        entries = [entry for entry in match.group(1).split(":") if entry]
        return [entry for entry in entries if "$ORIGIN" not in entry]


_CLI_EXAMPLES_EPILOG = """
Examples:
 # Nightly DEB (Ubuntu 24.04) - run inside matching container/VM
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # Prerelease DEB with GPG verification
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://rc.repo.amd.com/rocm/core/packages/ubuntu2404 \\
 --gfx-arch gfx94x --release-type prerelease --install-prefix /opt/rocm/core \\
 --gpg-key-url https://rc.repo.amd.com/rocm/core/packages/gpg/rocm.gpg

 # Nightly RPM (RHEL 8)
 python native_linux_package_install_test.py --os-profile rhel8 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/rpm/20260204-21658678136/rhel8/x86_64/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # --test-type full on RHEL 8 (rdhc needs pciutils/kmod on the host — install before running)
 python native_linux_package_install_test.py --test-type full --os-profile rhel8 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/rpm/20260204-21658678136/rhel8/x86_64/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # Prerelease RPM (RHEL 8)
 python native_linux_package_install_test.py --os-profile rhel8 \\
 --repo-url https://rc.repo.amd.com/rocm/core/packages/rhel8/x86_64/ \\
 --gfx-arch gfx94x --release-type prerelease --install-prefix /opt/rocm/core \\
 --gpg-key-url https://rc.repo.amd.com/rocm/core/packages/gpg/rocm.gpg

 # --test-type sanity (default): repo install + basic verification only
 python native_linux_package_install_test.py --test-type sanity --os-profile ubuntu2404 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # --test-type full: install + basic verification + rdhc
 python native_linux_package_install_test.py --test-type full --os-profile ubuntu2404 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # --test-type install: install only
 python native_linux_package_install_test.py --test-type install --os-profile ubuntu2404 \\
 --repo-url https://therock-dev-artifacts.s3.amazonaws.com/26299074718-linux/packages/deb \\
 --gfx-arch gfx94x --release-type dev --install-prefix /opt/rocm/core

 # Versioned + multiple --gfx-arch (metapackages amdrocm7.13-<arch> per arch)
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
 --rocm-version 7.12.1 --gfx-arch gfx94x gfx1100 --release-type nightly \\
 --install-prefix /opt/rocm/core

 # Comma-separated arches in one argument (equivalent normalization)
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
 --rocm-version 7.12 --gfx-arch gfx94x,gfx1100 --release-type nightly \\
 --install-prefix /opt/rocm/core

 # Semicolon-separated (quote for POSIX shells so ``;`` is not a command separator)
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://nightly.repo.amd.com/rocm/core/packages/deb/20260204-21658678136/ \\
 --rocm-version 7.12 --gfx-arch 'gfx94x;gfx1100' --release-type nightly \\
 --install-prefix /opt/rocm/core

 # --rocm-version without --gfx-arch: amdrocm7.13 / amdrocm-core-sdk7.13 only
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://therock-dev-artifacts.s3.amazonaws.com/25137154844-linux/packages/deb \\
 --rocm-version 7.13 --release-type dev --install-prefix /opt/rocm/core

 # without --rocm-version without --gfx-arch: amdrocm / amdrocm-core-sdk only
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://therock-dev-artifacts.s3.amazonaws.com/25137154844-linux/packages/deb \\
 --release-type dev --install-prefix /opt/rocm/core

 # Simulate install (dry-run) from local packages
 python native_linux_package_install_test.py --test-type simulate --packages-dir /path/to/pkgs --os-profile ubuntu2404
 python native_linux_package_install_test.py --test-type simulate --packages-dir /path/to/rpms --pkg-type rpm
"""


def _build_argument_parser(*, exit_on_error: bool = True) -> ArgumentParser:
    kwargs: dict = dict(
        description="Full installation and simulate-install test for ROCm native packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EXAMPLES_EPILOG,
    )
    if sys.version_info >= (3, 9):
        kwargs["exit_on_error"] = exit_on_error
    parser = ArgumentParser(**kwargs)
    parser.add_argument(
        "--os-profile",
        type=str,
        help="OS profile (e.g., ubuntu2404, rhel8, debian12, sles15, sles16, almalinux9, centos7, azl3). Required for sanity/full; for simulate, used only to derive pkg-type if --pkg-type is omitted.",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        help="Full repository URL (constructed in YAML workflow). Required for sanity/full; not used for simulate.",
    )
    parser.add_argument(
        "--gfx-arch",
        type=str,
        nargs="+",
        default=None,
        metavar="ARCH",
        help="GPU architecture(s), optional. Used in package names only with --rocm-version "
        "(e.g. amdrocm7.13-gfx94x). Without --rocm-version, arch is ignored for install targets "
        "(generic amdrocm / amdrocm-core-sdk). "
        "Repeat flag or list; commas and semicolons split within each value. Not used for simulate.",
    )
    parser.add_argument(
        "--rocm-version",
        type=str,
        default=None,
        metavar="VER",
        help=(
            "ROCm release (major.minor only in package names). With --gfx-arch: amdrocm7.13-ARCH per arch; "
            "without --gfx-arch: amdrocm7.13 and amdrocm-core-sdk7.13. Required for arch in package names. "
            "Optional. Not used for simulate."
        ),
    )
    parser.add_argument(
        "--release-type",
        type=str,
        choices=[
            "dev",
            "dev-bkc",
            "nightly",
            "nightly-bkc",
            "prerelease",
            "release",
            "ci",
        ],
        help="Type of release: 'dev', 'nightly', 'prerelease', 'release', or 'ci'",
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
        "--build-variant",
        type=str,
        default="",
        help="Build variant (e.g. 'asan'). Changes expected package names to match variant-suffixed packages.",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default="sanity",
        help="Test type: 'install' = repo install only; 'sanity' = install + basic verification; 'full' = sanity + rdhc; 'simulate' = dry-run local packages (requires --packages-dir). Also accepts CI test types: quick, standard, comprehensive.",
    )
    parser.add_argument(
        "--with-uninstall",
        action="store_true",
        help="After install verification succeeds (sanity or full), uninstall packages and verify clean teardown. Ignored for simulate and install test types.",
    )
    parser.add_argument(
        "--packages-dir",
        type=str,
        metavar="DIR",
        help="Directory containing .deb or .rpm files. Required when --test-type is 'simulate'.",
    )
    parser.add_argument(
        "--pkg-type",
        type=str,
        choices=["deb", "rpm"],
        help="Package type (deb or rpm). For --test-type simulate only; if omitted, derived from --os-profile.",
    )
    return parser


def _validate_cli_args(parser: ArgumentParser, args: Namespace) -> None:
    if args.test_type == "simulate":
        if not args.packages_dir:
            parser.error("--packages-dir is required when --test-type is 'simulate'")
        if not args.pkg_type and not args.os_profile:
            parser.error(
                "When --test-type is 'simulate', provide --pkg-type or --os-profile"
            )
        if args.os_profile and not args.pkg_type:
            try:
                NativeLinuxPackageInstallTest._derive_package_type(args.os_profile)
            except ValueError as e:
                parser.error(str(e))
        return
    if not args.os_profile:
        parser.error(
            "--os-profile is required when --test-type is 'install', 'sanity', or 'full'"
        )
    if not args.repo_url:
        parser.error(
            "--repo-url is required when --test-type is 'install', 'sanity', or 'full'"
        )
    if args.rocm_version:
        try:
            NativeLinuxPackageInstallTest._major_minor_rocm_version_from_input(
                args.rocm_version
            )
        except ValueError as e:
            parser.error(str(e))


def parse_cli_arguments(
    argv: list[str] | None = None, *, raise_instead_of_exit: bool = False
) -> Namespace:
    """Build parser, parse argv, validate.

    By default invalid input calls ``parser.error`` (exits the process). For pytest
    or other callers, pass ``raise_instead_of_exit=True`` to get ``ValueError``
    instead of ``sys.exit``.
    """
    exit_on_error = not raise_instead_of_exit
    parser = _build_argument_parser(exit_on_error=exit_on_error)
    if raise_instead_of_exit:

        def _raise(msg: str) -> None:
            raise ValueError(msg)

        parser.error = _raise  # type: ignore[method-assign]
    args = parser.parse_args(argv)
    try:
        args.test_type = _normalize_test_type(args.test_type)
    except ValueError as e:
        parser.error(str(e))
    _validate_cli_args(parser, args)
    return args


def run_tests(args: Namespace) -> int:
    """Run simulate or repo-based install test from parsed CLI args. Returns exit code (0 success)."""
    if args.test_type == "simulate":
        pkg_type = args.pkg_type or NativeLinuxPackageInstallTest._derive_package_type(
            args.os_profile
        )
        print("\n" + "=" * 80)
        print("SIMULATED INSTALL TEST")
        print("=" * 80)
        ok = run_simulate_install_test(pkg_type, args.packages_dir)
        if ok:
            return 0
        return 1

    try:
        derived_package_type = NativeLinuxPackageInstallTest._derive_package_type(
            args.os_profile
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"OS Profile: {args.os_profile}")
    print(f"Package Type (derived): {derived_package_type}")
    print(f"Release Type: {args.release_type}")
    print(f"Repository URL: {args.repo_url}")
    # Preview package-name rules before NativeLinuxPackageInstallTest is constructed.
    _norm = normalize_target_list(args.gfx_arch, lowercase=True, dedupe=True)
    if _norm:
        if args.rocm_version:
            print(
                f"GPU Architecture(s): {args.gfx_arch} "
                f"(normalized: {_norm}; used in package names with --rocm-version)"
            )
        else:
            print(
                f"GPU Architecture(s): {args.gfx_arch} "
                f"(normalized: {_norm}; not used in package names without --rocm-version)"
            )
    else:
        if args.rocm_version:
            _gv = NativeLinuxPackageInstallTest._major_minor_rocm_version_from_input(
                args.rocm_version
            )
            print(
                "GPU Architecture(s): (none — generic versioned packages "
                f"amdrocm{_gv}, amdrocm-core-sdk{_gv})"
            )
        else:
            print("GPU Architecture(s): (none — generic amdrocm, amdrocm-core-sdk)")
    if args.rocm_version:
        _rv = NativeLinuxPackageInstallTest._major_minor_rocm_version_from_input(
            args.rocm_version
        )
        print(
            f"ROCm version (for package names): {args.rocm_version} "
            f"(major.minor: {_rv})"
        )
    else:
        print("ROCm version (for package names): (not set)")
    print(f"Install Prefix: {args.install_prefix}")
    print(f"Test Type: {args.test_type}")
    if args.with_uninstall:
        print("With Uninstall: yes")
    if args.gpg_key_url:
        print(f"GPG Key URL: {args.gpg_key_url}")
    print("=" * 80)

    test_runner = NativeLinuxPackageInstallTest(
        os_profile=args.os_profile,
        repo_url=args.repo_url,
        release_type=args.release_type,
        install_prefix=args.install_prefix,
        gfx_arch=args.gfx_arch,
        rocm_version=args.rocm_version,
        gpg_key_url=args.gpg_key_url,
        build_variant=args.build_variant,
    )

    print("\n" + "=" * 80)
    print("INSTALLATION TEST - NATIVE LINUX PACKAGES")
    print("=" * 80)
    print(f"Release Type: {test_runner.release_type.upper()}")
    print(f"Install Prefix: {test_runner.install_prefix}")
    print(f"Test Type: {args.test_type}")
    print("=" * 80)

    try:
        if not test_runner.run_repo_setup_and_install():
            print("\n[FAIL] Step 1 (repo setup and install) failed.")
            return 1
        if args.test_type == "install":
            print("\n" + "=" * 80)
            print("[PASS] INSTALLATION TEST PASSED")
            print("(install: repo setup and package install completed)")
            print("=" * 80 + "\n")
            return 0
        if not test_runner.run_basic_verification():
            print("\n[FAIL] Step 2 (basic verification) failed.")
            return 1
        if args.test_type == "full":
            if not test_runner.run_full_verification():
                print("\n[FAIL] Step 3 (full verification) failed.")
                return 1
        if args.with_uninstall and args.test_type in ("sanity", "full"):
            if not test_runner.run_uninstall_and_verify():
                print("\n[FAIL] Step 4 (uninstall and verify) failed.")
                return 1
        print("\n" + "=" * 80)
        print("[PASS] INSTALLATION TEST PASSED")
        if args.test_type == "sanity":
            msg = "(sanity: basic verification completed"
            if args.with_uninstall:
                msg += " + uninstall verified"
            print(msg + ")")
        elif args.test_type == "full":
            msg = "ROCm has been successfully installed from repository and verified"
            if args.with_uninstall:
                msg += " and uninstalled cleanly"
            print(msg + "!")
        else:
            print("ROCm has been successfully installed from repository and verified!")
        print("=" * 80 + "\n")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Error during installation test: {e}")
        traceback.print_exc()
        return 1


def _argv_from_ci_env() -> list[str] | None:
    """Build CLI argv from workflow/container env (see ``test_native_linux_packages_install.yml``).

    Required for sanity/full: OS_PROFILE, REPO_URL, RELEASE_TYPE, INSTALL_PREFIX.
    Optional: GFX_ARCH, GPG_KEY_URL; ``NATIVE_LINUX_INSTALL_ROCM_VERSION`` maps to ``--rocm-version``
    only when versioned package names are needed (omit for unversioned installs).
    """
    test_type = (os.environ.get("TEST_TYPE") or "sanity").strip().lower() or "sanity"

    if test_type == "simulate":
        packages_dir = (os.environ.get("PACKAGES_DIR") or "").strip()
        if not packages_dir:
            return None
        argv: list[str] = [
            "--test-type",
            "simulate",
            "--packages-dir",
            packages_dir,
        ]
        os_profile = (os.environ.get("OS_PROFILE") or "").strip()
        if os_profile:
            argv.extend(["--os-profile", os_profile])
        pkg_type = (os.environ.get("SIMULATE_PKG_TYPE") or "").strip()
        if pkg_type in ("deb", "rpm"):
            argv.extend(["--pkg-type", pkg_type])
        prefix = (os.environ.get("INSTALL_PREFIX") or "").strip()
        if prefix:
            argv.extend(["--install-prefix", prefix])
        return argv

    os_profile = (os.environ.get("OS_PROFILE") or "").strip()
    repo_url = (os.environ.get("REPO_URL") or "").strip()
    gfx_raw = (os.environ.get("GFX_ARCH") or "").strip()
    # Semicolons delimit arches in CI; normalize to whitespace so split() does not
    # leave stray ';' on the first token (e.g. "gfx94x; gfx1100").
    gfx_arch = gfx_raw.replace(";", " ").split() if gfx_raw else []
    release_type = (os.environ.get("RELEASE_TYPE") or "").strip()
    install_prefix = (os.environ.get("INSTALL_PREFIX") or "").strip()

    if not (os_profile and repo_url and release_type and install_prefix):
        return None

    argv = [
        "--test-type",
        test_type,
        "--os-profile",
        os_profile,
        "--repo-url",
        repo_url,
        "--release-type",
        release_type,
        "--install-prefix",
        install_prefix,
    ]
    if gfx_arch:
        argv.extend(["--gfx-arch", *gfx_arch])
    rocm_version = (os.environ.get(ENV_NATIVE_LINUX_INSTALL_ROCM_VERSION) or "").strip()
    if rocm_version:
        argv.extend(["--rocm-version", rocm_version])
    gpg = (os.environ.get("GPG_KEY_URL") or "").strip()
    if gpg:
        argv.extend(["--gpg-key-url", gpg])
    build_variant = (os.environ.get("BUILD_VARIANT") or "").strip()
    if build_variant:
        argv.extend(["--build-variant", build_variant])
    run_uninstall = (os.environ.get(ENV_RUN_UNINSTALL) or "").strip().lower()
    if run_uninstall in ("1", "true", "yes"):
        argv.append("--with-uninstall")
    return argv


def test_native_linux_package_install() -> None:
    """Pytest entry: same run as CLI, driven by env vars in CI."""
    import pytest

    argv = _argv_from_ci_env()
    if argv is None:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(
                "Missing required environment variables for native install test "
                "(expected OS_PROFILE, REPO_URL, RELEASE_TYPE, INSTALL_PREFIX; "
                "optional GFX_ARCH, NATIVE_LINUX_INSTALL_ROCM_VERSION; or for simulate: PACKAGES_DIR)."
            )
        pytest.skip(
            "Set workflow env vars (OS_PROFILE, REPO_URL, RELEASE_TYPE, INSTALL_PREFIX); "
            "optional GFX_ARCH and NATIVE_LINUX_INSTALL_ROCM_VERSION."
        )

    args = parse_cli_arguments(argv, raise_instead_of_exit=True)
    rc = run_tests(args)
    assert rc == 0, f"run_tests exited with code {rc}"


def main() -> None:
    """Entry point: parse/validate CLI, then run tests."""
    args = parse_cli_arguments()
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
