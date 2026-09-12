#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Uninstall verification test for ROCm native Linux packages.

Run after a successful install test (same container/VM) to remove metapackages
installed by ``native_linux_package_install_test.py`` and assert clean teardown.

Steps:
  4a. Uninstall: remove metapackages in reverse install order.
      deb: ``sudo apt autoremove -y <metapackages>`` (matches public install docs);
      RHEL: ``dnf remove``; SLES: ``zypper remove --clean-deps``.
  4b. Verify: fail if the package-manager query fails, any ``rocm``/``amdrocm``
      packages remain, or any files remain under the install prefix.

Prerequisites:
- Run inside the same container/VM as the install test (packages must still be
  installed). Root or sudo may be required for package removal.
- CLI flags ``--gfx-arch``, ``--rocm-version``, and ``--build-variant`` must match
  the preceding install test so the same metapackage names are removed.

CI runs this module under pytest as a **separate workflow step** after install when
``run_uninstall: true`` in ``test_native_linux_packages_install.yml``::

    pytest build_tools/packaging/linux/native_linux_package_uninstall_test.py -vv --tb=short

Workflow/container ``env`` maps to CLI flags via :func:`_argv_from_ci_env` and
:func:`test_native_linux_package_uninstall`. Required env: ``OS_PROFILE``,
``INSTALL_PREFIX``. Optional: ``GFX_ARCH``, ``BUILD_VARIANT``,
``NATIVE_LINUX_INSTALL_ROCM_VERSION`` (same as install step).

Example invocations:

 # After install on Ubuntu 24.04
 python3 native_linux_package_uninstall_test.py \\
         --os-profile ubuntu2404 \\
         --gfx-arch gfx94x --install-prefix /opt/rocm/core

 # Versioned metapackages
 python3 native_linux_package_uninstall_test.py \\
         --os-profile ubuntu2404 \\
         --rocm-version 7.13 --gfx-arch gfx94x --install-prefix /opt/rocm/core

 # SLES: zypper remove --clean-deps is required
 python3 native_linux_package_uninstall_test.py \\
         --os-profile sles16 --install-prefix /opt/rocm/core
"""

import argparse
import os
import subprocess
import sys
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from native_linux_package_test_common import (
    ENV_NATIVE_LINUX_INSTALL_ROCM_VERSION,
    UNINSTALL_TIMEOUT_SEC,
    build_metapackage_names,
    derive_package_type,
    is_rocm_related_package_name,
    is_sles,
    major_minor_rocm_version_from_input,
    run_streaming,
)
from packaging_utils import normalize_target_list


class NativeLinuxPackageUninstallTest:
    """Runner for native Linux package uninstall and post-uninstall verification.

    Constructs the same metapackage name list as
    ``NativeLinuxPackageInstallTest`` via :func:`build_metapackage_names`, then
    removes those packages and verifies no ROCm-related packages remain installed.
    """

    def __init__(
        self,
        os_profile: str,
        install_prefix: str | None = None,
        gfx_arch: str | list[str] | None = None,
        rocm_version: str | None = None,
        build_variant: str = "",
    ):
        """Initialize the uninstall test runner.

        Args:
            os_profile: OS profile (e.g. ``ubuntu2404``, ``rhel8``, ``sles16``).
            install_prefix: Install prefix from the install test; Step 4b fails if
                any files remain under this path (default: ``/opt/rocm/core``).
            gfx_arch: GPU architecture(s) from the install test; must match install
                flags when arch-suffixed metapackages were installed.
            rocm_version: ROCm release from the install test (major.minor used in
                package names). Must match install when versioned names were used.
            build_variant: Build variant from the install test (e.g. ``asan``).
        """
        self.os_profile = os_profile.lower()
        self.package_type = derive_package_type(os_profile)
        self.install_prefix = install_prefix or "/opt/rocm/core"
        self.gfx_arch_list = normalize_target_list(
            gfx_arch, lowercase=True, dedupe=True
        )
        self.rocm_version_major_minor = major_minor_rocm_version_from_input(
            rocm_version
        )
        self.build_variant = build_variant.strip().lower()
        self.package_names = build_metapackage_names(
            gfx_arch=self.gfx_arch_list,
            rocm_version=rocm_version,
            build_variant=self.build_variant,
        )

    def list_installed_rocm_packages(self) -> list[str] | None:
        """Query the system package manager for installed ROCm-related packages.

        deb: parses ``dpkg -l`` lines with status ``ii``. rpm/SLES: parses
        ``rpm -qa`` output. Package names matching :func:`is_rocm_related_package_name`
        are collected.

        Returns:
            Sorted list of installed package names, ``[]`` when none match, or
            ``None`` when the query fails (distinct from a clean empty result).
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
                    if len(parts) >= 2 and is_rocm_related_package_name(parts[1]):
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
                if line.strip() and is_rocm_related_package_name(line.strip())
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
            print(f"[FAIL] Could not query installed packages: {stderr or e}")
            return None
        except OSError as e:
            print(f"[FAIL] Could not query installed packages: {e}")
            return None

    def uninstall_packages(self) -> bool:
        """Step 4a: remove configured metapackages in reverse install order.

        deb runs ``sudo apt autoremove -y`` with metapackage names (public docs).
        RHEL runs ``dnf remove``. SLES runs ``zypper remove --clean-deps``.

        Returns:
            True when all remove commands succeed; False on non-zero exit,
            timeout, or OS error.
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
            remove_cmd = ["sudo", "apt", "autoremove", "-y"] + packages_to_remove
        elif is_sles(self.os_profile):
            remove_cmd = [
                "zypper",
                "--non-interactive",
                "remove",
                "-y",
                "--clean-deps",
            ] + packages_to_remove
        else:
            remove_cmd = ["dnf", "remove", "-y"] + packages_to_remove

        print(f"\nRunning: {' '.join(remove_cmd)}")
        print("=" * 80)
        print("Uninstall progress (streaming output):\n")

        try:
            return_code = run_streaming(remove_cmd, UNINSTALL_TIMEOUT_SEC)
            if return_code != 0:
                print("\n" + "=" * 80)
                print(f"[FAIL] Failed to remove packages (exit code: {return_code})")
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

    def _verify_install_prefix_empty(self, install_path: Path) -> bool:
        """Return True when the install prefix is gone or contains no files."""
        if not install_path.exists():
            print(f"[PASS] Install prefix removed: {self.install_prefix}")
            return True

        leftover_paths = sorted(
            p.relative_to(install_path)
            for p in install_path.rglob("*")
            if p.is_file() or p.is_symlink()
        )
        if leftover_paths:
            print(f"\n[FAIL] Install prefix not empty: {self.install_prefix}")
            for rel_path in leftover_paths[:10]:
                print(f"  {rel_path}")
            if len(leftover_paths) > 10:
                print(f"  ... and {len(leftover_paths) - 10} more")
            return False

        print(f"[PASS] Install prefix is empty: {self.install_prefix}")
        return True

    def run_uninstall_verification(self) -> bool:
        """Step 4b: verify clean uninstall teardown.

        Fails when the package-manager query fails, any ROCm-related packages
        remain, or any files remain under the install prefix.

        Returns:
            True when package query succeeds, no ROCm packages remain, and the
            install prefix is gone or empty; False otherwise.
        """
        print("\n" + "=" * 80)
        print("STEP 4b: UNINSTALL VERIFICATION")
        print("=" * 80)

        remaining = self.list_installed_rocm_packages()
        if remaining is None:
            return False
        if remaining:
            print(f"\n[FAIL] {len(remaining)} ROCm package(s) still installed:")
            for pkg in remaining[:10]:
                print(f"  {pkg}")
            if len(remaining) > 10:
                print(f"  ... and {len(remaining) - 10} more")
            return False

        print("\n[PASS] No ROCm packages remain installed")

        if not self._verify_install_prefix_empty(Path(self.install_prefix)):
            return False

        print("\n[PASS] Uninstall verification PASSED")
        return True

    def run_uninstall_and_verify(self) -> bool:
        """Orchestrate Step 4a (uninstall) and Step 4b (verification).

        Logs a sample of installed ROCm packages before removal, then runs
        :meth:`uninstall_packages` and :meth:`run_uninstall_verification`.

        Returns:
            True only when both steps succeed.
        """
        print("\n" + "=" * 80)
        print("UNINSTALL AND VERIFY - NATIVE LINUX PACKAGES")
        print("=" * 80)

        before = self.list_installed_rocm_packages()
        if before is None:
            print("\n[FAIL] Could not query installed packages before uninstall")
            return False
        print(f"\nROCm packages before uninstall: {len(before)}")
        if before:
            print(" Sample packages (first 5):")
            for pkg in before[:5]:
                print(f"  {pkg}")

        if not self.uninstall_packages():
            return False
        return self.run_uninstall_verification()


_CLI_EXAMPLES_EPILOG = """
Examples:
 # After install on Ubuntu 24.04 (must match install --gfx-arch / prefix)
 python native_linux_package_uninstall_test.py --os-profile ubuntu2404 \\
 --gfx-arch gfx94x --install-prefix /opt/rocm/core

 # Versioned metapackages (match install --rocm-version and --gfx-arch)
 python native_linux_package_uninstall_test.py --os-profile ubuntu2404 \\
 --rocm-version 7.13 --gfx-arch gfx94x --install-prefix /opt/rocm/core

 # Multiple GPU architectures (same order/normalization as install)
 python native_linux_package_uninstall_test.py --os-profile ubuntu2404 \\
 --rocm-version 7.13 --gfx-arch gfx94x gfx1100 --install-prefix /opt/rocm/core

 # RHEL 8 after nightly install
 python native_linux_package_uninstall_test.py --os-profile rhel8 \\
 --gfx-arch gfx94x --install-prefix /opt/rocm/core

 # ASAN variant packages (match install --build-variant asan)
 python native_linux_package_uninstall_test.py --os-profile ubuntu2404 \\
 --rocm-version 7.13 --gfx-arch gfx94x --build-variant asan \\
 --install-prefix /opt/rocm/core

 # SLES: zypper remove --clean-deps is required
 python native_linux_package_uninstall_test.py --os-profile sles16 \\
 --install-prefix /opt/rocm/core

 # Unversioned generic metapackages (no --rocm-version / --gfx-arch)
 python native_linux_package_uninstall_test.py --os-profile ubuntu2404 \\
 --install-prefix /opt/rocm/core
"""


def _build_argument_parser(*, exit_on_error: bool = True) -> ArgumentParser:
    """Build the uninstall test CLI argument parser.

    Args:
        exit_on_error: When True (default), argparse exits on invalid input.
            Set False when the caller will handle errors (e.g. pytest with
            ``raise_instead_of_exit``).

    Returns:
        Configured :class:`ArgumentParser` with epilog examples.
    """
    kwargs: dict = dict(
        description="Uninstall and verify teardown for ROCm native Linux packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EXAMPLES_EPILOG,
    )
    if sys.version_info >= (3, 9):
        kwargs["exit_on_error"] = exit_on_error
    parser = ArgumentParser(**kwargs)
    parser.add_argument(
        "--os-profile",
        type=str,
        required=True,
        help="OS profile (e.g., ubuntu2404, rhel8, sles16).",
    )
    parser.add_argument(
        "--install-prefix",
        type=str,
        default="/opt/rocm/core",
        help="Installation prefix used during install (default: /opt/rocm/core).",
    )
    parser.add_argument(
        "--gfx-arch",
        type=str,
        nargs="+",
        default=None,
        metavar="ARCH",
        help="GPU architecture(s) used during install (must match install test).",
    )
    parser.add_argument(
        "--rocm-version",
        type=str,
        default=None,
        metavar="VER",
        help="ROCm release used during install (major.minor in package names).",
    )
    parser.add_argument(
        "--build-variant",
        type=str,
        default="",
        help="Build variant used during install (e.g. 'asan').",
    )
    return parser


def _validate_cli_args(parser: ArgumentParser, args: Namespace) -> None:
    """Validate parsed CLI arguments; call ``parser.error`` on failure.

    Args:
        parser: Parser used to report validation errors.
        args: Parsed namespace from :func:`parse_cli_arguments`.
    """
    try:
        derive_package_type(args.os_profile)
    except ValueError as e:
        parser.error(str(e))
    if args.rocm_version:
        try:
            major_minor_rocm_version_from_input(args.rocm_version)
        except ValueError as e:
            parser.error(str(e))


def parse_cli_arguments(
    argv: list[str] | None = None, *, raise_instead_of_exit: bool = False
) -> Namespace:
    """Build parser, parse argv, and validate uninstall CLI arguments.

    By default invalid input calls ``parser.error`` (exits the process). For pytest
    or other callers, pass ``raise_instead_of_exit=True`` to get ``ValueError``
    instead of ``sys.exit``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).
        raise_instead_of_exit: When True, validation errors raise ``ValueError``.

    Returns:
        Parsed and validated argument namespace.
    """
    exit_on_error = not raise_instead_of_exit
    parser = _build_argument_parser(exit_on_error=exit_on_error)
    if raise_instead_of_exit:

        def _raise(msg: str) -> None:
            raise ValueError(msg)

        parser.error = _raise  # type: ignore[method-assign]
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    return args


def run_tests(args: Namespace) -> int:
    """Run uninstall and verification from parsed CLI args.

    Instantiates :class:`NativeLinuxPackageUninstallTest` and calls
    :meth:`NativeLinuxPackageUninstallTest.run_uninstall_and_verify`.

    Args:
        args: Parsed CLI namespace from :func:`parse_cli_arguments`.

    Returns:
        Exit code (0 success, 1 failure).
    """
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"OS Profile: {args.os_profile}")
    print(f"Package Type (derived): {derive_package_type(args.os_profile).upper()}")
    print(f"Install Prefix: {args.install_prefix}")
    if args.gfx_arch:
        print(f"GPU Architecture(s): {args.gfx_arch}")
    if args.rocm_version:
        print(f"ROCm version (for package names): {args.rocm_version}")
    if args.build_variant:
        print(f"Build variant: {args.build_variant}")
    print("=" * 80)

    test_runner = NativeLinuxPackageUninstallTest(
        os_profile=args.os_profile,
        install_prefix=args.install_prefix,
        gfx_arch=args.gfx_arch,
        rocm_version=args.rocm_version,
        build_variant=args.build_variant,
    )

    try:
        if not test_runner.run_uninstall_and_verify():
            print("\n[FAIL] Uninstall and verify failed.")
            return 1
        print("\n" + "=" * 80)
        print("[PASS] UNINSTALL TEST PASSED")
        print("=" * 80 + "\n")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Error during uninstall test: {e}")
        traceback.print_exc()
        return 1


def _argv_from_ci_env() -> list[str] | None:
    """Build CLI argv from workflow/container env (see install workflow YAML).

    Required: ``OS_PROFILE``, ``INSTALL_PREFIX``. Optional: ``GFX_ARCH``,
    ``BUILD_VARIANT``, ``NATIVE_LINUX_INSTALL_ROCM_VERSION`` (maps to
    ``--rocm-version``). Semicolons in ``GFX_ARCH`` are normalized to spaces.

    Returns:
        Argument list suitable for :func:`parse_cli_arguments`, or ``None`` when
        required variables are missing.
    """
    os_profile = (os.environ.get("OS_PROFILE") or "").strip()
    install_prefix = (os.environ.get("INSTALL_PREFIX") or "").strip()
    if not (os_profile and install_prefix):
        return None

    argv = [
        "--os-profile",
        os_profile,
        "--install-prefix",
        install_prefix,
    ]
    gfx_raw = (os.environ.get("GFX_ARCH") or "").strip()
    gfx_arch = gfx_raw.replace(";", " ").split() if gfx_raw else []
    if gfx_arch:
        argv.extend(["--gfx-arch", *gfx_arch])
    rocm_version = (os.environ.get(ENV_NATIVE_LINUX_INSTALL_ROCM_VERSION) or "").strip()
    if rocm_version:
        argv.extend(["--rocm-version", rocm_version])
    build_variant = (os.environ.get("BUILD_VARIANT") or "").strip()
    if build_variant:
        argv.extend(["--build-variant", build_variant])
    return argv


def test_native_linux_package_uninstall() -> None:
    """Pytest entry: same run as CLI, driven by workflow env vars in CI.

    Skips locally when env is unset; fails in GitHub Actions when required env
    is missing. Failures are reported via :func:`run_tests` exit codes.
    """
    import pytest

    argv = _argv_from_ci_env()
    if argv is None:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(
                "Missing required environment variables for native uninstall test "
                "(expected OS_PROFILE, INSTALL_PREFIX; optional GFX_ARCH, "
                "BUILD_VARIANT, NATIVE_LINUX_INSTALL_ROCM_VERSION)."
            )
        pytest.skip(
            "Set workflow env vars (OS_PROFILE, INSTALL_PREFIX); optional GFX_ARCH, "
            "BUILD_VARIANT, NATIVE_LINUX_INSTALL_ROCM_VERSION."
        )

    args = parse_cli_arguments(argv, raise_instead_of_exit=True)
    rc = run_tests(args)
    if rc != 0:
        pytest.fail(f"Native Linux package uninstall test failed (exit code {rc})")


def main() -> None:
    """Entry point: parse/validate CLI, then run uninstall test."""
    args = parse_cli_arguments()
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
