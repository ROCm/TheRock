#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Uninstall the Windows MSI packages and verify a clean teardown.

Companion to ``native_windows_package_install_test.py``. For each requested
package it resolves the installed product's ProductCode (by DisplayName),
validates that the product under that code was installed for the requested
``--rocm-version`` (so the wrong version is never removed), runs
``msiexec /x {ProductCode} /qn`` to remove it, and verifies teardown: the
versioned install directory is gone (or empty) and the shared SDK-discovery
registry key no longer exists.

Note: the driver-supplied System32 DLLs are shared, reference-counted MSI
components and may legitimately remain after an uninstall (another ROCm install
may still reference them), so teardown does NOT assert their removal.

Same invocation modes as the install test (CLI / pytest / env vars) and the
same Windows-only guard, so the module imports and collects cleanly on Linux.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_msi_wxs import (  # noqa: E402
    PACKAGES,
    discovery_registry_key,
    parse_major_minor,
)

# Reuse the install test's shared helpers (same directory) so the two scripts
# agree on paths, msiexec handling, and env parsing.
from native_windows_package_install_test import (  # noqa: E402
    DEFAULT_PROGRAM_FILES,
    MSIEXEC_SUCCESS_CODES,
    MSIEXEC_TIMEOUT_SEC,
    _env,
    _print_log_tail,
    _split_packages,
    find_product_code,
    install_dir_for,
    installed_product_files,
)


def product_install_location(product_code: str) -> Path | None:
    """Return a product's install location from the Windows Installer API.

    Reads ``INSTALLPROPERTY_INSTALLLOCATION`` for ``product_code``; returns None
    when the product records no install location (older MSIs). Used to validate
    the installed version before removal.
    """
    import ctypes
    from ctypes import wintypes

    msi = ctypes.WinDLL("msi")
    ERROR_MORE_DATA = 234
    msi.MsiGetProductInfoW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    msi.MsiGetProductInfoW.restype = wintypes.UINT

    prop = "InstallLocation"  # INSTALLPROPERTY_INSTALLLOCATION
    pcch = wintypes.DWORD(0)
    rc = msi.MsiGetProductInfoW(product_code, prop, None, ctypes.byref(pcch))
    if rc not in (0, ERROR_MORE_DATA):
        return None
    buf = ctypes.create_unicode_buffer(pcch.value + 1)
    pcch = wintypes.DWORD(pcch.value + 1)
    rc = msi.MsiGetProductInfoW(product_code, prop, buf, ctypes.byref(pcch))
    if rc != 0 or not buf.value:
        return None
    return Path(buf.value)


def resolve_product_for_version(package: str, rocm_version: str) -> str:
    """Return the ProductCode for ``package`` at ``rocm_version``, or raise.

    Guards against removing the wrong version: the product is matched by
    DisplayName and then its installed footprint is confirmed to belong to the
    requested version's install directory before its ProductCode is returned.
    """
    product_code = find_product_code(PACKAGES[package].product_name)
    expected_dir = install_dir_for(package, rocm_version).resolve()

    # Prefer the product's recorded InstallLocation; fall back to its component
    # file paths when the MSI did not set InstallLocation.
    location = product_install_location(product_code)
    if location is not None:
        if location.resolve() != expected_dir:
            raise RuntimeError(
                f"product {package!r} ({product_code}) is installed at {location}, "
                f"not the requested version's directory {expected_dir}; refusing "
                "to uninstall a different version"
            )
        return product_code

    files = installed_product_files(product_code)
    if files and not any(expected_dir in f.resolve().parents for f in files):
        raise RuntimeError(
            f"product {package!r} ({product_code}) has no files under the "
            f"requested version's directory {expected_dir}; refusing to uninstall "
            "a different version"
        )
    return product_code


def uninstall_msi(product_code: str, log_path: Path) -> None:
    """Uninstall a product by ProductCode, raising on real failure.

    Exit code 3010 (reboot required) is treated as success.
    """
    cmd = [
        "msiexec",
        "/x",
        product_code,
        "/qn",
        "/norestart",
        "/l*v",
        str(log_path),
    ]
    print(f"Uninstalling: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, timeout=MSIEXEC_TIMEOUT_SEC)
    if result.returncode not in MSIEXEC_SUCCESS_CODES:
        _print_log_tail(log_path)
        raise RuntimeError(
            f"msiexec failed (exit {result.returncode}) uninstalling {product_code}; "
            "see the verbose log above."
        )
    if result.returncode == 3010:
        print("[note] msiexec returned 3010 (reboot required); treated as success")


def verify_install_dir_removed(
    package: str, rocm_version: str, program_files: Path = DEFAULT_PROGRAM_FILES
) -> None:
    """Assert the package's versioned install directory is gone or empty."""
    install_dir = install_dir_for(package, rocm_version, program_files)
    if install_dir.is_dir() and any(install_dir.iterdir()):
        raise RuntimeError(
            f"install directory still present after uninstall: {install_dir}"
        )
    print(f"[PASS] install directory removed: {install_dir}")


def verify_discovery_key_removed(rocm_version: str) -> None:
    """Assert the shared SDK-discovery key no longer exists.

    The key is ref-counted across all ROCm packages at the version, so it is
    removed only once the LAST package uninstalls; this is checked once after all
    requested packages are removed.
    """
    import winreg  # Windows-only; imported lazily so the module loads on Linux.

    major, minor = parse_major_minor(rocm_version)
    subkey = discovery_registry_key(major, minor)
    try:
        winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            subkey,
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ).Close()
    except FileNotFoundError:
        print(f"[PASS] discovery registry key removed: HKLM\\{subkey}")
        return
    raise RuntimeError(
        f"discovery registry key still present after uninstall: HKLM\\{subkey}"
    )


class WindowsPackageUninstallTest:
    """Uninstall the requested packages and verify clean teardown."""

    def __init__(
        self,
        packages: list[str],
        rocm_version: str,
        program_files: Path = DEFAULT_PROGRAM_FILES,
    ) -> None:
        unknown = [p for p in packages if p not in PACKAGES]
        if unknown:
            raise ValueError(
                f"unknown package(s): {', '.join(unknown)} "
                f"(choose from {', '.join(PACKAGES)})"
            )
        parse_major_minor(rocm_version)
        self.packages = packages
        self.rocm_version = rocm_version
        self.program_files = program_files

    def run(self) -> int:
        """Uninstall and verify teardown; return 0 on success, 1 on failure."""
        if sys.platform != "win32":
            print(
                "[ERROR] native_windows_package_uninstall_test requires Windows "
                f"(sys.platform={sys.platform!r}).",
                file=sys.stderr,
            )
            return 2
        try:
            with tempfile.TemporaryDirectory() as tmp:
                work_dir = Path(tmp)
                for package in self.packages:
                    # Validate the installed version before removing it.
                    product_code = resolve_product_for_version(
                        package, self.rocm_version
                    )
                    uninstall_msi(product_code, work_dir / f"{package}-uninstall.log")
                for package in self.packages:
                    verify_install_dir_removed(
                        package, self.rocm_version, self.program_files
                    )
                # The discovery key is shared and ref-counted across packages, so
                # it is gone only after the last package is removed above; check
                # it once.
                verify_discovery_key_removed(self.rocm_version)
        except Exception:  # noqa: BLE001 - top-level reporting boundary
            print("[FAIL] MSI uninstall test failed:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print("[PASS] all MSI packages uninstalled and teardown verified")
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Uninstall Windows MSI packages and verify clean teardown."
    )
    parser.add_argument(
        "--packages",
        default="runtime",
        help="Comma-separated packages to uninstall. Default: runtime.",
    )
    parser.add_argument(
        "--rocm-version",
        required=True,
        help="ROCm version the packages were installed with (e.g. '10.2.0').",
    )
    parser.add_argument(
        "--program-files",
        type=Path,
        default=DEFAULT_PROGRAM_FILES,
        help=f"Program Files root to verify under. Default: {DEFAULT_PROGRAM_FILES}.",
    )
    return parser


def parse_cli_arguments(
    argv: list[str] | None = None, *, raise_instead_of_exit: bool = False
) -> argparse.Namespace:
    parser = build_arg_parser()
    if raise_instead_of_exit:

        def _raise(message: str):
            raise ValueError(message)

        parser.error = _raise  # type: ignore[assignment]
    args = parser.parse_args(argv)
    args.packages = _split_packages(args.packages)
    if not args.packages:
        parser.error("--packages was empty")
    return args


def _argv_from_ci_env() -> list[str] | None:
    """Build argv from CI environment variables, or None if required ones unset."""
    rocm_version = _env("ROCM_VERSION")
    if not rocm_version:
        return None
    return [
        "--rocm-version",
        rocm_version,
        "--packages",
        _env("PACKAGES", "runtime"),
    ]


def _runner_from_args(args: argparse.Namespace) -> WindowsPackageUninstallTest:
    return WindowsPackageUninstallTest(
        packages=args.packages,
        rocm_version=args.rocm_version,
        program_files=args.program_files,
    )


def test_native_windows_package_uninstall() -> None:
    """Pytest entry: driven by CI environment variables."""
    import pytest

    if sys.platform != "win32":
        pytest.skip("Windows-only MSI uninstall test")
    argv = _argv_from_ci_env()
    if argv is None:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("Missing required environment variable (ROCM_VERSION).")
        pytest.skip("Set ROCM_VERSION to run this test.")
    args = parse_cli_arguments(argv, raise_instead_of_exit=True)
    rc = _runner_from_args(args).run()
    assert rc == 0, f"uninstall test exited with code {rc}"


def main() -> None:
    args = parse_cli_arguments()
    sys.exit(_runner_from_args(args).run())


if __name__ == "__main__":
    main()
