#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Uninstall the Windows MSI packages and verify a clean teardown.

Companion to ``native_windows_package_install_test.py``. For each requested
package it looks up the installed product's ProductCode by DisplayName (the
package's ``product_name``) under the Windows "Uninstall" registry hive, runs
``msiexec /x {ProductCode} /qn`` to remove it, and verifies teardown: the
versioned install directory is gone (or empty) and the package's HKLM registry
key no longer exists.

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
from generate_msi_wxs import PACKAGES  # noqa: E402

# Reuse the install test's shared helpers (same directory) so the two scripts
# agree on paths, version parsing, and env handling.
from native_windows_package_install_test import (  # noqa: E402
    DEFAULT_ARTIFACT_GITHUB_REPO,
    DEFAULT_PROGRAM_FILES,
    MSIEXEC_TIMEOUT_SEC,
    _env,
    _format_package_template,
    _print_log_tail,
    _split_packages,
    install_dir_for,
    major_minor,
)

UNINSTALL_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"


def find_product_code(display_name: str) -> str:
    """Return the MSI ProductCode for an installed product by DisplayName.

    Searches the 64-bit Uninstall hive for a subkey whose ``DisplayName`` value
    matches ``display_name`` and returns its ``{GUID}`` key name (the
    ProductCode msiexec /x expects).
    """
    import winreg  # Windows-only; imported lazily so the module loads on Linux.

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        UNINSTALL_KEY,
        0,
        winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
    ) as hive:
        index = 0
        while True:
            try:
                product_code = winreg.EnumKey(hive, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(hive, product_code) as entry:
                    name, _ = winreg.QueryValueEx(entry, "DisplayName")
            except (FileNotFoundError, OSError):
                continue
            if name == display_name:
                return product_code
    raise RuntimeError(
        f"no installed product found with DisplayName {display_name!r} under "
        f"HKLM\\{UNINSTALL_KEY}"
    )


def uninstall_msi(product_code: str, log_path: Path) -> None:
    """Uninstall a product by ProductCode, raising RuntimeError on failure."""
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
    if result.returncode != 0:
        _print_log_tail(log_path)
        raise RuntimeError(
            f"msiexec failed (exit {result.returncode}) uninstalling {product_code}; "
            "see the verbose log above."
        )


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


def verify_registry_key_removed(package: str, rocm_version: str) -> None:
    """Assert the package's HKLM registry key no longer exists."""
    import winreg  # Windows-only; imported lazily so the module loads on Linux.

    subkey = _format_package_template(PACKAGES[package].registry_key, rocm_version)
    try:
        winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            subkey,
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ).Close()
    except FileNotFoundError:
        print(f"[PASS] registry key removed: HKLM\\{subkey}")
        return
    raise RuntimeError(f"registry key still present after uninstall: HKLM\\{subkey}")


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
        major_minor(rocm_version)
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
                    product_code = find_product_code(PACKAGES[package].product_name)
                    uninstall_msi(product_code, work_dir / f"{package}-uninstall.log")
                for package in self.packages:
                    verify_install_dir_removed(
                        package, self.rocm_version, self.program_files
                    )
                    verify_registry_key_removed(package, self.rocm_version)
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
    # --artifact-run-id / --artifact-github-repo / --release-type are accepted for
    # symmetry with the install test (so the same env/args drive both), even
    # though uninstall needs only the installed product + version.
    parser.add_argument("--artifact-run-id", default="")
    parser.add_argument("--artifact-github-repo", default=DEFAULT_ARTIFACT_GITHUB_REPO)
    parser.add_argument("--release-type", default="nightly")
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
