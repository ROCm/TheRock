#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Install and verify the Windows MSI packages produced by a build run.

Installs each requested package's already-downloaded MSI with
``msiexec /i <msi> /qn`` and verifies the install:

  - the product's own installed files (queried from Windows Installer by the
    product's ProductCode, so each package is checked against *its* payload, not
    another package's) all exist on disk under the versioned install directory;
  - a default install has removed the package's legacy DLL names from
    ``System32`` (those are opt-in via ``LEGACY_INSTALL=1``);
  - the shared ``HKLM\\Software\\AMD\\ROCm\\{X.Y}`` discovery key records an
    ``InstallDir`` that matches the install directory and a matching ``Version``.

The MSIs are NOT downloaded here: the caller (CI workflow or a developer) places
them in a directory and passes ``--msi-dir`` / ``MSI_DIR`` so the files can be
re-used across repeated local runs without touching the network.

Can be run two ways:

  - CLI::

        python native_windows_package_install_test.py \\
            --rocm-version 10.2.0 --packages runtime --msi-dir path/to/msis

  - pytest, driven by environment variables (how CI invokes it)::

        MSI_DIR=... PACKAGES=... ROCM_VERSION=... \\
            python -m pytest native_windows_package_install_test.py

Windows-only: ``msiexec``, the Windows Installer API, and the registry checks
require Windows. On other platforms the CLI exits with a clear error and the
pytest entry skips, so the module still imports and collects cleanly on Linux CI.
"""

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

# Import the generator's package table and shared helpers (same directory) as
# the single source of truth for package metadata, MSI filenames, version
# parsing, install paths, and the discovery registry key.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_msi_wxs import (  # noqa: E402
    PACKAGES,
    discovery_registry_key,
    expected_msi_filename,
    install_subdir_for,
    parse_major_minor,
)

DEFAULT_PROGRAM_FILES = Path(r"C:\Program Files")
DEFAULT_SYSTEM32 = Path(r"C:\Windows\System32")
PRODUCT_SUBDIR = Path("AMD") / "ROCm"
MSIEXEC_TIMEOUT_SEC = 1800
# msiexec exit codes that count as a successful install/uninstall. 3010 is
# ERROR_SUCCESS_REBOOT_REQUIRED: the operation succeeded and a reboot is
# pending; it is not a failure.
MSIEXEC_SUCCESS_CODES = frozenset({0, 3010})

# Uninstall registry hive used to resolve an installed product's ProductCode
# from its DisplayName.
UNINSTALL_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"


def _env(key: str, default: str = "") -> str:
    """Return a stripped environment variable or the default when unset/empty."""
    value = os.environ.get(key, default)
    return value.strip() if value else default


def _split_packages(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def install_dir_for(
    package: str,
    rocm_version: str,
    program_files: Path = DEFAULT_PROGRAM_FILES,
) -> Path:
    """Return the versioned install directory for ``package``."""
    return program_files / PRODUCT_SUBDIR / install_subdir_for(package, rocm_version)


def collect_local_msis(packages: list[str], msi_dir: Path) -> list[Path]:
    """Return the expected MSI path for each package from a local directory."""
    found: list[Path] = []
    for package in packages:
        path = msi_dir / expected_msi_filename(package)
        if not path.is_file():
            raise FileNotFoundError(
                f"expected MSI for package {package!r} not found: {path}"
            )
        found.append(path)
    return found


def _print_log_tail(log_path: Path, lines: int = 40) -> None:
    """Print the tail of an msiexec verbose log to stderr for debugging."""
    try:
        text = log_path.read_text(encoding="utf-16", errors="replace")
    except (OSError, ValueError):
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            print(f"(could not read msiexec log {log_path})", file=sys.stderr)
            return
    tail = "\n".join(text.splitlines()[-lines:])
    print(f"---- msiexec log tail ({log_path}) ----\n{tail}", file=sys.stderr)


def install_msi(msi_path: Path, log_path: Path) -> None:
    """Install a single MSI silently, raising RuntimeError on real failure.

    Exit code 3010 (reboot required) is treated as success per Windows Installer
    semantics.
    """
    cmd = [
        "msiexec",
        "/i",
        str(msi_path),
        "/qn",
        "/norestart",
        "/l*v",
        str(log_path),
    ]
    print(f"Installing: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, timeout=MSIEXEC_TIMEOUT_SEC)
    if result.returncode not in MSIEXEC_SUCCESS_CODES:
        _print_log_tail(log_path)
        raise RuntimeError(
            f"msiexec failed (exit {result.returncode}) installing {msi_path}; "
            "see the verbose log above."
        )
    if result.returncode == 3010:
        print("[note] msiexec returned 3010 (reboot required); treated as success")


def find_product_code(display_name: str) -> str:
    """Return the MSI ProductCode for an installed product by DisplayName.

    Searches the 64-bit Uninstall hive for a subkey whose ``DisplayName`` value
    matches ``display_name`` and returns its ``{GUID}`` key name (the
    ProductCode msiexec /x and the Windows Installer API expect).
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


def installed_product_files(product_code: str) -> list[Path]:
    """Return the on-disk file paths Windows Installer records for a product.

    Enumerates every installed component (``MsiEnumComponentsW``) and keeps those
    whose key path belongs to ``product_code`` and resolves to a real filesystem
    path (``MsiGetComponentPathW`` returning INSTALLSTATE_LOCAL). Registry-backed
    components (whose key path is returned in the Windows Installer ``NN:\\...``
    registry-root encoding, not a filesystem path) are skipped — only payload
    files are returned.

    This attributes files to the specific product under test, so one package's
    payload cannot satisfy another's verification even when both install under
    the same directory. Non-file keypaths are skipped: registry-anchored
    components (RocmDiscovery, LongPaths) and directory-anchored components (the
    PATH/Environment component, whose keypath is the install dir itself).
    """
    import ctypes
    from ctypes import wintypes

    msi = ctypes.WinDLL("msi")
    ERROR_SUCCESS = 0
    ERROR_MORE_DATA = 234
    ERROR_NO_MORE_ITEMS = 259
    INSTALLSTATE_LOCAL = 3
    GUID_LEN = 39  # "{...}" plus NUL

    # UINT MsiEnumComponentsW(DWORD iComponentIndex, LPWSTR lpComponentBuf)
    msi.MsiEnumComponentsW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
    msi.MsiEnumComponentsW.restype = wintypes.UINT
    # INSTALLSTATE MsiGetComponentPathW(LPCWSTR szProduct, LPCWSTR szComponent,
    #                                   LPWSTR lpPathBuf, LPDWORD pcchBuf)
    msi.MsiGetComponentPathW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    msi.MsiGetComponentPathW.restype = ctypes.c_int

    def _is_registry_keypath(value: str) -> bool:
        # MsiGetComponentPath encodes registry key paths as "NN:\..." where NN is
        # a two-digit registry-root code; filesystem paths never match that.
        return len(value) >= 3 and value[0:2].isdigit() and value[2] == ":"

    files: list[Path] = []
    comp_buf = ctypes.create_unicode_buffer(GUID_LEN)
    index = 0
    while True:
        rc = msi.MsiEnumComponentsW(index, comp_buf)
        if rc == ERROR_NO_MORE_ITEMS:
            break
        if rc != ERROR_SUCCESS:
            raise RuntimeError(f"MsiEnumComponentsW failed at index {index}: {rc}")
        index += 1
        component = comp_buf.value

        # First call sizes the buffer (pcch excludes the NUL on return).
        pcch = wintypes.DWORD(0)
        state = msi.MsiGetComponentPathW(
            product_code, component, None, ctypes.byref(pcch)
        )
        if state != INSTALLSTATE_LOCAL:
            continue
        path_buf = ctypes.create_unicode_buffer(pcch.value + 1)
        pcch = wintypes.DWORD(pcch.value + 1)
        state = msi.MsiGetComponentPathW(
            product_code, component, path_buf, ctypes.byref(pcch)
        )
        if state != INSTALLSTATE_LOCAL:
            continue
        value = path_buf.value
        if not value or _is_registry_keypath(value):
            continue
        path = Path(value)
        # Some components are anchored by a directory keypath rather than a file
        # (e.g. the PATH/Environment component uses the install dir). Keep only
        # real payload files.
        if not path.is_file():
            continue
        files.append(path)
    return files


def verify_payload(package: str, install_dir: Path) -> None:
    """Assert the product's own installed files exist under the install dir.

    Resolves the package's ProductCode (by DisplayName) and asks Windows
    Installer which files it installed, then requires every such file to exist on
    disk and live under ``install_dir``. Because the file list comes from the
    product itself, this establishes that *this* package delivered its payload.
    """
    product_code = find_product_code(PACKAGES[package].product_name)
    # installed_product_files returns only real payload files that exist on disk,
    # attributed to this product's ProductCode.
    files = installed_product_files(product_code)
    if not files:
        raise RuntimeError(
            f"product {package!r} ({product_code}) reports no installed payload "
            "files on disk"
        )
    install_dir = install_dir.resolve()
    outside = [f for f in files if install_dir not in f.resolve().parents]
    if outside:
        raise RuntimeError(
            f"payload files for {package!r} installed outside {install_dir}: "
            f"{', '.join(str(f) for f in outside)}"
        )
    print(
        f"[PASS] payload present for {package!r}: {len(files)} files under "
        f"{install_dir}"
    )


def verify_system32_absent(package: str, system32: Path = DEFAULT_SYSTEM32) -> None:
    """Assert the package's legacy DLL names are absent from System32.

    A default install (no LEGACY_INSTALL) removes those names so a stale copy
    cannot shadow the runtime shipped under the install dir's bin/.
    """
    expected = PACKAGES[package].legacy_system32_dlls
    if not expected:
        print(f"[PASS] package {package!r} declares no System32 DLLs")
        return
    present = [name for name in expected if (system32 / name).is_file()]
    if present:
        raise RuntimeError(
            f"System32 DLLs still present in {system32} for package {package!r} "
            f"after a default install (should have been removed): "
            f"{', '.join(present)}"
        )
    print(f"[PASS] System32 DLLs absent from {system32} (default install)")


def verify_discovery_key(rocm_version: str, expected_install_dir: Path) -> None:
    """Assert the shared discovery key records the expected InstallDir and Version.

    The generator writes one shared ``HKLM\\Software\\AMD\\ROCm\\{X.Y}`` key
    (RFC0014) in the 64-bit view. Verifies InstallDir points at the actual
    install directory (not merely that it is non-empty) and that the directory
    exists, and that Version matches.
    """
    import winreg  # Windows-only; imported lazily so the module loads on Linux.

    major, minor = parse_major_minor(rocm_version)
    subkey = discovery_registry_key(major, minor)
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            subkey,
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as key:
            install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
            version, _ = winreg.QueryValueEx(key, "Version")
    except FileNotFoundError as e:
        raise RuntimeError(f"discovery registry key not found: HKLM\\{subkey}") from e
    except OSError as e:
        raise RuntimeError(
            f"could not read discovery registry values at HKLM\\{subkey}: {e}"
        ) from e
    if not isinstance(install_dir, str) or not install_dir:
        raise RuntimeError(f"registry InstallDir is empty at HKLM\\{subkey}")
    recorded = Path(install_dir)
    if recorded.resolve() != expected_install_dir.resolve():
        raise RuntimeError(
            f"registry InstallDir at HKLM\\{subkey} is {recorded}, "
            f"expected {expected_install_dir}"
        )
    if not recorded.is_dir():
        raise RuntimeError(
            f"registry InstallDir at HKLM\\{subkey} does not exist on disk: "
            f"{recorded}"
        )
    if version != rocm_version:
        raise RuntimeError(
            f"registry Version at HKLM\\{subkey} is {version!r}, "
            f"expected {rocm_version!r}"
        )
    print(
        f"[PASS] discovery key present: HKLM\\{subkey} "
        f"(InstallDir={install_dir}, Version={version})"
    )


class WindowsPackageInstallTest:
    """Install the requested MSIs from a local directory and verify each."""

    def __init__(
        self,
        packages: list[str],
        rocm_version: str,
        msi_dir: Path,
        program_files: Path = DEFAULT_PROGRAM_FILES,
        system32: Path = DEFAULT_SYSTEM32,
    ) -> None:
        unknown = [p for p in packages if p not in PACKAGES]
        if unknown:
            raise ValueError(
                f"unknown package(s): {', '.join(unknown)} "
                f"(choose from {', '.join(PACKAGES)})"
            )
        # Validate the version eagerly so a bad value fails before any install.
        parse_major_minor(rocm_version)
        self.packages = packages
        self.rocm_version = rocm_version
        self.msi_dir = msi_dir
        self.program_files = program_files
        self.system32 = system32

    def run(self) -> int:
        """Install and verify; return 0 on success, 1 on failure, 2 if not Windows."""
        if sys.platform != "win32":
            print(
                "[ERROR] native_windows_package_install_test requires Windows "
                f"(sys.platform={sys.platform!r}).",
                file=sys.stderr,
            )
            return 2
        try:
            msis = collect_local_msis(self.packages, self.msi_dir)
            for package, msi_path in zip(self.packages, msis):
                install_msi(msi_path, self.msi_dir / f"{package}-install.log")
            for package in self.packages:
                install_dir = install_dir_for(
                    package, self.rocm_version, self.program_files
                )
                verify_payload(package, install_dir)
                verify_system32_absent(package, self.system32)
            # The discovery key is shared/version-scoped (not per-package); verify
            # it once against the (shared) install directory.
            shared_install_dir = install_dir_for(
                self.packages[0], self.rocm_version, self.program_files
            )
            verify_discovery_key(self.rocm_version, shared_install_dir)
        except Exception:  # noqa: BLE001 - top-level reporting boundary
            print("[FAIL] MSI install test failed:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print("[PASS] all MSI packages installed and verified")
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install and verify already-downloaded Windows MSI packages."
    )
    parser.add_argument(
        "--packages",
        default="runtime",
        help="Comma-separated packages to test (e.g. 'runtime,core'). Default: runtime.",
    )
    parser.add_argument(
        "--rocm-version",
        required=True,
        help="ROCm version stamped on the MSI (e.g. '10.2.0'); sets install path + key.",
    )
    parser.add_argument(
        "--msi-dir",
        type=Path,
        required=True,
        help="Directory holding the already-downloaded MSIs to install.",
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
    msi_dir = _env("MSI_DIR")
    rocm_version = _env("ROCM_VERSION")
    if not msi_dir or not rocm_version:
        return None
    return [
        "--rocm-version",
        rocm_version,
        "--msi-dir",
        msi_dir,
        "--packages",
        _env("PACKAGES", "runtime"),
    ]


def _runner_from_args(args: argparse.Namespace) -> WindowsPackageInstallTest:
    return WindowsPackageInstallTest(
        packages=args.packages,
        rocm_version=args.rocm_version,
        msi_dir=args.msi_dir,
        program_files=args.program_files,
    )


def test_native_windows_package_install() -> None:
    """Pytest entry: same run as the CLI, driven by CI environment variables."""
    import pytest

    if sys.platform != "win32":
        pytest.skip("Windows-only MSI install test")
    argv = _argv_from_ci_env()
    if argv is None:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(
                "Missing required environment variables (MSI_DIR, ROCM_VERSION)."
            )
        pytest.skip("Set MSI_DIR / ROCM_VERSION to run this test.")
    args = parse_cli_arguments(argv, raise_instead_of_exit=True)
    rc = _runner_from_args(args).run()
    assert rc == 0, f"install test exited with code {rc}"


def main() -> None:
    args = parse_cli_arguments()
    sys.exit(_runner_from_args(args).run())


if __name__ == "__main__":
    main()
