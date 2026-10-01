#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Install and verify the Windows MSI packages produced by a build run.

Downloads the MSI(s) for the requested packages from the public artifacts
prefix of a build run (``{run_id}-windows/packages/msi/``), installs each with
``msiexec /i <msi> /qn``, and verifies the install: the versioned install
directory exists and contains the expected runtime DLLs, the driver-supplied
legacy DLLs are present in ``System32``, and the package's HKLM registry key
records an install location.

The MSI filenames are derived from the generator's PACKAGES table
(``<output_stem>.msi``), so no S3 listing is needed and the download is a plain
public HTTPS request (no credentials -- works from forks).

Can be run three ways:

  - CLI (local iteration against a locally built MSI)::

        python native_windows_package_install_test.py \\
            --artifact-run-id local --rocm-version 10.2.0 \\
            --packages runtime --msi-dir path/to/msis

  - CLI (download a real build run's MSIs by run id)::

        python native_windows_package_install_test.py \\
            --artifact-run-id 36648157930 --release-type nightly \\
            --rocm-version 10.2.0 --packages runtime

  - pytest, driven by environment variables (how CI invokes it)::

        ARTIFACT_RUN_ID=... RELEASE_TYPE=... PACKAGES=... ROCM_VERSION=... \\
            python -m pytest native_windows_package_install_test.py

Windows-only: ``msiexec`` and the registry checks require Windows. On other
platforms the CLI exits with a clear error and the pytest entry skips, so the
module still imports and collects cleanly on Linux CI.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
from pathlib import Path

# Import the generator's package table (same directory) as the single source of
# truth for output_stem / install_subdir / registry_key / legacy_system32_dlls.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_msi_wxs import PACKAGES  # noqa: E402

# Allow _therock_utils imports (build_tools/ on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _therock_utils.workflow_outputs import WorkflowOutputRoot  # noqa: E402

PLATFORM = "windows"
DEFAULT_ARTIFACT_GITHUB_REPO = "ROCm/TheRock"
DEFAULT_PROGRAM_FILES = Path(r"C:\Program Files")
DEFAULT_SYSTEM32 = Path(r"C:\Windows\System32")
PRODUCT_SUBDIR = Path("AMD") / "ROCm"
MSIEXEC_TIMEOUT_SEC = 1800

# Runtime payload DLLs expected somewhere under a package's install tree. The
# MSI lays these out under the install dir (DLLs live in the lib component on
# Windows), so verification globs the tree rather than hardcoding subpaths.
RUNTIME_PAYLOAD_DLL_GLOBS = [
    "amdhip64_7.dll",
    "hiprtc*.dll",
    "amd_comgr.dll",
    "rocm_kpack.dll",
]


def _env(key: str, default: str = "") -> str:
    """Return a stripped environment variable or the default when unset/empty."""
    value = os.environ.get(key, default)
    return value.strip() if value else default


def major_minor(rocm_version: str) -> tuple[str, str]:
    """Return (major, minor) from a dotted ROCm version like ``10.2.0``."""
    parts = rocm_version.split(".")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"rocm version must be at least major.minor (got {rocm_version!r})"
        )
    return parts[0], parts[1]


def _format_package_template(template: str, rocm_version: str) -> str:
    """Expand a PackageDef template using the version placeholders."""
    major, minor = major_minor(rocm_version)
    return template.format(version=rocm_version, major=major, minor=minor)


def expected_msi_filename(package: str) -> str:
    """Return the MSI filename the generator writes for ``package``."""
    try:
        return f"{PACKAGES[package].output_stem}.msi"
    except KeyError as e:
        raise ValueError(
            f"unknown package {package!r}; choose from {', '.join(PACKAGES)}"
        ) from e


def resolve_msi_prefix_url(
    artifact_run_id: str, artifact_github_repo: str, release_type: str
) -> str:
    """Return the public HTTPS URL of the run's msi package prefix.

    Resolves the artifacts bucket from the run id + release type without any
    GitHub API call (lookup_workflow_run=False).
    """
    root = WorkflowOutputRoot.from_workflow_run(
        run_id=artifact_run_id,
        platform=PLATFORM,
        github_repository=artifact_github_repo,
        release_type=release_type,
        lookup_workflow_run=False,
    )
    return root.native_windows_packages("msi").https_url


def download_msis(prefix_url: str, packages: list[str], dest_dir: Path) -> list[Path]:
    """Download each package's MSI from the public prefix into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for package in packages:
        filename = expected_msi_filename(package)
        url = f"{prefix_url.rstrip('/')}/{filename}"
        dest = dest_dir / filename
        print(f"Downloading {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.URLError as e:
            raise FileNotFoundError(
                f"could not download MSI for package {package!r} from {url}: {e}. "
                "The build run may not have produced this package, or its MSIs "
                "were not uploaded to the public prefix (fork build runs do not "
                "upload -- see the module docstring)."
            ) from e
        downloaded.append(dest)
    return downloaded


def _collect_local_msis(packages: list[str], msi_dir: Path) -> list[Path]:
    """Return the expected MSI paths from a local directory (local iteration)."""
    found: list[Path] = []
    for package in packages:
        path = msi_dir / expected_msi_filename(package)
        if not path.is_file():
            raise FileNotFoundError(
                f"expected MSI for package {package!r} not found: {path}"
            )
        found.append(path)
    return found


def install_msi(msi_path: Path, log_path: Path) -> None:
    """Install a single MSI silently, raising RuntimeError with the log on failure."""
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
    if result.returncode != 0:
        _print_log_tail(log_path)
        raise RuntimeError(
            f"msiexec failed (exit {result.returncode}) installing {msi_path}; "
            "see the verbose log above."
        )


def _print_log_tail(log_path: Path, lines: int = 40) -> None:
    """Print the tail of an msiexec verbose log to stderr for debugging."""
    try:
        # MSI logs are UTF-16; decode leniently.
        content = log_path.read_text(encoding="utf-16", errors="replace")
    except (OSError, UnicodeError):
        try:
            content = log_path.read_text(errors="replace")
        except OSError:
            print(f"(could not read msiexec log {log_path})", file=sys.stderr)
            return
    tail = "\n".join(content.splitlines()[-lines:])
    print(f"---- msiexec log tail ({log_path}) ----\n{tail}", file=sys.stderr)


def install_dir_for(
    package: str,
    rocm_version: str,
    program_files: Path = DEFAULT_PROGRAM_FILES,
) -> Path:
    """Return the versioned install directory for ``package``."""
    subdir = _format_package_template(PACKAGES[package].install_subdir, rocm_version)
    return program_files / PRODUCT_SUBDIR / subdir


def verify_install_dir(install_dir: Path) -> None:
    """Assert the install directory exists and is non-empty."""
    if not install_dir.is_dir():
        raise RuntimeError(f"install directory not found: {install_dir}")
    if not any(install_dir.iterdir()):
        raise RuntimeError(f"install directory is empty: {install_dir}")
    print(f"[PASS] install directory present: {install_dir}")


def verify_payload_dlls(install_dir: Path, package: str) -> None:
    """Assert the expected runtime DLLs are present somewhere under the tree."""
    missing = [
        pattern
        for pattern in RUNTIME_PAYLOAD_DLL_GLOBS
        if not any(install_dir.rglob(pattern))
    ]
    if missing:
        raise RuntimeError(
            f"payload DLLs missing under {install_dir} for package {package!r}: "
            f"{', '.join(missing)}"
        )
    print(f"[PASS] payload DLLs present under {install_dir}")


def verify_system32_dlls(package: str, system32: Path = DEFAULT_SYSTEM32) -> None:
    """Assert the package's declared legacy System32 DLLs are installed."""
    expected = PACKAGES[package].legacy_system32_dlls
    if not expected:
        print(f"[PASS] package {package!r} declares no System32 DLLs")
        return
    missing = [name for name in expected if not (system32 / name).is_file()]
    if missing:
        raise RuntimeError(
            f"System32 DLLs missing in {system32} for package {package!r}: "
            f"{', '.join(missing)}"
        )
    print(f"[PASS] System32 DLLs present in {system32}")


def verify_registry_key(package: str, rocm_version: str) -> None:
    """Assert the package's HKLM registry key records an install location."""
    import winreg  # Windows-only; imported lazily so the module loads on Linux.

    subkey = _format_package_template(PACKAGES[package].registry_key, rocm_version)
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            subkey,
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as key:
            value, _ = winreg.QueryValueEx(key, "InstallDir")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"registry key not found for package {package!r}: " f"HKLM\\{subkey}"
        ) from e
    except OSError as e:
        raise RuntimeError(
            f"could not read registry value for package {package!r} at "
            f"HKLM\\{subkey}: {e}"
        ) from e
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"registry InstallDir for package {package!r} is empty at HKLM\\{subkey}"
        )
    print(f"[PASS] registry key present: HKLM\\{subkey} (InstallDir={value})")


class WindowsPackageInstallTest:
    """Install the requested MSIs and verify each package's install."""

    def __init__(
        self,
        artifact_run_id: str,
        artifact_github_repo: str,
        release_type: str,
        packages: list[str],
        rocm_version: str,
        msi_dir: Path | None = None,
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
        major_minor(rocm_version)
        self.artifact_run_id = artifact_run_id
        self.artifact_github_repo = artifact_github_repo
        self.release_type = release_type
        self.packages = packages
        self.rocm_version = rocm_version
        self.msi_dir = msi_dir
        self.program_files = program_files
        self.system32 = system32

    def _obtain_msis(self, work_dir: Path) -> list[tuple[str, Path]]:
        """Return (package, msi_path) pairs, from --msi-dir or a public download."""
        if self.msi_dir is not None:
            paths = _collect_local_msis(self.packages, self.msi_dir)
        else:
            prefix_url = resolve_msi_prefix_url(
                self.artifact_run_id,
                self.artifact_github_repo,
                self.release_type,
            )
            paths = download_msis(prefix_url, self.packages, work_dir)
        return list(zip(self.packages, paths))

    def _verify_package(self, package: str) -> None:
        install_dir = install_dir_for(package, self.rocm_version, self.program_files)
        verify_install_dir(install_dir)
        verify_payload_dlls(install_dir, package)
        verify_system32_dlls(package, self.system32)
        verify_registry_key(package, self.rocm_version)

    def run(self) -> int:
        """Install and verify; return 0 on success, 1 on any failure."""
        if sys.platform != "win32":
            print(
                "[ERROR] native_windows_package_install_test requires Windows "
                f"(sys.platform={sys.platform!r}).",
                file=sys.stderr,
            )
            return 2
        try:
            with tempfile.TemporaryDirectory() as tmp:
                work_dir = Path(tmp)
                for package, msi_path in self._obtain_msis(work_dir):
                    install_msi(msi_path, work_dir / f"{package}-install.log")
                for package in self.packages:
                    self._verify_package(package)
        except Exception:  # noqa: BLE001 - top-level reporting boundary
            print("[FAIL] MSI install test failed:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print("[PASS] all MSI packages installed and verified")
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install and verify Windows MSI packages from a build run."
    )
    parser.add_argument(
        "--artifact-run-id",
        required=True,
        help="Run id whose published Windows MSIs to install and verify.",
    )
    parser.add_argument(
        "--artifact-github-repo",
        default=DEFAULT_ARTIFACT_GITHUB_REPO,
        help=f"Repository that owns the run id. Default: {DEFAULT_ARTIFACT_GITHUB_REPO}.",
    )
    parser.add_argument(
        "--release-type",
        default="nightly",
        help="Release type selecting the source artifacts bucket. Default: nightly.",
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
        default=None,
        help="Use already-built MSIs from this directory instead of downloading.",
    )
    parser.add_argument(
        "--program-files",
        type=Path,
        default=DEFAULT_PROGRAM_FILES,
        help=f"Program Files root to verify under. Default: {DEFAULT_PROGRAM_FILES}.",
    )
    return parser


def _split_packages(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


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
    run_id = _env("ARTIFACT_RUN_ID")
    rocm_version = _env("ROCM_VERSION")
    if not run_id or not rocm_version:
        return None
    argv = [
        "--artifact-run-id",
        run_id,
        "--rocm-version",
        rocm_version,
        "--release-type",
        _env("RELEASE_TYPE", "nightly"),
        "--packages",
        _env("PACKAGES", "runtime"),
    ]
    repo = _env("ARTIFACT_GITHUB_REPO")
    if repo:
        argv += ["--artifact-github-repo", repo]
    msi_dir = _env("MSI_DIR")
    if msi_dir:
        argv += ["--msi-dir", msi_dir]
    return argv


def _runner_from_args(args: argparse.Namespace) -> WindowsPackageInstallTest:
    return WindowsPackageInstallTest(
        artifact_run_id=args.artifact_run_id,
        artifact_github_repo=args.artifact_github_repo,
        release_type=args.release_type,
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
                "Missing required environment variables "
                "(ARTIFACT_RUN_ID, ROCM_VERSION)."
            )
        pytest.skip("Set ARTIFACT_RUN_ID / ROCM_VERSION to run this test.")
    args = parse_cli_arguments(argv, raise_instead_of_exit=True)
    rc = _runner_from_args(args).run()
    assert rc == 0, f"install test exited with code {rc}"


def main() -> None:
    args = parse_cli_arguments()
    sys.exit(_runner_from_args(args).run())


if __name__ == "__main__":
    main()
