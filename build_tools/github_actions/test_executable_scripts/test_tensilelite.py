#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Install and test TensileLite wheels from reconstructed ROCm artifacts."""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

import pytest_runner

logging.basicConfig(level=logging.INFO, format="%(message)s")

CANONICAL_PROJECT = canonicalize_name("tensilelite")
COMPATIBILITY_PROJECT = canonicalize_name("tensilelite_tensile_compat")
EXPECTED_PROJECTS = {CANONICAL_PROJECT, COMPATIBILITY_PROJECT}
CLIENT_VERSION_TIMEOUT_SECONDS = 5


class TensileLiteRunnerError(RuntimeError):
    """An artifact contract failure with a user-facing diagnostic."""


@dataclass(frozen=True)
class ReleaseWheels:
    canonical: Path
    compatibility: Path


def is_loader_failure(stderr: str) -> bool:
    """Recognize native loader diagnostics emitted before client main()."""
    stderr_lower = stderr.lower()
    loader_fragments = (
        "error while loading shared libraries",
        "cannot open shared object file",
        "library not loaded",
        "dll load failed",
        "the code execution cannot proceed",
    )
    return any(fragment in stderr_lower for fragment in loader_fragments)


def query_client_version(client_path: Path, env: dict[str, str]) -> Version:
    """Return the native client's canonical PEP 440 version."""
    command = [str(client_path), "--version"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CLIENT_VERSION_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise TensileLiteRunnerError(
            f"Failed to launch TensileLite client; file not found: {client_path}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TensileLiteRunnerError(
            "TensileLite client --version timed out after "
            f"{CLIENT_VERSION_TIMEOUT_SECONDS} seconds: {client_path}"
        ) from exc
    except OSError as exc:
        raise TensileLiteRunnerError(
            f"Failed to load or launch TensileLite client {client_path}: {exc}"
        ) from exc

    if result.returncode < 0:
        raise TensileLiteRunnerError(
            "TensileLite client --version terminated by signal "
            f"{-result.returncode}: {client_path}"
        )
    if result.returncode != 0 and is_loader_failure(result.stderr):
        raise TensileLiteRunnerError(
            f"Failed to load TensileLite client {client_path}: {result.stderr!r}"
        )
    if result.returncode != 0:
        raise TensileLiteRunnerError(
            "TensileLite client --version exited with status "
            f"{result.returncode}: {client_path}"
        )
    if result.stderr:
        raise TensileLiteRunnerError(
            "TensileLite client --version wrote unexpected stderr: "
            f"{result.stderr!r}"
        )

    lines = result.stdout.splitlines()
    if not lines or (len(lines) == 1 and not lines[0]):
        raise TensileLiteRunnerError(
            "TensileLite client --version produced no version line"
        )
    if len(lines) != 1:
        raise TensileLiteRunnerError(
            "TensileLite client --version must produce exactly one stdout line; "
            f"got {len(lines)}: {result.stdout!r}"
        )
    version_text = lines[0]
    if version_text != version_text.strip():
        raise TensileLiteRunnerError(
            "TensileLite client --version produced a non-canonical line: "
            f"{version_text!r}"
        )
    try:
        return Version(version_text)
    except InvalidVersion as exc:
        raise TensileLiteRunnerError(
            "TensileLite client --version produced malformed PEP 440 version: "
            f"{version_text!r}"
        ) from exc


def discover_release_wheels(
    wheels_dir: Path, expected_version: Version
) -> ReleaseWheels:
    """Select exactly one canonical and compatibility wheel for the client."""
    wheels_by_project: dict[str, list[tuple[Path, Version]]] = {
        project: [] for project in EXPECTED_PROJECTS
    }
    for wheel_path in sorted(wheels_dir.glob("*.whl")):
        try:
            project_name, version, _build, _tags = parse_wheel_filename(
                wheel_path.name
            )
        except ValueError as exc:
            raise TensileLiteRunnerError(
                f"Malformed wheel filename in TensileLite artifact: {wheel_path}"
            ) from exc
        project_name = canonicalize_name(project_name)
        if project_name not in EXPECTED_PROJECTS:
            raise TensileLiteRunnerError(
                "Unexpected wheel project in TensileLite artifact: "
                f"{project_name} ({wheel_path})"
            )
        wheels_by_project[project_name].append((wheel_path, version))

    selected: dict[str, Path] = {}
    for project_name, candidates in wheels_by_project.items():
        if not candidates:
            raise TensileLiteRunnerError(
                f"No {project_name} wheel found in {wheels_dir}"
            )
        if len(candidates) != 1:
            raise TensileLiteRunnerError(
                f"Expected exactly one {project_name} wheel in {wheels_dir}; "
                f"found {[path.name for path, _version in candidates]}"
            )
        wheel_path, wheel_version = candidates[0]
        if wheel_version != expected_version:
            raise TensileLiteRunnerError(
                f"{project_name} wheel version {wheel_version} does not match "
                f"TensileLite client version {expected_version}: {wheel_path}"
            )
        selected[project_name] = wheel_path

    return ReleaseWheels(
        canonical=selected[CANONICAL_PROJECT],
        compatibility=selected[COMPATIBILITY_PROJECT],
    )


def install_wheel(wheel_path: Path, env: dict[str, str]) -> int:
    """Force-install one exact artifact wheel with the active interpreter."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(wheel_path),
    ]
    logging.info("Installing artifact wheel: %s", wheel_path)
    try:
        return subprocess.run(command, check=False, env=env).returncode
    except OSError as exc:
        raise TensileLiteRunnerError(
            f"Failed to launch pip for artifact wheel {wheel_path}: {exc}"
        ) from exc


def run_test_phases(
    rocm_path: Path,
    test_type: str,
    amdgpu_families: str | None,
    env: dict[str, str],
) -> int:
    """Install canonical then compatibility wheels and run their test phases."""
    component_root = pytest_runner.resolve_component_path("tensilelite", rocm_path)
    client_name = (
        "tensilelite-client.exe" if os.name == "nt" else "tensilelite-client"
    )
    client_path = rocm_path / "libexec" / "hipblaslt" / "tensilelite" / client_name

    client_version = query_client_version(client_path, env)
    wheels = discover_release_wheels(component_root / "wheels", client_version)

    canonical_install_status = install_wheel(wheels.canonical, env)
    if canonical_install_status != 0:
        return canonical_install_status
    canonical_status = pytest_runner.run_phase(
        "tensilelite",
        test_type,
        amdgpu_families,
        rocm_path,
        env,
    )
    if canonical_status != 0:
        return canonical_status

    compatibility_install_status = install_wheel(wheels.compatibility, env)
    if compatibility_install_status != 0:
        return compatibility_install_status
    return pytest_runner.run_phase(
        "tensilelite",
        None,
        amdgpu_families,
        rocm_path,
        env,
        test_paths_override=["compat/tests"],
        marker_expression_override="",
        pytest_args_override=["--run-compat"],
    )


def main() -> int:
    if os.getenv("TEST_COMPONENT") != "tensilelite":
        logging.error("TEST_COMPONENT must be set to 'tensilelite'")
        return 1
    therock_bin_dir = os.getenv("THEROCK_BIN_DIR")
    if not therock_bin_dir:
        logging.error("THEROCK_BIN_DIR environment variable is required but not set.")
        return 1

    rocm_path = Path(therock_bin_dir).resolve().parent
    env = pytest_runner.build_environment(rocm_path, "tensilelite")
    try:
        return run_test_phases(
            rocm_path=rocm_path,
            test_type=os.getenv("TEST_TYPE", "quick"),
            amdgpu_families=os.getenv("AMDGPU_FAMILIES"),
            env=env,
        )
    except TensileLiteRunnerError as exc:
        logging.error("TensileLite artifact test setup failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
