#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Install and test TensileLite wheels from reconstructed ROCm artifacts."""

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

import pytest_runner

logging.basicConfig(level=logging.INFO, format="%(message)s")

CANONICAL_PROJECT = canonicalize_name("tensilelite")
COMPATIBILITY_PROJECT = canonicalize_name("tensilelite_tensile_compat")
EXPECTED_PROJECTS = {CANONICAL_PROJECT, COMPATIBILITY_PROJECT}


class TensileLiteRunnerError(RuntimeError):
    """An artifact contract failure with a user-facing diagnostic."""


@dataclass(frozen=True)
class ReleaseWheels:
    canonical: Path
    compatibility: Path


def discover_release_wheels(wheels_dir: Path) -> ReleaseWheels:
    """Select exactly one canonical and compatibility wheel with matching versions."""
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
    selected_versions: dict[str, Version] = {}
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
        selected[project_name] = wheel_path
        selected_versions[project_name] = wheel_version

    if selected_versions[CANONICAL_PROJECT] != selected_versions[COMPATIBILITY_PROJECT]:
        raise TensileLiteRunnerError(
            "Canonical and compatibility wheel versions do not match: "
            f"{selected_versions[CANONICAL_PROJECT]} != "
            f"{selected_versions[COMPATIBILITY_PROJECT]}"
        )

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


def verify_client_free_package_help(wheel_path: Path, env: dict[str, str]) -> None:
    """Prove the installed canonical wheel imports and prints help without a client."""
    try:
        _project, version, _build, _tags = parse_wheel_filename(wheel_path.name)
    except ValueError as exc:
        raise TensileLiteRunnerError(
            f"Malformed canonical wheel filename: {wheel_path}"
        ) from exc
    if not version.local or not version.local.startswith("rocm"):
        raise TensileLiteRunnerError(
            f"Canonical wheel has no ROCm local version tag: {wheel_path}"
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        rocm_root = temporary_root / "rocm"
        (rocm_root / ".info").mkdir(parents=True)
        (rocm_root / ".info" / "version").write_text(
            version.local[len("rocm") :] + "\n", encoding="utf-8"
        )
        sdk_module = temporary_root / "rocm_sdk"
        sdk_module.mkdir()
        (sdk_module / "__main__.py").write_text(
            "import sys\n"
            f"ROOT = {str(rocm_root)!r}\n"
            "if sys.argv[1:] == ['path', '--root']:\n"
            "    print(ROOT)\n"
            "else:\n"
            "    raise SystemExit(2)\n",
            encoding="utf-8",
        )

        help_env = dict(env)
        help_env["ROCM_PATH"] = str(rocm_root)
        help_env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(temporary_root), env.get("PYTHONPATH", "")])
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "tensilelite", "--help"],
                capture_output=True,
                text=True,
                check=False,
                env=help_env,
            )
        except OSError as exc:
            raise TensileLiteRunnerError(
                f"Failed to launch client-free TensileLite help: {exc}"
            ) from exc

    if result.returncode != 0:
        raise TensileLiteRunnerError(
            "Client-free TensileLite help failed with status "
            f"{result.returncode}: {result.stderr!r}"
        )
    if "create-library" not in result.stdout:
        raise TensileLiteRunnerError(
            "Client-free TensileLite help did not describe the package commands"
        )


def run_test_phases(
    rocm_path: Path,
    test_type: str,
    amdgpu_families: str | None,
    env: dict[str, str],
) -> int:
    """Install canonical then compatibility wheels and run their test phases."""
    component_root = pytest_runner.resolve_component_path("tensilelite", rocm_path)
    wheels = discover_release_wheels(component_root / "wheels")

    canonical_install_status = install_wheel(wheels.canonical, env)
    if canonical_install_status != 0:
        return canonical_install_status
    verify_client_free_package_help(wheels.canonical, env)
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
