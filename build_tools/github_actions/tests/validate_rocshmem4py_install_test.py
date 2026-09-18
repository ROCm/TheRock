# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the rocshmem4py CI wheel validator."""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(
    0,
    str(Path(__file__).parent.parent),
)

import validate_rocshmem4py_install as validator
from validate_rocshmem4py_install import build_install_command, validate_metadata


def test_gpu_smoke_subcommand_starts_without_site_packages():
    validator_path = Path(__file__).parent.parent / "validate_rocshmem4py_install.py"
    subprocess.run(
        [sys.executable, "-S", str(validator_path), "gpu-smoke-test", "--help"],
        check=True,
    )


def test_validation_requirements_install_into_scratch_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dependency_dir = tmp_path / "validation-deps"
    commands = []
    monkeypatch.setattr(validator, "_VALIDATION_DEPS_DIR", dependency_dir)
    monkeypatch.setattr(validator, "run_with_retries", commands.append)

    validator.install_validation_requirements()

    assert len(commands) == 1
    command = commands[0]
    assert command[:6] == [
        sys.executable,
        "-I",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    assert command[command.index("--target") + 1] == str(dependency_dir)
    assert "--no-cache-dir" in command
    assert "--only-binary=:all:" in command
    assert "packaging==25.0" in command


def test_install_command_pins_rocm_build_and_disables_cache(tmp_path: Path):
    rocm_version = "10.1.0.dev0+abc123"
    command = build_install_command(
        venv_dir=tmp_path / "venv",
        rocm_version=rocm_version,
        index_url="",
        find_links="https://example.test/index.html",
    )

    assert command[:7] == [
        sys.executable,
        "-I",
        "-m",
        "pip",
        "--python",
        str(tmp_path / "venv"),
        "install",
    ]
    assert "--no-cache-dir" in command
    assert "--only-binary=:all:" in command
    assert "--no-index" in command
    assert "rocshmem4py" in command
    assert f"rocm-sdk-core=={rocm_version}" in command


@pytest.mark.parametrize(
    ("rocm_version", "rocshmem4py_version"),
    [
        (
            "10.1.0.dev0+abc123",
            "0.1.0+devrocm10.1.0.dev0.abc123",
        ),
        (
            "7.10.0+dev1",
            "0.1.0+devrocm7.10.0.dev1",
        ),
    ],
)
def test_metadata_matches_expected_rocm_build(
    tmp_path: Path,
    rocm_version: str,
    rocshmem4py_version: str,
):
    rocshmem4py_metadata = tmp_path / "rocshmem4py-0.1.0.dist-info" / "METADATA"
    rocshmem4py_metadata.parent.mkdir()
    rocshmem4py_metadata.write_text(
        "Metadata-Version: 2.4\n"
        "Name: rocshmem4py\n"
        f"Version: {rocshmem4py_version}\n"
        f"Requires-Dist: rocm-sdk-core=={rocm_version}\n"
        "\n"
    )
    core_metadata = tmp_path / "rocm_sdk_core-10.1.0.dist-info" / "METADATA"
    core_metadata.parent.mkdir()
    core_metadata.write_text(
        "Metadata-Version: 2.4\n"
        "Name: rocm-sdk-core\n"
        f"Version: {rocm_version}\n"
        "\n"
    )

    validate_metadata(tmp_path, rocm_version)
