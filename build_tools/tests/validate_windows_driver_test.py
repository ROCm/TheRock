# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from validate_windows_driver import validate_artifacts


ARTIFACT_PATHS = (
    "compiler/amd-comgr/stage/bin/amd_comgr.dll",
    "compiler/amd-comgr/stage/bin/amd_comgr.pdb",
    "core/clr/stage/bin/amdhip64_7.dll",
    "core/clr/stage/bin/hiprtc0700.dll",
    "core/clr/stage/bin/hiprtc-builtins0700.dll",
    "core/ocl-clr/stage/bin/amdocl64.dll",
    "core/ocl-icd/stage/bin/OpenCL.dll",
    "core/rocm-kpack/stage/bin/rocm_kpack.dll",
)


@pytest.fixture
def staged_build(tmp_path: Path) -> Path:
    for relative_path in ARTIFACT_PATHS:
        artifact = tmp_path / relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"driver artifact fixture")
    return tmp_path


def test_complete_driver_artifacts(staged_build: Path) -> None:
    validate_artifacts(staged_build)


@pytest.mark.parametrize("relative_path", ARTIFACT_PATHS)
def test_missing_driver_artifact(staged_build: Path, relative_path: str) -> None:
    (staged_build / relative_path).unlink()
    with pytest.raises(ValueError, match="Expected one driver artifact"):
        validate_artifacts(staged_build)


@pytest.mark.parametrize("relative_path", ARTIFACT_PATHS)
def test_empty_driver_artifact(staged_build: Path, relative_path: str) -> None:
    (staged_build / relative_path).write_bytes(b"")
    with pytest.raises(ValueError, match="not a nonempty file"):
        validate_artifacts(staged_build)


def test_directory_is_not_an_artifact(staged_build: Path) -> None:
    artifact = staged_build / ARTIFACT_PATHS[0]
    artifact.unlink()
    artifact.mkdir()
    with pytest.raises(ValueError, match="not a nonempty file"):
        validate_artifacts(staged_build)


def test_stale_runtime_version_is_rejected(staged_build: Path) -> None:
    (staged_build / "core/clr/stage/bin/amdhip64_6.dll").write_bytes(b"stale DLL")
    with pytest.raises(ValueError, match="found 2"):
        validate_artifacts(staged_build)


def test_missing_build_tree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="found 0"):
        validate_artifacts(tmp_path / "missing-build")
