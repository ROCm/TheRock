# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared policy for PyTorch wheel test coverage levels."""

PYTORCH_TEST_LEVELS = ["sanity", "standard", "full"]
PYTORCH_GPU_TEST_PYTHON_VERSION = "3.12"
PYTORCH_FULL_TEST_PLATFORM = "linux"
PYTORCH_FULL_TEST_AMDGPU_FAMILY = "gfx94X-dcgpu"


def is_full_test_eligible(
    *,
    python_version: str,
    platform: str,
    amdgpu_families: list[str],
) -> bool:
    families_lower = {family.lower() for family in amdgpu_families}
    return (
        python_version == PYTORCH_GPU_TEST_PYTHON_VERSION
        and platform == PYTORCH_FULL_TEST_PLATFORM
        and PYTORCH_FULL_TEST_AMDGPU_FAMILY.lower() in families_lower
    )


def select_release_test_level(
    *,
    python_version: str,
    platform: str,
    amdgpu_families: list[str],
    run_full_pytorch_tests: bool,
) -> str:
    """Select the test level for one scheduled release matrix row."""
    if python_version != PYTORCH_GPU_TEST_PYTHON_VERSION:
        return "sanity"
    if run_full_pytorch_tests and is_full_test_eligible(
        python_version=python_version,
        platform=platform,
        amdgpu_families=amdgpu_families,
    ):
        return "full"
    return "standard"


def resolve_requested_test_level(
    *,
    requested_test_level: str,
    python_version: str,
    platform: str,
    amdgpu_families: list[str],
) -> str:
    """Resolve a direct build's requested level under full-test constraints."""
    if requested_test_level not in PYTORCH_TEST_LEVELS:
        raise ValueError(f"Unknown PyTorch test level: {requested_test_level!r}")
    if requested_test_level != "full":
        return requested_test_level
    if is_full_test_eligible(
        python_version=python_version,
        platform=platform,
        amdgpu_families=amdgpu_families,
    ):
        return "full"

    print(
        "Full PyTorch tests require "
        f"{PYTORCH_FULL_TEST_PLATFORM}, Python "
        f"{PYTORCH_GPU_TEST_PYTHON_VERSION}, and "
        f"{PYTORCH_FULL_TEST_AMDGPU_FAMILY}; using standard tests"
    )
    return "standard"
