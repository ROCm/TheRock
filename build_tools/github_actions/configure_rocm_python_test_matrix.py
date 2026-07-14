#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure ROCm Python package test matrix rows for multi-arch CI."""

from dataclasses import dataclass

from amdgpu_family_matrix import select_weighted_label

UBUNTU_24_04_CONTAINER = (
    "ghcr.io/rocm/no_rocm_image_ubuntu24_04@"
    "sha256:405945a40deaff9db90b9839c0f41d4cba4a383c1a7459b28627047bf6302a26"
)
UBI_10_CONTAINER = (
    "ghcr.io/rocm/no_rocm_image_ubi10@"
    "sha256:a10f34d6006a20d02cf688982de9dea147710927ed405a3b0d5c73b58a6030c0"
)


@dataclass(frozen=True)
class PythonTestEnvironment:
    python_version: str
    container_image_name: str
    container_image_url: str


LINUX_TEST_ENVIRONMENTS = [
    PythonTestEnvironment(
        python_version=python_version,
        container_image_name=container_image_name,
        container_image_url=container_image_url,
    )
    for python_version in ["3.10", "3.11", "3.12"]
    for container_image_name, container_image_url in [
        ("ubuntu24.04", UBUNTU_24_04_CONTAINER),
        ("ubi10", UBI_10_CONTAINER),
    ]
]

WINDOWS_TEST_ENVIRONMENTS = [
    PythonTestEnvironment(
        python_version="3.12",
        container_image_name="native",
        container_image_url="",
    )
]


def _test_environments_for_platform(platform: str) -> list[PythonTestEnvironment]:
    if platform == "linux":
        return LINUX_TEST_ENVIRONMENTS
    if platform == "windows":
        return WINDOWS_TEST_ENVIRONMENTS
    raise ValueError(f"Unknown platform: {platform}")


def _select_test_runs_on(family_info: dict) -> str:
    test_runs_on = str(family_info["test-runs-on"])
    if not test_runs_on:
        return ""

    labels_config = family_info.get("test-runs-on-labels")
    if labels_config:
        return select_weighted_label(
            labels_config=labels_config,
            context_name=f"python-test-runner ({family_info['amdgpu_family']})",
        )
    return test_runs_on


def build_rocm_python_test_matrix(
    *,
    per_family_info: list[dict],
    platform: str,
) -> list[dict[str, str]]:
    """Build one test row per runnable family and test environment."""
    test_environments = _test_environments_for_platform(platform)
    matrix: list[dict[str, str]] = []
    for family_info in per_family_info:
        amdgpu_family = str(family_info["amdgpu_family"])
        if not family_info["test-runs-on"]:
            continue

        # Example Linux output, note the Python versions, containers, and test
        # runners):
        #
        # [
        #   {
        #     "amdgpu_family": "gfx94X-dcgpu",
        #     "test_runs_on": "linux-gfx942-1gpu-ccs-csp-ossci-rocm",
        #     "python_version": "3.10",
        #     "container_image_name": "ubuntu24.04",
        #     "container_image_url": "ghcr.io/rocm/no_rocm_image_ubuntu24_04@sha256:..."
        #   },
        #   {
        #     "amdgpu_family": "gfx94X-dcgpu",
        #     "test_runs_on": "linux-gfx942-1gpu-ccs-ossci-rocm",
        #     "python_version": "3.10",
        #     "container_image_name": "ubi10",
        #     "container_image_url": "ghcr.io/rocm/no_rocm_image_ubi10@sha256:..."
        #   },
        #   ...
        #   {
        #     "amdgpu_family": "gfx94X-dcgpu",
        #     "test_runs_on": "linux-gfx942-1gpu-ccs-csp-ossci-rocm",
        #     "python_version": "3.12",
        #     "container_image_name": "ubi10",
        #     "container_image_url": "ghcr.io/rocm/no_rocm_image_ubi10@sha256:..."
        #   }
        # ]
        for environment in test_environments:
            test_runs_on = _select_test_runs_on(family_info)
            # These row keys are the contract with workflow files which use them
            # via matrix.<key> expressions. Empty values are allowed when the
            # workflow handles them explicitly, but undefined keys are not.
            matrix.append(
                {
                    "amdgpu_family": amdgpu_family,
                    "test_runs_on": test_runs_on,
                    "python_version": environment.python_version,
                    "container_image_name": environment.container_image_name,
                    # container_image_url is intentionally empty for native
                    # Windows test environments and non-empty for Linux
                    # container test environments.
                    "container_image_url": environment.container_image_url,
                }
            )
    return matrix
