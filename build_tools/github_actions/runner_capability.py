# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Preflight checks for capability-pinned CI test jobs."""

from __future__ import annotations

from pathlib import Path


def parse_dot_version(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.strip().split("."))


def read_amdgpu_driver_version() -> str | None:
    path = Path("/sys/module/amdgpu/version")
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def amdgpu_driver_meets_min(min_version: str) -> bool:
    current = read_amdgpu_driver_version()
    if not current:
        return False
    return parse_dot_version(current) >= parse_dot_version(min_version)


def check_runner_requirements(requirements: dict) -> None:
    min_driver = requirements.get("amdgpu_driver_min")
    if not min_driver:
        return
    if amdgpu_driver_meets_min(min_driver):
        return
    current = read_amdgpu_driver_version() or "unknown"
    raise SystemExit(
        "Runner capability check failed: "
        f"amdgpu driver {current} < required {min_driver}. "
        "This job must run on a pinned runner with the correct driver."
    )
