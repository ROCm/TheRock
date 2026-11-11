#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Limit memory-heavy builds using the available host/container resources.

CSAN kernel links have exceeded 19 GiB each in CI. Budget 32 GiB per job
and leave half the available memory for the build driver and other processes.
Use with THEROCK_BACKGROUND_BUILD_JOBS=1 to avoid concurrent nested builds.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import os
from pathlib import Path
import sys

GIB = 1024**3
MEMORY_PER_JOB = 32 * GIB
# Linux cgroup v1 represents an unlimited limit with a value near LONG_MAX.
UNLIMITED_MEMORY = 1 << 60


@dataclass(frozen=True)
class Resources:
    available_memory_bytes: int | None
    cpus: int


def cgroup_memory_available(limit: str | None, current: str | None) -> int | None:
    """Return finite cgroup headroom; missing usage fails conservatively."""
    try:
        limit_bytes = int(limit)
    except (TypeError, ValueError):
        return None  # Includes cgroup v2's "max" (unlimited).
    if limit_bytes < 0 or limit_bytes >= UNLIMITED_MEMORY:
        return None
    try:
        current_bytes = int(current)
    except (TypeError, ValueError):
        return 0
    return max(0, limit_bytes - max(0, current_bytes))


def available_memory(meminfo: str, cgroup_available: int | None) -> int | None:
    """Combine Linux MemAvailable (KiB) with optional container headroom."""
    limits = [] if cgroup_available is None else [max(0, cgroup_available)]
    for line in meminfo.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == "MemAvailable:" and fields[2] == "kB":
            try:
                limits.append(max(0, int(fields[1])) * 1024)
            except ValueError:
                pass
    return min(limits) if limits else None


def available_cpus(cpus: int | None, quota: str | None, period: str | None) -> int:
    """Cap the affinity CPU count by finite cgroup CPU quota (whole cores)."""
    result = max(1, cpus or 1)
    try:
        quota_value, period_value = int(quota), int(period)
    except (TypeError, ValueError):
        return result
    if quota_value > 0 and period_value > 0:
        result = min(result, max(1, quota_value // period_value))
    return result


def choose_build_jobs(resources: Resources) -> int:
    """Reserve half of memory, cap by CPUs, and always allow one job."""
    if resources.available_memory_bytes is None:
        return 1
    memory_jobs = max(0, resources.available_memory_bytes) // (2 * MEMORY_PER_JOB)
    return max(1, min(max(1, resources.cpus), memory_jobs))


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def detect_resources(
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Resources:
    """Read cgroup v2, falling back to the conventional v1 controller paths."""
    memory_limit = _read(cgroup_root / "memory.max")
    if memory_limit is not None:
        memory_current = _read(cgroup_root / "memory.current")
    else:
        memory_limit = _read(cgroup_root / "memory/memory.limit_in_bytes")
        memory_current = _read(cgroup_root / "memory/memory.usage_in_bytes")
    memory = available_memory(
        _read(proc_root / "meminfo") or "",
        cgroup_memory_available(memory_limit, memory_current),
    )
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count()
    cpu_max = (_read(cgroup_root / "cpu.max") or "").split()
    if len(cpu_max) == 2:
        quota, period = cpu_max
    else:
        cpu_dir = cgroup_root / "cpu"
        if not cpu_dir.exists():
            cpu_dir = cgroup_root / "cpu,cpuacct"
        quota = _read(cpu_dir / "cpu.cfs_quota_us")
        period = _read(cpu_dir / "cpu.cfs_period_us")
    return Resources(memory, available_cpus(cpus, quota, period))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--github-env",
        action="store_true",
        help="Set CMAKE_BUILD_PARALLEL_LEVEL for subsequent GitHub Actions steps",
    )
    args = parser.parse_args(argv)
    resources = detect_resources()
    jobs = choose_build_jobs(resources)
    memory = resources.available_memory_bytes
    memory_label = "unknown" if memory is None else f"{memory / GIB:.1f} GiB"
    print(
        f"Available memory: {memory_label}; CPUs: {resources.cpus}; "
        f"build jobs: {jobs} (half memory reserved, 32 GiB/job)",
        file=sys.stderr,
    )
    if args.github_env:
        from github_actions.github_actions_api import gha_set_env

        with redirect_stdout(sys.stderr):
            gha_set_env({"CMAKE_BUILD_PARALLEL_LEVEL": str(jobs)})
    else:
        print(jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
