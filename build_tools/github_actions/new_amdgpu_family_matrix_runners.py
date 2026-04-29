"""
GitHub Actions runner inventory and selection for the AMD GPU family matrix.

Two flat inventories live here:

  _TEST_RUNNER_INVENTORY        — GPU test runners, keyed by (platform, target, role).
                             Roles include "test", "test_multi_gpu", "benchmark",
                             plus arbitrary extra-role keys (e.g. "oem",
                             "test-sandbox") forwarded into GpuRunners.extra.
  _BUILD_RUNNER_INVENTORY  — build runners, keyed by (platform, variant pool).
                             Variant pool is "default" or "sanitizer"; sanitizer
                             builds (asan/tsan) need heavy-ramdisk runners.

Rows sharing the same key form a weighted pool. `_select_weighted_label` does
a single cumulative-weight random pick. Weights are relative; they need not sum
to 1.0. Solo entries are always returned regardless of weight.

Eager resolution: MatrixEntry.__post_init__ calls `_get_gpu_runners(...)` and
`_get_build_runner(..., "release")` once at construction time to seed each entry.
The lookup methods on AmdGpuFamilyMatrix re-bind BuildConfig.runs_on when a
non-release build_variant is requested.

Test injection: every selection function accepts an optional `inventory=` kwarg
to swap in a synthetic inventory.
"""

import random
from dataclasses import dataclass
from typing import Literal, TypeVar

from new_amdgpu_family_matrix_types import GpuRunners


##########################################################################################
# GPU runner inventory: flat list of every (platform, target, role) -> runner label.
#
# Each row is one concrete runner. Roles "test", "test_multi_gpu", "benchmark" map
# to the named fields on GpuRunners; any other role string (e.g. "oem",
# "test-sandbox") is forwarded as-is into GpuRunners.extra.
#
# When several rows share the same (platform, target, role), they form a weighted
# pool — `_get_gpu_runners(...)` performs one weighted random pick per role at module
# import time. Comment the `weight` value inline with its source-pool count where
# helpful (mirrors upstream amdgpu_family_matrix.py).
#
# To add or change a runner label, edit the matching row here. Targets without
# any entry on a platform get an empty GpuRunners (test job will not run).
##########################################################################################


@dataclass(frozen=True)
class _GpuRunnerEntry:
    """One row in the GPU runner inventory: a runner label for a given
    (platform, target, role). Rows sharing the same key form a weighted pool;
    `_get_gpu_runners(...)` does a single weighted random pick. Module-private
    — consumers go through `_get_gpu_runners(...)`."""

    platform: str
    """'linux' or 'windows'."""
    target: str
    """gfx target name, e.g. 'gfx942', 'gfx1151'."""
    role: str
    """'test', 'test_multi_gpu', 'benchmark', or any extra-role key (e.g. 'oem',
    'test-sandbox'). Extra keys are forwarded as-is into GpuRunners.extra."""
    label: str
    """The runner label string."""
    weight: float = 1.0
    """Relative weight; only meaningful when multiple rows share
    (platform, target, role). A solo row's weight is ignored."""


_TEST_RUNNER_INVENTORY: list[_GpuRunnerEntry] = [
    # gfx90a
    _GpuRunnerEntry(platform="linux", target="gfx90a", role="test", label="linux-gfx90a-gpu-rocm"),
    #
    # gfx94X (mi325) — split across 3 pools (vultr / cirrascale / core42)
    # 1-GPU distribution: 17N (vultr) + 4N (cirrascale) + 8N (core42)
    _GpuRunnerEntry(platform="linux", target="gfx942", role="test",
        label="linux-gfx942-1gpu-ossci-rocm",        weight=17),  # vultr (17/29)
    _GpuRunnerEntry(platform="linux", target="gfx942", role="test",
        label="linux-gfx942-1gpu-ccs-ossci-rocm",    weight=4),  # cirrascale (4/29)
    _GpuRunnerEntry(platform="linux", target="gfx942", role="test",
        label="linux-gfx942-1gpu-core42-ossci-rocm", weight=8),  # core42 (8/29)
    # 8-GPU distribution: 11N (cirrascale) + 7N (core42)
    _GpuRunnerEntry(platform="linux", target="gfx942", role="test_multi_gpu",
        label="linux-gfx942-8gpu-ossci-rocm",        weight=11),  # cirrascale (11/18)
    _GpuRunnerEntry(platform="linux", target="gfx942", role="test_multi_gpu",
        label="linux-gfx942-8gpu-core42-ossci-rocm", weight=7),  # core42 (7/18)
    # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
    _GpuRunnerEntry(platform="linux", target="gfx942", role="benchmark",
        label="linux-gfx942-8gpu-ossci-rocm"),
    # TODO(#3433): Remove sandbox label once ASAN tests are passing
    _GpuRunnerEntry(platform="linux", target="gfx942", role="test-sandbox",
        label="rocm-asan-mi325-sandbox"),
    #
    # gfx950 (mi355)
    _GpuRunnerEntry(platform="linux", target="gfx950", role="test", label="linux-gfx950-1gpu-ccs-ossci-rocm"),
    _GpuRunnerEntry(platform="linux", target="gfx950", role="test_multi_gpu", label="linux-gfx950-8gpu-ccs-ossci-rocm"),
    # gfx1030
    _GpuRunnerEntry(platform="linux", target="gfx1030", role="test", label="linux-gfx1030-gpu-rocm"),
    _GpuRunnerEntry(platform="windows", target="gfx1030", role="test", label="windows-gfx1030-gpu-rocm"),
    # gfx1101 (gfx110X family default)
    _GpuRunnerEntry(platform="linux", target="gfx1101", role="test", label="linux-gfx110X-gpu-rocm"),
    _GpuRunnerEntry(platform="windows", target="gfx1101", role="test", label="windows-gfx110X-gpu-rocm"),
    # gfx1150
    _GpuRunnerEntry(platform="linux", target="gfx1150", role="test", label="linux-gfx1150-gpu-rocm"),
    # gfx1151 (strix halo)
    _GpuRunnerEntry(platform="linux", target="gfx1151", role="test", label="linux-gfx1151-gpu-rocm"),
    _GpuRunnerEntry(platform="linux", target="gfx1151", role="oem", label="linux-strix-halo-gpu-rocm-oem"),
    _GpuRunnerEntry(platform="windows", target="gfx1151", role="test", label="windows-gfx1151-gpu-rocm"),
    # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
    _GpuRunnerEntry(platform="windows", target="gfx1151", role="benchmark", label="windows-gfx1151-gpu-rocm"),
    # gfx1153
    _GpuRunnerEntry(platform="linux", target="gfx1153", role="test", label="linux-gfx1153-gpu-rocm"),
    # gfx1201 (gfx120X family default)
    _GpuRunnerEntry(platform="linux", target="gfx1201", role="test", label="linux-gfx120X-gpu-rocm"),
    _GpuRunnerEntry(platform="windows", target="gfx1201", role="test", label="windows-gfx120X-gpu-rocm"),
]


##########################################################################################
# Build runner inventory: weighted pool of build runners per (platform, variant).
#
# Build runners are normally CPU-only and only dependent on the platform and build_variant.
# For example, sanitizer builds (asan / tsan) need the heavy-ramdisk runners (see #4899)
##########################################################################################


_BuildVariantPool = Literal["default", "sanitizer"]


@dataclass(frozen=True)
class _BuildRunnerEntry:
    """One row in the build runner inventory: a label for a given
    (platform, variant). `variant` is "default" or "sanitizer" — sanitizer
    routes asan/tsan builds to the heavy-ramdisk pool. Rows sharing the same
    key form a weighted pool. Module-private — consumers go through
    `_get_build_runner(...)`."""

    platform: str
    """'linux' or 'windows'."""
    variant: _BuildVariantPool
    """'default' or 'sanitizer' — selects the runner pool."""
    label: str
    """The runner label string."""
    weight: float = 1.0
    """Relative weight; ignored when only one row matches a (platform, variant)."""


# Mirrors upstream amdgpu_family_matrix.py BUILD_RUNNER_LABELS (PR #4899).
# Linux default split: 90% Azure / 10% AWS during the AWS migration ramp.
# Linux sanitizer: heavy-ramdisk runners only (Azure 100% — no AWS variant yet).
# Windows: single Azure pool; the workflow input is reserved for future use.
_BUILD_RUNNER_INVENTORY: list[_BuildRunnerEntry] = [
    _BuildRunnerEntry(platform="linux",   variant="default",   label="azure-linux-scale-rocm",                weight=90),
    _BuildRunnerEntry(platform="linux",   variant="default",   label="aws-linux-scale-rocm",                  weight=10),
    _BuildRunnerEntry(platform="linux",   variant="sanitizer", label="azure-linux-scale-rocm-heavy-ramdisk",  weight=1),
    _BuildRunnerEntry(platform="windows", variant="default",   label="azure-windows-scale-rocm",              weight=1),
]


##########################################################################################
# Shared selection helper: weighted random pick over a single pool of inventory rows.
##########################################################################################


_RunnerEntry = TypeVar("_RunnerEntry", _GpuRunnerEntry, _BuildRunnerEntry)


def _select_weighted_label(entries: list[_RunnerEntry]) -> str:
    """Weighted random pick across a non-empty pool. Solo entries are returned
    as-is regardless of weight; weights are relative and need not sum to 1.0."""
    if len(entries) == 1:
        return entries[0].label
    return random.choices(entries, weights=[e.weight for e in entries], k=1)[0].label


##########################################################################################
# GPU runner resolution: pick concrete labels per (platform, target) into a GpuRunners.
##########################################################################################


# Roles that map to the named fields on GpuRunners. Anything else goes into `extra`.
_NAMED_ROLES = {"test", "test_multi_gpu", "benchmark"}


def _get_gpu_runners(
    platform: str,
    target: str,
    *,
    inventory: list[_GpuRunnerEntry] | None = None,
) -> GpuRunners:
    """Resolve all roles for (platform, target) into a GpuRunners. Named roles
    fill their respective fields; everything else lands in `extra` under the
    role key. Returns an empty GpuRunners() if nothing matches. `inventory`
    is a test seam."""
    inv = inventory if inventory is not None else _TEST_RUNNER_INVENTORY
    matches = [e for e in inv if e.platform == platform and e.target == target]
    if not matches:
        return GpuRunners()

    by_role: dict[str, list[_GpuRunnerEntry]] = {}
    for e in matches:
        by_role.setdefault(e.role, []).append(e)

    named: dict[str, str] = {role: "" for role in _NAMED_ROLES}
    extra: dict[str, str] = {}
    for role, group in by_role.items():
        label = _select_weighted_label(group)
        if role in _NAMED_ROLES:
            named[role] = label
        else:
            extra[role] = label
    return GpuRunners(
        test=named["test"],
        test_multi_gpu=named["test_multi_gpu"],
        benchmark=named["benchmark"],
        extra=extra,
    )


##########################################################################################
# Build runner resolution: pick a single build runner label for (platform, build_variant).
##########################################################################################


def _build_variant_pool(build_variant: str) -> _BuildVariantPool:
    """Variant names containing 'san' (asan / tsan) route to the sanitizer
    pool; everything else uses default."""
    return "sanitizer" if "san" in build_variant else "default"


def _get_build_runner(
    platform: str,
    build_variant: str,
    *,
    inventory: list[_BuildRunnerEntry] | None = None,
) -> str:
    """Pick a build runner label for (platform, build_variant). Sanitizer
    requests fall back to the default pool when no sanitizer row exists for
    the platform. Returns "" if nothing matches. `inventory` is a test seam."""
    inv = inventory if inventory is not None else _BUILD_RUNNER_INVENTORY
    pool = _build_variant_pool(build_variant)
    matches = [e for e in inv if e.platform == platform and e.variant == pool]
    if not matches and pool == "sanitizer":
        # Fall back to default pool for platforms without a sanitizer-specific row.
        matches = [
            e for e in inv if e.platform == platform and e.variant == "default"
        ]
    if not matches:
        return ""
    return _select_weighted_label(matches)


