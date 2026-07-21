# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
GPU target definitions for CI/CD workflows.

This module defines specific GPU targets (not families) for building and testing.
Each target specifies exactly what to build and where to test.

Key principle: Build what you test.
- Old: gfx110X builds [gfx1100, gfx1101, gfx1102, gfx1103], tests on gfx1101
- New: gfx1101 builds [gfx1101], tests on gfx1101

Runner labels are loaded from external config (therock-ci-config) when available,
with local fallbacks defined here.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


class Trigger(Enum):
    """CI trigger types."""

    PRESUBMIT = "presubmit"  # Pull requests
    POSTSUBMIT = "postsubmit"  # Push to main
    NIGHTLY = "nightly"  # Scheduled runs


@dataclass
class RunnerConfig:
    """Runner labels for a platform."""

    test: str | None = None
    test_multi_gpu: str | None = None
    benchmark: str | None = None

    def to_dict(self) -> dict:
        result = {}
        if self.test:
            result["test-runs-on"] = self.test
        if self.test_multi_gpu:
            result["test-runs-on-multi-gpu"] = self.test_multi_gpu
        if self.benchmark:
            result["benchmark-runs-on"] = self.benchmark
        return result


@dataclass
class PlatformConfig:
    """Platform-specific configuration for a GPU target."""

    # What triggers include this target
    triggers: set[Trigger] = field(default_factory=lambda: {Trigger.PRESUBMIT})

    # Build variants to produce
    build_variants: list[str] = field(default_factory=lambda: ["release"])

    # Test configuration
    run_tests: bool = True
    test_scope: Literal["full", "smoke", "sanity"] = "full"

    # Runner labels (fallback - external config takes precedence)
    runners: RunnerConfig = field(default_factory=RunnerConfig)

    # Skip tests for release builds (used for consumer GPUs)
    bypass_tests_for_releases: bool = False

    # Only run tests on submodule bumps
    submodule_bump_tests_only: bool = False

    def to_dict(self, target: str) -> dict:
        """Serialize to legacy dict format for compatibility."""
        result = {
            "family": target,  # For now, family = target
            "fetch-gfx-targets": [target],
            "build_variants": self.build_variants,
        }
        result.update(self.runners.to_dict())

        if self.bypass_tests_for_releases:
            result["bypass_tests_for_releases"] = True
        if self.submodule_bump_tests_only:
            result["submodule_bump_tests_only"] = True
        if self.test_scope == "sanity":
            result["sanity_check_only_for_family"] = True
        if Trigger.NIGHTLY in self.triggers and Trigger.PRESUBMIT not in self.triggers:
            result["nightly_check_only_for_family"] = True

        return result


@dataclass
class GpuTarget:
    """A specific GPU target for CI/CD."""

    # The GPU target name (e.g., "gfx942", "gfx1101")
    target: str

    # What GFX targets to build (usually just [target], but can be multiple)
    build_targets: list[str] = field(default_factory=list)

    # Platform configurations
    linux: PlatformConfig | None = None
    windows: PlatformConfig | None = None

    def __post_init__(self):
        if not self.build_targets:
            self.build_targets = [self.target]

    def get_platform(self, platform: str) -> PlatformConfig | None:
        if platform == "linux":
            return self.linux
        elif platform == "windows":
            return self.windows
        return None

    def supports_trigger(self, trigger: Trigger, platform: str) -> bool:
        """Check if this target runs for the given trigger on the platform."""
        config = self.get_platform(platform)
        if not config:
            return False
        return trigger in config.triggers

    def to_dict(self, platform: str | None = None) -> dict:
        """Serialize to legacy dict format."""
        if platform:
            config = self.get_platform(platform)
            if not config:
                return {}
            return config.to_dict(self.target)

        result = {}
        if self.linux:
            result["linux"] = self.linux.to_dict(self.target)
        if self.windows:
            result["windows"] = self.windows.to_dict(self.target)
        return result


# =============================================================================
# GPU Target Definitions
# =============================================================================

# -----------------------------------------------------------------------------
# CDNA1 - MI100 series (Arcturus)
# -----------------------------------------------------------------------------
_GFX908 = GpuTarget(
    target="gfx908",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),  # No hardware available
        test_scope="sanity",
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),
    ),
)

# -----------------------------------------------------------------------------
# CDNA2 - MI200 series (Aldebaran)
# -----------------------------------------------------------------------------
_GFX90A = GpuTarget(
    target="gfx90a",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="linux-gfx90a-gpu-rocm"),
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),
    ),
)

# -----------------------------------------------------------------------------
# CDNA3 - MI300 series (Aqua Vanjaram)
# -----------------------------------------------------------------------------
_GFX942 = GpuTarget(
    target="gfx942",
    linux=PlatformConfig(
        triggers={Trigger.PRESUBMIT, Trigger.POSTSUBMIT, Trigger.NIGHTLY},
        build_variants=["release", "asan", "host-asan", "tsan"],
        runners=RunnerConfig(
            test="linux-gfx942-1gpu-ccs-csp-ossci-rocm",
            test_multi_gpu="linux-gfx942-8gpu-ossci-rocm",
            benchmark="linux-gfx942-8gpu-ossci-rocm",
        ),
    ),
)

_GFX950 = GpuTarget(
    target="gfx950",
    linux=PlatformConfig(
        triggers={Trigger.POSTSUBMIT, Trigger.NIGHTLY},
        build_variants=["release", "asan", "tsan"],
        runners=RunnerConfig(
            test="linux-gfx950-1gpu-ccs-ossci-rocm",
            test_multi_gpu="linux-gfx950-8gpu-ccs-ossci-rocm",
        ),
        submodule_bump_tests_only=True,
    ),
)

# -----------------------------------------------------------------------------
# CDNA5 - MI455 series
# -----------------------------------------------------------------------------
_GFX1250 = GpuTarget(
    target="gfx1250",
    linux=PlatformConfig(
        triggers={Trigger.PRESUBMIT, Trigger.POSTSUBMIT, Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="linux-mi455-gpu-rocm"),
    ),
)

# -----------------------------------------------------------------------------
# GCN5 - Vega (gfx900, gfx906)
# -----------------------------------------------------------------------------
_GFX900 = GpuTarget(
    target="gfx900",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),  # No hardware available
        test_scope="sanity",
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),
    ),
)

_GFX906 = GpuTarget(
    target="gfx906",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),  # No hardware available
        test_scope="sanity",
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),
    ),
)

# -----------------------------------------------------------------------------
# RDNA1 - Navi 10 (gfx1010, gfx1011, gfx1012, gfx1013)
# -----------------------------------------------------------------------------
_GFX1010 = GpuTarget(
    target="gfx1010",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),  # No hardware available
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),
    ),
)

# -----------------------------------------------------------------------------
# RDNA2 - Navi 2x (gfx1030, gfx1031, etc.)
# -----------------------------------------------------------------------------
_GFX1030 = GpuTarget(
    target="gfx1030",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="linux-gfx1030-gpu-rocm"),
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="windows-gfx1030-gpu-rocm"),
    ),
)

# -----------------------------------------------------------------------------
# RDNA3 - Navi 3x (gfx1100, gfx1101, gfx1102, gfx1103)
# -----------------------------------------------------------------------------
_GFX1100 = GpuTarget(
    target="gfx1100",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="linux-gfx110X-gpu-rocm"),
        bypass_tests_for_releases=True,
    ),
    windows=PlatformConfig(
        triggers={Trigger.PRESUBMIT, Trigger.POSTSUBMIT, Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="windows-gfx110X-gpu-rocm"),
        bypass_tests_for_releases=True,
    ),
)

# -----------------------------------------------------------------------------
# RDNA3.5 - Strix (gfx1150, gfx1151, gfx1152, gfx1153)
# -----------------------------------------------------------------------------
_GFX1150 = GpuTarget(
    target="gfx1150",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="linux-gfx1150-gpu-rocm"),
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),
    ),
)

_GFX1151 = GpuTarget(
    target="gfx1151",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="linux-gfx1151-gpu-rocm"),
        bypass_tests_for_releases=True,
    ),
    windows=PlatformConfig(
        triggers={Trigger.PRESUBMIT, Trigger.POSTSUBMIT, Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(
            test="windows-gfx1151-gpu-rocm",
            benchmark="windows-gfx1151-gpu-rocm",
        ),
    ),
)

_GFX1152 = GpuTarget(
    target="gfx1152",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),
    ),
)

_GFX1153 = GpuTarget(
    target="gfx1153",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="linux-gfx1153-gpu-rocm"),
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test=""),
    ),
)

# -----------------------------------------------------------------------------
# RDNA4 - Navi 4x (gfx1200, gfx1201)
# -----------------------------------------------------------------------------
_GFX1200 = GpuTarget(
    target="gfx1200",
    linux=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="linux-gfx120X-gpu-rocm"),
        bypass_tests_for_releases=True,
    ),
    windows=PlatformConfig(
        triggers={Trigger.NIGHTLY},
        build_variants=["release"],
        runners=RunnerConfig(test="windows-gfx120X-gpu-rocm"),
        bypass_tests_for_releases=True,
    ),
)


# =============================================================================
# Registry
# =============================================================================


class GpuTargetRegistry:
    """Registry of all GPU targets."""

    def __init__(self):
        self._targets: dict[str, GpuTarget] = {}
        self._external_config: dict | None = None

    def register(self, target: GpuTarget) -> None:
        """Register a GPU target."""
        self._targets[target.target.lower()] = target

    def get(self, target: str) -> GpuTarget | None:
        """Get a target by name (case-insensitive)."""
        return self._targets.get(target.lower())

    def all(self) -> list[GpuTarget]:
        """Get all registered targets."""
        return list(self._targets.values())

    def for_trigger(self, trigger: Trigger, platform: str) -> list[GpuTarget]:
        """Get all targets that run for a given trigger on a platform."""
        return [
            t for t in self._targets.values() if t.supports_trigger(trigger, platform)
        ]

    def load_external_config(self) -> None:
        """Load external runner config from therock-ci-config."""
        ci_config_path = os.environ.get("CI_CONFIG_PATH", "").strip()
        if not ci_config_path:
            return

        config_path = Path(ci_config_path)
        sys.path.insert(0, str(config_path))
        try:
            from ci_config_api import get_gpu_runner_labels, load_runner_config

            raw_config = load_runner_config(config_path)
            self._external_config = get_gpu_runner_labels(raw_config)
        except (ImportError, Exception):
            pass

    def to_legacy_matrix(
        self, trigger: Trigger | None = None, platform: str | None = None
    ) -> dict:
        """Export to legacy amdgpu_family_info_matrix format."""
        result = {}
        for target in self._targets.values():
            if trigger and platform:
                if not target.supports_trigger(trigger, platform):
                    continue
            entry = target.to_dict()
            if entry:
                result[target.target] = entry
        return result


# Global registry
_registry = GpuTargetRegistry()

# Register all targets
for _var in list(globals().values()):
    if isinstance(_var, GpuTarget):
        _registry.register(_var)


# =============================================================================
# Public API
# =============================================================================


def get_target(target: str) -> GpuTarget | None:
    """Get a GPU target by name."""
    return _registry.get(target)


def get_all_targets() -> list[GpuTarget]:
    """Get all registered GPU targets."""
    return _registry.all()


def get_targets_for_trigger(trigger: Trigger, platform: str) -> list[GpuTarget]:
    """Get all targets that run for a given trigger on a platform."""
    return _registry.for_trigger(trigger, platform)


def to_legacy_matrix(
    trigger: Trigger | None = None, platform: str | None = None
) -> dict:
    """Export to legacy amdgpu_family_info_matrix format for compatibility."""
    return _registry.to_legacy_matrix(trigger, platform)
