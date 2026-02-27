"""
Dataclass type definitions for the AMD GPU family matrix.

The actual matrix data lives in new_amdgpu_family_matrix_data.py.
This file defines the schema — field names, types, defaults, and validation.
Changes here affect all entries in the matrix.

-------------------------------------------------------------------------------
Terminology
-------------------------------------------------------------------------------

family        GPU family group name, e.g. "gfx94X", "gfx115X". Groups related
              architectures under one umbrella.

scope         Subgroup within a family. Generic scopes are "dcgpu", "dgpu", "all".
              Specific GPU names (e.g. "gfx1151") are also valid scopes.

canonical key The string used to identify a MatrixEntry in predefined groups and
              lookups. Matches the CMake AMDGPU target names defined in
              cmake/therock_amdgpu_targets.cmake. Generic scopes produce
              "{family}-{scope}" (e.g. "gfx94X-dcgpu"); specific GPU scopes
              use the scope alone (e.g. "gfx1151").
              CMake targets are defined in: cmake/therock_amdgpu_targets.cmake

-------------------------------------------------------------------------------
Class hierarchy
-------------------------------------------------------------------------------

AmdGpuFamilyMatrix
  └─ MatrixEntry          one (family, scope) row, e.g. gfx94X-dcgpu or gfx1151
       ├─ PlatformConfig  per-platform config (linux / windows); all fields Optional
       │    ├─ BuildConfig
       │    ├─ TestConfig
       │    │    └─ GpuRunners   runner labels (test, test_multi_gpu, benchmark, extra)
       │    └─ ReleaseConfig
       └─ ...
AllBuildVariants           CMake preset + artifact naming per platform and variant
  └─ BuildVariantInfo

-------------------------------------------------------------------------------
Key functions
-------------------------------------------------------------------------------

AmdGpuFamilyMatrix.get_entry(key)
    Look up a MatrixEntry by canonical key ("gfx94X-dcgpu", "gfx1151") or by
    family name alone ("gfx950"), which resolves to the is_family_default entry.

AmdGpuFamilyMatrix.get_default_for_family(family)
    Return the default MatrixEntry for a family, or None if no default is set
    (e.g. gfx115X where each GPU is registered individually).

AmdGpuFamilyMatrix.get_entries_for_groups(list[str])
    Return MatrixEntry list for a list of group keys (e.g. you can use amdgpu_presubmit).

AmdGpuFamilyMatrix.keys()
    Return all canonical keys in matrix definition order.

MatrixEntry.to_dict(platform=None)
    Serialize to dict. Without platform: nested linux/windows keys.
    With platform ("linux"/"windows"): flat dict with amdgpu_family + platform config.

AllBuildVariants.get(platform, variant)
    Return the BuildVariantInfo for a given platform and variant name.

-------------------------------------------------------------------------------
Adding a new field
-------------------------------------------------------------------------------

1. Add the field to the appropriate dataclass with a sensible default.
2. If the value should be inferred from other fields, add logic to __post_init__
   (see TestConfig.run_tests for an example).
3. Add the field to to_dict() if consumers need it in the serialized output.
4. Document the field with an inline docstring directly below the field definition.

-------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class BuildVariantInfo:
    """Configuration for a specific build variant (e.g. release, asan)."""

    label: str
    """Human-readable label, e.g. 'release', 'asan'."""
    suffix: str
    """Artifact-naming suffix; empty string for release, short id otherwise."""
    cmake_preset: str
    """CMake preset name, e.g. 'linux-release-asan'. Empty string for default."""
    expect_failure: bool = False
    """If True, failures in this variant are non-blocking (continue-on-error)."""

    def to_dict(self) -> dict:
        return {
            "build_variant_label": self.label,
            "build_variant_suffix": self.suffix,
            "build_variant_cmake_preset": self.cmake_preset,
            "expect_failure": self.expect_failure,
        }


@dataclass
class AllBuildVariants:
    """Build variant configurations grouped by platform."""

    linux: dict[str, BuildVariantInfo] = field(default_factory=dict)
    windows: dict[str, BuildVariantInfo] = field(default_factory=dict)

    def get(self, platform: str, variant: str) -> Optional[BuildVariantInfo]:
        """Look up a BuildVariantInfo by platform and variant name."""
        variants = getattr(self, platform, None)
        if variants is None:
            raise ValueError(f"Unknown platform: {platform!r}")
        return variants.get(variant)


@dataclass
class BuildConfig:
    """Build configuration for a single platform."""

    build_variants: list[str] = field(default_factory=lambda: ["release"])
    """Ordered list of variant names to build (e.g. ['release', 'asan'])."""
    expect_failure: bool = False
    """If True, build failures for this entry are non-blocking."""

    def to_dict(self) -> dict:
        return {
            "build_variants": list(self.build_variants),
            "expect_failure": self.expect_failure,
        }


@dataclass
class GpuRunners:
    """Runner labels for the various test job types."""

    test: str = ""
    """Label for the standard single-GPU test runner."""
    test_multi_gpu: str = ""
    """Label for the multi-GPU test runner."""
    benchmark: str = ""
    """Label for the benchmark runner."""
    extra: dict[str, str] = field(default_factory=dict)
    """Additional runners not covered by the named fields above (e.g. temporary or
    experimental runners). Keys are used as-is in the serialized output."""

    def __bool__(self) -> bool:
        # True if any runner is set; used by TestConfig.__post_init__ to infer run_tests.
        return bool(self.test or self.test_multi_gpu or self.benchmark or self.extra)

    def to_dict(self) -> dict:
        result: dict[str, str] = {
            "test": self.test,
            "test-multi-gpu": self.test_multi_gpu,
            "benchmark": self.benchmark,
        }
        result.update(self.extra)
        return result


@dataclass
class TestConfig:
    """Test configuration for a single platform."""

    __test__ = False  # Prevent pytest from collecting this as a test class.

    runs_on: GpuRunners = field(default_factory=GpuRunners)
    """Runner labels for test job types."""
    fetch_gfx_targets: list[str] = field(default_factory=list)
    """Individual GPU arch strings for fetching split artifacts (e.g. ['gfx942'])."""
    run_tests: Optional[bool] = None
    """Whether tests should actually be executed. Defaults to True any runner is set in runs_on,
    False otherwise. Can be set explicitly to False when a runner exists but is temporarily disabled."""
    sanity_check_only_for_family: bool = False
    """If True, only a sanity-check test subset is run, not the full suite."""
    test_scope: Literal["all", "smoke", "full"] = "all"
    """Which test subset to run: all (default), smoke (sanity only), full (skip smoke)."""
    expect_pytorch_failure: bool = False
    """If True, PyTorch builds are skipped because they are known to fail."""

    def __post_init__(self):
        # set run_tests to True if any runners are specified,
        # False otherwise, if not explicitly set
        if self.run_tests is None:
            self.run_tests = bool(self.runs_on)

    def to_dict(self) -> dict:
        return {
            "run_tests": self.run_tests,
            "runs_on": self.runs_on.to_dict(),
            "fetch-gfx-targets": list(self.fetch_gfx_targets),
            "sanity_check_only_for_family": self.sanity_check_only_for_family,
            "test_scope": self.test_scope,
            "expect_pytorch_failure": self.expect_pytorch_failure,
        }


@dataclass
class ReleaseConfig:
    """Release configuration for a single platform."""

    bypass_tests_for_releases: bool = False
    """If True, tests are skipped when creating release artifacts."""

    def to_dict(self) -> dict:
        return {
            "bypass_tests_for_releases": self.bypass_tests_for_releases,
        }


@dataclass
class PlatformConfig:
    """Complete configuration for one platform within a target entry."""

    build: BuildConfig = field(default_factory=BuildConfig)
    test: TestConfig = field(default_factory=TestConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)

    def to_dict(self) -> dict:
        return {
            "build": self.build.to_dict(),
            "test": self.test.to_dict(),
            "release": self.release.to_dict(),
        }


# Generic scope names that pair with the family name to form a lookup key.
# Specific GPU names (e.g. "gfx1151") are their own key.
_GENERIC_SCOPES: frozenset[str] = frozenset({"all", "dcgpu", "dgpu"})


@dataclass
class MatrixEntry:
    """A single (family, scope) row in the matrix."""

    family: str
    """GPU family group name, e.g. 'gfx94X', 'gfx115X'."""
    scope: str
    """Scope within the family, e.g. 'dcgpu', 'all', 'gfx1151'."""
    is_family_default: bool = False
    """If True, this entry is the default when only the family name is given (e.g. 'gfx110X')."""
    linux: Optional[PlatformConfig] = None
    """Linux platform config, or None if Linux is not supported."""
    windows: Optional[PlatformConfig] = None
    """Windows platform config, or None if Windows is not supported."""

    @property
    def key(self) -> str:
        """Canonical lookup key used in predefined group lists.

        Generic scopes (dcgpu, dgpu, all) → '{family}-{scope}'  e.g. 'gfx94X-dcgpu'
        Specific GPU scopes               → scope alone          e.g. 'gfx1151'
        """
        if self.scope in _GENERIC_SCOPES:
            return f"{self.family}-{self.scope}"
        return self.scope

    def platform_config(self, platform: str) -> Optional[PlatformConfig]:
        """Return the PlatformConfig for 'linux' or 'windows', or None."""
        if platform == "linux":
            return self.linux
        if platform == "windows":
            return self.windows
        raise ValueError(f"Unknown platform: {platform!r}")

    def to_dict(self, platform: Optional[str] = None) -> dict:
        d: dict = {"amdgpu_family": self.key}
        if platform is not None:
            cfg = self.platform_config(platform)
            if cfg is not None:
                d.update(cfg.to_dict())
        else:
            for p in ("linux", "windows"):
                cfg = self.platform_config(p)
                if cfg is not None:
                    d[p] = cfg.to_dict()
        return d


@dataclass
class AmdGpuFamilyMatrix:
    """The complete AMD GPU family matrix."""

    entries: list[MatrixEntry]

    def get_entry(self, key: str) -> Optional[MatrixEntry]:
        """Look up a MatrixEntry by its canonical key (e.g. 'gfx94X-dcgpu', 'gfx1151').

        If no exact match is found, treats the key as a family name and returns
        the default entry for that family (e.g. 'gfx950' → gfx950-dcgpu).
        Lookup is case-insensitive.
        """
        key_lower = key.lower()
        for entry in self.entries:
            if entry.key.lower() == key_lower:
                return entry
        return self.get_default_for_family(key)

    def get_default_for_family(self, family: str) -> Optional[MatrixEntry]:
        """Return the default entry for a family (e.g. 'gfx110X' → gfx110X-all entry).
        Lookup is case-insensitive.
        """
        family_lower = family.lower()
        for entry in self.entries:
            if entry.family.lower() == family_lower and entry.is_family_default:
                return entry
        return None

    def get_entries_for_groups(self, group_keys: list[str]) -> list[MatrixEntry]:
        """Return entries matching the given keys, preserving order and skipping unknowns."""
        result = []
        for key in group_keys:
            entry = self.get_entry(key)
            if entry is not None:
                result.append(entry)
        return result

    def keys(self) -> list[str]:
        """Return all canonical keys in matrix order."""
        return [e.key for e in self.entries]

    def to_nested_dict(self) -> dict:
        """Convert to the original nested-dict format: family → target → platform → ..."""
        result: dict = {}
        for entry in self.entries:
            result.setdefault(entry.family, {})[entry.scope] = entry.to_dict()
        return result
