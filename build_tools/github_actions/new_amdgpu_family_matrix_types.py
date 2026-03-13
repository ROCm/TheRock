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

scope         Subgroup within a family. Generic scopes are "dcgpu", "dgpu", "all", ...
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
  └─ MatrixEntry          one GPU target entry, e.g. gfx942 or gfx1151
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
    Look up a MatrixEntry by canonical key ("gfx942", "gfx1151") or by
    family name alone ("gfx94X-dcgpu"), which resolves to the is_family_default entry.

AmdGpuFamilyMatrix.get_default_for_family(family)
    Return the is_family_default entry whose family list contains the given name,
    or None if no default is set (e.g. gfx115X-all where GPUs are registered individually)
    or the requested family does not exist.

AmdGpuFamilyMatrix.get_entries_for_groups(list[str])
    Return a GroupLookupResult with matched MatrixEntry list and unmatched keys.

AmdGpuFamilyMatrix.keys()
    Return all canonical keys in alphabetical order.

AmdGpuFamilyMatrix.to_nested_dict()
    Serialize all entries as a flat dict keyed by target name, e.g.:
    {"gfx1101": {"amdgpu_family": "gfx1101",
                 "linux": {"build": {"build_variants": ["release"], "expect_failure": False},
                           "release": {"bypass_tests_for_releases": True},
                           "test": {"expect_pytorch_failure": False,
                                    "fetch-gfx-targets": ["gfx1101"],
                                    "run_tests": False,
                                    "runs_on": {"benchmark": "",
                                                "test": "linux-gfx110X-gpu-rocm",
                                                "test-multi-gpu": ""},
                                    "sanity_check_only_for_family": True,
                                    "test_scope": "all"}},
                 "windows": {"build": {"build_variants": ["release"], "expect_failure": False},
                             "release": {"bypass_tests_for_releases": True},
                             "test": {"expect_pytorch_failure": False,
                                      "fetch-gfx-targets": ["gfx1101"],
                                      "run_tests": True,
                                      "runs_on": {"benchmark": "",
                                                  "test": "windows-gfx110X-gpu-rocm",
                                                  "test-multi-gpu": ""},
                                      "sanity_check_only_for_family": True,
                                      "test_scope": "all"}}},
     ...}

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


@dataclass
class MatrixEntry:
    """A single GPU target entry in the matrix."""

    target: str
    """Specific GPU target name, e.g. 'gfx942', 'gfx1151'."""
    is_family_default: bool = False
    """If True, this entry is returned when looking up by a shared family name
    (e.g. 'gfx94X-dcgpu'). At most one entry per shared family name may have
    this set — validated at AmdGpuFamilyMatrix construction time."""
    linux: Optional[PlatformConfig] = None
    """Linux platform config, or None if Linux is not supported."""
    windows: Optional[PlatformConfig] = None
    """Windows platform config, or None if Windows is not supported."""
    family: list[str] = field(init=False, default_factory=list)
    """Family names this target belongs to (e.g. ['dcgpu-all', 'gfx94X-all', 'gfx94X-dcgpu']).
    Auto-populated by AmdGpuFamilyMatrix from cmake/therock_amdgpu_targets.cmake."""

    @property
    def key(self) -> str:
        """Canonical lookup key"""
        return self.target

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
class GroupLookupResult:
    """Result of get_entries_for_groups: matched entries and keys with no match."""

    entries: list[MatrixEntry]
    unmatched_keys: list[str]


@dataclass
class EntryLookupResult:
    """Result of a single-key lookup, preserving how the entry was resolved."""

    entry: MatrixEntry
    amdgpu_family: str
    """The original lookup key, e.g. 'gfx94X', 'gfx94X-dcgpu', 'gfx942'.
    Used as the family identifier in CI output."""
    resolved_via: Literal["target", "family", "family_prefix"]
    """How the entry was resolved:
      'target'        — exact target name match (e.g. 'gfx942')
      'family'        — exact family name match (e.g. 'gfx94X-dcgpu')
      'family_prefix' — prefix match via trailing X (e.g. 'gfx94X')
    """


@dataclass
class AmdGpuFamilyMatrix:
    """The complete AMD GPU family matrix."""

    entries: list[MatrixEntry]
    cmake_families: dict[str, list[str]] = field(default_factory=dict)
    """Mapping of gfx_target → [family names], parsed from cmake/therock_amdgpu_targets.cmake.
    Used to auto-populate MatrixEntry.family at construction time."""

    def __post_init__(self) -> None:
        self._populate_families()
        self._validate_is_family_default()

    def _populate_families(self) -> None:
        """Populate entry.family from cmake_families for each entry."""
        for entry in self.entries:
            families = self.cmake_families.get(entry.target, [])
            if not families:
                raise ValueError(
                    f"Target '{entry.target}' has no family entries in cmake_families. "
                    f"Check cmake/therock_amdgpu_targets.cmake."
                )
            entry.family = families

    def _validate_is_family_default(self) -> None:
        """Ensure at most one entry has is_family_default=True per gfx-prefixed family name.

        Only validates gfx-prefixed family names (e.g. 'gfx94X-dcgpu', 'gfx110X-all').
        Broad category names like 'dgpu-all', 'dcgpu-all', 'igpu-all' are intentionally
        shared across many entries and are not validated here.
        """
        family_defaults: dict[str, str] = {}
        for entry in self.entries:
            if not entry.is_family_default:
                continue
            for family in entry.family:
                if family == entry.target:
                    continue  # skip self-family (each target is its own family in cmake)
                if not family.startswith("gfx"):
                    continue  # skip broad categories like dgpu-all, dcgpu-all, igpu-all
                if family in family_defaults:
                    raise ValueError(
                        f"Multiple is_family_default entries for family '{family}': "
                        f"'{family_defaults[family]}' and '{entry.target}'"
                    )
                family_defaults[family] = entry.target

    def get_targets_for_family(self, family: str) -> list[str]:
        """Return all target names that belong to the given family name.

        Example: get_targets_for_family("gfx120X-all") → ["gfx1200", "gfx1201"]
        Lookup is case-insensitive. Returns an empty list if no targets match.
        """
        family_lower = family.lower()
        return [
            entry.target
            for entry in self.entries
            if any(f.lower() == family_lower for f in entry.family)
        ]

    def get_entry(self, key: str) -> Optional[MatrixEntry]:
        """Look up a MatrixEntry by target name (e.g. 'gfx942') or family name (e.g. 'gfx94X-dcgpu').

        Direct target lookup is tried first. If not found, treats the key as a family
        name and returns the is_family_default entry for that family.
        Lookup is case-insensitive. Returns None if no match found.
        """
        key_lower = key.lower()
        for entry in self.entries:
            if entry.key.lower() == key_lower:
                return entry
        return self.get_default_for_family(key)

    def get_default_for_family(self, family: str) -> Optional[MatrixEntry]:
        """Return the is_family_default entry whose family list contains the given name,
        or None if no default is set.

        Supports two match modes (case-insensitive):
          - Exact: 'gfx94X-dcgpu' matches the family name literally.
          - Prefix: if the key ends with 'X' or 'x', it matches any family name that
            starts with key + '-'. E.g. 'gfx94X' matches 'gfx94X-dcgpu' and 'gfx94X-all'.

        Family names like 'gfx115X-all' that have no is_family_default entry return None.
        """
        family_lower = family.lower()
        is_prefix = family_lower.endswith("x")
        for entry in self.entries:
            if not entry.is_family_default:
                continue
            for f in entry.family:
                f_lower = f.lower()
                if f_lower == family_lower:
                    return entry
                if is_prefix and f_lower.startswith(family_lower + "-"):
                    return entry
        return None

    def lookup(self, key: str) -> Optional[EntryLookupResult]:
        """Look up a single key and return the entry with resolution metadata.

        Returns an EntryLookupResult or None if no match found. resolved_via indicates
        how the entry was found:
          'target'        — exact target name match (e.g. 'gfx942' → gfx942)
          'family'        — exact family name match (e.g. 'gfx94X-dcgpu' → gfx942)
          'family_prefix' — prefix match via trailing X (e.g. 'gfx94X' → gfx942)
        """
        key_lower = key.lower()
        for entry in self.entries:
            if entry.key.lower() == key_lower:
                return EntryLookupResult(
                    entry=entry, amdgpu_family=key, resolved_via="target"
                )
        family_lower = key_lower
        is_prefix = family_lower.endswith("x")
        for entry in self.entries:
            if not entry.is_family_default:
                continue
            for f in entry.family:
                f_lower = f.lower()
                if f_lower == family_lower:
                    return EntryLookupResult(
                        entry=entry, amdgpu_family=key, resolved_via="family"
                    )
                if is_prefix and f_lower.startswith(family_lower + "-"):
                    return EntryLookupResult(
                        entry=entry, amdgpu_family=key, resolved_via="family_prefix"
                    )
        return None

    def get_entries_for_groups(
        self, group_keys: list[str], deduplicate: bool = True
    ) -> GroupLookupResult:
        """Look up entries for a list of keys, returning both matches and misses.
           By default deduplicates group_keys entries resolving to the same target.

        Args:
            group_keys: list of keys to look up.
            deduplicate: if True, entries with the same target are included only once
                         (first occurrence wins). Unmatched keys are always reported.

        Returns a GroupLookupResult with:
            entries: matched MatrixEntry objects, in the order of group_keys
            unmatched_keys: keys from group_keys that had no match in the matrix
        """
        entries = []
        unmatched_keys = []
        seen: set[str] = set()
        for key in group_keys:
            entry = self.get_entry(key)
            if entry is None:
                unmatched_keys.append(key)
                continue
            if deduplicate and entry.target in seen:
                continue
            seen.add(entry.target)
            entries.append(entry)
        return GroupLookupResult(entries=entries, unmatched_keys=unmatched_keys)

    def keys(self) -> list[str]:
        """Return all canonical keys in alphabetical order."""
        return sorted(e.key for e in self.entries)

    def to_nested_dict(self) -> dict:
        """Serialize all entries as a flat dict keyed by target name."""
        return {entry.target: entry.to_dict() for entry in self.entries}
