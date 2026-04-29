# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
       └─ PlatformConfig
            ├─ ...
AllBuildVariants           CMake preset + artifact naming per platform and variant
  └─ BuildVariantInfo

-------------------------------------------------------------------------------
Key functions
-------------------------------------------------------------------------------

AmdGpuFamilyMatrix.get_entry(key, *, build_variant="release")
    Look up by exact target ("gfx942") or family name ("gfx94X-dcgpu") which
    resolves to the is_family_default entry. Always returns a deep copy scoped
    to `build_variant`: platforms not supporting the variant are dropped, and
    non-release variants re-bind BuildConfig.runs_on to a matching build runner.
    Returns None if no entry matches or no platform supports the build_variant.

AmdGpuFamilyMatrix.get_default_for_family(family)
    Return the is_family_default entry for the given family name, or None.

AmdGpuFamilyMatrix.get_entries_for_groups(list[str], *, build_variant="release")
    Look up many keys at once. Returns a GroupLookupResult with the matched
    entries and the keys that had no match (either unknown, or the entry exists
    but does not support `build_variant`).

AmdGpuFamilyMatrix.keys()
    All canonical keys, alphabetically.

AmdGpuFamilyMatrix.to_nested_dict()
    Serialize the matrix as {target: entry_dict}; each entry_dict has
    "amdgpu_family" plus per-platform "linux"/"windows" sub-dicts containing
    "build", "test", and "release" blocks (see MatrixEntry.to_dict).

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

import copy
from dataclasses import dataclass, field
from typing import Literal


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

    def get(self, platform: str, variant: str) -> BuildVariantInfo | None:
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
    runs_on: str | None = None
    """Build runner label. Auto-filled by MatrixEntry.__post_init__ from the
    'release' inventory pool; non-release lookups re-bind it to the matching
    pool. None only if the inventory has no matching row — to_dict() then raises."""

    def to_dict(self) -> dict:
        if self.runs_on is None:
            raise ValueError(
                "BuildConfig.runs_on is None — no matching row in the build runner "
                "inventory. Build jobs always require a runner."
            )
        return {
            "build_variants": list(self.build_variants),
            "expect_failure": self.expect_failure,
            "build_runs_on": self.runs_on,
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

    def has_any_runner(self) -> bool:
        """Return True if any runner field is set; used to infer TestConfig.run_tests."""
        return any([self.test, self.test_multi_gpu, self.benchmark, self.extra])

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
    run_tests: bool | None = None
    """Whether tests should actually run. None means "infer from runs_on", and
    is resolved by MatrixEntry.__post_init__ *after* the inventory fill (so an
    inferred value sees the populated runners). Set explicitly to True/False to
    bypass inference. to_dict() raises if still None at serialization time."""
    sanity_check_only_for_family: bool = False
    """If True, only a sanity-check test subset is run, not the full suite."""
    test_scope: Literal["quick", "comprehensive", "full"] = "comprehensive"
    """Which test subset to run: comprehensive (default), quick (subset), full (extended)."""
    bypass_tests_for_unscheduled: bool = False
    """If True, tests are skipped on non-scheduled triggers (PR / push) and only run
    on the schedule (nightly) workflow trigger."""

    def to_dict(self) -> dict:
        if self.run_tests is None:
            raise ValueError(
                "TestConfig.run_tests is None — inference normally happens in "
                "MatrixEntry.__post_init__; bare TestConfig instances must set it."
            )
        return {
            "run_tests": self.run_tests,
            "runs_on": self.runs_on.to_dict(),
            "fetch-gfx-targets": list(self.fetch_gfx_targets),
            "sanity_check_only_for_family": self.sanity_check_only_for_family,
            "test_scope": self.test_scope,
            "bypass_tests_for_unscheduled": self.bypass_tests_for_unscheduled,
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

    def supports_variant(self, build_variant: str) -> bool:
        """True if this platform's BuildConfig lists the given variant."""
        return build_variant in self.build.build_variants

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
    """If True, this entry is returned for family-name lookups like 'gfx94X-dcgpu'.
    At most one entry per shared family name (validated at construction time).
    Intended for user-facing inputs (workflow dispatch); internal CI code
    should use exact target names."""
    linux: PlatformConfig | None = None
    """Linux platform config, or None if Linux is not supported."""
    windows: PlatformConfig | None = None
    """Windows platform config, or None if Windows is not supported."""
    family: list[str] = field(init=False, default_factory=list)
    """Family names this target belongs to (e.g. ['dcgpu-all', 'gfx94X-all', 'gfx94X-dcgpu']).
    Auto-populated by AmdGpuFamilyMatrix from cmake/therock_amdgpu_targets.cmake."""

    def __post_init__(self):
        # Fill runner labels from the inventories on each platform that did not
        # specify them, then infer test.run_tests if the caller left it as None.
        # build.runs_on is always release-pool here; lookup paths rebind for
        # non-release variants. Function-local import breaks the types <->
        # runners cycle.
        from new_amdgpu_family_matrix_runners import _get_build_runner, _get_gpu_runners

        for platform in ("linux", "windows"):
            cfg = self.platform_config(platform)
            if cfg is None:
                continue
            if not cfg.test.runs_on.has_any_runner():
                cfg.test.runs_on = _get_gpu_runners(platform, self.target)
            if cfg.test.run_tests is None:
                cfg.test.run_tests = cfg.test.runs_on.has_any_runner()
            if cfg.build.runs_on is None:
                cfg.build.runs_on = _get_build_runner(platform, "release")

    @property
    def key(self) -> str:
        """Canonical lookup key"""
        return self.target

    def platform_config(self, platform: str) -> PlatformConfig | None:
        """Return the PlatformConfig for 'linux' or 'windows', or None."""
        if platform == "linux":
            return self.linux
        if platform == "windows":
            return self.windows
        raise ValueError(f"Unknown platform: {platform!r}")

    def to_dict(self, platform: str | None = None) -> dict:
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


def _entry_supports_variant(entry: "MatrixEntry", build_variant: str) -> bool:
    """True if any platform on the entry lists `build_variant`."""
    for p in ("linux", "windows"):
        cfg = entry.platform_config(p)
        if cfg is not None and cfg.supports_variant(build_variant):
            return True
    return False


def _scope_to_variant(entry: "MatrixEntry", build_variant: str) -> None:
    """Drop platforms that do not support `build_variant`; trim survivors'
    build_variants to [build_variant]. Mutates in place — caller must deep-copy."""
    for platform in ("linux", "windows"):
        cfg = entry.platform_config(platform)
        if cfg is None:
            continue
        if not cfg.supports_variant(build_variant):
            setattr(entry, platform, None)
            continue
        cfg.build.build_variants = [build_variant]


def _rebind_build_runners(entry: "MatrixEntry", build_variant: str) -> None:
    """Re-pick BuildConfig.runs_on from `build_variant`'s pool on every kept
    platform. Function-local import breaks the types <-> runners cycle."""
    from new_amdgpu_family_matrix_runners import _get_build_runner

    for platform in ("linux", "windows"):
        cfg = entry.platform_config(platform)
        if cfg is None:
            continue
        cfg.build.runs_on = _get_build_runner(platform, build_variant)


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

    def get_entry(
        self, key: str, *, build_variant: str = "release"
    ) -> MatrixEntry | None:
        """Look up by target name ('gfx942') or family name ('gfx94X-dcgpu',
        falls through to the is_family_default entry). Case-insensitive.

        Returns a deep copy scoped to `build_variant`: unsupported platforms are
        set to None and survivors' build_variants is trimmed to [build_variant].
        Non-release variants also re-bind BuildConfig.runs_on. Returns None if
        the key is unknown or no platform supports the variant.

        Deep-copying lets consumers mutate freely without leaking changes into
        the shared module-level matrix.
        """
        key_lower = key.lower()
        match: MatrixEntry | None = None
        for entry in self.entries:
            if entry.key.lower() == key_lower:
                match = entry
                break
        if match is None:
            match = self.get_default_for_family(key)
        if match is None:
            return None
        if not _entry_supports_variant(match, build_variant):
            return None
        match = copy.deepcopy(match)
        _scope_to_variant(match, build_variant)
        if build_variant != "release":
            _rebind_build_runners(match, build_variant)
        return match

    def get_default_for_family(self, family: str) -> MatrixEntry | None:
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

    def get_entries_for_groups(
        self,
        group_keys: list[str],
        deduplicate: bool = True,
        *,
        build_variant: str = "release",
    ) -> GroupLookupResult:
        """Look up many keys at once via get_entry. Returns matches in input
        order plus a list of keys that produced no entry (unknown key, or the
        entry does not support `build_variant`).

        With `deduplicate=True` (default), keys that resolve to the same target
        appear only once (first occurrence wins).
        """
        entries = []
        unmatched_keys = []
        seen: set[str] = set()
        for key in group_keys:
            entry = self.get_entry(key, build_variant=build_variant)
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
