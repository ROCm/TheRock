import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from new_amdgpu_family_matrix_data import (
    amdgpu_family_info_matrix_all,
    amdgpu_family_predefined_groups,
    all_build_variants,
)
from new_amdgpu_family_matrix_runners import (
    _BuildRunnerEntry,
    _GpuRunnerEntry,
    _get_gpu_runners,
    _get_build_runner,
)
from new_amdgpu_family_matrix_types import (
    AmdGpuFamilyMatrix,
    BuildConfig,
    GroupLookupResult,
    GpuRunners,
    MatrixEntry,
    PlatformConfig,
    ReleaseConfig,
    TestConfig,
)


# -----------------------------------------------------------------------------
# Helpers for building synthetic matrices used by mechanics tests.
# -----------------------------------------------------------------------------


def _make_synthetic_matrix() -> AmdGpuFamilyMatrix:
    """Build a small synthetic matrix with two GPUs in one family.

    The intent is to exercise the matrix mechanics (lookup, family resolution,
    dedup, to_dict) without coupling to the real GPU inventory.
    """
    fake_default = MatrixEntry(
        target="gfxFAKE1",
        is_family_default=True,
        linux=PlatformConfig(
            build=BuildConfig(runs_on="fake-build-runner"),
            test=TestConfig(
                runs_on=GpuRunners(test="fake-linux-runner"),
                fetch_gfx_targets=["gfxFAKE1"],
            ),
        ),
    )
    fake_sibling = MatrixEntry(
        target="gfxFAKE2",
        linux=PlatformConfig(
            build=BuildConfig(runs_on="fake-build-runner"),
        ),
    )
    return AmdGpuFamilyMatrix(
        entries=[fake_default, fake_sibling],
        cmake_families={
            "gfxFAKE1": ["gfxFAKEX-all", "gfxFAKEX-fakegroup"],
            "gfxFAKE2": ["gfxFAKEX-all", "gfxFAKEX-fakegroup"],
        },
    )


# -----------------------------------------------------------------------------
# Smoke tests over the real matrix — kept to catch egregious regressions.
# These are intentionally minimal; resist adding per-GPU assertions here.
# -----------------------------------------------------------------------------


class TestRealMatrixSmoke(unittest.TestCase):
    """Lightweight integrity checks against the real matrix."""

    def test_keys_are_unique(self):
        keys = amdgpu_family_info_matrix_all.keys()
        self.assertEqual(len(keys), len(set(keys)), "Duplicate canonical keys found")

    def test_predefined_groups_reference_valid_keys(self):
        for group_name, group_keys in amdgpu_family_predefined_groups.items():
            for key in group_keys:
                self.assertIsNotNone(
                    amdgpu_family_info_matrix_all.get_entry(key),
                    f"Group {group_name!r} references unresolvable key {key!r}",
                )

    def test_matrix_is_non_empty(self):
        self.assertGreater(len(amdgpu_family_info_matrix_all.keys()), 0)


# -----------------------------------------------------------------------------
# Lookup mechanics — exercised on synthetic data so they are stable across
# changes to the real matrix.
# -----------------------------------------------------------------------------


class TestLookupMechanics(unittest.TestCase):
    """Mechanics of get_entry / get_default_for_family, on synthetic data."""

    def setUp(self):
        self.matrix = _make_synthetic_matrix()

    def test_exact_target_lookup(self):
        entry = self.matrix.get_entry("gfxFAKE1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfxFAKE1")

    def test_lookup_is_case_insensitive(self):
        # Each lookup returns a fresh deep copy, so compare by target rather than identity.
        self.assertEqual(
            self.matrix.get_entry("GFXFAKE1").target,
            self.matrix.get_entry("gfxFAKE1").target,
        )

    def test_unknown_key_returns_none(self):
        self.assertIsNone(self.matrix.get_entry("gfxNOPE"))

    def test_exact_family_name_resolves_to_default(self):
        entry = self.matrix.get_entry("gfxFAKEX-fakegroup")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfxFAKE1")

    def test_family_prefix_resolves_to_default(self):
        # Trailing 'X' triggers the prefix-match path.
        entry = self.matrix.get_entry("gfxFAKEX")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "gfxFAKE1")

    def test_family_prefix_with_unknown_subgroup_returns_none(self):
        # Negative case: a prefix-shaped key that does not match anything.
        self.assertIsNone(self.matrix.get_entry("gfxFAKEX-unknowngroup"))

    def test_get_targets_for_family(self):
        self.assertCountEqual(
            self.matrix.get_targets_for_family("gfxFAKEX-all"),
            ["gfxFAKE1", "gfxFAKE2"],
        )

    def test_get_targets_for_unknown_family_returns_empty(self):
        self.assertEqual(self.matrix.get_targets_for_family("gfxNOPE-x"), [])

    def test_is_family_default_validation_prevents_duplicates(self):
        with self.assertRaises(ValueError):
            AmdGpuFamilyMatrix(
                entries=[
                    MatrixEntry(target="gfxFAKE1", is_family_default=True),
                    MatrixEntry(target="gfxFAKE2", is_family_default=True),
                ],
                cmake_families={
                    "gfxFAKE1": ["gfxFAKEX-fakegroup"],
                    "gfxFAKE2": ["gfxFAKEX-fakegroup"],
                },
            )


# -----------------------------------------------------------------------------
# get_entries_for_groups mechanics, exercised on synthetic data.
# -----------------------------------------------------------------------------


class TestGetEntriesForGroups(unittest.TestCase):
    def setUp(self):
        self.matrix = _make_synthetic_matrix()

    def test_returns_group_lookup_result(self):
        result = self.matrix.get_entries_for_groups(["gfxFAKE1"])
        self.assertIsInstance(result, GroupLookupResult)
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.unmatched_keys, [])

    def test_empty_input_returns_empty_result(self):
        result = self.matrix.get_entries_for_groups([])
        self.assertEqual(result.entries, [])
        self.assertEqual(result.unmatched_keys, [])

    def test_unmatched_keys_preserve_order_with_matches(self):
        result = self.matrix.get_entries_for_groups(
            ["gfxFAKE1", "gfxNOPE", "gfxFAKE2", "gfxBAD"]
        )
        self.assertEqual([e.target for e in result.entries], ["gfxFAKE1", "gfxFAKE2"])
        self.assertEqual(result.unmatched_keys, ["gfxNOPE", "gfxBAD"])

    def test_deduplicate_removes_keys_resolving_to_same_target(self):
        # All three of these resolve to gfxFAKE1.
        result = self.matrix.get_entries_for_groups(
            ["gfxFAKEX", "gfxFAKEX-fakegroup", "gfxFAKE1"], deduplicate=True
        )
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0].target, "gfxFAKE1")

    def test_deduplicate_false_allows_duplicates(self):
        result = self.matrix.get_entries_for_groups(
            ["gfxFAKEX", "gfxFAKEX-fakegroup", "gfxFAKE1"], deduplicate=False
        )
        self.assertEqual(len(result.entries), 3)


# -----------------------------------------------------------------------------
# Serialization mechanics (to_dict / to_nested_dict), exercised on synthetic data.
# -----------------------------------------------------------------------------


class TestSerializationMechanics(unittest.TestCase):
    def setUp(self):
        self.matrix = _make_synthetic_matrix()

    def test_runners_to_dict_includes_named_and_extra_keys(self):
        runners = GpuRunners(
            test="t", test_multi_gpu="m", benchmark="b", extra={"oem": "o"}
        )
        d = runners.to_dict()
        self.assertEqual(d["test"], "t")
        self.assertEqual(d["test-multi-gpu"], "m")
        self.assertEqual(d["benchmark"], "b")
        self.assertEqual(d["oem"], "o")

    def test_matrix_entry_to_dict_no_platform(self):
        entry = self.matrix.get_entry("gfxFAKE1")
        d = entry.to_dict()
        self.assertEqual(d["amdgpu_family"], "gfxFAKE1")
        self.assertIn("linux", d)
        # gfxFAKE1 has no windows config → key omitted.
        self.assertNotIn("windows", d)

    def test_matrix_entry_to_dict_with_platform_is_flat(self):
        entry = self.matrix.get_entry("gfxFAKE1")
        d = entry.to_dict("linux")
        self.assertEqual(d["amdgpu_family"], "gfxFAKE1")
        # Flat layout: no nested 'linux' key.
        self.assertNotIn("linux", d)
        self.assertIn("build", d)
        self.assertIn("test", d)
        self.assertIn("release", d)

    def test_to_nested_dict_keyed_by_target(self):
        d = self.matrix.to_nested_dict()
        self.assertIn("gfxFAKE1", d)
        self.assertIn("gfxFAKE2", d)
        # Family names are NOT used as top-level keys.
        self.assertNotIn("gfxFAKEX", d)
        self.assertNotIn("gfxFAKEX-all", d)

    def test_to_nested_dict_full_layout(self):
        d = self.matrix.to_nested_dict()
        self.assertEqual(
            d["gfxFAKE1"],
            {
                "amdgpu_family": "gfxFAKE1",
                "linux": {
                    "build": {
                        "build_variants": ["release"],
                        "expect_failure": False,
                        "build_runs_on": "fake-build-runner",
                    },
                    "release": {"bypass_tests_for_releases": False},
                    "test": {
                        "fetch-gfx-targets": ["gfxFAKE1"],
                        "run_tests": True,
                        "runs_on": {
                            "benchmark": "",
                            "test": "fake-linux-runner",
                            "test-multi-gpu": "",
                        },
                        "sanity_check_only_for_family": False,
                        "test_scope": "comprehensive",
                        "bypass_tests_for_unscheduled": False,
                    },
                },
            },
        )


# -----------------------------------------------------------------------------
# Family auto-population from cmake (mechanic of AmdGpuFamilyMatrix).
# -----------------------------------------------------------------------------


class TestFamilyPopulation(unittest.TestCase):
    """Verify that entry.family is auto-populated from the cmake_families mapping."""

    def test_family_is_populated_from_cmake_families(self):
        matrix = _make_synthetic_matrix()
        entry = matrix.get_entry("gfxFAKE1")
        self.assertCountEqual(entry.family, ["gfxFAKEX-all", "gfxFAKEX-fakegroup"])

    def test_missing_family_in_cmake_raises(self):
        with self.assertRaises(ValueError):
            AmdGpuFamilyMatrix(
                entries=[MatrixEntry(target="gfxFAKE1")],
                cmake_families={},  # no entry for gfxFAKE1
            )


# -----------------------------------------------------------------------------
# Build variants lookup (small, real-data API).
# -----------------------------------------------------------------------------


class TestBuildVariants(unittest.TestCase):
    """Verify all_build_variants lookup behavior."""

    def test_known_variant_returns_info(self):
        bvi = all_build_variants.get("linux", "release")
        self.assertIsNotNone(bvi)
        self.assertEqual(bvi.label, "release")

    def test_unknown_platform_raises(self):
        with self.assertRaises(ValueError):
            all_build_variants.get("macos", "release")

    def test_unknown_variant_returns_none(self):
        self.assertIsNone(all_build_variants.get("linux", "nonexistent"))


# -----------------------------------------------------------------------------
# Weighted runner selection — exercise _get_gpu_runners via injected inventories so
# tests do not depend on the live matrix.
# -----------------------------------------------------------------------------


class TestWeightedRunnerSelection(unittest.TestCase):
    """Exercise the weighted-pick path inside `_get_gpu_runners`."""

    def test_single_label_no_random(self):
        # A solo entry is always returned, regardless of weight.
        inv = [
            _GpuRunnerEntry(
                platform="linux",
                target="gfxFAKE",
                role="test",
                label="solo-runner",
                weight=0.0,
            ),
        ]
        runners = _get_gpu_runners("linux", "gfxFAKE", inventory=inv)
        self.assertEqual(runners.test, "solo-runner")

    def test_weighted_pick_respects_zero_weight(self):
        # Mixed pool with one zero-weight entry: the zero-weight label must never
        # be chosen. Repeat to make a stuck-at-wrong-label bug obvious.
        inv = [
            _GpuRunnerEntry(
                platform="linux",
                target="gfxFAKE",
                role="test",
                label="never-pick",
                weight=0.0,
            ),
            _GpuRunnerEntry(
                platform="linux",
                target="gfxFAKE",
                role="test",
                label="always-pick",
                weight=1.0,
            ),
        ]
        for _ in range(50):
            runners = _get_gpu_runners("linux", "gfxFAKE", inventory=inv)
            self.assertEqual(runners.test, "always-pick")

    def test_extras_passed_through(self):
        # Roles outside the named-roles set land in `extra`, keyed by role name.
        inv = [
            _GpuRunnerEntry(
                platform="linux",
                target="gfxFAKE",
                role="test",
                label="t-runner",
            ),
            _GpuRunnerEntry(
                platform="linux",
                target="gfxFAKE",
                role="oem",
                label="oem-runner",
            ),
            _GpuRunnerEntry(
                platform="linux",
                target="gfxFAKE",
                role="test-sandbox",
                label="sandbox-runner",
            ),
        ]
        runners = _get_gpu_runners("linux", "gfxFAKE", inventory=inv)
        self.assertEqual(runners.test, "t-runner")
        self.assertEqual(
            runners.extra,
            {
                "oem": "oem-runner",
                "test-sandbox": "sandbox-runner",
            },
        )

    def test_no_match_returns_empty_runners(self):
        inv = [
            _GpuRunnerEntry(
                platform="linux",
                target="gfxFAKE",
                role="test",
                label="x",
            ),
        ]
        runners = _get_gpu_runners("windows", "gfxFAKE", inventory=inv)
        self.assertFalse(runners.has_any_runner())


# -----------------------------------------------------------------------------
# Build runner selection (separate inventory from GPU runners).
# -----------------------------------------------------------------------------


class TestBuildRunnerSelection(unittest.TestCase):
    """Exercise `_get_build_runner` against synthetic inventories."""

    def test_default_pool_picked_for_release(self):
        inv = [
            _BuildRunnerEntry(
                platform="linux", variant="default", label="default-runner", weight=1.0
            ),
            _BuildRunnerEntry(
                platform="linux", variant="sanitizer", label="san-runner", weight=1.0
            ),
        ]
        for variant in ("release", "release-package", ""):
            self.assertEqual(
                _get_build_runner("linux", variant, inventory=inv),
                "default-runner",
            )

    def test_sanitizer_pool_picked_for_san_variants(self):
        inv = [
            _BuildRunnerEntry(
                platform="linux", variant="default", label="default-runner"
            ),
            _BuildRunnerEntry(
                platform="linux", variant="sanitizer", label="san-runner"
            ),
        ]
        for variant in ("asan", "tsan", "linux-release-asan"):
            self.assertEqual(
                _get_build_runner("linux", variant, inventory=inv),
                "san-runner",
            )

    def test_sanitizer_falls_back_to_default_when_no_san_pool(self):
        # Windows has no sanitizer pool today; sanitizer requests must fall back.
        inv = [
            _BuildRunnerEntry(
                platform="windows", variant="default", label="windows-default"
            ),
        ]
        self.assertEqual(
            _get_build_runner("windows", "asan", inventory=inv),
            "windows-default",
        )

    def test_unknown_platform_returns_empty(self):
        inv = [
            _BuildRunnerEntry(platform="linux", variant="default", label="x"),
        ]
        self.assertEqual(_get_build_runner("macos", "release", inventory=inv), "")

# -----------------------------------------------------------------------------
# Build-variant filtering and re-binding on lookup paths.
# -----------------------------------------------------------------------------


class TestBuildVariantLookup(unittest.TestCase):
    """Exercise the build_variant kwarg on get_entry / get_entries_for_groups."""

    def _matrix(self) -> AmdGpuFamilyMatrix:
        # Two entries: one supports release+asan on linux, release on windows;
        # the other supports release-only on both platforms.
        multi = MatrixEntry(
            target="gfxMULTI",
            is_family_default=True,
            linux=PlatformConfig(
                build=BuildConfig(
                    build_variants=["release", "asan"],
                    runs_on="L-build-preset",
                ),
                test=TestConfig(runs_on=GpuRunners(test="L-test")),
            ),
            windows=PlatformConfig(
                build=BuildConfig(
                    build_variants=["release"],
                    runs_on="W-build-preset",
                ),
                test=TestConfig(runs_on=GpuRunners(test="W-test")),
            ),
        )
        single = MatrixEntry(
            target="gfxRELONLY",
            linux=PlatformConfig(
                build=BuildConfig(
                    build_variants=["release"],
                    runs_on="L-build-preset",
                ),
                test=TestConfig(runs_on=GpuRunners(test="L-test")),
            ),
        )
        return AmdGpuFamilyMatrix(
            entries=[multi, single],
            cmake_families={
                "gfxMULTI": ["gfxMULTIX-all"],
                "gfxRELONLY": ["gfxMULTIX-all"],
            },
        )

    def test_release_default_returns_entry_unchanged(self):
        matrix = self._matrix()
        entry = matrix.get_entry("gfxMULTI")
        self.assertIsNotNone(entry)
        # Caller-supplied label survives — no re-bind for release.
        self.assertEqual(entry.linux.build.runs_on, "L-build-preset")
        self.assertEqual(entry.windows.build.runs_on, "W-build-preset")
        # build_variants is scoped to the requested variant on every supported platform.
        self.assertEqual(entry.linux.build.build_variants, ["release"])
        self.assertEqual(entry.windows.build.build_variants, ["release"])

    def test_unsupported_variant_returns_none(self):
        # gfxRELONLY has no asan support on either platform.
        matrix = self._matrix()
        self.assertIsNone(matrix.get_entry("gfxRELONLY", build_variant="asan"))

    def test_partially_supported_variant_scopes_entry(self):
        # gfxMULTI supports asan on linux but not windows; the entry is scoped:
        # windows is dropped, linux's build_variants is trimmed to [asan].
        matrix = self._matrix()
        entry = matrix.get_entry("gfxMULTI", build_variant="asan")
        self.assertIsNotNone(entry)
        self.assertIsNotNone(entry.linux)
        self.assertIsNone(entry.windows)
        self.assertEqual(entry.linux.build.build_variants, ["asan"])
        # Serialization reflects the scope: no windows key in the dict.
        d = entry.to_dict()
        self.assertIn("linux", d)
        self.assertNotIn("windows", d)
        # The "preset" label is overwritten by the sanitizer pool pick.
        self.assertNotEqual(entry.linux.build.runs_on, "L-build-preset")
        self.assertIn("ramdisk", entry.linux.build.runs_on)

    def test_get_entries_for_groups_reports_unsupported_as_unmatched(self):
        matrix = self._matrix()
        result = matrix.get_entries_for_groups(
            ["gfxMULTI", "gfxRELONLY"], build_variant="asan"
        )
        self.assertEqual([e.target for e in result.entries], ["gfxMULTI"])
        self.assertEqual(result.unmatched_keys, ["gfxRELONLY"])

    def test_platform_config_supports_variant(self):
        # Direct predicate check — the scoping that get_entry applies to copies
        # would mask this if we went through a lookup.
        cfg = PlatformConfig(build=BuildConfig(build_variants=["release", "asan"]))
        self.assertTrue(cfg.supports_variant("asan"))
        self.assertFalse(cfg.supports_variant("tsan"))

    def test_lookups_return_independent_copies(self):
        # Re-binding for a non-release variant must not leak into a sibling
        # release lookup, and consumer-side mutation must not leak back into
        # the matrix.
        matrix = self._matrix()
        release = matrix.get_entry("gfxMULTI", build_variant="release")
        asan = matrix.get_entry("gfxMULTI", build_variant="asan")
        self.assertIsNot(release, asan)
        self.assertEqual(release.linux.build.runs_on, "L-build-preset")
        self.assertNotEqual(asan.linux.build.runs_on, release.linux.build.runs_on)
        # Mutate one copy; the other and a fresh lookup are unaffected.
        release.linux.build.runs_on = "mutated"
        fresh = matrix.get_entry("gfxMULTI")
        self.assertEqual(fresh.linux.build.runs_on, "L-build-preset")


# -----------------------------------------------------------------------------
# __post_init__ eager-fill behavior on MatrixEntry: build runner, GPU runners,
# and the deferred run_tests inference.
# -----------------------------------------------------------------------------


class TestEagerFill(unittest.TestCase):
    """Verify MatrixEntry.__post_init__ fills runners and infers run_tests."""

    def test_post_init_fills_build_runs_on_when_none(self):
        # Real inventory has linux+release rows, so the field gets a real label.
        entry = MatrixEntry(
            target="gfx1101",
            linux=PlatformConfig(build=BuildConfig()),
        )
        self.assertTrue(entry.linux.build.runs_on)
        self.assertNotEqual(entry.linux.build.runs_on, "")

    def test_post_init_preserves_caller_supplied_runs_on(self):
        entry = MatrixEntry(
            target="gfx1101",
            linux=PlatformConfig(build=BuildConfig(runs_on="explicit-label")),
        )
        self.assertEqual(entry.linux.build.runs_on, "explicit-label")

    def test_post_init_infers_run_tests_false_when_no_runner(self):
        # gfxNORUNNER has no rows in the real inventory → inferred False.
        entry = MatrixEntry(target="gfxNORUNNER", linux=PlatformConfig())
        self.assertFalse(entry.linux.test.run_tests)

    def test_explicit_run_tests_false_survives_post_init(self):
        # Caller-set False must not be overridden even when a runner is present.
        entry = MatrixEntry(
            target="gfx1101",
            linux=PlatformConfig(test=TestConfig(run_tests=False)),
        )
        self.assertFalse(entry.linux.test.run_tests)

    def test_to_dict_raises_when_build_runs_on_unresolved(self):
        # Bare BuildConfig (not part of an entry) keeps runs_on=None.
        bare = BuildConfig()
        with self.assertRaises(ValueError) as ctx:
            bare.to_dict()
        self.assertIn("inventory", str(ctx.exception))

    def test_to_dict_raises_when_run_tests_unresolved(self):
        # Bare TestConfig keeps run_tests=None — to_dict must refuse.
        bare = TestConfig()
        with self.assertRaises(ValueError) as ctx:
            bare.to_dict()
        self.assertIn("run_tests", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
