# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Key regressions: estimate helpers and relational ArtifactKey.

compute_cache_key returns a derived estimate key. Database lookup requires ArtifactKey.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_key import (
    CACHE_EPOCH, ArtifactIdentity, CacheKey, compute_cache_key,
    derive_fallback_fingerprint,
)
from build_tools.github_actions.artifact_cache.manifest import ArtifactKey


class ComputeCacheKeyTest(unittest.TestCase):
    def test_deterministic(self):
        a = compute_cache_key(source_fingerprint="abc", build_variant="release")
        b = compute_cache_key(source_fingerprint="abc", build_variant="release")
        self.assertEqual(a, b)
        self.assertEqual(a.value, b.value)

    def test_variant_changes_key(self):
        a = compute_cache_key(source_fingerprint="abc", build_variant="release")
        b = compute_cache_key(source_fingerprint="abc", build_variant="asan")
        self.assertNotEqual(a.value, b.value)

    def test_source_fingerprint_changes_key(self):
        a = compute_cache_key(source_fingerprint="abc", build_variant="release")
        b = compute_cache_key(source_fingerprint="xyz", build_variant="release")
        self.assertNotEqual(a.value, b.value)

    def test_cmake_hash_and_extra_inputs_change_key(self):
        options = dict(source_fingerprint="abc", build_variant="release")
        base = compute_cache_key(**options)
        self.assertNotEqual(base, compute_cache_key(**options, cmake_files_hash="recipe"))
        self.assertNotEqual(base, compute_cache_key(**options, extra_inputs=["docker:1"]))

    def test_epoch_in_value_and_bump_changes_key(self):
        options = dict(source_fingerprint="abc", build_variant="release")
        key = compute_cache_key(**options)
        self.assertTrue(key.value.startswith(f"e{CACHE_EPOCH}:release:"))
        self.assertNotEqual(key, compute_cache_key(**options, epoch=CACHE_EPOCH + 1))

    def test_key_short_form(self):
        key = CacheKey(fingerprint="0123456789abcdef" * 4, build_variant="release")
        self.assertEqual(key.short, "release:0123456789ab")


class DeriveFallbackFingerprintTest(unittest.TestCase):
    def setUp(self):
        self.artifact = ArtifactIdentity("blas", "bundle", "gfx94X-dcgpu")

    def test_prefixed_so_no_collision_with_real(self):
        fp = derive_fallback_fingerprint(source_commit="c0ffee", artifact=self.artifact)
        self.assertTrue(fp.startswith("srcfp1:"))
        self.assertTrue(compute_cache_key(source_fingerprint=fp,
                                         build_variant="release").is_fallback_fingerprint)
        with self.assertRaises(ValueError):
            ArtifactKey("blas_lib_gfx94X-dcgpu", fp)

    def test_commit_changes_fingerprint(self):
        a = derive_fallback_fingerprint(source_commit="aaaa", artifact=self.artifact)
        b = derive_fallback_fingerprint(source_commit="bbbb", artifact=self.artifact)
        self.assertNotEqual(a, b)

    def test_artifact_identity_changes_fingerprint(self):
        other = ArtifactIdentity("fft", "bundle", "gfx94X-dcgpu")
        a = derive_fallback_fingerprint(source_commit="aaaa", artifact=self.artifact)
        b = derive_fallback_fingerprint(source_commit="aaaa", artifact=other)
        self.assertNotEqual(a, b)

    def test_dep_fingerprints_order_independent(self):
        args = dict(source_commit="aaaa", artifact=self.artifact)
        self.assertEqual(derive_fallback_fingerprint(**args, dep_fingerprints=["x", "y"]),
                         derive_fallback_fingerprint(**args, dep_fingerprints=["y", "x"]))


class RelationalKeyTest(unittest.TestCase):
    def test_raw_fingerprint_is_preserved_and_epoch_is_separate(self):
        raw = "a" * 64
        a = ArtifactKey("hipdnn_lib_generic", raw, epoch=1)
        b = ArtifactKey("hipdnn_lib_generic", raw, epoch=2)
        self.assertEqual(a.fingerprint, raw)
        self.assertEqual(b.fingerprint, raw)
        self.assertNotEqual(a, b)

    def test_same_fingerprint_does_not_merge_components_or_bundle_membership(self):
        raw = "a" * 64
        members = ("hipdnn_dev_generic", "hipdnn_lib_generic")
        lib = ArtifactKey(members[1], raw)
        dev = ArtifactKey(members[0], raw)
        full = ArtifactKey("hipdnn/bundle/generic", raw, components=members)
        partial = ArtifactKey("hipdnn/bundle/generic", raw, components=(members[1],))
        self.assertEqual(len({lib, dev, full, partial}), 4)


if __name__ == "__main__":
    unittest.main()
