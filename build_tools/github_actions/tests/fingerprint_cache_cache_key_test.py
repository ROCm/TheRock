# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for fingerprint_cache.cache_key."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprint_cache.cache_key import (
    CACHE_EPOCH,
    ArtifactIdentity,
    CacheKey,
    compute_cache_key,
    derive_fallback_fingerprint,
)


class ComputeCacheKeyTest(unittest.TestCase):
    def test_deterministic(self):
        k1 = compute_cache_key(source_fingerprint="abc", build_variant="release")
        k2 = compute_cache_key(source_fingerprint="abc", build_variant="release")
        self.assertEqual(k1, k2)
        self.assertEqual(k1.value, k2.value)

    def test_variant_changes_key(self):
        k1 = compute_cache_key(source_fingerprint="abc", build_variant="release")
        k2 = compute_cache_key(source_fingerprint="abc", build_variant="asan")
        self.assertNotEqual(k1.value, k2.value)

    def test_source_fingerprint_changes_key(self):
        k1 = compute_cache_key(source_fingerprint="abc", build_variant="release")
        k2 = compute_cache_key(source_fingerprint="xyz", build_variant="release")
        self.assertNotEqual(k1.value, k2.value)

    def test_cmake_hash_and_extra_inputs_change_key(self):
        base = compute_cache_key(source_fingerprint="abc", build_variant="release")
        with_cmake = compute_cache_key(
            source_fingerprint="abc",
            build_variant="release",
            cmake_files_hash="deadbeef",
        )
        with_extra = compute_cache_key(
            source_fingerprint="abc", build_variant="release", extra_inputs=["docker:1"]
        )
        self.assertNotEqual(base.value, with_cmake.value)
        self.assertNotEqual(base.value, with_extra.value)

    def test_epoch_in_value_and_bump_changes_key(self):
        k = compute_cache_key(source_fingerprint="abc", build_variant="release")
        self.assertTrue(k.value.startswith(f"e{CACHE_EPOCH}:release:"))
        bumped = compute_cache_key(
            source_fingerprint="abc", build_variant="release", epoch=CACHE_EPOCH + 1
        )
        self.assertNotEqual(k.value, bumped.value)

    def test_key_short_form(self):
        k = CacheKey(fingerprint="0123456789abcdef" * 4, build_variant="release")
        self.assertEqual(k.short, "release:0123456789ab")


class DeriveFallbackFingerprintTest(unittest.TestCase):
    def setUp(self):
        self.artifact = ArtifactIdentity("blas", "bundle", "gfx94X-dcgpu")

    def test_prefixed_so_no_collision_with_real(self):
        fp = derive_fallback_fingerprint(source_commit="c0ffee", artifact=self.artifact)
        self.assertTrue(fp.startswith("srcfp1:"))
        key = compute_cache_key(source_fingerprint=fp, build_variant="release")
        self.assertTrue(key.is_fallback_fingerprint)

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
        a = derive_fallback_fingerprint(
            source_commit="aaaa", artifact=self.artifact, dep_fingerprints=["x", "y"]
        )
        b = derive_fallback_fingerprint(
            source_commit="aaaa", artifact=self.artifact, dep_fingerprints=["y", "x"]
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
