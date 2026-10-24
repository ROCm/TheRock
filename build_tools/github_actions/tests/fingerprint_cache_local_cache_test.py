# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for local builds (local_store + local_cache)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprint_cache.local_store import LocalObjectStore, sha256_file
from fingerprint_cache import local_cache


class LocalObjectStoreTest(unittest.TestCase):
    def test_upload_exists_digest_download_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "art.tar.zst"
            src.write_text("hello artifact")
            store = LocalObjectStore(d / "store")
            obj = store.upload_file(src)
            self.assertTrue(store.exists(obj.uri))
            self.assertEqual(obj.digest, sha256_file(src))
            self.assertEqual(store.fetch_digest(obj.uri), obj.digest)
            out = store.download_file(obj.uri, d / "out.tar.zst")
            self.assertEqual(out.read_text(), "hello artifact")

    def test_missing_object(self):
        with tempfile.TemporaryDirectory() as d:
            store = LocalObjectStore(Path(d) / "store")
            self.assertFalse(store.exists("local://nope.tar.zst"))
            self.assertIsNone(store.fetch_digest("local://nope.tar.zst"))


class LocalCacheReuseTest(unittest.TestCase):
    def _run(self, *argv) -> int:
        return local_cache.main(list(argv))

    def test_build_once_stores_then_reuses(self):
        with tempfile.TemporaryDirectory() as root:
            build = str(Path(root) / "build")
            cache = str(Path(root) / "cache")
            self._run("synth", "--out", build)
            self._run(
                "cache", "--build", build, "--cache", cache
            )  # first build: stores
            self._run("cache", "--build", build, "--cache", cache)  # again: reuses
            scan = local_cache.scan_build(Path(build))
            self.assertTrue(scan.units)
            # Warm with identical sources: every cacheable unit must be a hit.
            from fingerprint_cache.cache_db import CacheDB, ReadPolicy
            from fingerprint_cache.cache_key import compute_cache_key
            from fingerprint_cache.manifest import TrustLevel, ValidationState
            from fingerprint_cache.local_store import LocalObjectStore
            from fingerprint_cache.restore import restore_manifest

            store = LocalObjectStore(Path(cache) / "objects")
            policy = ReadPolicy(
                min_trust=TrustLevel.TRUSTED,
                min_validation=ValidationState.BUILD_PASSED,
            )
            with CacheDB(str(Path(cache) / "cache.sqlite")) as db:
                for unit in scan.units:
                    key = compute_cache_key(
                        source_fingerprint=unit.source_fingerprint,
                        build_variant="release",
                    )
                    m = db.lookup(key.value, policy)
                    self.assertIsNotNone(m, f"{unit.identity} should be cached")
                    self.assertTrue(restore_manifest(m, store).reusable)

    def test_split_artifact_is_uncacheable(self):
        # miopen is synthesized as a split artifact with no .fprint (the real gap).
        with tempfile.TemporaryDirectory() as root:
            build = str(Path(root) / "build")
            self._run("synth", "--out", build)
            scan = local_cache.scan_build(Path(build))
            cacheable_names = {u.identity.name for u in scan.units}
            self.assertNotIn("miopen", cacheable_names)
            self.assertTrue(any(s.startswith("miopen_") for s in scan.uncacheable))

    def test_source_change_rebuilds_only_changed_unit(self):
        with tempfile.TemporaryDirectory() as root:
            build = str(Path(root) / "build")
            cache = str(Path(root) / "cache")
            self._run("synth", "--out", build)
            self._run("cache", "--build", build, "--cache", cache)
            # Flip only blas's fingerprint (models a rocBLAS source edit).
            self._run("synth", "--out", build, "--change", "blas")

            from fingerprint_cache.cache_db import CacheDB, ReadPolicy
            from fingerprint_cache.cache_key import compute_cache_key
            from fingerprint_cache.manifest import TrustLevel, ValidationState

            policy = ReadPolicy(
                min_trust=TrustLevel.TRUSTED,
                min_validation=ValidationState.BUILD_PASSED,
            )
            scan = local_cache.scan_build(Path(build))
            misses = []
            with CacheDB(str(Path(cache) / "cache.sqlite")) as db:
                for unit in scan.units:
                    key = compute_cache_key(
                        source_fingerprint=unit.source_fingerprint,
                        build_variant="release",
                    )
                    if db.lookup(key.value, policy) is None:
                        misses.append(unit.identity.name)
            self.assertEqual(misses, ["blas"])


if __name__ == "__main__":
    unittest.main()
