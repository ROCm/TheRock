# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for fingerprint_cache.cache_db (publish ordering, batch lookup, policy)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprint_cache.cache_key import ArtifactIdentity
from fingerprint_cache.cache_db import CacheDB, ReadPolicy
from fingerprint_cache.manifest import (
    Manifest,
    OutputObject,
    Provenance,
    PublicationState,
    TrustLevel,
    ValidationState,
)


def _manifest(
    key: str,
    *,
    trust=TrustLevel.TRUSTED,
    validation=ValidationState.BUILD_PASSED,
    publication=PublicationState.PUBLISHED,
    created_at="2026-09-28T00:00:00Z",
    run_id="123",
    uri="s3://b/obj.tar.zst",
) -> Manifest:
    return Manifest(
        cache_key=key,
        artifact=ArtifactIdentity("blas", "bundle", "gfx94X-dcgpu"),
        outputs=(OutputObject("blas_lib_gfx94X-dcgpu.tar.zst", uri, "d1", 10),),
        provenance=Provenance(
            repository="ROCm/TheRock",
            run_id=run_id,
            branch="main",
            commit="c0ffee",
            workflow="multi_arch_ci.yml",
            created_at=created_at,
        ),
        trust=trust,
        validation=validation,
        publication=publication,
    )


class PublishGuardTest(unittest.TestCase):
    def test_refuses_pending(self):
        with CacheDB() as db:
            with self.assertRaises(ValueError):
                db.publish(_manifest("k1", publication=PublicationState.PENDING))

    def test_refuses_incomplete(self):
        with CacheDB() as db:
            m = _manifest("k1", uri="")  # no URI -> incomplete
            with self.assertRaises(ValueError):
                db.publish(m)

    def test_publish_is_idempotent(self):
        with CacheDB() as db:
            db.publish(_manifest("k1"))
            db.publish(_manifest("k1", run_id="999"))
            self.assertEqual(db.count(), 1)
            self.assertEqual(db.lookup("k1").provenance.run_id, "999")


class BatchLookupTest(unittest.TestCase):
    def test_batch_hits_and_misses(self):
        with CacheDB() as db:
            db.publish(_manifest("k1"))
            db.publish(_manifest("k2"))
            result = db.lookup_batch(["k1", "k2", "k3"])
            self.assertIsNotNone(result["k1"])
            self.assertIsNotNone(result["k2"])
            self.assertIsNone(result["k3"])

    def test_batch_dedupes_keys(self):
        with CacheDB() as db:
            db.publish(_manifest("k1"))
            result = db.lookup_batch(["k1", "k1", "k1"])
            self.assertEqual(set(result.keys()), {"k1"})
            self.assertIsNotNone(result["k1"])


class ReadPolicyTest(unittest.TestCase):
    def test_trust_filter(self):
        with CacheDB() as db:
            db.publish(_manifest("untrusted", trust=TrustLevel.UNTRUSTED))
            strict = db.lookup("untrusted", ReadPolicy(min_trust=TrustLevel.TRUSTED))
            loose = db.lookup("untrusted", ReadPolicy(min_trust=TrustLevel.UNTRUSTED))
            self.assertIsNone(strict)
            self.assertIsNotNone(loose)

    def test_validation_filter(self):
        with CacheDB() as db:
            db.publish(_manifest("unval", validation=ValidationState.UNVALIDATED))
            # UNVALIDATED can still be published? No -- default policy publishes
            # build_passed only, but the DB itself does not gate validation. It
            # requires PUBLISHED + complete, which this is.
            strict = db.lookup(
                "unval", ReadPolicy(min_validation=ValidationState.BUILD_PASSED)
            )
            loose = db.lookup(
                "unval", ReadPolicy(min_validation=ValidationState.UNVALIDATED)
            )
            self.assertIsNone(strict)
            self.assertIsNotNone(loose)

    def test_point_in_time_filter(self):
        with CacheDB() as db:
            db.publish(_manifest("late", created_at="2026-09-28T12:00:00Z"))
            before = db.lookup(
                "late", ReadPolicy(created_before="2026-09-28T06:00:00Z")
            )
            after = db.lookup("late", ReadPolicy(created_before="2026-09-28T18:00:00Z"))
            self.assertIsNone(before)  # entry created after the cutoff
            self.assertIsNotNone(after)


class EvictionTest(unittest.TestCase):
    def test_evict_before(self):
        with CacheDB() as db:
            db.publish(_manifest("old", created_at="2026-01-01T00:00:00Z"))
            db.publish(_manifest("new", created_at="2026-09-01T00:00:00Z"))
            removed = db.evict_before("2026-06-01T00:00:00Z")
            self.assertEqual(removed, 1)
            self.assertIsNone(db.lookup("old"))
            self.assertIsNotNone(db.lookup("new"))

    def test_evict_where_missing(self):
        with CacheDB() as db:
            db.publish(_manifest("present", uri="s3://b/here.tar.zst"))
            db.publish(_manifest("gone", uri="s3://b/gone.tar.zst"))
            removed = db.evict_where_missing(lambda uri: uri != "s3://b/gone.tar.zst")
            self.assertEqual(removed, 1)
            self.assertIsNone(db.lookup("gone"))
            self.assertIsNotNone(db.lookup("present"))


class StatsAndDiagnosticsTest(unittest.TestCase):
    def test_stats_and_run_lookup(self):
        with CacheDB() as db:
            db.publish(_manifest("k1", run_id="500"))
            db.publish(_manifest("k2", run_id="500", trust=TrustLevel.UNTRUSTED))
            self.assertEqual(db.stats()["total"], 2)
            self.assertEqual(len(db.get_by_run_id("500")), 2)


if __name__ == "__main__":
    unittest.main()
