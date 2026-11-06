# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for fingerprint_cache.publish (flowchart 3) and restore (flowchart 2)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprint_cache.cache_key import ArtifactIdentity
from fingerprint_cache.cache_db import CacheDB
from fingerprint_cache.manifest import (
    OutputObject,
    Provenance,
    PublicationState,
    TrustLevel,
    ValidationState,
)
from fingerprint_cache.planner import PlanEntry, RequiredUnit
from fingerprint_cache.cache_key import compute_cache_key
from fingerprint_cache.publish import publish_build_outputs
from fingerprint_cache.restore import (
    InMemoryObjectStore,
    restore_manifest,
    restore_units,
)

ARTIFACT = ArtifactIdentity("blas", "bundle", "gfx94X-dcgpu")
PROV = Provenance(
    "ROCm/TheRock", "123", "main", "c0ffee", "multi_arch_ci.yml", "2026-09-01T00:00:00Z"
)


def _outputs(uri="s3://b/blas_lib.tar.zst", digest=None):
    return [OutputObject("blas_lib_gfx94X-dcgpu.tar.zst", uri, digest, 10)]


class PublishGatesTest(unittest.TestCase):
    def test_publishes_complete_build_passed(self):
        with CacheDB() as db:
            m = publish_build_outputs(
                db=db,
                artifact=ARTIFACT,
                outputs=_outputs(),
                provenance=PROV,
                source_fingerprint="fp1",
                trust=TrustLevel.TRUSTED,
                validation=ValidationState.BUILD_PASSED,
            )
            self.assertIsNotNone(m)
            self.assertEqual(m.publication, PublicationState.PUBLISHED)
            self.assertEqual(db.count(), 1)

    def test_not_published_when_outputs_incomplete(self):
        with CacheDB() as db:
            m = publish_build_outputs(
                db=db,
                artifact=ARTIFACT,
                outputs=_outputs(uri=""),  # missing URI
                provenance=PROV,
                source_fingerprint="fp1",
            )
            self.assertIsNone(m)
            self.assertEqual(db.count(), 0)

    def test_not_published_when_no_outputs(self):
        with CacheDB() as db:
            m = publish_build_outputs(
                db=db,
                artifact=ARTIFACT,
                outputs=[],
                provenance=PROV,
                source_fingerprint="fp1",
            )
            self.assertIsNone(m)

    def test_not_published_when_validation_insufficient(self):
        with CacheDB() as db:
            m = publish_build_outputs(
                db=db,
                artifact=ARTIFACT,
                outputs=_outputs(),
                provenance=PROV,
                source_fingerprint="fp1",
                validation=ValidationState.UNVALIDATED,
            )
            self.assertIsNone(m)
            self.assertEqual(db.count(), 0)

    def test_untrusted_is_published_but_tagged(self):
        with CacheDB() as db:
            m = publish_build_outputs(
                db=db,
                artifact=ARTIFACT,
                outputs=_outputs(),
                provenance=PROV,
                source_fingerprint="fp1",
                trust=TrustLevel.UNTRUSTED,
                validation=ValidationState.BUILD_PASSED,
            )
            self.assertIsNotNone(m)
            self.assertEqual(m.trust, TrustLevel.UNTRUSTED)


class RestoreTest(unittest.TestCase):
    def _published_manifest(self, db, digest=None):
        return publish_build_outputs(
            db=db,
            artifact=ARTIFACT,
            outputs=_outputs(digest=digest),
            provenance=PROV,
            source_fingerprint="fp1",
            trust=TrustLevel.TRUSTED,
            validation=ValidationState.BUILD_PASSED,
        )

    def test_reusable_when_objects_present(self):
        with CacheDB() as db:
            m = self._published_manifest(db)
            result = restore_manifest(m, InMemoryObjectStore())
            self.assertTrue(result.reusable)

    def test_needs_build_when_object_missing(self):
        with CacheDB() as db:
            m = self._published_manifest(db)
            store = InMemoryObjectStore(missing=[m.outputs[0].uri])
            result = restore_manifest(m, store)
            self.assertFalse(result.reusable)
            self.assertIn("missing", result.reason)

    def test_needs_build_when_digest_mismatch(self):
        with CacheDB() as db:
            m = self._published_manifest(db, digest="expected_digest")
            store = InMemoryObjectStore(bad_digest=[m.outputs[0].uri])
            result = restore_manifest(m, store, verify_digests=True)
            self.assertFalse(result.reusable)
            self.assertIn("digest", result.reason)

    def test_restore_units_passes_misses_through(self):
        # A miss plan entry (no manifest) resolves to needs_build.
        key = compute_cache_key(source_fingerprint="fp", build_variant="release")
        unit = RequiredUnit(ARTIFACT, "math", key, source_changed=True)
        entry = PlanEntry(unit, "miss", "source changed")
        results = restore_units([entry], InMemoryObjectStore())
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].reusable)


if __name__ == "__main__":
    unittest.main()
