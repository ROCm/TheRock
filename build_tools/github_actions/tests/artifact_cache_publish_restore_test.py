# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Relational publication gates and actual archive restoration.

Synthetic producer metadata isolates the DB/storage boundary. Separate managed
build tests prove that configure-only or failed builds cannot publish outputs.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_db import CacheDB, ReadPolicy
from build_tools.github_actions.artifact_cache.manifest import (
    ArtifactKey, Manifest, OutputObject, Provenance, PublicationState, TrustLevel,
)
from build_tools.github_actions.artifact_cache.cache_key import ArtifactIdentity
from build_tools.github_actions.artifact_cache.planner import plan_cache
from build_tools.github_actions.artifact_cache.publish import publish_build_outputs
from build_tools.github_actions.artifact_cache.restore import (
    InMemoryObjectStore, restore_local_key, restore_manifest, restore_units,
)
from build_tools.github_actions.tests import _artifact_cache_fixtures as fixtures


class PublishGatesTest(fixtures.RelationalFixture):
    def test_legacy_output_list_cannot_assert_a_successful_build(self):
        with CacheDB(self.database) as db:
            with self.assertRaises(TypeError):
                publish_build_outputs(
                    db=db, artifact=ArtifactIdentity("blas", "lib", "generic"),
                    outputs=[OutputObject("blas_lib_generic.tar.zst", "s3://b/old-output")],
                    source_fingerprint=fixtures.FPRINT,
                )
            self.assertEqual(db.stats()["cache_entries"], 0)

    def test_publishes_complete_objects_after_success_record(self):
        manifest = self.publish_fixture()
        self.assertEqual(manifest.validation_state, "build_passed")
        self.assertEqual(set(manifest.expected_files), {"payload.tar", "inventory.json"})
        for file in manifest.files:
            self.assertTrue((Path(manifest.local_directory) / file.relative_path).is_file())
        with CacheDB(self.database) as db:
            self.assertEqual(db.count(), 1)
        self.assertFalse(manifest.payloads_verified)

    def test_not_published_when_outputs_incomplete(self):
        run, stored = self.producer(), self.stored()
        with CacheDB(self.database) as db:
            with self.assertRaisesRegex(ValueError, "archive and inventory"):
                db.register_local_outputs(producer_run_id=run,
                    outputs=[(self.key(), stored.directory, stored.files[:1])])
            self.assertEqual(db.count(), 0)
            self.assertEqual(db.stats()["cache_entries"], 0)
            self.assertEqual(db.stats()["artifact_files"], 0)

    def test_not_published_when_no_outputs(self):
        run = self.producer()
        with CacheDB(self.database) as db:
            with self.assertRaises(ValueError):
                db.register_local_outputs(producer_run_id=run, outputs=[])
            self.assertEqual(db.stats()["cache_entries"], 0)

    def test_not_published_without_managed_build_evidence(self):
        run = self.producer(successful_build=False)
        stored = self.stored()
        with CacheDB(self.database) as db:
            with self.assertRaisesRegex(ValueError, "managed-build"):
                db.register_local_outputs(producer_run_id=run,
                    outputs=[(self.key(), stored.directory, stored.files)])
            self.assertEqual(db.stats()["cache_entries"], 0)

    def test_untrusted_is_published_but_filtered_on_read(self):
        manifest = self.publish_fixture(trust=TrustLevel.UNTRUSTED)
        self.assertEqual(manifest.trust, TrustLevel.UNTRUSTED)
        with CacheDB(self.database) as db:
            self.assertIsNone(db.lookup(self.key(), ReadPolicy(min_trust=TrustLevel.TRUSTED)))
            self.assertIsNotNone(db.lookup(self.key(), ReadPolicy(min_trust=TrustLevel.UNTRUSTED)))

    def test_mismatched_observed_key_cannot_publish_existing_outputs(self):
        run, stored = self.producer(), self.stored()
        changed = ArtifactKey(fixtures.IDENTITY, "b" * 64)
        with CacheDB(self.database) as db:
            with self.assertRaisesRegex(ValueError, "exact artifact key"):
                db.register_local_outputs(producer_run_id=run,
                    outputs=[(changed, stored.directory, stored.files)])
            self.assertEqual(db.stats()["cache_entries"], 0)


class RestoreTest(fixtures.RelationalFixture):
    def restore(self):
        with CacheDB(self.database) as db:
            return restore_local_key(db, self.key(), self.root / "objects", self.root / "restored")

    def test_restores_verified_bytes_when_objects_present(self):
        self.publish_fixture()
        result = self.restore()
        self.assertEqual(result["outcome"], "restored")
        self.assertTrue(result["payloads_verified"])
        self.assertEqual((self.root / "restored/stage/payload.txt").read_bytes(), b"artifact version one\n")

    def test_needs_build_when_object_missing(self):
        manifest = self.publish_fixture()
        (Path(manifest.local_directory) / "payload.tar").unlink()
        result = self.restore()
        self.assertEqual(result["outcome"], "needs_build")
        self.assertEqual(result["rejected_entries"][0]["cache_entry_id"], manifest.cache_entry_id)
        self.assertFalse((self.root / "restored").exists())

    def test_needs_build_when_digest_mismatch(self):
        manifest = self.publish_fixture()
        archive = Path(manifest.local_directory) / "payload.tar"
        data = bytearray(archive.read_bytes())
        data[0] ^= 1
        archive.write_bytes(data)
        result = self.restore()
        self.assertEqual(result["outcome"], "needs_build")
        self.assertIn("digest", result["rejected_entries"][0]["reason"])
        self.assertFalse((self.root / "restored").exists())

    def test_restore_units_passes_misses_through_without_creating_destination(self):
        with CacheDB(self.database) as db:
            plan = plan_cache(db=db, records=[({"artifact_identity": fixtures.IDENTITY}, self.key())])
            results = restore_units(plan.entries, db=db, cache_root=self.root / "objects",
                                    destination=self.root / "restored")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["outcome"], "needs_build")
        self.assertFalse((self.root / "restored").exists())


class RetainedRestoreModelTest(unittest.TestCase):
    def test_unknown_digest_is_not_a_verified_reuse(self):
        manifest = Manifest(
            cache_key="legacy-estimate", artifact=ArtifactIdentity("blas", "lib", "generic"),
            outputs=(OutputObject("blas_lib_generic.tar.zst", "s3://b/object"),),
            provenance=Provenance("ROCm/TheRock", "1", "main", "commit", "ci.yml", "2026-09-01T00:00:00Z"),
            publication=PublicationState.PUBLISHED,
        )
        self.assertFalse(restore_manifest(manifest, InMemoryObjectStore()).reusable)


if __name__ == "__main__":
    unittest.main()
