# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CI metadata model and relational manifest invariants.

Manifest.is_servable checks metadata only. Neither that method
nor LocalManifest construction verifies bytes or authorizes stage skipping.
"""

from dataclasses import replace
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_key import ArtifactIdentity
from build_tools.github_actions.artifact_cache.manifest import (
    ArtifactFile, ArtifactKey, LocalManifest, Manifest, OutputObject, Provenance,
    PublicationState, TrustLevel, ValidationState,
)


def make_manifest(**overrides):
    values = dict(
        cache_key="e1:release:deadbeef",
        artifact=ArtifactIdentity("blas", "bundle", "gfx94X-dcgpu"),
        outputs=(OutputObject("blas_lib_gfx94X-dcgpu.tar.zst", "s3://b/blas", "a" * 64, 10),),
        provenance=Provenance("ROCm/TheRock", "123", "main", "c0ffee", "ci.yml",
                              "2026-09-01T00:00:00Z"),
        trust=TrustLevel.TRUSTED, validation=ValidationState.BUILD_PASSED,
        publication=PublicationState.PUBLISHED,
    )
    values.update(overrides)
    return Manifest(**values)


class EnumOrderingTest(unittest.TestCase):
    def test_trust_meets(self):
        self.assertTrue(TrustLevel.TRUSTED.meets(TrustLevel.UNTRUSTED))
        self.assertFalse(TrustLevel.UNTRUSTED.meets(TrustLevel.TRUSTED))

    def test_validation_meets(self):
        self.assertTrue(ValidationState.TESTS_PASSED.meets(ValidationState.BUILD_PASSED))
        self.assertFalse(ValidationState.UNVALIDATED.meets(ValidationState.BUILD_PASSED))


class ManifestStateTest(unittest.TestCase):
    def test_complete_and_servable_metadata(self):
        manifest = make_manifest()
        self.assertTrue(manifest.is_complete())
        self.assertTrue(manifest.is_servable())

    def test_incomplete_when_missing_uri(self):
        manifest = make_manifest(outputs=(OutputObject("x.tar.zst", ""),))
        self.assertFalse(manifest.is_complete())
        self.assertFalse(manifest.is_servable())

    def test_not_servable_when_pending(self):
        manifest = make_manifest(publication=PublicationState.PENDING)
        self.assertTrue(manifest.is_complete())
        self.assertFalse(manifest.is_servable())

    def test_not_servable_when_no_outputs(self):
        self.assertFalse(make_manifest(outputs=()).is_complete())


class ManifestRoundTripTest(unittest.TestCase):
    def test_json_roundtrip(self):
        manifest = make_manifest()
        restored = Manifest.from_json(manifest.to_json())
        self.assertEqual(restored, manifest)
        self.assertEqual(restored.to_dict(), manifest.to_dict())


class RelationalManifestTest(unittest.TestCase):
    def setUp(self):
        self.manifest = LocalManifest(
            cache_entry_id=1, artifact_key_id=2,
            key=ArtifactKey("blas_lib_generic", "a" * 64),
            producer_run_id=3, producer_commit="commit", producer_label="test",
            local_directory="/not-read-by-this-test",
            expected_files=("inventory.json", "payload.tar"),
            files=(ArtifactFile("inventory.json", "b" * 64, 10),
                   ArtifactFile("payload.tar", "c" * 64, 20)),
            validation_state="build_passed", validated_at=10,
            created_at=10, expires_at=None,
        )

    def test_complete_metadata_still_requires_physical_verification(self):
        self.assertFalse(self.manifest.payloads_verified)
        data = self.manifest.to_dict()
        self.assertFalse(data["payloads_verified"])
        self.assertEqual(data["lookup_scope"], "database_metadata")
        self.assertEqual(self.manifest.trust, TrustLevel.UNTRUSTED)

    def test_missing_or_duplicate_inventory_record_is_rejected(self):
        for files in (self.manifest.files[:1], self.manifest.files * 2):
            with self.subTest(files=files), self.assertRaises(ValueError):
                replace(self.manifest, files=files)


if __name__ == "__main__":
    unittest.main()
