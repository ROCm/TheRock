# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for fingerprint_cache.manifest."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprint_cache.cache_key import ArtifactIdentity
from fingerprint_cache.manifest import (
    Manifest,
    OutputObject,
    Provenance,
    PublicationState,
    TrustLevel,
    ValidationState,
)


def _manifest(**overrides) -> Manifest:
    defaults = dict(
        cache_key="e1:release:deadbeef",
        artifact=ArtifactIdentity("blas", "bundle", "gfx94X-dcgpu"),
        outputs=(
            OutputObject("blas_lib_gfx94X-dcgpu.tar.zst", "s3://b/blas", "d1", 10),
        ),
        provenance=Provenance(
            repository="ROCm/TheRock",
            run_id="123",
            branch="main",
            commit="c0ffee",
            workflow="multi_arch_ci.yml",
            created_at="2026-09-28T00:00:00Z",
        ),
        trust=TrustLevel.TRUSTED,
        validation=ValidationState.BUILD_PASSED,
        publication=PublicationState.PUBLISHED,
    )
    defaults.update(overrides)
    return Manifest(**defaults)


class EnumOrderingTest(unittest.TestCase):
    def test_trust_meets(self):
        self.assertTrue(TrustLevel.TRUSTED.meets(TrustLevel.UNTRUSTED))
        self.assertFalse(TrustLevel.UNTRUSTED.meets(TrustLevel.TRUSTED))

    def test_validation_meets(self):
        self.assertTrue(
            ValidationState.TESTS_PASSED.meets(ValidationState.BUILD_PASSED)
        )
        self.assertFalse(
            ValidationState.UNVALIDATED.meets(ValidationState.BUILD_PASSED)
        )


class ManifestStateTest(unittest.TestCase):
    def test_complete_and_servable(self):
        m = _manifest()
        self.assertTrue(m.is_complete())
        self.assertTrue(m.is_servable())

    def test_incomplete_when_missing_uri(self):
        m = _manifest(outputs=(OutputObject("x.tar.zst", ""),))
        self.assertFalse(m.is_complete())
        self.assertFalse(m.is_servable())

    def test_not_servable_when_pending(self):
        m = _manifest(publication=PublicationState.PENDING)
        self.assertTrue(m.is_complete())
        self.assertFalse(m.is_servable())

    def test_not_servable_when_no_outputs(self):
        m = _manifest(outputs=())
        self.assertFalse(m.is_complete())


class ManifestRoundTripTest(unittest.TestCase):
    def test_json_roundtrip(self):
        m = _manifest()
        restored = Manifest.from_json(m.to_json())
        self.assertEqual(restored.cache_key, m.cache_key)
        self.assertEqual(restored.artifact, m.artifact)
        self.assertEqual(restored.outputs, m.outputs)
        self.assertEqual(restored.provenance, m.provenance)
        self.assertEqual(restored.trust, m.trust)
        self.assertEqual(restored.validation, m.validation)
        self.assertEqual(restored.publication, m.publication)
        self.assertEqual(restored.to_dict(), m.to_dict())


if __name__ == "__main__":
    unittest.main()
