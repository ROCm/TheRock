# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the review-attestation digest in pr_quality_check.py.

Run locally:
    python -m unittest skills/rocm-pr-quality/tools/pr_quality_check_test.py -v
    # or
    pytest skills/rocm-pr-quality/tools/pr_quality_check_test.py
"""

import hashlib
import sys
import unittest
from pathlib import Path

# Make `pr_quality_check` importable regardless of the working directory.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pr_quality_check as pq  # noqa: E402


def _expected(key: str) -> str:
    """The documented algorithm (reference.md), computed independently."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


class AttestationDigestTest(unittest.TestCase):
    def test_known_value(self):
        # Pinned so an accidental algorithm change is caught.
        self.assertEqual(
            pq.attestation_digest("https://github.com/ROCm/TheRock/pull/6653"),
            "46012600",
        )

    def test_matches_documented_algorithm(self):
        self.assertEqual(
            pq.attestation_digest("https://github.com/ROCm/TheRock/pull/6653"),
            _expected("rocm/therock#6653"),
        )

    def test_is_eight_lowercase_hex_chars(self):
        digest = pq.attestation_digest("https://github.com/ROCm/TheRock/pull/1")
        self.assertRegex(digest, r"^[0-9a-f]{8}$")

    def test_case_insensitive_on_owner_and_repo(self):
        self.assertEqual(
            pq.attestation_digest("https://github.com/ROCm/TheRock/pull/6653"),
            pq.attestation_digest("https://github.com/rocm/therock/pull/6653"),
        )

    def test_differs_by_pr_number(self):
        self.assertNotEqual(
            pq.attestation_digest("https://github.com/ROCm/TheRock/pull/6653"),
            pq.attestation_digest("https://github.com/ROCm/TheRock/pull/6654"),
        )

    def test_differs_by_repo(self):
        # PR numbers collide across repositories, so the repo must be in the input.
        self.assertNotEqual(
            pq.attestation_digest("https://github.com/ROCm/TheRock/pull/6653"),
            pq.attestation_digest("https://github.com/ROCm/rocm-libraries/pull/6653"),
        )

    def test_differs_by_owner(self):
        self.assertNotEqual(
            pq.attestation_digest("https://github.com/ROCm/TheRock/pull/6653"),
            pq.attestation_digest("https://github.com/someone/TheRock/pull/6653"),
        )

    def test_stable_across_trailing_pr_url_path(self):
        # The same PR reached via a sub-page must yield the same digest.
        base = pq.attestation_digest("https://github.com/ROCm/TheRock/pull/6653")
        self.assertEqual(
            pq.attestation_digest("https://github.com/ROCm/TheRock/pull/6653/files"),
            base,
        )

    def test_non_pr_url_returns_none(self):
        for bad in (
            "",
            None,
            "garbage",
            "https://github.com/ROCm/TheRock/issues/6653",
            "https://github.com/ROCm/TheRock/pull/",
            "https://github.com/ROCm/TheRock/pull/abc",
        ):
            with self.subTest(url=bad):
                self.assertIsNone(pq.attestation_digest(bad))


if __name__ == "__main__":
    unittest.main()
