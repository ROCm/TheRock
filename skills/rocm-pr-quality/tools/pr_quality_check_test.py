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
from datetime import datetime, timedelta, timezone
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


PR_URL = "https://github.com/ROCm/TheRock/pull/6653"
OTHER_PR_URL = "https://github.com/ROCm/TheRock/pull/6654"
FULL_SHA = "d9c0aae856895ed1ddc99e71654d47b1649e8e6c"

# The block as the skill drafts it (see reference.md), for PR_URL.
FULL_BLOCK = f"""<!-- rocm-pr-quality:attestation v1 -->
> **Reviewed with the ROCm PR-quality skill.**
> - Skill: `rocm-pr-quality` (base) - overlays: `therock-pr-quality`
> - Action: `review` - Result: **APPROVED** (`0` BLOCKING; `0` IMPORTANT)
> - Risk: `1`
> - Reviewed commit: `{FULL_SHA}`
> - Attestation: `46012600`
>
> <sub>Machine-checkable attestation. The digest ties this review to this PR; later commits
> are allowed. The reviewed commit is informational (what the review looked at).</sub>
"""

ONE_LINE = (
    "<!-- rocm-pr-quality:attestation v1 --> rocm-pr-quality review APPROVED - commit "
    f"{FULL_SHA} - attestation 46012600"
)


def _src(body, author="tony-davis", kind="comment"):
    return {
        "kind": kind,
        "author": author,
        "association": "MEMBER",
        "created_at": "2026-10-14T00:00:00Z",
        "updated_at": "2026-10-14T00:00:00Z",
        "url": "https://example.test/c/1",
        "body": body,
    }


class ParseAttestationsTest(unittest.TestCase):
    def test_full_block(self):
        (blk,) = pq.parse_attestations("Looks good.\n\n" + FULL_BLOCK)
        self.assertEqual(blk["digest"], "46012600")
        self.assertEqual(blk["commit"], FULL_SHA)
        self.assertTrue(blk["approved"])

    def test_one_line_form(self):
        (blk,) = pq.parse_attestations(ONE_LINE)
        self.assertEqual(blk["digest"], "46012600")
        self.assertEqual(blk["commit"], FULL_SHA)
        self.assertTrue(blk["approved"])

    def test_no_marker_means_no_block(self):
        self.assertEqual(pq.parse_attestations("LGTM, attestation 46012600"), [])
        self.assertEqual(pq.parse_attestations(""), [])
        self.assertEqual(pq.parse_attestations(None), [])

    def test_prose_before_marker_is_ignored(self):
        body = "the attestation deadbeef is old\n\n" + FULL_BLOCK
        (blk,) = pq.parse_attestations(body)
        self.assertEqual(blk["digest"], "46012600")

    def test_multiple_blocks_are_parsed_separately(self):
        blocks = pq.parse_attestations(FULL_BLOCK + "\n" + ONE_LINE)
        self.assertEqual(len(blocks), 2)

    def test_marker_without_digest(self):
        (blk,) = pq.parse_attestations(
            "<!-- rocm-pr-quality:attestation v1 --> review APPROVED - commit "
            + FULL_SHA
        )
        self.assertIsNone(blk["digest"])

    def test_not_approved_result_is_not_marked_approved(self):
        body = FULL_BLOCK.replace("**APPROVED**", "**CHANGES REQUESTED**")
        (blk,) = pq.parse_attestations(body)
        self.assertFalse(blk["approved"])


class VerifyAttestationsTest(unittest.TestCase):
    def test_verified_for_matching_pr(self):
        result = pq.verify_attestations(PR_URL, [_src(FULL_BLOCK)])
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["expected_digest"], "46012600")
        self.assertTrue(result["attestations"][0]["valid"])

    def test_one_line_form_verifies(self):
        result = pq.verify_attestations(PR_URL, [_src(ONE_LINE)])
        self.assertEqual(result["status"], "VERIFIED")

    def test_none_when_no_block_anywhere(self):
        result = pq.verify_attestations(PR_URL, [_src("LGTM"), _src("thanks!")])
        self.assertEqual(result["status"], "NONE")
        self.assertEqual(result["attestations"], [])

    def test_none_with_no_sources(self):
        self.assertEqual(pq.verify_attestations(PR_URL, [])["status"], "NONE")

    def test_block_copied_from_another_pr_is_invalid(self):
        # The point of the digest: the same block pasted onto a different PR must fail.
        result = pq.verify_attestations(OTHER_PR_URL, [_src(FULL_BLOCK)])
        self.assertEqual(result["status"], "INVALID")
        self.assertEqual(
            result["attestations"][0]["problem"], "digest does not match this PR"
        )

    def test_template_with_placeholder_digest_is_invalid(self):
        body = FULL_BLOCK.replace("46012600", "<8-hex digest>")
        result = pq.verify_attestations(PR_URL, [_src(body)])
        self.assertEqual(result["status"], "INVALID")
        self.assertEqual(result["attestations"][0]["problem"], "missing digest")

    def test_wrong_digest_is_invalid(self):
        body = FULL_BLOCK.replace("46012600", "deadbeef")
        self.assertEqual(
            pq.verify_attestations(PR_URL, [_src(body)])["status"], "INVALID"
        )

    def test_non_approved_block_is_invalid(self):
        body = FULL_BLOCK.replace("**APPROVED**", "**CHANGES REQUESTED**")
        result = pq.verify_attestations(PR_URL, [_src(body)])
        self.assertEqual(result["status"], "INVALID")
        self.assertEqual(result["attestations"][0]["problem"], "not an APPROVED result")

    def test_one_valid_block_among_invalid_ones_verifies(self):
        bad = FULL_BLOCK.replace("46012600", "deadbeef")
        result = pq.verify_attestations(PR_URL, [_src(bad), _src(FULL_BLOCK)])
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual([a["valid"] for a in result["attestations"]], [False, True])

    def test_review_bodies_count_like_comments(self):
        result = pq.verify_attestations(PR_URL, [_src(FULL_BLOCK, kind="review")])
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["attestations"][0]["kind"], "review")

    def test_reviewed_commit_is_not_compared_to_head(self):
        # Later pushes are allowed: a different reviewed commit does not matter.
        body = FULL_BLOCK.replace(FULL_SHA, "1" * 40)
        self.assertEqual(
            pq.verify_attestations(PR_URL, [_src(body)])["status"], "VERIFIED"
        )

    def test_unparseable_pr_url_never_verifies(self):
        result = pq.verify_attestations("garbage", [_src(FULL_BLOCK)])
        self.assertEqual(result["status"], "INVALID")


class CommitsSinceTest(unittest.TestCase):
    COMMITS = [{"oid": "aaa1111"}, {"oid": "bbb2222"}, {"oid": "ccc3333"}]

    def test_counts_commits_after(self):
        self.assertEqual(pq.commits_since(self.COMMITS, "aaa1111"), 2)
        self.assertEqual(pq.commits_since(self.COMMITS, "bbb2222"), 1)
        self.assertEqual(pq.commits_since(self.COMMITS, "ccc3333"), 0)

    def test_accepts_abbreviated_sha(self):
        self.assertEqual(pq.commits_since(self.COMMITS, "bbb"), 1)

    def test_none_when_commit_not_in_history(self):
        self.assertIsNone(pq.commits_since(self.COMMITS, "ddd4444"))

    def test_none_for_missing_inputs(self):
        self.assertIsNone(pq.commits_since(self.COMMITS, None))
        self.assertIsNone(pq.commits_since(None, "aaa1111"))
        self.assertIsNone(pq.commits_since([], "aaa1111"))


ZERO_DATE = "0001-01-01T00:00:00Z"  # how gh reports an unset time


def _ci_findings(rollup):
    findings = []
    pq.check_ci_currency({"statusCheckRollup": rollup}, findings)
    return findings


class CiCurrencyTest(unittest.TestCase):
    def test_queued_check_with_zero_completed_at_is_not_stale(self):
        # Regression: the zero date used to be read as a ~2000-year-old CI result.
        now = datetime.now(timezone.utc).isoformat()
        (f,) = _ci_findings([{"startedAt": now, "completedAt": ZERO_DATE}])
        self.assertEqual(f.severity, "INFO")

    def test_only_zero_dates_means_no_timestamps(self):
        (f,) = _ci_findings([{"startedAt": ZERO_DATE, "completedAt": ZERO_DATE}])
        self.assertEqual(f.severity, "INFO")
        self.assertIn("No CI completion timestamps", f.message)

    def test_genuinely_old_result_is_flagged(self):
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        (f,) = _ci_findings([{"completedAt": old}])
        self.assertEqual(f.severity, "IMPORTANT")

    def test_recent_result_is_fine(self):
        recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        (f,) = _ci_findings([{"completedAt": recent}])
        self.assertEqual(f.severity, "INFO")

    def test_no_checks(self):
        (f,) = _ci_findings([])
        self.assertEqual(f.severity, "INFO")


if __name__ == "__main__":
    unittest.main()
