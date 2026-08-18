# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for post_bump_breadcrumbs.py."""

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import post_bump_breadcrumbs


# --- Pure-function tests: comment-building helpers ---


class HistoryHasEventTest(unittest.TestCase):
    def test_true_only_when_marker_present(self):
        # The None case is already covered incidentally: every ProcessBumpTest
        # test calls this with existing_body=None via the default mock.
        key = "abc123_rocm-systems_False"
        marker = post_bump_breadcrumbs.event_key_marker(key)

        self.assertFalse(
            post_bump_breadcrumbs.history_has_event("unrelated comment", key)
        )
        self.assertTrue(post_bump_breadcrumbs.history_has_event(f"...{marker}...", key))


class BuildTimelineEntryTest(unittest.TestCase):
    def test_wording_matches_reverted_flag(self):
        included = post_bump_breadcrumbs.build_timeline_entry(
            "2026-07-21",
            reverted=False,
            therock_pr_number=42,
            submodule="rocm-systems",
            event_key="k1",
        )
        self.assertIn("2026-07-21", included)
        self.assertIn("Included in TheRock via", included)
        self.assertIn("ROCm/TheRock#42", included)
        self.assertIn(post_bump_breadcrumbs.event_key_marker("k1"), included)

        reverted = post_bump_breadcrumbs.build_timeline_entry(
            "2026-07-21",
            reverted=True,
            therock_pr_number=42,
            submodule="rocm-systems",
            event_key="k1",
        )
        self.assertIn("Reverted out of TheRock via", reverted)
        self.assertNotIn("Included in TheRock via", reverted)

    def test_falls_back_when_no_therock_pr_found(self):
        entry = post_bump_breadcrumbs.build_timeline_entry(
            "2026-07-21",
            reverted=False,
            therock_pr_number=None,
            submodule="rocm-libraries",
            event_key="k1",
        )
        self.assertIn("rocm-libraries", entry)
        self.assertNotIn("#None", entry)


class BuildBreadcrumbBodyTest(unittest.TestCase):
    def test_first_event_has_no_prior_history(self):
        body = post_bump_breadcrumbs.build_breadcrumb_body(
            existing_body=None,
            reverted=False,
            therock_pr_number=100,
            submodule="rocm-systems",
            event_date="2026-07-01",
            event_key="k1",
        )
        self.assertTrue(body.startswith(post_bump_breadcrumbs.BREADCRUMB_MARKER))
        self.assertIn(post_bump_breadcrumbs.HISTORY_HEADER, body)
        self.assertIn("2026-07-01", body)
        self.assertIn("ROCm/TheRock#100", body)

    def test_new_entry_is_prepended_above_prior_history(self):
        existing_body = (
            f"{post_bump_breadcrumbs.BREADCRUMB_MARKER}\n"
            f"{post_bump_breadcrumbs.HISTORY_HEADER}\n\n"
            f"- **2026-07-01** — Included in TheRock via ROCm/TheRock#100. "
            f"{post_bump_breadcrumbs.event_key_marker('k1')}\n"
        )

        body = post_bump_breadcrumbs.build_breadcrumb_body(
            existing_body=existing_body,
            reverted=True,
            therock_pr_number=101,
            submodule="rocm-systems",
            event_date="2026-07-05",
            event_key="k2",
        )

        # The new (reverted) entry must sort ABOVE the older one.
        reverted_idx = body.index("2026-07-05")
        included_idx = body.index("2026-07-01")
        self.assertLess(reverted_idx, included_idx)
        self.assertIn("ROCm/TheRock#101", body)
        self.assertIn("ROCm/TheRock#100", body)
        # Exactly one marker/comment -- not a second, separately-marked comment.
        self.assertEqual(body.count(post_bump_breadcrumbs.BREADCRUMB_MARKER), 1)

    def test_ignores_prior_body_without_history_header(self):
        """A malformed/foreign existing body degrades to a fresh history."""
        body = post_bump_breadcrumbs.build_breadcrumb_body(
            existing_body="some unrelated comment body",
            reverted=False,
            therock_pr_number=42,
            submodule="rocm-systems",
            event_date="2026-07-21",
            event_key="k1",
        )
        self.assertEqual(body.count(post_bump_breadcrumbs.HISTORY_HEADER), 1)
        self.assertIn("2026-07-21", body)


class BuildUnmappedSummaryBodyTest(unittest.TestCase):
    def test_lists_commits_with_wording_matching_reverted_flag(self):
        included = post_bump_breadcrumbs.build_unmapped_summary_body(
            reverted=False,
            submodule="rocm-systems",
            repo="ROCm/rocm-systems",
            unmapped_shas=["a" * 40, "b" * 40],
        )
        self.assertTrue(included.startswith(post_bump_breadcrumbs.UNMAPPED_MARKER))
        self.assertIn("included in", included)
        self.assertIn(("a" * 40)[:7], included)
        self.assertIn(("b" * 40)[:7], included)
        self.assertIn(
            "https://github.com/ROCm/rocm-systems/commit/" + "a" * 40, included
        )

        reverted = post_bump_breadcrumbs.build_unmapped_summary_body(
            reverted=True,
            submodule="rocm-systems",
            repo="ROCm/rocm-systems",
            unmapped_shas=["c" * 40],
        )
        self.assertIn("removed from", reverted)


# --- process_bump() orchestration tests: each collaborator mocked directly ---


class ProcessBumpTest(unittest.TestCase):
    """Tests for process_bump() orchestration."""

    REPO = "ROCm/rocm-systems"
    TOKENS = {
        "systems": "systems-tok",
        "libraries": "libraries-tok",
        "rocgdb": "rocgdb-tok",
    }
    THEROCK_AFTER_SHA = "a" * 40
    OLD_SHA = "b" * 40
    NEW_SHA = "c" * 40
    API_BASE = "https://api.github.com/repos/ROCm/rocm-systems"

    def setUp(self):
        self.changed = {
            "name": "rocm-systems",
            "repo": self.REPO,
            "old_sha": self.OLD_SHA,
            "new_sha": self.NEW_SHA,
        }
        # Stand-in for the GitHubAPI instance process_bump() constructs.
        self.fake_github_api = mock.sentinel.github_api

        self.mocks = {}
        for target, default_return in (
            ("run", "https://github.com/ROCm/rocm-systems.git"),
            ("resolve_therock_pr_number", 99),
            ("is_revert", False),
            ("fetch_commits_in_range", [{"sha": "d" * 40}]),
            ("resolve_prs_for_commits", ({10}, [])),
            ("find_existing_comment_body", None),
        ):
            patcher = mock.patch.object(
                post_bump_breadcrumbs, target, return_value=default_return
            )
            self.mocks[target] = patcher.start()
            self.addCleanup(patcher.stop)
        gha_update_patcher = mock.patch.object(
            post_bump_breadcrumbs, "gha_update_pr_comment"
        )
        self.mocks["gha_update_pr_comment"] = gha_update_patcher.start()
        self.addCleanup(gha_update_patcher.stop)
        github_api_cls_patcher = mock.patch.object(
            post_bump_breadcrumbs, "GitHubAPI", return_value=self.fake_github_api
        )
        github_api_cls_patcher.start()
        self.addCleanup(github_api_cls_patcher.stop)

    def _run(self, **kwargs):
        post_bump_breadcrumbs.process_bump(
            self.changed, self.THEROCK_AFTER_SHA, self.TOKENS, **kwargs
        )

    def test_revert_swaps_fetch_range(self):
        self.mocks["is_revert"].return_value = True

        self._run()

        self.mocks["fetch_commits_in_range"].assert_called_once_with(
            repo_name=self.REPO,
            start_sha=self.NEW_SHA,
            end_sha=self.OLD_SHA,
            api_base=self.API_BASE,
            github_api=self.fake_github_api,
        )

    def test_posts_comment_to_each_resolved_pr(self):
        """A bump range spanning multiple upstream PRs must post a comment
        to each independently, not just the first."""
        self.mocks["resolve_prs_for_commits"].return_value = ({10, 20}, [])

        self._run()

        calls = self.mocks["gha_update_pr_comment"].call_args_list
        self.assertEqual({c.kwargs["pr_number"] for c in calls}, {10, 20})
        for c in calls:
            self.assertEqual(c.kwargs["github_repository"], self.REPO)
            self.assertIn("ROCm/TheRock#99", c.kwargs["body"])

    def test_unmapped_commits_posted_as_summary_on_therock_pr(self):
        unmapped_sha = "e" * 40
        self.mocks["resolve_prs_for_commits"].return_value = (set(), [unmapped_sha])

        self._run()

        self.mocks["gha_update_pr_comment"].assert_called_once()
        call = self.mocks["gha_update_pr_comment"].call_args
        self.assertEqual(call.kwargs["pr_number"], 99)
        self.assertEqual(
            call.kwargs["github_repository"], post_bump_breadcrumbs.THEROCK_REPO
        )
        self.assertEqual(call.kwargs["marker"], post_bump_breadcrumbs.UNMAPPED_MARKER)
        self.assertIn(unmapped_sha[:7], call.kwargs["body"])

    def test_dry_run_never_calls_gha_update_pr_comment(self):
        self.mocks["resolve_prs_for_commits"].return_value = ({10}, ["f" * 40])

        self._run(dry_run=True)

        self.mocks["gha_update_pr_comment"].assert_not_called()


class HandlePostBreadcrumbsDispatchTest(unittest.TestCase):
    """Tests for handle_post_breadcrumbs() dispatch wiring."""

    def test_skips_processing_when_no_submodule_changed(self):
        with mock.patch.object(
            post_bump_breadcrumbs, "detect_changed_submodules", return_value=[]
        ), mock.patch.object(post_bump_breadcrumbs, "process_bump") as process_mock:
            post_bump_breadcrumbs.handle_post_breadcrumbs("aaa", "bbb", {})

        process_mock.assert_not_called()

    def test_processes_each_changed_submodule_independently(self):
        """A single push touching two monitored submodules must post
        breadcrumbs for both, not just the first detected."""
        changed_a = {"name": "rocm-systems", "repo": "ROCm/rocm-systems"}
        changed_b = {"name": "rocm-libraries", "repo": "ROCm/rocm-libraries"}
        tokens = {"systems": "tok"}
        with mock.patch.object(
            post_bump_breadcrumbs,
            "detect_changed_submodules",
            return_value=[changed_a, changed_b],
        ), mock.patch.object(post_bump_breadcrumbs, "process_bump") as process_mock:
            post_bump_breadcrumbs.handle_post_breadcrumbs(
                "aaa", "bbb", tokens, dry_run=True
            )

        self.assertEqual(
            process_mock.call_args_list,
            [
                mock.call(changed_a, "bbb", tokens, dry_run=True),
                mock.call(changed_b, "bbb", tokens, dry_run=True),
            ],
        )

    def test_one_submodule_failure_does_not_block_the_next(self):
        changed_a = {"name": "rocm-systems", "repo": "ROCm/rocm-systems"}
        changed_b = {"name": "rocm-libraries", "repo": "ROCm/rocm-libraries"}
        tokens = {"systems": "tok"}
        with mock.patch.object(
            post_bump_breadcrumbs,
            "detect_changed_submodules",
            return_value=[changed_a, changed_b],
        ), mock.patch.object(
            post_bump_breadcrumbs,
            "process_bump",
            side_effect=[RuntimeError("boom"), None],
        ) as process_mock:
            post_bump_breadcrumbs.handle_post_breadcrumbs("aaa", "bbb", tokens)

        self.assertEqual(process_mock.call_count, 2)


class DetectChangedSubmodulesTest(unittest.TestCase):
    """Tests for detect_changed_submodules()."""

    def test_returns_details_for_every_changed_submodule(self):
        """A push touching both rocm-systems and rocm-libraries must return
        correct details for both, not silently drop the second one."""
        with mock.patch.object(
            post_bump_breadcrumbs,
            "submodule_changed",
            side_effect=lambda before, after, name: name
            in ("rocm-systems", "rocm-libraries"),
        ), mock.patch.object(
            post_bump_breadcrumbs,
            "get_submodule_sha",
            side_effect=lambda ref, name: f"{ref}-{name}-sha",
        ):
            result = post_bump_breadcrumbs.detect_changed_submodules("before", "after")

        self.assertEqual(
            result,
            [
                {
                    "name": "rocm-systems",
                    "repo": "ROCm/rocm-systems",
                    "old_sha": "before-rocm-systems-sha",
                    "new_sha": "after-rocm-systems-sha",
                },
                {
                    "name": "rocm-libraries",
                    "repo": "ROCm/rocm-libraries",
                    "old_sha": "before-rocm-libraries-sha",
                    "new_sha": "after-rocm-libraries-sha",
                },
            ],
        )

    def test_returns_empty_list_when_no_monitored_submodule_changed(self):
        with mock.patch.object(
            post_bump_breadcrumbs, "submodule_changed", return_value=False
        ):
            result = post_bump_breadcrumbs.detect_changed_submodules("before", "after")

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
