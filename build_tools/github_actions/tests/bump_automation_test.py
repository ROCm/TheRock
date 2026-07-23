# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import bump_automation
from bump_automation import (
    _clone_url,
    close_stale_prs,
    create_therock_bump,
    generate_pr_body,
    get_submodule_sha,
    handle_push,
    latest_commit,
    submodule_changed,
    update_ci_env_file,
    update_ref_in_file,
)
from github_actions_api import GitHubAPI


class CloneUrlTest(unittest.TestCase):
    def test_formats_url_with_token(self):
        url = _clone_url("ROCm/TheRock", "mytoken")
        self.assertEqual(
            url, "https://x-access-token:mytoken@github.com/ROCm/TheRock.git"
        )

    def test_formats_url_with_different_repo(self):
        url = _clone_url("ROCm/rocgdb", "tok123")
        self.assertEqual(
            url, "https://x-access-token:tok123@github.com/ROCm/rocgdb.git"
        )


class GeneratePrBodyTest(unittest.TestCase):
    def test_contains_repo_links(self):
        body = generate_pr_body("ROCm/rocgdb", "aabbcc1", "ddeeff2")
        self.assertIn("ROCm/rocgdb", body)
        self.assertIn("aabbcc1", body)
        self.assertIn("ddeeff2", body)

    def test_contains_compare_url(self):
        body = generate_pr_body("ROCm/rocgdb", "aabbcc1", "ddeeff2")
        self.assertIn("compare/aabbcc1...ddeeff2", body)


class GetSubmoduleShaTest(unittest.TestCase):
    def test_parses_sha_from_ls_tree_output(self):
        ls_tree_output = "160000 commit abc123def456  debug-tools/rocgdb/source"
        with patch("bump_automation.run", return_value=ls_tree_output):
            sha = get_submodule_sha("HEAD", "debug-tools/rocgdb/source")
        self.assertEqual(sha, "abc123def456")


class LatestCommitTest(unittest.TestCase):
    def test_queries_default_branch_when_no_branch(self):
        with patch(
            "bump_automation.gh_api", return_value=[{"sha": "deadbeef"}]
        ) as mock_api:
            sha = latest_commit("ROCm/rocm-systems", "token")
        self.assertEqual(sha, "deadbeef")
        self.assertEqual(mock_api.call_args.args[1], "repos/ROCm/rocm-systems/commits")

    def test_queries_specific_branch_when_provided(self):
        with patch(
            "bump_automation.gh_api", return_value=[{"sha": "cafef00d"}]
        ) as mock_api:
            sha = latest_commit("ROCm/rocgdb", "token", "amd-staging-rocgdb-16")
        self.assertEqual(sha, "cafef00d")
        self.assertEqual(
            mock_api.call_args.args[1],
            "repos/ROCm/rocgdb/commits?sha=amd-staging-rocgdb-16",
        )


class SubmoduleChangedTest(unittest.TestCase):
    def test_returns_true_when_diff_nonempty(self):
        with patch("bump_automation.run", return_value="some diff output"):
            self.assertTrue(submodule_changed("abc", "def", "rocm-systems"))

    def test_returns_false_when_diff_empty(self):
        with patch("bump_automation.run", return_value=""):
            self.assertFalse(submodule_changed("abc", "def", "rocm-systems"))

    def test_returns_false_when_diff_whitespace_only(self):
        with patch("bump_automation.run", return_value="   \n  "):
            self.assertFalse(submodule_changed("abc", "def", "rocm-systems"))


class UpdateRefInFileTest(unittest.TestCase):
    def _run(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            update_ref_in_file(path, "newsha1234567")
            return Path(path).read_text()
        finally:
            os.unlink(path)

    def test_updates_ref_line(self):
        content = textwrap.dedent(
            """\
            uses: actions/checkout@v3
            with:
              repository: "ROCm/TheRock"
              ref: oldsha1234567 # 2024-01-01 commit
        """
        )
        result = self._run(content)
        self.assertIn("ref: newsha1234567", result)
        self.assertNotIn("oldsha1234567", result)

    def test_preserves_other_lines(self):
        content = textwrap.dedent(
            """\
            uses: actions/checkout@v3
            with:
              repository: "ROCm/TheRock"
              ref: oldsha1234567
            other: value
        """
        )
        result = self._run(content)
        self.assertIn("uses: actions/checkout@v3", result)
        self.assertIn("other: value", result)

    def test_handles_path_line_between_repository_and_ref(self):
        content = textwrap.dedent(
            """\
            with:
              repository: "ROCm/TheRock"
              path: "TheRock"
              ref: oldsha1234567
        """
        )
        result = self._run(content)
        self.assertIn('path: "TheRock"', result)
        self.assertIn("ref: newsha1234567", result)

    def test_no_change_when_no_matching_repository(self):
        content = textwrap.dedent(
            """\
            uses: actions/checkout@v3
            with:
              repository: "ROCm/SomeOtherRepo"
              ref: oldsha1234567
        """
        )
        result = self._run(content)
        self.assertIn("oldsha1234567", result)

    def test_updates_multiple_occurrences(self):
        content = textwrap.dedent(
            """\
            - uses: actions/checkout@v3
              with:
                repository: "ROCm/TheRock"
                ref: oldsha0000001
            - uses: actions/checkout@v3
              with:
                repository: "ROCm/TheRock"
                ref: oldsha0000002
        """
        )
        result = self._run(content)
        self.assertEqual(result.count("ref: newsha1234567"), 2)
        self.assertNotIn("oldsha0000001", result)
        self.assertNotIn("oldsha0000002", result)


class UpdateCiEnvFileTest(unittest.TestCase):
    def _run(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            update_ci_env_file(path, "newsha1234567")
            return Path(path).read_text()
        finally:
            os.unlink(path)

    def test_updates_value_under_therock_ref(self):
        content = textwrap.dedent(
            """\
            outputs:
              therock-ref:
                description: "TheRock commit ref"
                value: "oldsha1234567" # 2024-01-01 commit
        """
        )
        result = self._run(content)
        self.assertIn('"newsha1234567"', result)
        self.assertNotIn("oldsha1234567", result)

    def test_preserves_description_line(self):
        content = textwrap.dedent(
            """\
            outputs:
              therock-ref:
                description: "TheRock commit ref"
                value: "oldsha1234567"
        """
        )
        result = self._run(content)
        self.assertIn('description: "TheRock commit ref"', result)

    def test_preserves_other_outputs(self):
        content = textwrap.dedent(
            """\
            outputs:
              therock-ref:
                description: "TheRock commit ref"
                value: "oldsha1234567"
              other-output:
                value: "unchanged"
        """
        )
        result = self._run(content)
        self.assertIn('"newsha1234567"', result)
        self.assertIn('"unchanged"', result)

    def test_no_change_when_no_therock_ref(self):
        content = textwrap.dedent(
            """\
            outputs:
              some-other-ref:
                value: "oldsha1234567"
        """
        )
        result = self._run(content)
        self.assertIn("oldsha1234567", result)


class CloseStalePrsTest(unittest.TestCase):
    def _make_pr(self, number: int, title: str) -> dict:
        return {"number": number, "title": title}

    def test_closes_matching_pr(self):
        prs = [self._make_pr(42, "Bump rocm-systems from abc1234 to xyz5678")]
        with patch("bump_automation.gh_api", return_value=prs) as mock_api:
            close_stale_prs("rocm-systems", "abc1234567890", "token")

        patch_calls = [
            c for c in mock_api.call_args_list if c.kwargs.get("method") == "PATCH"
        ]
        self.assertEqual(len(patch_calls), 1)
        self.assertIn("pulls/42", patch_calls[0].args[1])
        self.assertEqual(patch_calls[0].kwargs["data"]["state"], "closed")

    def test_skips_non_matching_pr(self):
        prs = [self._make_pr(99, "Bump rocm-libraries from def9876 to uvw5432")]
        with patch("bump_automation.gh_api", return_value=prs) as mock_api:
            close_stale_prs("rocm-systems", "abc1234567890", "token")

        patch_calls = [
            c for c in mock_api.call_args_list if c.kwargs.get("method") == "PATCH"
        ]
        self.assertEqual(len(patch_calls), 0)

    def test_posts_comment_before_closing(self):
        prs = [self._make_pr(42, "Bump rocm-systems from abc1234 to xyz5678")]
        with patch("bump_automation.gh_api", return_value=prs) as mock_api:
            close_stale_prs("rocm-systems", "abc1234567890", "token")

        post_calls = [
            c for c in mock_api.call_args_list if c.kwargs.get("method") == "POST"
        ]
        self.assertTrue(any("comments" in c.args[1] for c in post_calls))


class HandlePushTest(unittest.TestCase):
    def test_noop_when_no_submodule_changed(self):
        with patch("bump_automation.submodule_changed", return_value=False):
            with patch("bump_automation.get_submodule_sha") as mock_sha:
                handle_push("before", "after", {"systems": "t"})
        mock_sha.assert_not_called()

    def test_submodule_only_closes_stale_prs_then_returns(self):
        def changed(before, after, path):
            return path == "debug-tools/rocgdb/source"

        with patch("bump_automation.submodule_changed", side_effect=changed):
            with patch(
                "bump_automation.get_submodule_sha", return_value="oldsha1234567"
            ):
                with patch("bump_automation.close_stale_prs") as mock_close:
                    with patch(
                        "bump_automation.tempfile.TemporaryDirectory"
                    ) as mock_tmp:
                        handle_push(
                            "before",
                            "after",
                            {
                                "systems": "systems-token",
                                "libraries": "libraries-token",
                                "rocgdb": "rocgdb-token",
                            },
                        )

        mock_close.assert_called_once()
        # rocgdb reuses the systems token, so close_stale_prs must receive it.
        self.assertEqual(mock_close.call_args.args[2], "systems-token")
        # submodule-only entries have no upstream ref files, so the handler must
        # bail out before cloning the upstream repo.
        mock_tmp.assert_not_called()


class CreateTheRockBumpTest(unittest.TestCase):
    def test_skips_when_pr_already_open(self):
        """create_therock_bump must bail out without cloning when an open PR
        already targets the exact bump branch for the latest commit."""
        with patch("bump_automation.latest_commit", return_value="abc1234567890"):
            with patch(
                "bump_automation.gh_api",
                return_value=[{"number": 99}],
            ) as mock_api:
                with patch("bump_automation.tempfile.TemporaryDirectory") as mock_tmp:
                    create_therock_bump("rocm-systems", "token")

        mock_tmp.assert_not_called()
        # Only the open-PR check should have hit the API.
        self.assertEqual(mock_api.call_count, 1)
        endpoint = mock_api.call_args.args[1]
        self.assertIn("pulls?state=open", endpoint)
        self.assertIn("bump-rocm-systems-abc1234", endpoint)

    def test_skips_when_submodule_already_at_latest(self):
        """create_therock_bump must bail out without cloning when the submodule
        is already pinned to the latest upstream commit."""
        latest_sha = "abc1234567890"
        api_responses = [
            [],  # open-PR check returns nothing
        ]

        def _gh_api_side_effect(*args, **kwargs):
            return api_responses.pop(0) if api_responses else {}

        with patch("bump_automation.latest_commit", return_value=latest_sha):
            with patch("bump_automation.gh_api", side_effect=_gh_api_side_effect):
                with patch("bump_automation.run") as mock_run:
                    with patch("bump_automation.os.chdir"):
                        with patch("bump_automation.os.path.exists", return_value=True):
                            with patch(
                                "bump_automation.get_submodule_sha",
                                return_value=latest_sha,
                            ):
                                with patch(
                                    "bump_automation.tempfile.TemporaryDirectory"
                                ) as mock_tmp:
                                    create_therock_bump("rocm-systems", "token")

        # The clone is created before we detect the no-op, so TemporaryDirectory
        # will have been called. What must NOT happen is any git commit or push.
        git_calls = [c for c in mock_run.call_args_list if "commit" in c.args[0]]
        self.assertEqual(
            git_calls, [], "git commit must not run when already at latest"
        )
        push_calls = [c for c in mock_run.call_args_list if "push" in c.args[0]]
        self.assertEqual(push_calls, [], "git push must not run when already at latest")

    def test_proceeds_when_no_open_pr(self):
        """create_therock_bump must proceed to clone when no open PR exists."""
        # First gh_api call is the open-PR check (returns []); subsequent calls
        # are PR creation (returns a PR dict) and label addition (ignored).
        api_responses = [[], {"number": 1}, {}]

        def _gh_api_side_effect(*args, **kwargs):
            return api_responses.pop(0) if api_responses else {}

        with patch("bump_automation.latest_commit", return_value="abc1234567890"):
            with patch(
                "bump_automation.gh_api", side_effect=_gh_api_side_effect
            ) as mock_api:
                with patch("bump_automation.run"):
                    with patch("bump_automation.os.chdir"):
                        with patch("bump_automation.os.path.exists", return_value=True):
                            with patch(
                                "bump_automation.get_submodule_sha",
                                return_value="old1234",
                            ):
                                with patch(
                                    "bump_automation.generate_pr_body",
                                    return_value="body",
                                ):
                                    with patch("bump_automation._git_commit"):
                                        create_therock_bump("rocm-systems", "token")

        # Must have made at least the open-PR check + PR creation calls.
        self.assertGreaterEqual(mock_api.call_count, 2)


class HandleScheduleTest(unittest.TestCase):
    def test_rocgdb_maps_to_correct_submodule_and_token(self):
        """handle_schedule('rocgdb') must call create_therock_bump with
        'debug-tools/rocgdb/source' and tokens['rocgdb'], not any other key."""
        tokens = {
            "systems": "systems-token",
            "libraries": "libraries-token",
            "rocgdb": "rocgdb-token",
        }
        with patch("bump_automation.create_therock_bump") as mock_bump:
            from bump_automation import handle_schedule

            handle_schedule(tokens, "rocgdb")

        mock_bump.assert_called_once_with("debug-tools/rocgdb/source", "rocgdb-token")

    def test_rocgdb_does_not_invoke_other_submodules(self):
        tokens = {
            "systems": "systems-token",
            "libraries": "libraries-token",
            "rocgdb": "rocgdb-token",
        }
        with patch("bump_automation.create_therock_bump") as mock_bump:
            from bump_automation import handle_schedule

            handle_schedule(tokens, "rocgdb")

        called_submodules = [c.args[0] for c in mock_bump.call_args_list]
        self.assertNotIn("rocm-systems", called_submodules)
        self.assertNotIn("rocm-libraries", called_submodules)


class CreateTheRockBumpRocgdbConfigTest(unittest.TestCase):
    def test_reads_repo_and_branch_from_submodule_config(self):
        """create_therock_bump('debug-tools/rocgdb/source') must derive repo and
        branch from SUBMODULE_CONFIG, not hard-code them."""
        api_responses = [[], {"number": 1}, {}]

        def _gh_api_side_effect(*args, **kwargs):
            return api_responses.pop(0) if api_responses else {}

        with patch(
            "bump_automation.latest_commit", return_value="abc1234567890"
        ) as mock_latest:
            with patch("bump_automation.gh_api", side_effect=_gh_api_side_effect):
                with patch("bump_automation.run"):
                    with patch("bump_automation.os.chdir"):
                        with patch("bump_automation.os.path.exists", return_value=True):
                            with patch(
                                "bump_automation.get_submodule_sha",
                                return_value="old1234",
                            ):
                                with patch(
                                    "bump_automation.generate_pr_body",
                                    return_value="body",
                                ):
                                    with patch("bump_automation._git_commit"):
                                        create_therock_bump(
                                            "debug-tools/rocgdb/source", "token"
                                        )

        mock_latest.assert_called_once_with(
            "ROCm/rocgdb", "token", "amd-staging-rocgdb-16"
        )


# --- Bump breadcrumbs (--event_type post_breadcrumbs) ---
#
# The tests below cover the "post breadcrumb comments on upstream PRs" half
# of this module: detect_changed_submodule(), resolve_app_token(),
# resolve_therock_pr_number(), resolve_prs_for_commits(),
# find_existing_comment_body(), build_timeline_entry()/build_breadcrumb_body(),
# build_unmapped_summary_body(), process_bump(), and the
# handle_post_breadcrumbs()/main() wiring for --event_type post_breadcrumbs.


class DetectChangedSubmoduleTest(unittest.TestCase):
    """Tests for detect_changed_submodule()."""

    def test_returns_none_when_nothing_changed(self):
        with mock.patch.object(
            bump_automation, "submodule_changed", return_value=False
        ):
            result = bump_automation.detect_changed_submodule("before", "after")

        self.assertIsNone(result)

    def test_detects_rocm_systems_change(self):
        with mock.patch.object(
            bump_automation,
            "submodule_changed",
            side_effect=lambda before, after, name: name == "rocm-systems",
        ), mock.patch.object(
            bump_automation,
            "get_submodule_sha",
            side_effect=lambda commit, name: f"{commit}-{name}",
        ):
            result = bump_automation.detect_changed_submodule("before", "after")

        self.assertEqual(
            result,
            {
                "name": "rocm-systems",
                "repo": "ROCm/rocm-systems",
                "old_sha": "before-rocm-systems",
                "new_sha": "after-rocm-systems",
            },
        )

    def test_detects_rocm_libraries_change(self):
        with mock.patch.object(
            bump_automation,
            "submodule_changed",
            side_effect=lambda before, after, name: name == "rocm-libraries",
        ), mock.patch.object(
            bump_automation,
            "get_submodule_sha",
            side_effect=lambda commit, name: f"{commit}-{name}",
        ):
            result = bump_automation.detect_changed_submodule("before", "after")

        self.assertEqual(result["name"], "rocm-libraries")
        self.assertEqual(result["repo"], "ROCm/rocm-libraries")

    def test_stops_at_first_match(self):
        """If (impossibly) both looked changed, the first configured wins."""
        calls = []

        def fake_changed(before, after, name):
            calls.append(name)
            return True

        with mock.patch.object(
            bump_automation, "submodule_changed", side_effect=fake_changed
        ), mock.patch.object(bump_automation, "get_submodule_sha", return_value="sha"):
            result = bump_automation.detect_changed_submodule("before", "after")

        self.assertEqual(result["name"], "rocm-systems")
        self.assertEqual(calls, ["rocm-systems"])


class ResolveAppTokenTest(unittest.TestCase):
    """Tests for resolve_app_token()."""

    def test_systems_submodule_uses_systems_token(self):
        tokens = {"systems": "systems-tok", "libraries": "libraries-tok"}
        token = bump_automation.resolve_app_token("rocm-systems", tokens)
        self.assertEqual(token, "systems-tok")

    def test_libraries_submodule_uses_libraries_token(self):
        tokens = {"systems": "systems-tok", "libraries": "libraries-tok"}
        token = bump_automation.resolve_app_token("rocm-libraries", tokens)
        self.assertEqual(token, "libraries-tok")

    def test_rocgdb_submodule_reuses_systems_token(self):
        tokens = {
            "systems": "systems-tok",
            "libraries": "libraries-tok",
            "rocgdb": "rocgdb-tok",
        }
        token = bump_automation.resolve_app_token("debug-tools/rocgdb/source", tokens)
        self.assertEqual(token, "systems-tok")


class ResolveTherockPrNumberTest(unittest.TestCase):
    """Tests for resolve_therock_pr_number()."""

    def test_returns_pr_number_when_found(self):
        fake_api = mock.create_autospec(GitHubAPI, instance=True)
        with mock.patch.object(
            bump_automation, "gha_query_prs_for_commit", return_value=[{"number": 123}]
        ) as query:
            result = bump_automation.resolve_therock_pr_number("abc123", fake_api)

        self.assertEqual(result, 123)
        query.assert_called_once_with("ROCm/TheRock", "abc123", github_api=fake_api)

    def test_returns_none_when_no_pr_found(self):
        fake_api = mock.create_autospec(GitHubAPI, instance=True)
        with mock.patch.object(
            bump_automation, "gha_query_prs_for_commit", return_value=[]
        ):
            result = bump_automation.resolve_therock_pr_number("abc123", fake_api)

        self.assertIsNone(result)

    def test_uses_first_pr_when_multiple_found(self):
        fake_api = mock.create_autospec(GitHubAPI, instance=True)
        with mock.patch.object(
            bump_automation,
            "gha_query_prs_for_commit",
            return_value=[{"number": 5}, {"number": 6}],
        ):
            result = bump_automation.resolve_therock_pr_number("abc123", fake_api)

        self.assertEqual(result, 5)


class ResolvePrsForCommitsTest(unittest.TestCase):
    """Tests for resolve_prs_for_commits()."""

    def test_dedupes_pr_numbers_and_collects_unmapped(self):
        commits = [{"sha": "a"}, {"sha": "b"}, {"sha": "c"}]
        fake_api = mock.create_autospec(GitHubAPI, instance=True)

        def fake_query(repo, sha, github_api=None):
            self.assertEqual(repo, "ROCm/rocm-systems")
            self.assertIs(github_api, fake_api)
            return {
                "a": [{"number": 1}],
                "b": [{"number": 1}, {"number": 2}],
                "c": [],
            }[sha]

        with mock.patch.object(
            bump_automation, "gha_query_prs_for_commit", side_effect=fake_query
        ):
            pr_numbers, unmapped = bump_automation.resolve_prs_for_commits(
                "ROCm/rocm-systems", commits, fake_api
            )

        self.assertEqual(pr_numbers, {1, 2})
        self.assertEqual(unmapped, ["c"])

    def test_all_commits_unmapped(self):
        commits = [{"sha": "x"}, {"sha": "y"}]
        fake_api = mock.create_autospec(GitHubAPI, instance=True)

        with mock.patch.object(
            bump_automation, "gha_query_prs_for_commit", return_value=[]
        ):
            pr_numbers, unmapped = bump_automation.resolve_prs_for_commits(
                "ROCm/rocm-libraries", commits, fake_api
            )

        self.assertEqual(pr_numbers, set())
        self.assertEqual(unmapped, ["x", "y"])


class FindExistingCommentBodyTest(unittest.TestCase):
    """Tests for find_existing_comment_body()."""

    def test_returns_none_when_no_comment_matches(self):
        fake_api = mock.create_autospec(GitHubAPI, instance=True)
        fake_api.send_request.return_value = [{"id": 1, "body": "unrelated"}]

        result = bump_automation.find_existing_comment_body(
            42, bump_automation.BREADCRUMB_MARKER, "ROCm/rocm-systems", fake_api
        )

        self.assertIsNone(result)
        fake_api.send_request.assert_called_once_with(
            "https://api.github.com/repos/ROCm/rocm-systems/issues/42/comments"
            "?per_page=100&page=1"
        )

    def test_returns_body_of_matching_comment(self):
        marker = bump_automation.BREADCRUMB_MARKER
        matching_body = f"{marker}\nhistory here"
        fake_api = mock.create_autospec(GitHubAPI, instance=True)
        fake_api.send_request.return_value = [
            {"id": 1, "body": "unrelated"},
            {"id": 2, "body": matching_body},
        ]

        result = bump_automation.find_existing_comment_body(
            42, marker, "ROCm/rocm-systems", fake_api
        )

        self.assertEqual(result, matching_body)

    def test_paginates_until_marker_found(self):
        marker = bump_automation.BREADCRUMB_MARKER
        matching_body = f"{marker}\nold history"
        page_1 = [{"id": i, "body": f"comment {i}"} for i in range(100)]
        page_2 = [{"id": 200, "body": matching_body}]
        fake_api = mock.create_autospec(GitHubAPI, instance=True)
        fake_api.send_request.side_effect = [page_1, page_2]

        result = bump_automation.find_existing_comment_body(
            7, marker, "ROCm/rocm-libraries", fake_api
        )

        self.assertEqual(result, matching_body)
        self.assertEqual(fake_api.send_request.call_count, 2)

    def test_returns_none_when_response_is_not_a_list(self):
        fake_api = mock.create_autospec(GitHubAPI, instance=True)
        fake_api.send_request.return_value = {"message": "not found"}

        result = bump_automation.find_existing_comment_body(
            42, bump_automation.BREADCRUMB_MARKER, "ROCm/rocm-systems", fake_api
        )

        self.assertIsNone(result)


class BuildTimelineEntryTest(unittest.TestCase):
    """Tests for build_timeline_entry()."""

    def test_inclusion_entry_references_therock_pr(self):
        entry = bump_automation.build_timeline_entry(
            "2026-07-21", reverted=False, therock_pr_number=42, submodule="rocm-systems"
        )
        self.assertIn("2026-07-21", entry)
        self.assertIn("Included in TheRock via", entry)
        self.assertIn("ROCm/TheRock#42", entry)

    def test_reverted_entry_uses_reverted_wording(self):
        entry = bump_automation.build_timeline_entry(
            "2026-07-21", reverted=True, therock_pr_number=42, submodule="rocm-systems"
        )
        self.assertIn("Reverted out of TheRock via", entry)
        self.assertNotIn("Included in TheRock via", entry)

    def test_falls_back_when_no_therock_pr_found(self):
        entry = bump_automation.build_timeline_entry(
            "2026-07-21",
            reverted=False,
            therock_pr_number=None,
            submodule="rocm-libraries",
        )
        self.assertIn("rocm-libraries", entry)
        self.assertNotIn("#None", entry)


class BuildBreadcrumbBodyTest(unittest.TestCase):
    """Tests for build_breadcrumb_body(): single comment, newest-first history."""

    def test_first_event_has_no_prior_history(self):
        body = bump_automation.build_breadcrumb_body(
            existing_body=None,
            reverted=False,
            therock_pr_number=100,
            submodule="rocm-systems",
            event_date="2026-07-01",
        )

        self.assertTrue(body.startswith(bump_automation.BREADCRUMB_MARKER))
        self.assertIn(bump_automation.HISTORY_HEADER, body)
        self.assertIn("2026-07-01", body)
        self.assertIn("ROCm/TheRock#100", body)

    def test_new_entry_is_prepended_above_prior_history(self):
        existing_body = (
            f"{bump_automation.BREADCRUMB_MARKER}\n{bump_automation.HISTORY_HEADER}\n\n"
            f"- **2026-07-01** — Included in TheRock via ROCm/TheRock#100.\n"
        )

        body = bump_automation.build_breadcrumb_body(
            existing_body=existing_body,
            reverted=True,
            therock_pr_number=101,
            submodule="rocm-systems",
            event_date="2026-07-05",
        )

        # Both entries must be present, and the new (reverted) one must sort
        # ABOVE the older (included) one -- this is the whole point of
        # maintaining one comment with an explicit history list, instead of
        # relying on GitHub's fixed comment-creation ordering.
        reverted_idx = body.index("2026-07-05")
        included_idx = body.index("2026-07-01")
        self.assertLess(reverted_idx, included_idx)
        self.assertIn("ROCm/TheRock#101", body)
        self.assertIn("ROCm/TheRock#100", body)
        # Exactly one marker/comment -- not a second, separately-marked comment.
        self.assertEqual(body.count(bump_automation.BREADCRUMB_MARKER), 1)

    def test_three_events_preserve_full_history_newest_first(self):
        body_after_first = bump_automation.build_breadcrumb_body(
            existing_body=None,
            reverted=False,
            therock_pr_number=100,
            submodule="rocm-systems",
            event_date="2026-07-01",
        )
        body_after_revert = bump_automation.build_breadcrumb_body(
            existing_body=body_after_first,
            reverted=True,
            therock_pr_number=101,
            submodule="rocm-systems",
            event_date="2026-07-05",
        )
        body_after_reinclusion = bump_automation.build_breadcrumb_body(
            existing_body=body_after_revert,
            reverted=False,
            therock_pr_number=150,
            submodule="rocm-systems",
            event_date="2026-07-10",
        )

        for pr_number in (100, 101, 150):
            self.assertIn(f"ROCm/TheRock#{pr_number}", body_after_reinclusion)
        # Newest-first: 07-10 (re-included) above 07-05 (reverted) above 07-01.
        idx_10 = body_after_reinclusion.index("2026-07-10")
        idx_05 = body_after_reinclusion.index("2026-07-05")
        idx_01 = body_after_reinclusion.index("2026-07-01")
        self.assertLess(idx_10, idx_05)
        self.assertLess(idx_05, idx_01)
        # Still exactly one comment/marker throughout.
        self.assertEqual(
            body_after_reinclusion.count(bump_automation.BREADCRUMB_MARKER), 1
        )

    def test_ignores_prior_body_without_history_header(self):
        """A malformed/foreign existing body degrades to a fresh history."""
        body = bump_automation.build_breadcrumb_body(
            existing_body="some unrelated comment body",
            reverted=False,
            therock_pr_number=42,
            submodule="rocm-systems",
            event_date="2026-07-21",
        )

        self.assertEqual(body.count(bump_automation.HISTORY_HEADER), 1)
        self.assertIn("2026-07-21", body)


class BuildUnmappedSummaryBodyTest(unittest.TestCase):
    """Tests for build_unmapped_summary_body()."""

    def test_lists_all_unmapped_commits(self):
        body = bump_automation.build_unmapped_summary_body(
            reverted=False,
            submodule="rocm-systems",
            repo="ROCm/rocm-systems",
            unmapped_shas=["a" * 40, "b" * 40],
        )
        self.assertTrue(body.startswith(bump_automation.UNMAPPED_MARKER))
        self.assertIn("included in", body)
        self.assertIn(("a" * 40)[:7], body)
        self.assertIn(("b" * 40)[:7], body)
        self.assertIn("https://github.com/ROCm/rocm-systems/commit/" + "a" * 40, body)

    def test_reverted_summary_uses_removed_wording(self):
        body = bump_automation.build_unmapped_summary_body(
            reverted=True,
            submodule="rocm-systems",
            repo="ROCm/rocm-systems",
            unmapped_shas=["c" * 40],
        )
        self.assertIn("removed from", body)


class ProcessBumpTest(unittest.TestCase):
    """Tests for process_bump(): end-to-end wiring of a single submodule bump."""

    def _changed(self):
        return {
            "name": "rocm-systems",
            "repo": "ROCm/rocm-systems",
            "old_sha": "0" * 40,
            "new_sha": "1" * 40,
        }

    def _tokens(self):
        return {"systems": "systems-tok", "libraries": "libraries-tok"}

    def test_forward_bump_posts_breadcrumb_and_unmapped_summary(self):
        changed = self._changed()
        fake_api = mock.create_autospec(GitHubAPI, instance=True)

        with mock.patch.object(
            bump_automation, "GitHubAPI", return_value=fake_api
        ) as api_cls, mock.patch.object(
            bump_automation,
            "get_submodule_url",
            return_value="https://github.com/ROCm/rocm-systems.git",
        ), mock.patch.object(
            bump_automation,
            "get_api_base_from_url",
            return_value="https://api.github.com/repos/ROCm/rocm-systems",
        ) as get_api_base, mock.patch.object(
            bump_automation, "resolve_therock_pr_number", return_value=42
        ) as resolve_pr, mock.patch.object(
            bump_automation, "is_revert", return_value=False
        ) as is_revert_mock, mock.patch.object(
            bump_automation,
            "fetch_commits_in_range",
            return_value=[{"sha": "c1"}, {"sha": "c2"}],
        ) as fetch_commits, mock.patch.object(
            bump_automation,
            "resolve_prs_for_commits",
            return_value=({7, 8}, ["c2"]),
        ) as resolve_prs, mock.patch.object(
            bump_automation, "find_existing_comment_body", return_value=None
        ) as find_existing, mock.patch.object(
            bump_automation, "gha_update_pr_comment"
        ) as update_comment:
            bump_automation.process_bump(changed, self._tokens())

        api_cls.assert_called_once_with(github_token="systems-tok")
        get_api_base.assert_called_once_with(
            "https://github.com/ROCm/rocm-systems.git", "rocm-systems"
        )
        resolve_pr.assert_called_once_with(changed["new_sha"], fake_api)
        is_revert_mock.assert_called_once_with(
            changed["old_sha"],
            changed["new_sha"],
            "https://api.github.com/repos/ROCm/rocm-systems",
        )
        fetch_commits.assert_called_once_with(
            repo_name="ROCm/rocm-systems",
            start_sha=changed["old_sha"],
            end_sha=changed["new_sha"],
            api_base="https://api.github.com/repos/ROCm/rocm-systems",
        )
        resolve_prs.assert_called_once_with(
            "ROCm/rocm-systems", [{"sha": "c1"}, {"sha": "c2"}], fake_api
        )

        # One find-then-update per associated PR, both via the shared client.
        self.assertEqual(find_existing.call_count, 2)
        for call in find_existing.call_args_list:
            self.assertEqual(call.args[1], bump_automation.BREADCRUMB_MARKER)
            self.assertEqual(call.args[2], "ROCm/rocm-systems")
            self.assertIs(call.args[3], fake_api)

        self.assertEqual(update_comment.call_count, 3)  # 2 PR comments + 1 summary
        pr_calls = [
            c for c in update_comment.call_args_list if c.kwargs["pr_number"] in (7, 8)
        ]
        self.assertEqual(len(pr_calls), 2)
        for call in pr_calls:
            self.assertEqual(call.kwargs["marker"], bump_automation.BREADCRUMB_MARKER)
            self.assertEqual(call.kwargs["github_repository"], "ROCm/rocm-systems")
            self.assertIs(call.kwargs["github_api"], fake_api)
            self.assertIn("ROCm/TheRock#42", call.kwargs["body"])

        summary_call = next(
            c for c in update_comment.call_args_list if c.kwargs["pr_number"] == 42
        )
        self.assertEqual(summary_call.kwargs["marker"], bump_automation.UNMAPPED_MARKER)
        self.assertEqual(summary_call.kwargs["github_repository"], "ROCm/TheRock")
        self.assertIs(summary_call.kwargs["github_api"], fake_api)
        self.assertIn("c2", summary_call.kwargs["body"])

    def test_preserves_prior_history_when_updating_existing_comment(self):
        changed = self._changed()
        fake_api = mock.create_autospec(GitHubAPI, instance=True)
        prior_body = (
            f"{bump_automation.BREADCRUMB_MARKER}\n{bump_automation.HISTORY_HEADER}\n\n"
            f"- **2026-07-01** — Included in TheRock via ROCm/TheRock#100.\n"
        )

        with mock.patch.object(
            bump_automation, "GitHubAPI", return_value=fake_api
        ), mock.patch.object(
            bump_automation, "get_submodule_url", return_value="url"
        ), mock.patch.object(
            bump_automation, "get_api_base_from_url", return_value="api_base"
        ), mock.patch.object(
            bump_automation, "resolve_therock_pr_number", return_value=101
        ), mock.patch.object(
            bump_automation, "is_revert", return_value=True
        ), mock.patch.object(
            bump_automation, "fetch_commits_in_range", return_value=[{"sha": "r1"}]
        ), mock.patch.object(
            bump_automation, "resolve_prs_for_commits", return_value=({3}, [])
        ), mock.patch.object(
            bump_automation, "find_existing_comment_body", return_value=prior_body
        ), mock.patch.object(
            bump_automation, "gha_update_pr_comment"
        ) as update_comment:
            bump_automation.process_bump(changed, self._tokens())

        update_comment.assert_called_once()
        body = update_comment.call_args.kwargs["body"]
        # The new reverted entry AND the prior inclusion entry must both be
        # present in the single updated comment.
        self.assertIn("ROCm/TheRock#101", body)
        self.assertIn("ROCm/TheRock#100", body)
        self.assertEqual(body.count(bump_automation.BREADCRUMB_MARKER), 1)

    def test_libraries_bump_uses_libraries_token(self):
        changed = {
            "name": "rocm-libraries",
            "repo": "ROCm/rocm-libraries",
            "old_sha": "0" * 40,
            "new_sha": "1" * 40,
        }
        fake_api = mock.create_autospec(GitHubAPI, instance=True)

        with mock.patch.object(
            bump_automation, "GitHubAPI", return_value=fake_api
        ) as api_cls, mock.patch.object(
            bump_automation, "get_submodule_url", return_value="url"
        ), mock.patch.object(
            bump_automation, "get_api_base_from_url", return_value="api_base"
        ), mock.patch.object(
            bump_automation, "resolve_therock_pr_number", return_value=None
        ), mock.patch.object(
            bump_automation, "is_revert", return_value=False
        ), mock.patch.object(
            bump_automation, "fetch_commits_in_range", return_value=[]
        ), mock.patch.object(
            bump_automation, "resolve_prs_for_commits"
        ) as resolve_prs, mock.patch.object(
            bump_automation, "gha_update_pr_comment"
        ) as update_comment:
            bump_automation.process_bump(changed, self._tokens())

        api_cls.assert_called_once_with(github_token="libraries-tok")
        # No commits in range -> nothing further should be queried or posted.
        resolve_prs.assert_not_called()
        update_comment.assert_not_called()

    def test_rocgdb_bump_reuses_systems_token(self):
        changed = {
            "name": "debug-tools/rocgdb/source",
            "repo": "ROCm/rocgdb",
            "old_sha": "0" * 40,
            "new_sha": "1" * 40,
        }
        fake_api = mock.create_autospec(GitHubAPI, instance=True)
        tokens = {
            "systems": "systems-tok",
            "libraries": "libraries-tok",
            "rocgdb": "rocgdb-tok",
        }

        with mock.patch.object(
            bump_automation, "GitHubAPI", return_value=fake_api
        ) as api_cls, mock.patch.object(
            bump_automation, "get_submodule_url", return_value="url"
        ), mock.patch.object(
            bump_automation, "get_api_base_from_url", return_value="api_base"
        ), mock.patch.object(
            bump_automation, "resolve_therock_pr_number", return_value=None
        ), mock.patch.object(
            bump_automation, "is_revert", return_value=False
        ), mock.patch.object(
            bump_automation, "fetch_commits_in_range", return_value=[]
        ):
            bump_automation.process_bump(changed, tokens)

        # rocgdb's token_key is "systems" (see SUBMODULE_CONFIG), not "rocgdb".
        api_cls.assert_called_once_with(github_token="systems-tok")

    def test_revert_uses_swapped_range_and_reverted_wording(self):
        changed = self._changed()
        fake_api = mock.create_autospec(GitHubAPI, instance=True)

        with mock.patch.object(
            bump_automation, "GitHubAPI", return_value=fake_api
        ), mock.patch.object(
            bump_automation, "get_submodule_url", return_value="url"
        ), mock.patch.object(
            bump_automation, "get_api_base_from_url", return_value="api_base"
        ), mock.patch.object(
            bump_automation, "resolve_therock_pr_number", return_value=99
        ), mock.patch.object(
            bump_automation, "is_revert", return_value=True
        ), mock.patch.object(
            bump_automation, "fetch_commits_in_range", return_value=[{"sha": "r1"}]
        ) as fetch_commits, mock.patch.object(
            bump_automation, "resolve_prs_for_commits", return_value=({3}, [])
        ), mock.patch.object(
            bump_automation, "find_existing_comment_body", return_value=None
        ), mock.patch.object(
            bump_automation, "gha_update_pr_comment"
        ) as update_comment:
            bump_automation.process_bump(changed, self._tokens())

        # Reverted: fetch range swaps new_sha -> old_sha (commits being undone).
        fetch_commits.assert_called_once_with(
            repo_name="ROCm/rocm-systems",
            start_sha=changed["new_sha"],
            end_sha=changed["old_sha"],
            api_base="api_base",
        )

        # No unmapped commits -> only the PR comment, no summary comment.
        update_comment.assert_called_once()
        call_kwargs = update_comment.call_args.kwargs
        self.assertEqual(call_kwargs["pr_number"], 3)
        self.assertEqual(call_kwargs["marker"], bump_automation.BREADCRUMB_MARKER)
        self.assertIn("Reverted out of TheRock via", call_kwargs["body"])
        self.assertIs(call_kwargs["github_api"], fake_api)

    def test_unmapped_commits_without_therock_pr_are_logged_not_posted(self):
        changed = self._changed()
        fake_api = mock.create_autospec(GitHubAPI, instance=True)

        with mock.patch.object(
            bump_automation, "GitHubAPI", return_value=fake_api
        ), mock.patch.object(
            bump_automation, "get_submodule_url", return_value="url"
        ), mock.patch.object(
            bump_automation, "get_api_base_from_url", return_value="api_base"
        ), mock.patch.object(
            bump_automation, "resolve_therock_pr_number", return_value=None
        ), mock.patch.object(
            bump_automation, "is_revert", return_value=False
        ), mock.patch.object(
            bump_automation, "fetch_commits_in_range", return_value=[{"sha": "c1"}]
        ), mock.patch.object(
            bump_automation, "resolve_prs_for_commits", return_value=(set(), ["c1"])
        ), mock.patch.object(
            bump_automation, "gha_update_pr_comment"
        ) as update_comment:
            bump_automation.process_bump(changed, self._tokens())

        # No PR found for the bump commit -> can't post an unmapped summary
        # anywhere; must not raise, must not post any comment.
        update_comment.assert_not_called()


class HandlePostBreadcrumbsTest(unittest.TestCase):
    """Tests for handle_post_breadcrumbs()'s wiring of detect -> process."""

    def test_no_change_returns_without_processing(self):
        with mock.patch.object(
            bump_automation, "detect_changed_submodule", return_value=None
        ) as detect, mock.patch.object(bump_automation, "process_bump") as process_bump:
            bump_automation.handle_post_breadcrumbs(
                "aaa", "bbb", {"systems": "s-tok", "libraries": "l-tok"}
            )

        detect.assert_called_once_with("aaa", "bbb")
        process_bump.assert_not_called()

    def test_detected_change_is_processed(self):
        changed = {"name": "rocm-systems", "repo": "ROCm/rocm-systems"}
        tokens = {"systems": "s-tok", "libraries": "l-tok"}
        with mock.patch.object(
            bump_automation, "detect_changed_submodule", return_value=changed
        ) as detect, mock.patch.object(bump_automation, "process_bump") as process_bump:
            bump_automation.handle_post_breadcrumbs("aaa", "bbb", tokens)

        detect.assert_called_once_with("aaa", "bbb")
        process_bump.assert_called_once_with(changed, tokens)


class MainDispatchTest(unittest.TestCase):
    """Tests that main() routes --event_type post_breadcrumbs correctly."""

    def test_post_breadcrumbs_event_dispatches_to_handler(self):
        with mock.patch.object(bump_automation, "run"), mock.patch.object(
            bump_automation, "handle_post_breadcrumbs"
        ) as handle_mock:
            bump_automation.main(
                [
                    "--event_type",
                    "post_breadcrumbs",
                    "--before",
                    "aaa",
                    "--after",
                    "bbb",
                    "--systems_token",
                    "s-tok",
                    "--libraries_token",
                    "l-tok",
                    "--rocgdb_token",
                    "r-tok",
                ]
            )

        handle_mock.assert_called_once_with(
            "aaa",
            "bbb",
            {"systems": "s-tok", "libraries": "l-tok", "rocgdb": "r-tok"},
        )

    def test_required_arguments_enforced(self):
        with mock.patch.object(bump_automation, "run"):
            with self.assertRaises(SystemExit):
                bump_automation.main(["--event_type", "post_breadcrumbs"])


if __name__ == "__main__":
    unittest.main()
