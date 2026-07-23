# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
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


if __name__ == "__main__":
    unittest.main()
