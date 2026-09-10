# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from bump_automation import (
    _baseline_gate_jobs_succeeded,
    _clone_url,
    _list_run_jobs,
    close_stale_prs,
    close_stale_therock_ref_prs,
    create_therock_bump,
    find_therock_workflow_files,
    generate_pr_body,
    get_baseline_run_id_from_merged_pr,
    GITHUB_SEARCH_PAGE_SIZE,
    GITHUB_SEARCH_RESULT_LIMIT,
    get_submodule_sha,
    handle_push,
    latest_commit,
    LIBRARIES_BASELINE_GATE_JOB_NAME,
    search_issues,
    submodule_changed,
    update_ci_env_file,
    update_ref_in_file,
    update_therock_workflow_file,
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


def _gate_job(conclusion: str | None, name: str = LIBRARIES_BASELINE_GATE_JOB_NAME):
    return {
        "name": f"Linux::release / Build Multi-Arch Stages / {name}",
        "conclusion": conclusion,
    }


def _jobs_page(jobs: list[dict], total_count: int | None = None) -> dict:
    return {
        "jobs": jobs,
        "total_count": total_count if total_count is not None else len(jobs),
    }


class ListRunJobsTest(unittest.TestCase):
    def test_returns_single_page_without_further_requests(self):
        with patch(
            "bump_automation.gh_api",
            return_value=_jobs_page([_gate_job("success")]),
        ) as mock_api:
            jobs = _list_run_jobs("ROCm/TheRock", "token", 111)
        self.assertEqual(len(jobs), 1)
        mock_api.assert_called_once()

    def test_follows_pagination_until_all_jobs_are_retrieved(self):
        first_page = _jobs_page([_gate_job("success")] * 100, total_count=101)
        second_page = _jobs_page([_gate_job("success")], total_count=101)
        with patch(
            "bump_automation.gh_api", side_effect=[first_page, second_page]
        ) as mock_api:
            jobs = _list_run_jobs("ROCm/TheRock", "token", 111)
        self.assertEqual(len(jobs), 101)
        self.assertEqual(mock_api.call_count, 2)


class BaselineGateJobsSucceededTest(unittest.TestCase):
    def test_true_when_the_single_gate_job_succeeded(self):
        with patch(
            "bump_automation.gh_api", return_value=_jobs_page([_gate_job("success")])
        ):
            self.assertTrue(_baseline_gate_jobs_succeeded("ROCm/TheRock", "token", 111))

    def test_true_only_when_every_platform_gate_job_succeeded(self):
        jobs = [
            {
                "name": f"Linux::release / Build Multi-Arch Stages / {LIBRARIES_BASELINE_GATE_JOB_NAME}",
                "conclusion": "success",
            },
            {
                "name": f"Windows::release / Build Multi-Arch Stages / {LIBRARIES_BASELINE_GATE_JOB_NAME}",
                "conclusion": "failure",
            },
        ]
        with patch("bump_automation.gh_api", return_value=_jobs_page(jobs)):
            self.assertFalse(
                _baseline_gate_jobs_succeeded("ROCm/TheRock", "token", 111)
            )

    def test_false_when_no_gate_job_is_found(self):
        with patch(
            "bump_automation.gh_api",
            return_value=_jobs_page([{"name": "setup", "conclusion": "success"}]),
        ):
            self.assertFalse(
                _baseline_gate_jobs_succeeded("ROCm/TheRock", "token", 111)
            )

    def test_false_when_gate_job_conclusion_is_missing(self):
        with patch(
            "bump_automation.gh_api", return_value=_jobs_page([_gate_job(None)])
        ):
            self.assertFalse(
                _baseline_gate_jobs_succeeded("ROCm/TheRock", "token", 111)
            )


class GetBaselineRunIdFromMergedPrTest(unittest.TestCase):
    MERGE_SHA = "df3d451a3c054e14705ddf94e58498e1208df8d5"
    HEAD_SHA = "23bc501d4b826695062a657d0b582076c354dd77"

    def _pr(self, number: int = 7999) -> dict:
        return {
            "number": number,
            "merged_at": "2026-09-08T20:33:17Z",
            "merge_commit_sha": self.MERGE_SHA,
            "head": {"sha": self.HEAD_SHA},
        }

    def _run(self, run_id: int, conclusion: str | None, name: str = "Multi-Arch CI"):
        return {"id": run_id, "name": name, "conclusion": conclusion}

    def test_returns_the_run_id_when_its_gate_job_succeeded(self):
        with patch(
            "bump_automation.gh_api",
            side_effect=[
                [self._pr()],
                {"workflow_runs": [self._run(111, "success")]},
                _jobs_page([_gate_job("success")]),
            ],
        ) as mock_api:
            run_id = get_baseline_run_id_from_merged_pr(
                "ROCm/TheRock", "token", self.MERGE_SHA
            )
        self.assertEqual(run_id, "111")
        # The run lookup must key off the PR's pre-merge head SHA, not the
        # merge commit itself.
        self.assertIn(self.HEAD_SHA, mock_api.call_args_list[1].args[1])

    def test_returns_the_run_id_even_when_overall_run_conclusion_is_failure(self):
        # This is the real scenario that motivated the gate-job check: an
        # unrelated math-libs/compiler-runtime failure elsewhere in the
        # matrix makes the whole run "failure" even though every stage
        # rocm-libraries reuses succeeded.
        with patch(
            "bump_automation.gh_api",
            side_effect=[
                [self._pr()],
                {"workflow_runs": [self._run(111, "failure")]},
                _jobs_page([_gate_job("success")]),
            ],
        ):
            run_id = get_baseline_run_id_from_merged_pr(
                "ROCm/TheRock", "token", self.MERGE_SHA
            )
        self.assertEqual(run_id, "111")

    def test_skips_runs_whose_gate_job_did_not_succeed(self):
        with patch(
            "bump_automation.gh_api",
            side_effect=[
                [self._pr()],
                {
                    "workflow_runs": [
                        self._run(111, "success"),
                        self._run(112, "success"),
                    ]
                },
                _jobs_page([_gate_job("failure")]),  # run 111's gate job
                _jobs_page([_gate_job("success")]),  # run 112's gate job
            ],
        ):
            run_id = get_baseline_run_id_from_merged_pr(
                "ROCm/TheRock", "token", self.MERGE_SHA
            )
        self.assertEqual(run_id, "112")

    def test_returns_none_when_no_run_gate_job_succeeded(self):
        with patch(
            "bump_automation.gh_api",
            side_effect=[
                [self._pr()],
                {
                    "workflow_runs": [
                        self._run(111, "failure"),
                        self._run(112, "cancelled"),
                    ]
                },
                _jobs_page([_gate_job("failure")]),
                _jobs_page([_gate_job("cancelled")]),
            ],
        ):
            run_id = get_baseline_run_id_from_merged_pr(
                "ROCm/TheRock", "token", self.MERGE_SHA
            )
        self.assertIsNone(run_id)

    def test_returns_none_when_gate_job_is_missing(self):
        # A run that predates the gate job (or is otherwise structurally
        # different) must not be treated as a usable baseline.
        with patch(
            "bump_automation.gh_api",
            side_effect=[
                [self._pr()],
                {"workflow_runs": [self._run(111, "success")]},
                _jobs_page([{"name": "setup", "conclusion": "success"}]),
            ],
        ):
            run_id = get_baseline_run_id_from_merged_pr(
                "ROCm/TheRock", "token", self.MERGE_SHA
            )
        self.assertIsNone(run_id)

    def test_returns_none_when_no_matching_workflow_name(self):
        with patch(
            "bump_automation.gh_api",
            side_effect=[
                [self._pr()],
                {"workflow_runs": [self._run(111, "success", name="pre-commit")]},
            ],
        ):
            run_id = get_baseline_run_id_from_merged_pr(
                "ROCm/TheRock", "token", self.MERGE_SHA
            )
        self.assertIsNone(run_id)

    def test_returns_none_when_no_merged_pr_found(self):
        with patch("bump_automation.gh_api", return_value=[]) as mock_api:
            run_id = get_baseline_run_id_from_merged_pr(
                "ROCm/TheRock", "token", self.MERGE_SHA
            )
        self.assertIsNone(run_id)
        mock_api.assert_called_once()


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


class UpdateTheRockWorkflowFileTest(unittest.TestCase):
    OLD_SHA = "1" * 40
    OTHER_SHA = "2" * 40
    NEW_SHA = "3" * 40
    STALE_COMMENT_SHA = "4" * 40

    def _run(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            update_therock_workflow_file(path, self.NEW_SHA)
            return Path(path).read_text()
        finally:
            os.unlink(path)

    def test_updates_reusable_workflow_and_matching_source_refs(self):
        content = textwrap.dedent(
            f"""\
            # TheRock ref: pinned to ROCm/TheRock commit {self.STALE_COMMENT_SHA}.
            jobs:
              setup:
                uses: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@{self.OLD_SHA} # 2026-01-02
                with:
                  repository: ROCm/TheRock
                  ref: {self.OLD_SHA} # 2026-01-02
            """
        )
        result = self._run(content)
        self.assertEqual(result.count(self.NEW_SHA), 3)
        self.assertNotIn(self.OLD_SHA, result)
        self.assertNotIn(self.STALE_COMMENT_SHA, result)
        self.assertNotIn("2026-01-02", result)

    def test_updates_therock_checkout_without_reusable_workflow(self):
        content = textwrap.dedent(
            f"""\
            steps:
              - uses: actions/checkout@{self.OTHER_SHA}
                with:
                  repository: "ROCm/TheRock"
                  ref: {self.OLD_SHA} # 2026-01-02
            """
        )
        result = self._run(content)
        self.assertIn(f"actions/checkout@{self.OTHER_SHA}", result)
        self.assertIn(f"ref: {self.NEW_SHA}", result)
        self.assertNotIn(self.OLD_SHA, result)

    def test_updates_therock_ref_override_default(self):
        content = textwrap.dedent(
            f"""\
            steps:
              - uses: actions/checkout@{self.OTHER_SHA}
                with:
                  repository: "ROCm/TheRock"
                  ref: ${{{{ inputs.therock_ref_override || '{self.OLD_SHA}' }}}}
            """
        )
        result = self._run(content)
        self.assertIn(self.NEW_SHA, result)
        self.assertNotIn(self.OLD_SHA, result)

    def test_fails_when_workflow_has_no_pinned_therock_ref(self):
        with self.assertRaisesRegex(RuntimeError, "No pinned TheRock refs"):
            self._run("name: unrelated workflow\n")


class FindTheRockWorkflowFilesTest(unittest.TestCase):
    def test_discovers_all_yaml_extensions_with_pinned_refs(self):
        old_sha = "1" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflows = root / "workflows"
            actions = root / "actions"
            workflows.mkdir()
            actions.mkdir()
            (workflows / "reusable.yml").write_text(
                f"uses: ROCm/TheRock/.github/workflows/ci.yml@{old_sha}\n",
                encoding="utf-8",
            )
            (actions / "checkout.yaml").write_text(
                textwrap.dedent(
                    f"""\
                    repository: ROCm/TheRock
                    ref: {old_sha}
                    """
                ),
                encoding="utf-8",
            )
            (workflows / "unrelated.yml").write_text(
                "uses: actions/checkout@v4\n", encoding="utf-8"
            )

            result = find_therock_workflow_files(root)

        self.assertEqual(
            result,
            [
                (actions / "checkout.yaml").as_posix(),
                (workflows / "reusable.yml").as_posix(),
            ],
        )

    def test_fails_when_no_pinned_workflows_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "unrelated.yml").write_text(
                "uses: actions/checkout@v4\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "No pinned TheRock workflow files"
            ):
                find_therock_workflow_files(root)


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


def _search_page(numbers: range, total_count: int) -> dict:
    return {
        "total_count": total_count,
        "items": [{"number": n} for n in numbers],
    }


class SearchIssuesTest(unittest.TestCase):
    def test_returns_single_page_without_further_requests(self):
        page = _search_page(range(3), total_count=3)
        with patch("bump_automation.gh_api", return_value=page) as mock_api:
            items = search_issues("token", "repo:ROCm/rocm-libraries is:pr")

        self.assertEqual(len(items), 3)
        mock_api.assert_called_once()
        self.assertIn("page=1", mock_api.call_args.args[1])

    def test_follows_pagination_until_all_matches_are_retrieved(self):
        total = GITHUB_SEARCH_PAGE_SIZE + 5
        pages = [
            _search_page(range(GITHUB_SEARCH_PAGE_SIZE), total_count=total),
            _search_page(range(GITHUB_SEARCH_PAGE_SIZE, total), total_count=total),
        ]
        with patch("bump_automation.gh_api", side_effect=pages) as mock_api:
            items = search_issues("token", "repo:ROCm/rocm-libraries is:pr")

        self.assertEqual([item["number"] for item in items], list(range(total)))
        requested_pages = [call.args[1] for call in mock_api.call_args_list]
        self.assertEqual(len(requested_pages), 2)
        self.assertIn("page=1", requested_pages[0])
        self.assertIn("page=2", requested_pages[1])

    def test_stops_at_the_search_api_result_limit(self):
        full_page = _search_page(range(GITHUB_SEARCH_PAGE_SIZE), total_count=10**6)
        with patch("bump_automation.gh_api", return_value=full_page) as mock_api:
            items = search_issues("token", "repo:ROCm/rocm-libraries is:pr")

        self.assertEqual(len(items), GITHUB_SEARCH_RESULT_LIMIT)
        self.assertEqual(
            mock_api.call_count, GITHUB_SEARCH_RESULT_LIMIT // GITHUB_SEARCH_PAGE_SIZE
        )

    def test_quotes_the_query(self):
        with patch(
            "bump_automation.gh_api", return_value=_search_page(range(0), 0)
        ) as mock_api:
            search_issues("token", 'repo:ROCm/rocm-libraries in:title "Update"')

        endpoint = mock_api.call_args.args[1]
        self.assertTrue(endpoint.startswith("search/issues?q="))
        self.assertNotIn(" ", endpoint)
        self.assertNotIn('"', endpoint)


class CloseStaleTheRockRefPrsTest(unittest.TestCase):
    NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)

    def _make_pr(self, number: int, title: str, author: str, *, age: timedelta) -> dict:
        created = self.NOW - age
        return {
            "number": number,
            "title": title,
            "user": {"login": author},
            "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def test_closes_older_bot_reference_prs_only(self):
        search_result = {
            "items": [
                self._make_pr(
                    10,
                    "Update TheRock reference to (1111111)",
                    "assistant-librarian[bot]",
                    age=timedelta(days=3),
                ),
                self._make_pr(
                    20,
                    "Update TheRock reference to (2222222)",
                    "assistant-librarian[bot]",
                    age=timedelta(days=3),
                ),
                self._make_pr(
                    30,
                    "Update TheRock reference to (3333333)",
                    "human-author",
                    age=timedelta(days=3),
                ),
            ]
        }
        with patch("bump_automation.gh_api", return_value=search_result) as mock_api:
            close_stale_therock_ref_prs(
                "ROCm/rocm-libraries",
                current_pr_number=20,
                token="token",
                bot_author="assistant-librarian[bot]",
                now=self.NOW,
            )

        closed_endpoints = [
            call.args[1]
            for call in mock_api.call_args_list
            if call.kwargs.get("method") == "PATCH"
        ]
        self.assertEqual(closed_endpoints, ["repos/ROCm/rocm-libraries/pulls/10"])

    def test_leaves_recent_bot_reference_prs_open(self):
        search_result = {
            "items": [
                self._make_pr(
                    10,
                    "Update TheRock reference to (1111111)",
                    "assistant-librarian[bot]",
                    age=timedelta(days=1, hours=23),
                ),
                self._make_pr(
                    11,
                    "Update TheRock reference to (aaaaaaa)",
                    "assistant-librarian[bot]",
                    age=timedelta(days=2),
                ),
            ]
        }
        with patch("bump_automation.gh_api", return_value=search_result) as mock_api:
            close_stale_therock_ref_prs(
                "ROCm/rocm-libraries",
                current_pr_number=20,
                token="token",
                bot_author="assistant-librarian[bot]",
                now=self.NOW,
            )

        closed_endpoints = [
            call.args[1]
            for call in mock_api.call_args_list
            if call.kwargs.get("method") == "PATCH"
        ]
        self.assertEqual(closed_endpoints, ["repos/ROCm/rocm-libraries/pulls/11"])

    def test_searches_all_open_reference_prs(self):
        with patch("bump_automation.gh_api", return_value={"items": []}) as mock_api:
            close_stale_therock_ref_prs(
                "ROCm/rocm-libraries",
                current_pr_number=20,
                token="token",
                bot_author="assistant-librarian[bot]",
            )

        endpoint = mock_api.call_args.args[1]
        self.assertTrue(endpoint.startswith("search/issues?q="))
        self.assertIn("per_page=100", endpoint)

    def test_closes_stale_prs_found_beyond_the_first_search_page(self):
        # A full first page of unrelated PRs must not hide stale bot PRs that
        # only appear on a later page.
        first_page = {
            "total_count": GITHUB_SEARCH_PAGE_SIZE + 1,
            "items": [
                self._make_pr(
                    number,
                    "Update TheRock reference to (1111111)",
                    "human-author",
                    age=timedelta(days=3),
                )
                for number in range(GITHUB_SEARCH_PAGE_SIZE)
            ],
        }
        second_page = {
            "total_count": GITHUB_SEARCH_PAGE_SIZE + 1,
            "items": [
                self._make_pr(
                    2000,
                    "Update TheRock reference to (2222222)",
                    "assistant-librarian[bot]",
                    age=timedelta(days=3),
                )
            ],
        }
        with patch(
            "bump_automation.gh_api",
            side_effect=[first_page, second_page, {}, {}],
        ) as mock_api:
            close_stale_therock_ref_prs(
                "ROCm/rocm-libraries",
                current_pr_number=20,
                token="token",
                bot_author="assistant-librarian[bot]",
                now=self.NOW,
            )

        closed_endpoints = [
            call.args[1]
            for call in mock_api.call_args_list
            if call.kwargs.get("method") == "PATCH"
        ]
        self.assertEqual(closed_endpoints, ["repos/ROCm/rocm-libraries/pulls/2000"])


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

    def test_mesa_fork_submodule_only_closes_stale_prs_then_returns(self):
        """mesa-fork is submodule-only: handle_push must close stale PRs using
        the systems token and skip cloning the upstream repo."""

        def changed(before, after, path):
            return path == "third-party/sysdeps/common/mesa-fork"

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
                                "mesa-fork": "systems-token",
                            },
                        )

        mock_close.assert_called_once()
        # mesa-fork reuses the systems token.
        self.assertEqual(mock_close.call_args.args[2], "systems-token")
        # submodule-only: must not clone any upstream repo.
        mock_tmp.assert_not_called()

    def test_ref_updater_closes_stale_prs_with_its_own_bot_author(self):
        """rocm-systems' bump PRs are opened by "systems-assistant[bot]", a
        different GitHub App than rocm-libraries' "assistant-librarian[bot]".
        handle_push must pass each repo's own bot identity through to
        close_stale_therock_ref_prs rather than a single hardcoded value."""

        def changed(before, after, path):
            return path == "rocm-systems"

        with patch("bump_automation.submodule_changed", side_effect=changed):
            with patch(
                "bump_automation.get_submodule_sha", return_value="oldsha1234567"
            ):
                with patch("bump_automation.close_stale_prs"):
                    with patch("bump_automation.run"):
                        with patch("bump_automation.os.chdir"):
                            with patch(
                                "bump_automation.os.path.exists", return_value=True
                            ):
                                with patch("bump_automation.update_ref_in_file"):
                                    with patch(
                                        "bump_automation.gh_api",
                                        return_value={"number": 1},
                                    ):
                                        with patch(
                                            "bump_automation.close_stale_therock_ref_prs"
                                        ) as mock_close_ref:
                                            with patch(
                                                "bump_automation.create_therock_bump"
                                            ):
                                                handle_push(
                                                    "before",
                                                    "after",
                                                    {
                                                        "systems": "systems-token",
                                                        "libraries": "libraries-token",
                                                    },
                                                )

        mock_close_ref.assert_called_once()
        self.assertEqual(mock_close_ref.call_args.args[0], "ROCm/rocm-systems")
        self.assertEqual(mock_close_ref.call_args.args[3], "systems-assistant[bot]")

    def test_ci_env_updater_closes_stale_prs_with_its_own_bot_author(self):
        """rocm-libraries must keep using its own "assistant-librarian[bot]"
        identity, not rocm-systems'."""

        def changed(before, after, path):
            return path == "rocm-libraries"

        with patch("bump_automation.submodule_changed", side_effect=changed):
            with patch(
                "bump_automation.get_submodule_sha", return_value="oldsha1234567"
            ):
                with patch("bump_automation.close_stale_prs"):
                    with patch(
                        "bump_automation.get_baseline_run_id_from_merged_pr",
                        return_value=None,
                    ):
                        with patch("bump_automation.run"):
                            with patch("bump_automation.os.chdir"):
                                with patch(
                                    "bump_automation.os.path.exists",
                                    return_value=True,
                                ):
                                    with patch("bump_automation.update_ci_env_file"):
                                        with patch(
                                            "bump_automation.find_therock_workflow_files",
                                            return_value=[],
                                        ):
                                            with patch(
                                                "bump_automation.gh_api",
                                                return_value={"number": 1},
                                            ):
                                                with patch(
                                                    "bump_automation.close_stale_therock_ref_prs"
                                                ) as mock_close_ref:
                                                    with patch(
                                                        "bump_automation.create_therock_bump"
                                                    ):
                                                        handle_push(
                                                            "before",
                                                            "after",
                                                            {
                                                                "systems": "systems-token",
                                                                "libraries": "libraries-token",
                                                            },
                                                        )

        mock_close_ref.assert_called_once()
        self.assertEqual(mock_close_ref.call_args.args[0], "ROCm/rocm-libraries")
        self.assertEqual(mock_close_ref.call_args.args[3], "assistant-librarian[bot]")

    def test_only_submodule_bypasses_auto_detection(self):
        """The manual workflow_dispatch replay path (bump_submodules.yml's
        pin_before/pin_after inputs) passes only_submodule to target one
        submodule directly; submodule_changed must never be consulted."""
        with patch("bump_automation.submodule_changed") as mock_changed:
            with patch(
                "bump_automation.get_submodule_sha", return_value="oldsha1234567"
            ):
                with patch("bump_automation.close_stale_prs"):
                    with patch(
                        "bump_automation.get_baseline_run_id_from_merged_pr",
                        return_value=None,
                    ):
                        with patch("bump_automation.run"):
                            with patch("bump_automation.os.chdir"):
                                with patch(
                                    "bump_automation.os.path.exists",
                                    return_value=True,
                                ):
                                    with patch("bump_automation.update_ci_env_file"):
                                        with patch(
                                            "bump_automation.find_therock_workflow_files",
                                            return_value=[],
                                        ):
                                            with patch(
                                                "bump_automation.gh_api",
                                                return_value={"number": 1},
                                            ):
                                                with patch(
                                                    "bump_automation.close_stale_therock_ref_prs"
                                                ):
                                                    with patch(
                                                        "bump_automation.create_therock_bump"
                                                    ) as mock_bump:
                                                        handle_push(
                                                            "before",
                                                            "after",
                                                            {
                                                                "systems": "systems-token",
                                                                "libraries": "libraries-token",
                                                            },
                                                            only_submodule="rocm-libraries",
                                                        )

        mock_changed.assert_not_called()
        mock_bump.assert_called_once_with("rocm-libraries", "libraries-token")

    def test_skip_next_bump_does_not_queue_next_bump(self):
        """The manual replay path sets skip_next_bump so it never opens a new
        Bump <submodule> PR in TheRock after updating the downstream pin,
        unlike a real push event which always queues the next bump."""
        with patch("bump_automation.get_submodule_sha", return_value="oldsha1234567"):
            with patch("bump_automation.close_stale_prs"):
                with patch(
                    "bump_automation.get_baseline_run_id_from_merged_pr",
                    return_value=None,
                ):
                    with patch("bump_automation.run"):
                        with patch("bump_automation.os.chdir"):
                            with patch(
                                "bump_automation.os.path.exists",
                                return_value=True,
                            ):
                                with patch("bump_automation.update_ci_env_file"):
                                    with patch(
                                        "bump_automation.find_therock_workflow_files",
                                        return_value=[],
                                    ):
                                        with patch(
                                            "bump_automation.gh_api",
                                            return_value={"number": 1},
                                        ):
                                            with patch(
                                                "bump_automation.close_stale_therock_ref_prs"
                                            ):
                                                with patch(
                                                    "bump_automation.create_therock_bump"
                                                ) as mock_bump:
                                                    handle_push(
                                                        "before",
                                                        "after",
                                                        {
                                                            "systems": "systems-token",
                                                            "libraries": "libraries-token",
                                                        },
                                                        only_submodule="rocm-libraries",
                                                        skip_next_bump=True,
                                                    )

        mock_bump.assert_not_called()


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

    def test_mesa_fork_maps_to_correct_submodule_and_token(self):
        """handle_schedule('mesa-fork') must call create_therock_bump with
        'third-party/sysdeps/common/mesa-fork' and tokens['mesa-fork']."""
        tokens = {
            "systems": "systems-token",
            "libraries": "libraries-token",
            "rocgdb": "rocgdb-token",
            "mesa-fork": "mesa-fork-token",
        }
        with patch("bump_automation.create_therock_bump") as mock_bump:
            from bump_automation import handle_schedule

            handle_schedule(tokens, "mesa-fork")

        mock_bump.assert_called_once_with(
            "third-party/sysdeps/common/mesa-fork", "mesa-fork-token"
        )

    def test_mesa_fork_does_not_invoke_other_submodules(self):
        tokens = {
            "systems": "systems-token",
            "libraries": "libraries-token",
            "rocgdb": "rocgdb-token",
            "mesa-fork": "mesa-fork-token",
        }
        with patch("bump_automation.create_therock_bump") as mock_bump:
            from bump_automation import handle_schedule

            handle_schedule(tokens, "mesa-fork")

        called_submodules = [c.args[0] for c in mock_bump.call_args_list]
        self.assertNotIn("rocm-systems", called_submodules)
        self.assertNotIn("rocm-libraries", called_submodules)
        self.assertNotIn("debug-tools/rocgdb/source", called_submodules)


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
