#!/usr/bin/env python
"""Unit tests for post_manifest_diff_pr_comment.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# Add build_tools to path so _therock_utils is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
# Add github_actions to path so post_manifest_diff_pr_comment is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
import post_manifest_diff_pr_comment as post_comment


def _make_output_root(run_id="12345", platform="linux"):
    return WorkflowOutputRoot(
        bucket="therock-ci-artifacts",
        external_repo="",
        run_id=run_id,
        platform=platform,
    )


class BuildCommentBodyTest(unittest.TestCase):
    """Tests for build_comment_body()."""

    def test_includes_marker_link_and_summary(self):
        body = post_comment.build_comment_body(
            "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/logs/manifest-diff/index.html",
            "**Commit Range:** `abc12345` -> `def67890` (2 submodules changed)",
        )

        self.assertIn(post_comment.MARKER, body)
        self.assertIn(
            "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/logs/manifest-diff/index.html",
            body,
        )
        self.assertIn("2 submodules changed", body)
        # Marker must appear so gha_update_pr_comment can find/update this comment.
        self.assertTrue(body.startswith(post_comment.MARKER))

    def test_omits_summary_line_when_blank(self):
        body = post_comment.build_comment_body("https://example.com/index.html", "")

        self.assertIn(post_comment.MARKER, body)
        self.assertIn("https://example.com/index.html", body)
        self.assertNotIn("Commit Range", body)


class RunTest(unittest.TestCase):
    """Tests for run(): URL computation + gha_update_pr_comment dispatch."""

    def test_posts_comment_with_computed_report_url(self):
        args = post_comment.argparse.Namespace(
            run_id="99999",
            pr_number=1234,
            commit_range_summary="**Commit Range:** `aaa` -> `bbb` (1 submodule changed)",
            github_repository="ROCm/TheRock",
            platform="linux",
        )

        with mock.patch.object(
            post_comment.WorkflowOutputRoot,
            "from_workflow_run",
            return_value=_make_output_root(run_id="99999"),
        ) as from_workflow_run, mock.patch.object(
            post_comment, "gha_update_pr_comment"
        ) as gha_update_pr_comment:
            post_comment.run(args)

        from_workflow_run.assert_called_once_with(run_id="99999", platform="linux")
        gha_update_pr_comment.assert_called_once()
        call_kwargs = gha_update_pr_comment.call_args.kwargs
        self.assertEqual(call_kwargs["pr_number"], 1234)
        self.assertEqual(call_kwargs["marker"], post_comment.MARKER)
        self.assertEqual(call_kwargs["github_repository"], "ROCm/TheRock")
        self.assertIn(
            "99999-linux/logs/manifest-diff/index.html", call_kwargs["body"]
        )
        self.assertIn("1 submodule changed", call_kwargs["body"])


class MainArgParsingTest(unittest.TestCase):
    """Tests for main()'s CLI argument handling."""

    def test_missing_run_id_errors_without_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                post_comment.main(["--pr-number", "1"])

    def test_pr_number_is_required(self):
        with self.assertRaises(SystemExit):
            post_comment.main(["--run-id", "123"])

    def test_run_id_falls_back_to_env_var(self):
        with mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "555"}, clear=True):
            with mock.patch.object(post_comment, "run") as run:
                result = post_comment.main(["--pr-number", "1"])

        self.assertEqual(result, 0)
        run.assert_called_once()
        self.assertEqual(run.call_args[0][0].run_id, "555")


if __name__ == "__main__":
    unittest.main()
