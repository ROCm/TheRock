#!/usr/bin/env python
"""Unit tests for s3_buckets.py."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.s3_buckets import (
    get_artifacts_bucket_config,
    get_artifacts_bucket_config_for_workflow_run,
)


# ---------------------------------------------------------------------------
# get_artifacts_bucket_config
# ---------------------------------------------------------------------------


class TestGetArtifactsBucketConfig(unittest.TestCase):
    def test_ci_rocm_therock(self):
        config = get_artifacts_bucket_config(
            release_type="", repository="ROCm/TheRock", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-ci-artifacts")

    def test_ci_fork_pr(self):
        config = get_artifacts_bucket_config(
            release_type="", repository="ROCm/TheRock", is_pr_from_fork=True
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")

    def test_ci_external_repo(self):
        config = get_artifacts_bucket_config(
            release_type="", repository="ROCm/rocm-libraries", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")

    def test_release_type_dev(self):
        config = get_artifacts_bucket_config(
            release_type="dev", repository="ROCm/TheRock", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-dev-artifacts")
        self.assertEqual(config.iam_role, "therock-dev")

    def test_release_type_from_rockrel(self):
        config = get_artifacts_bucket_config(
            release_type="nightly", repository="ROCm/rockrel", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-nightly-artifacts")

    def test_release_type_invalid_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_artifacts_bucket_config(
                release_type="bogus",
                repository="ROCm/TheRock",
                is_pr_from_fork=False,
            )
        self.assertIn("bogus", str(cm.exception))

    def test_release_type_disallowed_repo_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_artifacts_bucket_config(
                release_type="dev",
                repository="ROCm/rocm-libraries",
                is_pr_from_fork=False,
            )
        self.assertIn("ROCm/rocm-libraries", str(cm.exception))


# ---------------------------------------------------------------------------
# get_artifacts_bucket_config_for_workflow_run
# ---------------------------------------------------------------------------


class TestGetArtifactsBucketConfigForWorkflowRun(unittest.TestCase):
    """Test the workflow-run-aware wrapper."""

    def setUp(self):
        self.api_patcher = mock.patch(
            "github_actions.github_actions_api.gha_query_workflow_run_by_id"
        )
        self.mock_api = self.api_patcher.start()

        self.env_patcher = mock.patch.dict(os.environ)
        self.env_patcher.start()
        os.environ.pop("GITHUB_REPOSITORY", None)
        os.environ.pop("GITHUB_EVENT_NAME", None)
        os.environ.pop("GITHUB_EVENT_PATH", None)
        os.environ.pop("RELEASE_TYPE", None)

    def tearDown(self):
        self.env_patcher.stop()
        self.api_patcher.stop()

    def test_default_ci(self):
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock"
        )
        self.assertEqual(config.name, "therock-ci-artifacts")

    def test_explicit_release_type(self):
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", release_type="nightly"
        )
        self.assertEqual(config.name, "therock-nightly-artifacts")

    def test_release_type_from_env(self):
        os.environ["RELEASE_TYPE"] = "dev"
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock"
        )
        self.assertEqual(config.name, "therock-dev-artifacts")

    def test_explicit_release_type_overrides_env(self):
        os.environ["RELEASE_TYPE"] = "dev"
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", release_type="nightly"
        )
        self.assertEqual(config.name, "therock-nightly-artifacts")

    def test_workflow_run_same_repo(self):
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts")

    def test_workflow_run_from_fork(self):
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "SomeUser/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")

    def test_workflow_run_id_triggers_api_call(self):
        self.mock_api.return_value = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run_id="12345"
        )
        self.mock_api.assert_called_once_with("ROCm/TheRock", "12345")
        self.assertEqual(config.name, "therock-ci-artifacts")

    def _write_event(self, event: dict) -> str:
        """Write a synthetic GitHub event payload to a temp file.

        Returns the path. Caller must os.unlink() after use.
        Uses delete=False because NamedTemporaryFile(delete=True) holds an
        exclusive lock on Windows, preventing the code under test from reading.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(event, f)
            return f.name

    def test_fork_pr_from_event_payload(self):
        """Fork PR detected via event payload (no workflow_run dict)."""
        event_path = self._write_event(
            {"pull_request": {"head": {"repo": {"fork": True}}}}
        )
        try:
            os.environ["GITHUB_EVENT_NAME"] = "pull_request"
            os.environ["GITHUB_EVENT_PATH"] = event_path
            config = get_artifacts_bucket_config_for_workflow_run(
                github_repository="ROCm/TheRock"
            )
            self.assertEqual(config.name, "therock-ci-artifacts-external")
        finally:
            os.unlink(event_path)

    def test_same_repo_pr_from_event_payload(self):
        """Same-repo PR detected via event payload (no workflow_run dict)."""
        event_path = self._write_event(
            {"pull_request": {"head": {"repo": {"fork": False}}}}
        )
        try:
            os.environ["GITHUB_EVENT_NAME"] = "pull_request"
            os.environ["GITHUB_EVENT_PATH"] = event_path
            config = get_artifacts_bucket_config_for_workflow_run(
                github_repository="ROCm/TheRock"
            )
            self.assertEqual(config.name, "therock-ci-artifacts")
        finally:
            os.unlink(event_path)

    def test_workflow_run_id_ignored_when_workflow_run_provided(self):
        """workflow_run takes priority over workflow_run_id."""
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock",
            workflow_run=fake_run,
            workflow_run_id="99999",
        )
        self.mock_api.assert_not_called()
        self.assertEqual(config.name, "therock-ci-artifacts")


if __name__ == "__main__":
    unittest.main()
