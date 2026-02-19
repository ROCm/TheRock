#!/usr/bin/env python
"""Unit tests for run_outputs.py."""

import os
import sys
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.run_outputs import OutputLocation, RunOutputRoot


# ---------------------------------------------------------------------------
# OutputLocation
# ---------------------------------------------------------------------------


class TestOutputLocation(unittest.TestCase):
    def test_s3_uri(self):
        loc = OutputLocation("my-bucket", "12345-linux/file.tar.xz")
        self.assertEqual(loc.s3_uri, "s3://my-bucket/12345-linux/file.tar.xz")

    def test_https_url(self):
        loc = OutputLocation("my-bucket", "12345-linux/file.tar.xz")
        self.assertEqual(
            loc.https_url,
            "https://my-bucket.s3.amazonaws.com/12345-linux/file.tar.xz",
        )

    def test_local_path(self):
        loc = OutputLocation("my-bucket", "12345-linux/logs/group/build.log")
        result = loc.local_path(Path("/tmp/staging"))
        expected = Path("/tmp/staging/12345-linux/logs/group/build.log")
        self.assertEqual(result, expected)

    def test_frozen(self):
        loc = OutputLocation("bucket", "path")
        with self.assertRaises(AttributeError):
            loc.bucket = "other"


# ---------------------------------------------------------------------------
# RunOutputRoot — prefix
# ---------------------------------------------------------------------------


class TestRunOutputRootPrefix(unittest.TestCase):
    def _make_root(self, **kwargs):
        defaults = dict(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="12345",
            platform="linux",
        )
        defaults.update(kwargs)
        return RunOutputRoot(**defaults)

    def test_prefix_no_external_repo(self):
        root = self._make_root()
        self.assertEqual(root.prefix, "12345-linux")

    def test_prefix_with_external_repo(self):
        root = self._make_root(external_repo="Fork-TheRock/")
        self.assertEqual(root.prefix, "Fork-TheRock/12345-linux")

    def test_prefix_windows(self):
        root = self._make_root(platform="windows")
        self.assertEqual(root.prefix, "12345-windows")

    def test_frozen(self):
        root = self._make_root()
        with self.assertRaises(AttributeError):
            root.run_id = "99999"


# ---------------------------------------------------------------------------
# RunOutputRoot — location methods
# ---------------------------------------------------------------------------


class TestRunOutputRootLocations(unittest.TestCase):
    """Test that each location method returns correct relative paths."""

    def setUp(self):
        self.root = RunOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="99999",
            platform="linux",
        )

    def _assert_relative_path(self, loc: OutputLocation, expected_path: str):
        self.assertIsInstance(loc, OutputLocation)
        self.assertEqual(loc.bucket, "therock-ci-artifacts")
        self.assertEqual(loc.relative_path, expected_path)

    # -- Artifacts --

    def test_artifact(self):
        loc = self.root.artifact("blas_lib_gfx94X.tar.xz")
        self._assert_relative_path(loc, "99999-linux/blas_lib_gfx94X.tar.xz")

    def test_artifact_sha256sum(self):
        loc = self.root.artifact("blas_lib_gfx94X.tar.xz.sha256sum")
        self._assert_relative_path(loc, "99999-linux/blas_lib_gfx94X.tar.xz.sha256sum")

    def test_artifact_index(self):
        loc = self.root.artifact_index("gfx94X-dcgpu")
        self._assert_relative_path(loc, "99999-linux/index-gfx94X-dcgpu.html")

    # -- Logs --

    def test_log_dir(self):
        loc = self.root.log_dir("gfx94X-dcgpu")
        self._assert_relative_path(loc, "99999-linux/logs/gfx94X-dcgpu")

    def test_log_file(self):
        loc = self.root.log_file("gfx94X-dcgpu", "build.log")
        self._assert_relative_path(loc, "99999-linux/logs/gfx94X-dcgpu/build.log")

    def test_log_file_ninja_archive(self):
        loc = self.root.log_file("gfx94X-dcgpu", "ninja_logs.tar.gz")
        self._assert_relative_path(
            loc, "99999-linux/logs/gfx94X-dcgpu/ninja_logs.tar.gz"
        )

    def test_log_index(self):
        loc = self.root.log_index("gfx94X-dcgpu")
        self._assert_relative_path(loc, "99999-linux/logs/gfx94X-dcgpu/index.html")

    def test_build_observability(self):
        loc = self.root.build_observability("gfx94X-dcgpu")
        self._assert_relative_path(
            loc, "99999-linux/logs/gfx94X-dcgpu/build_observability.html"
        )

    # -- Manifests --

    def test_manifest(self):
        loc = self.root.manifest("gfx94X-dcgpu")
        self._assert_relative_path(
            loc,
            "99999-linux/manifests/gfx94X-dcgpu/therock_manifest.json",
        )

    # -- Python packages --

    def test_python_packages(self):
        loc = self.root.python_packages("gfx110X-all")
        self._assert_relative_path(loc, "99999-linux/python/gfx110X-all")


class TestRunOutputRootLocationsExternalRepo(unittest.TestCase):
    """Verify external_repo prefix propagates through location methods."""

    def test_artifact_with_external_repo(self):
        root = RunOutputRoot(
            bucket="therock-ci-artifacts-external",
            external_repo="Fork-TheRock/",
            run_id="12345",
            platform="windows",
        )
        loc = root.artifact("blas_lib_gfx110X.tar.zst")
        self.assertEqual(
            loc.relative_path,
            "Fork-TheRock/12345-windows/blas_lib_gfx110X.tar.zst",
        )
        self.assertEqual(
            loc.s3_uri,
            "s3://therock-ci-artifacts-external/Fork-TheRock/12345-windows/blas_lib_gfx110X.tar.zst",
        )

    def test_log_dir_with_external_repo(self):
        root = RunOutputRoot(
            bucket="therock-ci-artifacts-external",
            external_repo="Fork-TheRock/",
            run_id="12345",
            platform="linux",
        )
        loc = root.log_dir("gfx94X-dcgpu")
        self.assertEqual(
            loc.relative_path,
            "Fork-TheRock/12345-linux/logs/gfx94X-dcgpu",
        )


# ---------------------------------------------------------------------------
# RunOutputRoot — end-to-end (s3_uri, https_url, local_path via OutputLocation)
# ---------------------------------------------------------------------------


class TestOutputLocationEndToEnd(unittest.TestCase):
    """Verify the full chain: RunOutputRoot → OutputLocation → final strings."""

    def setUp(self):
        self.root = RunOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="42",
            platform="linux",
        )

    def test_artifact_s3_uri(self):
        self.assertEqual(
            self.root.artifact("f.tar.xz").s3_uri,
            "s3://therock-ci-artifacts/42-linux/f.tar.xz",
        )

    def test_artifact_https_url(self):
        self.assertEqual(
            self.root.artifact("f.tar.xz").https_url,
            "https://therock-ci-artifacts.s3.amazonaws.com/42-linux/f.tar.xz",
        )

    def test_artifact_local_path(self):
        self.assertEqual(
            self.root.artifact("f.tar.xz").local_path(Path("/s")),
            Path("/s/42-linux/f.tar.xz"),
        )

    def test_manifest_s3_uri(self):
        self.assertEqual(
            self.root.manifest("gfx94X-dcgpu").s3_uri,
            "s3://therock-ci-artifacts/42-linux/manifests/gfx94X-dcgpu/therock_manifest.json",
        )

    def test_log_index_https_url(self):
        self.assertEqual(
            self.root.log_index("gfx94X-dcgpu").https_url,
            "https://therock-ci-artifacts.s3.amazonaws.com/42-linux/logs/gfx94X-dcgpu/index.html",
        )


# ---------------------------------------------------------------------------
# RunOutputRoot — factory methods
# ---------------------------------------------------------------------------


class TestRunOutputRootForLocal(unittest.TestCase):
    def test_defaults(self):
        root = RunOutputRoot.for_local()
        self.assertEqual(root.bucket, "local")
        self.assertEqual(root.external_repo, "")
        self.assertEqual(root.run_id, "local")
        # Platform depends on system, just check it's set
        self.assertIn(root.platform, ("linux", "windows", "darwin"))

    def test_custom_values(self):
        root = RunOutputRoot.for_local(
            run_id="test-42", platform="linux", bucket="test-bucket"
        )
        self.assertEqual(root.run_id, "test-42")
        self.assertEqual(root.platform, "linux")
        self.assertEqual(root.bucket, "test-bucket")
        self.assertEqual(root.prefix, "test-42-linux")


class TestRunOutputRootFromWorkflowRun(unittest.TestCase):
    """Test from_workflow_run() with mocked _retrieve_bucket_info."""

    @mock.patch("_therock_utils.run_outputs._retrieve_bucket_info")
    def test_basic(self, mock_retrieve):
        mock_retrieve.return_value = ("", "therock-ci-artifacts")
        root = RunOutputRoot.from_workflow_run(run_id="12345", platform="linux")
        self.assertEqual(root.bucket, "therock-ci-artifacts")
        self.assertEqual(root.external_repo, "")
        self.assertEqual(root.run_id, "12345")
        self.assertEqual(root.platform, "linux")
        mock_retrieve.assert_called_once_with(
            github_repository=None,
            workflow_run_id="12345",
            workflow_run=None,
        )

    @mock.patch("_therock_utils.run_outputs._retrieve_bucket_info")
    def test_with_explicit_repo(self, mock_retrieve):
        mock_retrieve.return_value = ("Fork-Repo/", "therock-ci-artifacts-external")
        root = RunOutputRoot.from_workflow_run(
            run_id="99999",
            platform="windows",
            github_repository="SomeUser/TheRock",
        )
        self.assertEqual(root.external_repo, "Fork-Repo/")
        self.assertEqual(root.bucket, "therock-ci-artifacts-external")
        mock_retrieve.assert_called_once_with(
            github_repository="SomeUser/TheRock",
            workflow_run_id="99999",
            workflow_run=None,
        )

    @mock.patch("_therock_utils.run_outputs._retrieve_bucket_info")
    def test_with_workflow_run_dict(self, mock_retrieve):
        mock_retrieve.return_value = ("", "therock-ci-artifacts")
        fake_run = {"id": 12345, "updated_at": "2026-01-01T00:00:00Z"}
        root = RunOutputRoot.from_workflow_run(
            run_id="12345",
            platform="linux",
            workflow_run=fake_run,
        )
        mock_retrieve.assert_called_once_with(
            github_repository=None,
            workflow_run_id="12345",
            workflow_run=fake_run,
        )


# ---------------------------------------------------------------------------
# _retrieve_bucket_info
# ---------------------------------------------------------------------------


class TestRetrieveBucketInfo(unittest.TestCase):
    """Test _retrieve_bucket_info with mocked environment."""

    def setUp(self):
        # Patch gha_query_workflow_run_by_id so we never make real API calls
        self.api_patcher = mock.patch(
            "_therock_utils.run_outputs.gha_query_workflow_run_by_id"
        )
        self.mock_api = self.api_patcher.start()

    def tearDown(self):
        self.api_patcher.stop()

    def _call(self, **kwargs):
        from _therock_utils.run_outputs import _retrieve_bucket_info

        return _retrieve_bucket_info(**kwargs)

    @mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "ROCm/TheRock"}, clear=False)
    def test_rocm_therock_default(self):
        external_repo, bucket = self._call()
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-ci-artifacts")

    @mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "ROCm/TheRock"}, clear=False)
    def test_rocm_therock_explicit(self):
        external_repo, bucket = self._call(github_repository="ROCm/TheRock")
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-ci-artifacts")

    @mock.patch.dict(
        os.environ,
        {"GITHUB_REPOSITORY": "SomeUser/TheRock", "IS_PR_FROM_FORK": "true"},
        clear=False,
    )
    def test_fork_pr(self):
        external_repo, bucket = self._call()
        self.assertEqual(external_repo, "SomeUser-TheRock/")
        self.assertEqual(bucket, "therock-ci-artifacts-external")

    @mock.patch.dict(
        os.environ,
        {"GITHUB_REPOSITORY": "ROCm/TheRock", "RELEASE_TYPE": "nightly"},
        clear=False,
    )
    def test_release_type_nightly(self):
        external_repo, bucket = self._call()
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-nightly-artifacts")

    @mock.patch.dict(
        os.environ,
        {"GITHUB_REPOSITORY": "ROCm/TheRock", "RELEASE_TYPE": "release"},
        clear=False,
    )
    def test_release_type_release(self):
        external_repo, bucket = self._call()
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-release-artifacts")

    def test_with_workflow_run_recent(self):
        """Recent workflow run should use therock-ci-artifacts."""
        fake_run = {
            "id": 12345,
            "updated_at": "2026-01-15T12:00:00Z",
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        external_repo, bucket = self._call(
            github_repository="ROCm/TheRock",
            workflow_run=fake_run,
        )
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-ci-artifacts")

    def test_with_workflow_run_old(self):
        """Old workflow run (before cutover) should use therock-artifacts."""
        fake_run = {
            "id": 99999,
            "updated_at": "2025-10-01T00:00:00Z",
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        external_repo, bucket = self._call(
            github_repository="ROCm/TheRock",
            workflow_run=fake_run,
        )
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-artifacts")

    def test_with_workflow_run_from_fork(self):
        """Workflow run from a fork should use external bucket.

        The external_repo prefix uses the base repo (github_repository), not
        the head repo.  When head != base, is_pr_from_fork is True.
        """
        fake_run = {
            "id": 12345,
            "updated_at": "2026-01-15T12:00:00Z",
            "head_repository": {"full_name": "SomeUser/TheRock"},
        }
        external_repo, bucket = self._call(
            github_repository="ROCm/TheRock",
            workflow_run=fake_run,
        )
        self.assertEqual(external_repo, "ROCm-TheRock/")
        self.assertEqual(bucket, "therock-ci-artifacts-external")

    def test_workflow_run_id_triggers_api_call(self):
        """When workflow_run_id is provided without workflow_run, API is called."""
        self.mock_api.return_value = {
            "id": 12345,
            "updated_at": "2026-01-15T12:00:00Z",
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        external_repo, bucket = self._call(
            github_repository="ROCm/TheRock",
            workflow_run_id="12345",
        )
        self.mock_api.assert_called_once_with("ROCm/TheRock", "12345")
        self.assertEqual(bucket, "therock-ci-artifacts")

    def test_internal_releases_repo(self):
        """therock-releases-internal should use therock-artifacts-internal."""
        external_repo, bucket = self._call(
            github_repository="ROCm/therock-releases-internal"
        )
        self.assertEqual(external_repo, "ROCm-therock-releases-internal/")
        self.assertEqual(bucket, "therock-artifacts-internal")


if __name__ == "__main__":
    unittest.main()
