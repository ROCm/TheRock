#!/usr/bin/env python
"""Unit tests for run_outputs.py (RunOutputRoot)."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.run_outputs import RunOutputRoot
from github_actions.github_actions_utils import is_authenticated_github_api_available


def _skip_unless_authenticated_github_api_is_available(test_func):
    """Decorator to skip tests unless GitHub API is available.

    Checks for GITHUB_TOKEN env var or authenticated gh CLI.
    """
    return unittest.skipUnless(
        is_authenticated_github_api_available(),
        "No authenticated GitHub API auth available (need GITHUB_TOKEN or authenticated gh CLI)",
    )(test_func)


class TestRunOutputRootProperties(unittest.TestCase):
    """Tests for RunOutputRoot basic properties."""

    def test_prefix_without_external_repo(self):
        """Test prefix for main ROCm/TheRock repo (no external_repo)."""
        root = RunOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="12345678901",
            platform="linux",
        )
        self.assertEqual(root.prefix, "12345678901-linux")

    def test_prefix_with_external_repo(self):
        """Test prefix for external/fork repos."""
        root = RunOutputRoot(
            bucket="therock-ci-artifacts-external",
            external_repo="ROCm-TheRock/",
            run_id="12345678901",
            platform="linux",
        )
        self.assertEqual(root.prefix, "ROCm-TheRock/12345678901-linux")

    def test_s3_uri(self):
        """Test S3 URI construction."""
        root = RunOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="12345678901",
            platform="linux",
        )
        self.assertEqual(root.s3_uri, "s3://therock-ci-artifacts/12345678901-linux")

    def test_s3_uri_with_external_repo(self):
        """Test S3 URI with external repo prefix."""
        root = RunOutputRoot(
            bucket="therock-ci-artifacts-external",
            external_repo="ROCm-TheRock/",
            run_id="12345678901",
            platform="windows",
        )
        self.assertEqual(
            root.s3_uri,
            "s3://therock-ci-artifacts-external/ROCm-TheRock/12345678901-windows",
        )

    def test_https_url(self):
        """Test HTTPS URL construction."""
        root = RunOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="12345678901",
            platform="linux",
        )
        self.assertEqual(
            root.https_url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345678901-linux",
        )

    def test_local_path(self):
        """Test local path construction."""
        root = RunOutputRoot(
            bucket="local",
            external_repo="",
            run_id="local-123",
            platform="linux",
        )
        staging_dir = Path("/tmp/staging")
        expected = staging_dir / "local-123-linux"
        self.assertEqual(root.local_path(staging_dir), expected)

    def test_local_path_with_external_repo(self):
        """Test local path with external repo prefix."""
        root = RunOutputRoot(
            bucket="local",
            external_repo="ROCm-TheRock/",
            run_id="12345",
            platform="windows",
        )
        staging_dir = Path("/tmp/staging")
        expected = staging_dir / "ROCm-TheRock" / "12345-windows"
        self.assertEqual(root.local_path(staging_dir), expected)


class TestRunOutputRootArtifactPaths(unittest.TestCase):
    """Tests for artifact-related path methods."""

    def setUp(self):
        self.root = RunOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="12345678901",
            platform="linux",
        )

    def test_artifact_s3_key(self):
        """Test S3 key for artifact file."""
        key = self.root.artifact_s3_key("blas_lib_gfx94X.tar.xz")
        self.assertEqual(key, "12345678901-linux/blas_lib_gfx94X.tar.xz")

    def test_artifact_s3_uri(self):
        """Test S3 URI for artifact file."""
        uri = self.root.artifact_s3_uri("blas_lib_gfx94X.tar.xz")
        self.assertEqual(
            uri, "s3://therock-ci-artifacts/12345678901-linux/blas_lib_gfx94X.tar.xz"
        )

    def test_artifact_https_url(self):
        """Test HTTPS URL for artifact file."""
        url = self.root.artifact_https_url("blas_lib_gfx94X.tar.xz")
        self.assertEqual(
            url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345678901-linux/blas_lib_gfx94X.tar.xz",
        )

    def test_artifact_index_s3_key(self):
        """Test S3 key for artifact index file."""
        key = self.root.artifact_index_s3_key("gfx94X-dcgpu")
        self.assertEqual(key, "12345678901-linux/index-gfx94X-dcgpu.html")

    def test_artifact_index_s3_uri(self):
        """Test S3 URI for artifact index file."""
        uri = self.root.artifact_index_s3_uri("gfx94X-dcgpu")
        self.assertEqual(
            uri, "s3://therock-ci-artifacts/12345678901-linux/index-gfx94X-dcgpu.html"
        )

    def test_artifact_index_url(self):
        """Test HTTPS URL for artifact index file."""
        url = self.root.artifact_index_url("gfx94X-dcgpu")
        self.assertEqual(
            url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345678901-linux/index-gfx94X-dcgpu.html",
        )


class TestRunOutputRootLogPaths(unittest.TestCase):
    """Tests for log-related path methods."""

    def setUp(self):
        self.root = RunOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="12345678901",
            platform="linux",
        )

    def test_logs_prefix(self):
        """Test S3 prefix for logs directory."""
        prefix = self.root.logs_prefix("gfx94X-dcgpu")
        self.assertEqual(prefix, "12345678901-linux/logs/gfx94X-dcgpu")

    def test_logs_s3_uri(self):
        """Test S3 URI for logs directory."""
        uri = self.root.logs_s3_uri("gfx94X-dcgpu")
        self.assertEqual(
            uri, "s3://therock-ci-artifacts/12345678901-linux/logs/gfx94X-dcgpu"
        )

    def test_logs_https_url(self):
        """Test HTTPS URL for logs directory."""
        url = self.root.logs_https_url("gfx94X-dcgpu")
        self.assertEqual(
            url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345678901-linux/logs/gfx94X-dcgpu",
        )

    def test_log_file_s3_key(self):
        """Test S3 key for a specific log file."""
        key = self.root.log_file_s3_key("gfx94X-dcgpu", "build.log")
        self.assertEqual(key, "12345678901-linux/logs/gfx94X-dcgpu/build.log")

    def test_log_index_url(self):
        """Test HTTPS URL for log index."""
        url = self.root.log_index_url("gfx94X-dcgpu")
        self.assertEqual(
            url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345678901-linux/logs/gfx94X-dcgpu/index.html",
        )

    def test_build_time_analysis_url(self):
        """Test HTTPS URL for build time analysis."""
        url = self.root.build_time_analysis_url("gfx94X-dcgpu")
        self.assertEqual(
            url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345678901-linux/logs/gfx94X-dcgpu/build_time_analysis.html",
        )


class TestRunOutputRootManifestPaths(unittest.TestCase):
    """Tests for manifest-related path methods."""

    def setUp(self):
        self.root = RunOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="12345678901",
            platform="linux",
        )

    def test_manifests_prefix(self):
        """Test S3 prefix for manifests directory."""
        prefix = self.root.manifests_prefix("gfx94X-dcgpu")
        self.assertEqual(prefix, "12345678901-linux/manifests/gfx94X-dcgpu")

    def test_manifest_s3_key(self):
        """Test S3 key for manifest file."""
        key = self.root.manifest_s3_key("gfx94X-dcgpu")
        self.assertEqual(
            key, "12345678901-linux/manifests/gfx94X-dcgpu/therock_manifest.json"
        )

    def test_manifest_s3_uri(self):
        """Test S3 URI for manifest file."""
        uri = self.root.manifest_s3_uri("gfx94X-dcgpu")
        self.assertEqual(
            uri,
            "s3://therock-ci-artifacts/12345678901-linux/manifests/gfx94X-dcgpu/therock_manifest.json",
        )

    def test_manifest_url(self):
        """Test HTTPS URL for manifest file."""
        url = self.root.manifest_url("gfx94X-dcgpu")
        self.assertEqual(
            url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345678901-linux/manifests/gfx94X-dcgpu/therock_manifest.json",
        )


class TestRunOutputRootFactoryMethods(unittest.TestCase):
    """Tests for RunOutputRoot factory methods."""

    def test_for_local_default_platform(self):
        """Test for_local with default platform detection."""
        root = RunOutputRoot.for_local(run_id="test-123")
        self.assertEqual(root.bucket, "local")
        self.assertEqual(root.external_repo, "")
        self.assertEqual(root.run_id, "test-123")
        # Platform should be detected from system
        self.assertIn(root.platform, ["linux", "windows", "darwin"])

    def test_for_local_explicit_platform(self):
        """Test for_local with explicit platform."""
        root = RunOutputRoot.for_local(run_id="test-456", platform="windows")
        self.assertEqual(root.run_id, "test-456")
        self.assertEqual(root.platform, "windows")
        self.assertEqual(root.prefix, "test-456-windows")

    def test_for_local_custom_bucket(self):
        """Test for_local with custom bucket name."""
        root = RunOutputRoot.for_local(
            run_id="test-789", platform="linux", bucket="custom-bucket"
        )
        self.assertEqual(root.bucket, "custom-bucket")

    def test_from_workflow_run_main_repo(self):
        """Test from_workflow_run for main ROCm/TheRock repo."""
        # Provide workflow_run dict to avoid API call - this is the standard
        # pattern since from_workflow_run always passes run_id to retrieve_bucket_info
        workflow_run = {
            "id": 12345678901,
            "head_repository": {"full_name": "ROCm/TheRock"},
            "updated_at": "2025-12-01T12:00:00Z",  # After bucket cutover date
            "status": "completed",
            "html_url": "https://github.com/ROCm/TheRock/actions/runs/12345678901",
        }

        root = RunOutputRoot.from_workflow_run(
            run_id="12345678901",
            platform="linux",
            github_repository="ROCm/TheRock",
            workflow_run=workflow_run,
        )

        self.assertEqual(root.bucket, "therock-ci-artifacts")
        self.assertEqual(root.external_repo, "")
        self.assertEqual(root.run_id, "12345678901")
        self.assertEqual(root.platform, "linux")
        self.assertEqual(root.prefix, "12345678901-linux")

    def test_from_workflow_run_external_repo(self):
        """Test from_workflow_run for external repo (non-TheRock)."""
        workflow_run = {
            "id": 12345678901,
            "head_repository": {"full_name": "SomeOrg/SomeRepo"},
            "updated_at": "2025-12-01T12:00:00Z",
            "status": "completed",
            "html_url": "https://github.com/SomeOrg/SomeRepo/actions/runs/12345678901",
        }

        root = RunOutputRoot.from_workflow_run(
            run_id="12345678901",
            platform="windows",
            github_repository="SomeOrg/SomeRepo",
            workflow_run=workflow_run,
        )

        self.assertEqual(root.bucket, "therock-ci-artifacts-external")
        self.assertEqual(root.external_repo, "SomeOrg-SomeRepo/")
        self.assertEqual(root.prefix, "SomeOrg-SomeRepo/12345678901-windows")

    def test_from_workflow_run_fork(self):
        """Test from_workflow_run for fork PR (head_repo != base_repo)."""
        # Workflow from fork - head_repository differs from github_repository
        workflow_run = {
            "id": 12345678901,
            "head_repository": {"full_name": "SomeUser/TheRock"},  # Fork
            "updated_at": "2025-12-01T12:00:00Z",
            "status": "completed",
            "html_url": "https://github.com/ROCm/TheRock/actions/runs/12345678901",
        }

        root = RunOutputRoot.from_workflow_run(
            run_id="12345678901",
            platform="linux",
            github_repository="ROCm/TheRock",
            workflow_run=workflow_run,
        )

        self.assertEqual(root.bucket, "therock-ci-artifacts-external")
        self.assertEqual(root.external_repo, "ROCm-TheRock/")
        self.assertEqual(root.prefix, "ROCm-TheRock/12345678901-linux")

    def test_from_workflow_run_old_bucket(self):
        """Test from_workflow_run with old workflow date uses legacy bucket."""
        workflow_run = {
            "id": 12345678901,
            "head_repository": {"full_name": "ROCm/TheRock"},
            "updated_at": "2025-10-01T12:00:00Z",  # Before bucket cutover date
            "status": "completed",
            "html_url": "https://github.com/ROCm/TheRock/actions/runs/12345678901",
        }

        root = RunOutputRoot.from_workflow_run(
            run_id="12345678901",
            platform="linux",
            github_repository="ROCm/TheRock",
            workflow_run=workflow_run,
        )

        # Old runs use legacy bucket
        self.assertEqual(root.bucket, "therock-artifacts")
        self.assertEqual(root.external_repo, "")


class TestRunOutputRootImmutability(unittest.TestCase):
    """Tests that RunOutputRoot is immutable (frozen dataclass)."""

    def test_cannot_modify_bucket(self):
        """Test that bucket cannot be modified after creation."""
        root = RunOutputRoot(
            bucket="original",
            external_repo="",
            run_id="123",
            platform="linux",
        )
        with self.assertRaises(AttributeError):
            root.bucket = "modified"

    def test_cannot_modify_run_id(self):
        """Test that run_id cannot be modified after creation."""
        root = RunOutputRoot(
            bucket="bucket",
            external_repo="",
            run_id="original",
            platform="linux",
        )
        with self.assertRaises(AttributeError):
            root.run_id = "modified"


class TestRunOutputRootReleaseType(unittest.TestCase):
    """Tests for RELEASE_TYPE environment variable handling."""

    def setUp(self):
        """Save and clean environment state."""
        self._saved_env = {}
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY", "IS_PR_FROM_FORK"]:
            if key in os.environ:
                self._saved_env[key] = os.environ[key]
                del os.environ[key]

    def tearDown(self):
        """Restore environment state."""
        for key in ["RELEASE_TYPE", "GITHUB_REPOSITORY", "IS_PR_FROM_FORK"]:
            if key in os.environ:
                del os.environ[key]
        for key, value in self._saved_env.items():
            os.environ[key] = value

    def test_nightly_release_bucket(self):
        """Test that RELEASE_TYPE=nightly uses nightly bucket."""
        os.environ["RELEASE_TYPE"] = "nightly"
        workflow_run = {
            "id": 12345678901,
            "head_repository": {"full_name": "ROCm/TheRock"},
            "updated_at": "2025-12-01T12:00:00Z",
            "status": "completed",
            "html_url": "https://github.com/ROCm/TheRock/actions/runs/12345678901",
        }

        root = RunOutputRoot.from_workflow_run(
            run_id="12345678901",
            platform="linux",
            github_repository="ROCm/TheRock",
            workflow_run=workflow_run,
        )

        self.assertEqual(root.bucket, "therock-nightly-artifacts")
        self.assertEqual(root.external_repo, "")

    def test_release_bucket(self):
        """Test that RELEASE_TYPE=release uses release bucket."""
        os.environ["RELEASE_TYPE"] = "release"
        workflow_run = {
            "id": 12345678901,
            "head_repository": {"full_name": "ROCm/TheRock"},
            "updated_at": "2025-12-01T12:00:00Z",
            "status": "completed",
            "html_url": "https://github.com/ROCm/TheRock/actions/runs/12345678901",
        }

        root = RunOutputRoot.from_workflow_run(
            run_id="12345678901",
            platform="linux",
            github_repository="ROCm/TheRock",
            workflow_run=workflow_run,
        )

        self.assertEqual(root.bucket, "therock-release-artifacts")
        self.assertEqual(root.external_repo, "")


class TestRunOutputRootIntegration(unittest.TestCase):
    """Integration tests that use real GitHub API calls.

    These tests require authenticated GitHub API access (GITHUB_TOKEN or gh CLI)
    and make network requests. They verify that from_workflow_run() correctly
    determines bucket info for known historical workflow runs.
    """

    @_skip_unless_authenticated_github_api_is_available
    def test_from_workflow_run_older_bucket(self):
        """Test workflow run from before bucket cutover uses legacy bucket."""
        # https://github.com/ROCm/TheRock/actions/runs/18022609292?pr=1597
        root = RunOutputRoot.from_workflow_run(
            run_id="18022609292",
            platform="linux",
            github_repository="ROCm/TheRock",
        )
        self.assertEqual(root.external_repo, "")
        self.assertEqual(root.bucket, "therock-artifacts")

    @_skip_unless_authenticated_github_api_is_available
    def test_from_workflow_run_newer_bucket(self):
        """Test workflow run from after bucket cutover uses new bucket."""
        # https://github.com/ROCm/TheRock/actions/runs/19680190301
        root = RunOutputRoot.from_workflow_run(
            run_id="19680190301",
            platform="linux",
            github_repository="ROCm/TheRock",
        )
        self.assertEqual(root.external_repo, "")
        self.assertEqual(root.bucket, "therock-ci-artifacts")

    @_skip_unless_authenticated_github_api_is_available
    def test_from_workflow_run_fork(self):
        """Test workflow run from fork uses external bucket with repo prefix."""
        # https://github.com/ROCm/TheRock/actions/runs/18023442478?pr=1596
        root = RunOutputRoot.from_workflow_run(
            run_id="18023442478",
            platform="linux",
            github_repository="ROCm/TheRock",
        )
        self.assertEqual(root.external_repo, "ROCm-TheRock/")
        self.assertEqual(root.bucket, "therock-artifacts-external")

    @_skip_unless_authenticated_github_api_is_available
    def test_from_workflow_run_external_repo_older(self):
        """Test workflow run from external repo (rocm-libraries) older bucket."""
        # https://github.com/ROCm/rocm-libraries/actions/runs/18020401326?pr=1828
        root = RunOutputRoot.from_workflow_run(
            run_id="18020401326",
            platform="linux",
            github_repository="ROCm/rocm-libraries",
        )
        self.assertEqual(root.external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(root.bucket, "therock-artifacts-external")

    @_skip_unless_authenticated_github_api_is_available
    def test_from_workflow_run_external_repo_newer(self):
        """Test workflow run from external repo (rocm-libraries) newer bucket."""
        # https://github.com/ROCm/rocm-libraries/actions/runs/19784318631
        root = RunOutputRoot.from_workflow_run(
            run_id="19784318631",
            platform="linux",
            github_repository="ROCm/rocm-libraries",
        )
        self.assertEqual(root.external_repo, "ROCm-rocm-libraries/")
        self.assertEqual(root.bucket, "therock-ci-artifacts-external")


if __name__ == "__main__":
    unittest.main()
