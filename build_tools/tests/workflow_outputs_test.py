#!/usr/bin/env python
"""Unit tests for workflow_outputs.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils import s3_buckets, workflow_outputs
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_location import StorageLocation
from _therock_utils.s3_buckets import (
    CdnRule,
    S3BucketConfig,
    require_bucket_config,
)


# ---------------------------------------------------------------------------
# StorageLocation
# ---------------------------------------------------------------------------


class TestStorageLocation(unittest.TestCase):
    def test_s3_uri(self):
        loc = StorageLocation("my-bucket", "12345-linux/file.tar.xz")
        self.assertEqual(loc.s3_uri, "s3://my-bucket/12345-linux/file.tar.xz")

    def test_https_url(self):
        loc = StorageLocation("my-bucket", "12345-linux/file.tar.xz")
        self.assertEqual(
            loc.https_url,
            "https://my-bucket.s3.amazonaws.com/12345-linux/file.tar.xz",
        )

    def test_local_path(self):
        loc = StorageLocation("my-bucket", "12345-linux/logs/group/build.log")
        result = loc.local_path(Path("/tmp/staging"))
        expected = Path("/tmp/staging/12345-linux/logs/group/build.log")
        self.assertEqual(result, expected)

    def test_frozen(self):
        loc = StorageLocation("bucket", "path")
        with self.assertRaises(AttributeError):
            loc.bucket = "other"


# ---------------------------------------------------------------------------
# WorkflowOutputRoot — prefix
# ---------------------------------------------------------------------------


class TestWorkflowOutputRootPrefix(unittest.TestCase):
    def _make_root(self, **kwargs):
        defaults = dict(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="12345",
            platform="linux",
        )
        defaults.update(kwargs)
        return WorkflowOutputRoot(**defaults)

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

    # -- key_prefix --

    def test_prefix_defaults_to_no_key_prefix(self):
        """Omitting key_prefix must reproduce the pre-existing layout exactly."""
        self.assertEqual(self._make_root().key_prefix, "")
        self.assertEqual(self._make_root().prefix, "12345-linux")

    def test_prefix_with_key_prefix(self):
        root = self._make_root(key_prefix="v3/artifacts/")
        self.assertEqual(root.prefix, "v3/artifacts/12345-linux")

    def test_prefix_with_key_prefix_and_external_repo(self):
        """key_prefix is the bucket's layout; external_repo namespaces within it."""
        root = self._make_root(
            key_prefix="v3/artifacts/", external_repo="Fork-TheRock/"
        )
        self.assertEqual(root.prefix, "v3/artifacts/Fork-TheRock/12345-linux")

    def test_key_prefix_reaches_location_methods(self):
        """relative_path must be the full S3 key, not a key missing its prefix."""
        root = self._make_root(key_prefix="v3/artifacts/")
        self.assertEqual(
            root.artifact("blas_lib_gfx94X.tar.xz").relative_path,
            "v3/artifacts/12345-linux/blas_lib_gfx94X.tar.xz",
        )
        self.assertEqual(
            root.log_file("gfx94X-dcgpu", "build.log").relative_path,
            "v3/artifacts/12345-linux/logs/gfx94X-dcgpu/build.log",
        )
        self.assertEqual(
            root.artifact_index().s3_uri,
            "s3://therock-ci-artifacts/v3/artifacts/12345-linux/index.html",
        )

    def test_key_prefix_local_path_uses_posix_separators_on_all_platforms(self):
        """A slashed key_prefix must not break local staging paths on Windows."""
        root = self._make_root(key_prefix="v3/artifacts/")
        result = root.artifact("a.tar.xz").local_path(Path("/tmp/staging"))
        self.assertEqual(result, Path("/tmp/staging/v3/artifacts/12345-linux/a.tar.xz"))


# ---------------------------------------------------------------------------
# WorkflowOutputRoot — location methods
# ---------------------------------------------------------------------------


class TestWorkflowOutputRootLocations(unittest.TestCase):
    """Test that each location method returns correct relative paths."""

    def setUp(self):
        self.root = WorkflowOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="99999",
            platform="linux",
        )

    def _assert_relative_path(self, loc: StorageLocation, expected_path: str):
        self.assertIsInstance(loc, StorageLocation)
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
        loc = self.root.artifact_index()
        self._assert_relative_path(loc, "99999-linux/index.html")

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

    # -- Stage logs (multi-arch CI) --

    def test_log_stage_dir_per_arch(self):
        loc = self.root.log_stage_dir("math-libs", "gfx1151")
        self._assert_relative_path(loc, "99999-linux/logs/math-libs/gfx1151")

    def test_log_stage_dir_generic(self):
        loc = self.root.log_stage_dir("foundation")
        self._assert_relative_path(loc, "99999-linux/logs/foundation")

    def test_log_stage_dir_generic_empty_string(self):
        loc = self.root.log_stage_dir("compiler-runtime", "")
        self._assert_relative_path(loc, "99999-linux/logs/compiler-runtime")

    # -- Manifests --

    def test_manifest_dir(self):
        loc = self.root.manifest_dir("gfx94X-dcgpu")
        self._assert_relative_path(loc, "99999-linux/manifests/gfx94X-dcgpu")

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

    def test_python_packages_no_artifact_group(self):
        loc = self.root.python_packages()
        self._assert_relative_path(loc, "99999-linux/python")


class TestWorkflowOutputRootLocationsExternalRepo(unittest.TestCase):
    """Verify external_repo prefix propagates through location methods."""

    def test_artifact_with_external_repo(self):
        root = WorkflowOutputRoot(
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
        root = WorkflowOutputRoot(
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
# WorkflowOutputRoot — end-to-end (s3_uri, https_url, local_path via StorageLocation)
# ---------------------------------------------------------------------------


class TestStorageLocationEndToEnd(unittest.TestCase):
    """Verify the full chain: WorkflowOutputRoot → StorageLocation → final strings."""

    def setUp(self):
        self.root = WorkflowOutputRoot(
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
# WorkflowOutputRoot — factory methods
# ---------------------------------------------------------------------------


class TestWorkflowOutputRootForLocal(unittest.TestCase):
    def test_defaults(self):
        root = WorkflowOutputRoot.for_local()
        self.assertEqual(root.bucket, "local")
        self.assertEqual(root.external_repo, "")
        self.assertEqual(root.run_id, "local")
        # Platform depends on system, just check it's set
        self.assertIn(root.platform, ("linux", "windows", "darwin"))

    def test_custom_values(self):
        root = WorkflowOutputRoot.for_local(
            run_id="test-42", platform="linux", bucket="test-bucket"
        )
        self.assertEqual(root.run_id, "test-42")
        self.assertEqual(root.platform, "linux")
        self.assertEqual(root.bucket, "test-bucket")
        self.assertEqual(root.prefix, "test-42-linux")


class TestWorkflowOutputRootFromWorkflowRun(unittest.TestCase):
    """Test from_workflow_run() with mocked _retrieve_bucket_info."""

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_basic_does_not_trigger_api(self, mock_retrieve):
        """By default, run_id is NOT passed as workflow_run_id."""
        mock_retrieve.return_value = ("", require_bucket_config("therock-ci-artifacts"))
        root = WorkflowOutputRoot.from_workflow_run(run_id="12345", platform="linux")
        self.assertEqual(root.bucket, "therock-ci-artifacts")
        self.assertEqual(root.external_repo, "")
        self.assertEqual(root.run_id, "12345")
        self.assertEqual(root.platform, "linux")
        mock_retrieve.assert_called_once_with(
            github_repository=None,
            workflow_run_id=None,
            workflow_run=None,
            release_type=None,
        )

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_lookup_workflow_run_triggers_api(self, mock_retrieve):
        """With lookup_workflow_run=True, run_id IS passed as workflow_run_id."""
        mock_retrieve.return_value = (
            "Fork-Repo/",
            require_bucket_config("therock-ci-artifacts-external"),
        )
        root = WorkflowOutputRoot.from_workflow_run(
            run_id="99999",
            platform="windows",
            github_repository="SomeUser/TheRock",
            lookup_workflow_run=True,
        )
        self.assertEqual(root.external_repo, "Fork-Repo/")
        self.assertEqual(root.bucket, "therock-ci-artifacts-external")
        mock_retrieve.assert_called_once_with(
            github_repository="SomeUser/TheRock",
            workflow_run_id="99999",
            workflow_run=None,
            release_type=None,
        )

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_with_workflow_run_dict(self, mock_retrieve):
        """When workflow_run is provided, it's passed through (no API call)."""
        mock_retrieve.return_value = ("", require_bucket_config("therock-ci-artifacts"))
        fake_run = {"id": 12345}
        root = WorkflowOutputRoot.from_workflow_run(
            run_id="12345",
            platform="linux",
            workflow_run=fake_run,
        )
        mock_retrieve.assert_called_once_with(
            github_repository=None,
            workflow_run_id=None,
            workflow_run=fake_run,
            release_type=None,
        )

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_lookup_ignored_when_workflow_run_provided(self, mock_retrieve):
        """lookup_workflow_run is irrelevant when workflow_run is provided."""
        mock_retrieve.return_value = ("", require_bucket_config("therock-ci-artifacts"))
        fake_run = {"id": 12345}
        root = WorkflowOutputRoot.from_workflow_run(
            run_id="12345",
            platform="linux",
            workflow_run=fake_run,
            lookup_workflow_run=True,
        )
        # workflow_run_id is still None because workflow_run was provided
        # directly — no API lookup needed.
        mock_retrieve.assert_called_once_with(
            github_repository=None,
            workflow_run_id=None,
            workflow_run=fake_run,
            release_type=None,
        )


class TestKeyPrefixAndCdnRoundTrip(unittest.TestCase):
    """key_prefix and cdn_rules are independent, and compose correctly.

    This is the downstream (rocm-npi-dev) shape: a bucket whose objects all live
    under a versioned key prefix, fronted by a CDN that serves that prefix at its
    root. The prefix must appear in the S3 key and be stripped from the CDN URL.
    """

    KEY_PREFIX = "v3/artifacts/"
    CDN = "https://artifacts.example.com/"

    def _register(self, **overrides):
        config = S3BucketConfig(
            name="downstream-artifacts",
            key_prefix=self.KEY_PREFIX,
            cdn_rules=(CdnRule(self.KEY_PREFIX, self.CDN),),
            **overrides,
        )
        patcher = mock.patch.object(
            s3_buckets,
            "s3_bucket_configs",
            list(s3_buckets.s3_bucket_configs) + [config],
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        s3_buckets.reset_bucket_registry()
        self.addCleanup(s3_buckets.reset_bucket_registry)
        return config

    def test_prefix_in_s3_key_stripped_in_cdn_url(self):
        config = self._register()
        root = WorkflowOutputRoot(
            bucket=config.name,
            external_repo="",
            run_id="12345",
            platform="linux",
            key_prefix=config.key_prefix,
        )
        loc = root.artifact("blas_lib_gfx94X.tar.xz")

        # The prefix is part of the S3 key...
        self.assertEqual(
            loc.s3_uri,
            "s3://downstream-artifacts/v3/artifacts/12345-linux/blas_lib_gfx94X.tar.xz",
        )
        self.assertEqual(
            loc.https_url,
            "https://downstream-artifacts.s3.amazonaws.com/"
            "v3/artifacts/12345-linux/blas_lib_gfx94X.tar.xz",
        )
        # ...and is stripped by the CDN rule, which serves it at the root.
        self.assertEqual(
            loc.public_url,
            "https://artifacts.example.com/12345-linux/blas_lib_gfx94X.tar.xz",
        )

    def test_independent_of_each_other(self):
        """A bucket may have cdn_rules with no key_prefix, and vice versa.

        TheRock's own release buckets are the first case; deriving one field from
        the other would break them.
        """
        release = require_bucket_config("therock-nightly-python")
        self.assertEqual(release.key_prefix, "")
        self.assertTrue(release.cdn_rules)

        artifacts = require_bucket_config("therock-ci-artifacts")
        self.assertEqual(artifacts.key_prefix, "")
        self.assertEqual(artifacts.cdn_rules, ())


class TestNamespaceExternalRepos(unittest.TestCase):
    """external_repo is driven by the bucket config flag, not a literal name."""

    def _retrieve(self, config, github_repository):
        with mock.patch.object(
            workflow_outputs,
            "get_artifacts_bucket_config_for_workflow_run",
            return_value=config,
        ):
            return workflow_outputs._retrieve_bucket_info(
                github_repository=github_repository
            )

    def test_shared_external_bucket_namespaces(self):
        config = require_bucket_config("therock-ci-artifacts-external")
        self.assertTrue(config.namespace_external_repos)
        external_repo, returned = self._retrieve(config, "SomeUser/TheRock")
        self.assertEqual(external_repo, "SomeUser-TheRock/")
        self.assertIs(returned, config)

    def test_dedicated_bucket_does_not_namespace(self):
        config = require_bucket_config("therock-ci-artifacts")
        self.assertFalse(config.namespace_external_repos)
        external_repo, returned = self._retrieve(config, "ROCm/TheRock")
        self.assertEqual(external_repo, "")
        self.assertIs(returned, config)

    def test_returns_config_not_name(self):
        """The whole config is returned so from_workflow_run can read key_prefix."""
        config = S3BucketConfig(name="downstream", key_prefix="v3/artifacts/")
        _, returned = self._retrieve(config, "ROCm/TheRock")
        self.assertIsInstance(returned, S3BucketConfig)
        self.assertEqual(returned.key_prefix, "v3/artifacts/")


if __name__ == "__main__":
    unittest.main()
