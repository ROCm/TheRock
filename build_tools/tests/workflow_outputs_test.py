#!/usr/bin/env python
"""Unit tests for workflow_outputs.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_location import StorageConfig, StorageLocation


# ---------------------------------------------------------------------------
# StorageConfig
# ---------------------------------------------------------------------------


class TestStorageConfig(unittest.TestCase):
    """Test StorageConfig dataclass."""

    def test_defaults(self):
        """Test default values are set correctly."""
        config = StorageConfig()
        self.assertEqual(config.s3_url_schema, "s3://{bucket}/{path}")
        self.assertEqual(
            config.https_url_schema, "https://{bucket}.s3.amazonaws.com/{path}"
        )
        self.assertEqual(config.bucket_schema, "therock-{release_type}-artifacts")

    def test_custom_values(self):
        """Test custom values are stored."""
        config = StorageConfig(
            s3_url_schema="custom://{bucket}/{path}",
            https_url_schema="https://cdn.example.com/{path}",
            bucket_schema="my-{release_type}-bucket",
        )
        self.assertEqual(config.s3_url_schema, "custom://{bucket}/{path}")
        self.assertEqual(config.https_url_schema, "https://cdn.example.com/{path}")
        self.assertEqual(config.bucket_schema, "my-{release_type}-bucket")

    def test_frozen(self):
        """Test that StorageConfig is immutable."""
        config = StorageConfig()
        with self.assertRaises(AttributeError):
            config.s3_url_schema = "other"

    def test_validation_s3_url_schema_invalid_placeholder(self):
        """Test validation rejects unknown placeholders in s3_url_schema."""
        with self.assertRaises(ValueError) as cm:
            StorageConfig(s3_url_schema="s3://{bucket}.{region}/{path}")
        self.assertIn("s3_url_schema", str(cm.exception))
        self.assertIn("region", str(cm.exception))

    def test_validation_https_url_schema_invalid_placeholder(self):
        """Test validation rejects unknown placeholders in https_url_schema."""
        with self.assertRaises(ValueError) as cm:
            StorageConfig(https_url_schema="https://{bucket}.{zone}.example.com/{path}")
        self.assertIn("https_url_schema", str(cm.exception))
        self.assertIn("zone", str(cm.exception))

    def test_validation_bucket_schema_invalid_placeholder(self):
        """Test validation rejects unknown placeholders in bucket_schema."""
        with self.assertRaises(ValueError) as cm:
            StorageConfig(bucket_schema="my-{env}-{release_type}-bucket")
        self.assertIn("bucket_schema", str(cm.exception))
        self.assertIn("env", str(cm.exception))

    def test_validation_allows_subset_of_placeholders(self):
        """Test that schemas can omit allowed placeholders."""
        # CDN URL without {bucket}
        config = StorageConfig(https_url_schema="https://cdn.example.com/{path}")
        self.assertEqual(config.https_url_schema, "https://cdn.example.com/{path}")

    def test_from_json_valid(self):
        """Test from_json with valid JSON."""
        config = StorageConfig.from_json(
            '{"https_url_schema": "https://cdn.example.com/{path}"}'
        )
        self.assertEqual(config.https_url_schema, "https://cdn.example.com/{path}")
        # Other fields should have defaults
        self.assertEqual(config.s3_url_schema, "s3://{bucket}/{path}")

    def test_from_json_all_fields(self):
        """Test from_json with all fields."""
        config = StorageConfig.from_json(
            '{"s3_url_schema": "custom://{bucket}/{path}", '
            '"https_url_schema": "https://cdn/{path}", '
            '"bucket_schema": "my-{release_type}"}'
        )
        self.assertEqual(config.s3_url_schema, "custom://{bucket}/{path}")
        self.assertEqual(config.https_url_schema, "https://cdn/{path}")
        self.assertEqual(config.bucket_schema, "my-{release_type}")

    def test_from_json_empty_object(self):
        """Test from_json with empty object returns defaults."""
        config = StorageConfig.from_json("{}")
        self.assertEqual(config.s3_url_schema, "s3://{bucket}/{path}")

    def test_from_json_unknown_key(self):
        """Test from_json rejects unknown keys."""
        with self.assertRaises(ValueError) as cm:
            StorageConfig.from_json('{"unknown_field": "value"}')
        self.assertIn("unknown_field", str(cm.exception))

    def test_from_json_invalid_json(self):
        """Test from_json rejects invalid JSON syntax."""
        import json

        with self.assertRaises(json.JSONDecodeError):
            StorageConfig.from_json("not valid json")

    def test_from_json_non_object(self):
        """Test from_json rejects non-object JSON."""
        with self.assertRaises(ValueError) as cm:
            StorageConfig.from_json('"just a string"')
        self.assertIn("must be a JSON object", str(cm.exception))


# ---------------------------------------------------------------------------
# StorageLocation
# ---------------------------------------------------------------------------


class TestStorageLocation(unittest.TestCase):
    def test_s3_uri(self):
        loc = StorageLocation("my-bucket", "12345-linux/file.tar.xz")
        self.assertEqual(loc.s3_uri, "s3://my-bucket/12345-linux/file.tar.xz")

    def test_s3_uri_custom_schema(self):
        config = StorageConfig(s3_url_schema="custom://{bucket}/prefix/{path}")
        loc = StorageLocation("my-bucket", "12345-linux/file.tar.xz", config)
        self.assertEqual(
            loc.s3_uri, "custom://my-bucket/prefix/12345-linux/file.tar.xz"
        )

    def test_https_url(self):
        loc = StorageLocation("my-bucket", "12345-linux/file.tar.xz")
        self.assertEqual(
            loc.https_url,
            "https://my-bucket.s3.amazonaws.com/12345-linux/file.tar.xz",
        )

    def test_https_url_custom_schema(self):
        config = StorageConfig(
            https_url_schema="https://cdn.example.com/{bucket}/{path}"
        )
        loc = StorageLocation("my-bucket", "12345-linux/file.tar.xz", config)
        self.assertEqual(
            loc.https_url, "https://cdn.example.com/my-bucket/12345-linux/file.tar.xz"
        )

    def test_https_url_default_schema(self):
        """Test https_url uses default S3 pattern when no custom schema provided."""
        loc = StorageLocation("therock-ci-artifacts", "12345-linux/file.tar.xz")
        self.assertEqual(
            loc.https_url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/file.tar.xz",
        )

    def test_local_path(self):
        loc = StorageLocation("my-bucket", "12345-linux/logs/group/build.log")
        result = loc.local_path(Path("/tmp/staging"))
        expected = Path("/tmp/staging/12345-linux/logs/group/build.log")
        self.assertEqual(result, expected)

    def test_s3_uri_and_https_url_with_both_schemas(self):
        """Test that both custom schemas work together."""
        config = StorageConfig(
            s3_url_schema="custom-s3://{bucket}/prefix/{path}",
            https_url_schema="https://cdn.example.com/{bucket}/{path}",
        )
        loc = StorageLocation("my-bucket", "12345-linux/file.tar.xz", config)
        self.assertEqual(
            loc.s3_uri, "custom-s3://my-bucket/prefix/12345-linux/file.tar.xz"
        )
        self.assertEqual(
            loc.https_url, "https://cdn.example.com/my-bucket/12345-linux/file.tar.xz"
        )

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


class TestWorkflowOutputRootCustomSchemas(unittest.TestCase):
    """Test that custom URL schemas propagate through to StorageLocation."""

    def test_custom_s3_schema_propagates(self):
        config = StorageConfig(s3_url_schema="custom-s3://{bucket}/prefix/{path}")
        root = WorkflowOutputRoot(
            bucket="my-bucket",
            external_repo="",
            run_id="12345",
            platform="linux",
            storage_config=config,
        )
        loc = root.artifact("test.tar.xz")
        self.assertEqual(
            loc.s3_uri, "custom-s3://my-bucket/prefix/12345-linux/test.tar.xz"
        )

    def test_custom_https_schema_propagates(self):
        config = StorageConfig(
            https_url_schema="https://cdn.example.com/{bucket}/{path}"
        )
        root = WorkflowOutputRoot(
            bucket="my-bucket",
            external_repo="",
            run_id="12345",
            platform="linux",
            storage_config=config,
        )
        loc = root.artifact("test.tar.xz")
        self.assertEqual(
            loc.https_url, "https://cdn.example.com/my-bucket/12345-linux/test.tar.xz"
        )

    def test_both_schemas_propagate(self):
        config = StorageConfig(
            s3_url_schema="custom-s3://{bucket}/data/{path}",
            https_url_schema="https://custom.example.com/{bucket}/{path}",
        )
        root = WorkflowOutputRoot(
            bucket="my-bucket",
            external_repo="",
            run_id="12345",
            platform="linux",
            storage_config=config,
        )
        loc = root.log_file("gfx94X-dcgpu", "build.log")
        self.assertEqual(
            loc.s3_uri,
            "custom-s3://my-bucket/data/12345-linux/logs/gfx94X-dcgpu/build.log",
        )
        self.assertEqual(
            loc.https_url,
            "https://custom.example.com/my-bucket/12345-linux/logs/gfx94X-dcgpu/build.log",
        )

    def test_default_schemas_when_no_config(self):
        """When storage_config uses defaults, StorageLocation should use defaults."""
        root = WorkflowOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="12345",
            platform="linux",
            # storage_config omitted - uses default StorageConfig()
        )
        loc = root.artifact("test.tar.xz")
        # StorageLocation should apply default schemas
        self.assertEqual(
            loc.s3_uri, "s3://therock-ci-artifacts/12345-linux/test.tar.xz"
        )
        self.assertEqual(
            loc.https_url,
            "https://therock-ci-artifacts.s3.amazonaws.com/12345-linux/test.tar.xz",
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
        # storage_config should use defaults
        self.assertEqual(root.storage_config, StorageConfig())

    def test_custom_values(self):
        root = WorkflowOutputRoot.for_local(
            run_id="test-42", platform="linux", bucket="test-bucket"
        )
        self.assertEqual(root.run_id, "test-42")
        self.assertEqual(root.platform, "linux")
        self.assertEqual(root.bucket, "test-bucket")
        self.assertEqual(root.prefix, "test-42-linux")

    def test_custom_schemas(self):
        config = StorageConfig(
            https_url_schema="https://custom.example.com/{bucket}/{path}"
        )
        root = WorkflowOutputRoot.for_local(storage_config=config)
        loc = root.artifact("test.tar.xz")
        self.assertEqual(
            loc.https_url,
            f"https://custom.example.com/local/local-{root.platform}/test.tar.xz",
        )


class TestWorkflowOutputRootFromWorkflowRun(unittest.TestCase):
    """Test from_workflow_run() with mocked _retrieve_bucket_info."""

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_basic_does_not_trigger_api(self, mock_retrieve):
        """By default, run_id is NOT passed as workflow_run_id."""
        mock_retrieve.return_value = ("", "therock-ci-artifacts")
        root = WorkflowOutputRoot.from_workflow_run(run_id="12345", platform="linux")
        self.assertEqual(root.bucket, "therock-ci-artifacts")
        self.assertEqual(root.external_repo, "")
        self.assertEqual(root.run_id, "12345")
        self.assertEqual(root.platform, "linux")
        # storage_config should use defaults
        self.assertEqual(root.storage_config, StorageConfig())
        mock_retrieve.assert_called_once_with(
            github_repository=None,
            workflow_run_id=None,
            workflow_run=None,
            storage_config=StorageConfig(),
        )

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_custom_schemas(self, mock_retrieve):
        """StorageConfig is passed through to WorkflowOutputRoot."""
        mock_retrieve.return_value = ("", "therock-ci-artifacts")
        config = StorageConfig(
            https_url_schema="https://cdn.example.com/{bucket}/{path}"
        )
        root = WorkflowOutputRoot.from_workflow_run(
            run_id="12345",
            platform="linux",
            storage_config=config,
        )
        self.assertEqual(
            root.storage_config.https_url_schema,
            "https://cdn.example.com/{bucket}/{path}",
        )
        loc = root.artifact("test.tar.xz")
        self.assertEqual(
            loc.https_url,
            "https://cdn.example.com/therock-ci-artifacts/12345-linux/test.tar.xz",
        )

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_lookup_workflow_run_triggers_api(self, mock_retrieve):
        """With lookup_workflow_run=True, run_id IS passed as workflow_run_id."""
        mock_retrieve.return_value = ("Fork-Repo/", "therock-ci-artifacts-external")
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
            storage_config=StorageConfig(),
        )

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_with_workflow_run_dict(self, mock_retrieve):
        """When workflow_run is provided, it's passed through (no API call)."""
        mock_retrieve.return_value = ("", "therock-ci-artifacts")
        fake_run = {"id": 12345, "updated_at": "2026-01-01T00:00:00Z"}
        root = WorkflowOutputRoot.from_workflow_run(
            run_id="12345",
            platform="linux",
            workflow_run=fake_run,
        )
        mock_retrieve.assert_called_once_with(
            github_repository=None,
            workflow_run_id=None,
            workflow_run=fake_run,
            storage_config=StorageConfig(),
        )

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_lookup_ignored_when_workflow_run_provided(self, mock_retrieve):
        """lookup_workflow_run is irrelevant when workflow_run is provided."""
        mock_retrieve.return_value = ("", "therock-ci-artifacts")
        fake_run = {"id": 12345, "updated_at": "2026-01-01T00:00:00Z"}
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
            storage_config=StorageConfig(),
        )


# ---------------------------------------------------------------------------
# _retrieve_bucket_info
# ---------------------------------------------------------------------------


class TestRetrieveBucketInfo(unittest.TestCase):
    """Test _retrieve_bucket_info with mocked environment."""

    def setUp(self):
        # Patch gha_query_workflow_run_by_id so we never make real API calls.
        self.api_patcher = mock.patch(
            "_therock_utils.workflow_outputs.gha_query_workflow_run_by_id"
        )
        self.mock_api = self.api_patcher.start()

        # Isolate from ambient env vars that _retrieve_bucket_info reads.
        # mock.patch.dict records the original state; individual tests add
        # specific vars via @mock.patch.dict decorators on top.
        self.env_patcher = mock.patch.dict(os.environ)
        self.env_patcher.start()
        os.environ.pop("GITHUB_REPOSITORY", None)
        os.environ.pop("IS_PR_FROM_FORK", None)
        os.environ.pop("RELEASE_TYPE", None)

    def tearDown(self):
        self.env_patcher.stop()
        self.api_patcher.stop()

    def _call(self, **kwargs):
        from _therock_utils.workflow_outputs import _retrieve_bucket_info

        # Provide default storage_config if not specified
        if "storage_config" not in kwargs:
            kwargs["storage_config"] = StorageConfig()
        return _retrieve_bucket_info(**kwargs)

    def test_no_env_defaults_to_rocm_therock(self):
        """When GITHUB_REPOSITORY is not set, defaults to ROCm/TheRock."""
        external_repo, bucket = self._call()
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-ci-artifacts")

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
        {"GITHUB_REPOSITORY": "ROCm/TheRock", "RELEASE_TYPE": "prerelease"},
        clear=False,
    )
    def test_release_type_prerelease(self):
        external_repo, bucket = self._call()
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "therock-prerelease-artifacts")

    @mock.patch.dict(
        os.environ,
        {"GITHUB_REPOSITORY": "ROCm/TheRock", "RELEASE_TYPE": "bogus"},
        clear=False,
    )
    def test_release_type_invalid_raises(self):
        with self.assertRaises(ValueError) as cm:
            self._call()
        self.assertIn("bogus", str(cm.exception))

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

    @mock.patch.dict(
        os.environ,
        {"GITHUB_REPOSITORY": "ROCm/TheRock", "RELEASE_TYPE": "dev"},
        clear=False,
    )
    def test_custom_bucket_schema(self):
        """Custom bucket schema should be used when provided."""
        config = StorageConfig(bucket_schema="custom-{release_type}-bucket")
        external_repo, bucket = self._call(storage_config=config)
        self.assertEqual(external_repo, "")
        self.assertEqual(bucket, "custom-dev-bucket")

    @mock.patch.dict(
        os.environ,
        {"GITHUB_REPOSITORY": "ROCm/TheRock"},
        clear=False,
    )
    def test_bucket_schema_ignored_without_release_type(self):
        """Custom bucket schema is ignored when RELEASE_TYPE is not set."""
        config = StorageConfig(bucket_schema="custom-{release_type}-bucket")
        external_repo, bucket = self._call(storage_config=config)
        self.assertEqual(external_repo, "")
        # Should use default logic, not custom schema
        self.assertEqual(bucket, "therock-ci-artifacts")


if __name__ == "__main__":
    unittest.main()
