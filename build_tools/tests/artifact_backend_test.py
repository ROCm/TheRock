#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for artifact_backend.py."""

import os
import shutil
import sys
import tempfile
import unittest
from botocore import UNSIGNED
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifact_backend import (
    ArtifactBackend,
    ArtifactConflictError,
    LocalDirectoryBackend,
    S3Backend,
    _parse_sha256sum_content,
    _read_local_sha256sum,
    create_backend_from_env,
)
from _therock_utils.workflow_outputs import WorkflowOutputRoot


def _make_local_root(run_id="test-run-123", platform="linux"):
    return WorkflowOutputRoot.for_local(run_id=run_id, platform=platform)


def _make_s3_root(
    bucket="test-bucket",
    run_id="test-run-456",
    platform="linux",
    external_repo="external/",
):
    return WorkflowOutputRoot(
        bucket=bucket,
        external_repo=external_repo,
        run_id=run_id,
        platform=platform,
    )


class TestLocalDirectoryBackend(unittest.TestCase):
    """Tests for LocalDirectoryBackend."""

    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_root = _make_local_root()
        self.backend = LocalDirectoryBackend(
            staging_dir=Path(self.temp_dir),
            output_root=self.output_root,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_base_uri(self):
        """Test that base_uri returns the correct path."""
        expected = str(Path(self.temp_dir) / "test-run-123-linux")
        self.assertEqual(self.backend.base_uri, expected)

    def test_base_path_created(self):
        """Test that the base path directory is created."""
        self.assertTrue(self.backend.base_path.exists())
        self.assertTrue(self.backend.base_path.is_dir())

    def test_list_artifacts_empty(self):
        """Test listing artifacts when none exist."""
        artifacts = self.backend.list_artifacts()
        self.assertEqual(artifacts, [])

    def test_list_artifacts_with_files(self):
        """Test listing artifacts with tar.zst and tar.xz files present."""
        # Create some test artifact files (both zstd and xz)
        (self.backend.base_path / "blas_lib_gfx94X.tar.zst").touch()
        (self.backend.base_path / "blas_dev_gfx94X.tar.xz").touch()
        (self.backend.base_path / "fft_lib_generic.tar.zst").touch()
        # Create a sha256sum file (should be excluded)
        (self.backend.base_path / "blas_lib_gfx94X.tar.zst.sha256sum").touch()
        # Create a non-artifact file (should be excluded)
        (self.backend.base_path / "README.txt").touch()

        artifacts = self.backend.list_artifacts()
        self.assertEqual(len(artifacts), 3)
        self.assertIn("blas_lib_gfx94X.tar.zst", artifacts)
        self.assertIn("blas_dev_gfx94X.tar.xz", artifacts)
        self.assertIn("fft_lib_generic.tar.zst", artifacts)

    def test_list_artifacts_with_name_filter(self):
        """Test filtering artifacts by name prefix."""
        (self.backend.base_path / "blas_lib_gfx94X.tar.xz").touch()
        (self.backend.base_path / "blas_dev_gfx94X.tar.xz").touch()
        (self.backend.base_path / "fft_lib_generic.tar.xz").touch()

        # Filter by "blas"
        artifacts = self.backend.list_artifacts(name_filter="blas")
        self.assertEqual(len(artifacts), 2)
        self.assertIn("blas_lib_gfx94X.tar.xz", artifacts)
        self.assertIn("blas_dev_gfx94X.tar.xz", artifacts)

        # Filter by "fft"
        artifacts = self.backend.list_artifacts(name_filter="fft")
        self.assertEqual(len(artifacts), 1)
        self.assertIn("fft_lib_generic.tar.xz", artifacts)

        # Filter with no matches
        artifacts = self.backend.list_artifacts(name_filter="nonexistent")
        self.assertEqual(len(artifacts), 0)

    def test_upload_and_download_artifact_xz(self):
        """Test uploading and downloading a .tar.xz artifact."""
        # Create a source file to upload
        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        source_file = source_dir / "test_artifact.tar.xz"
        source_file.write_text("test artifact content xz")

        # Also create a sha256sum file
        sha_file = source_dir / "test_artifact.tar.xz.sha256sum"
        sha_file.write_text("abc123  test_artifact.tar.xz\n")

        # Upload with sha256sum_path
        self.backend.upload_artifact(
            source_file, "test_artifact.tar.xz", sha256sum_path=sha_file
        )

        # Verify it exists in the backend
        self.assertTrue(self.backend.artifact_exists("test_artifact.tar.xz"))
        self.assertTrue(
            (self.backend.base_path / "test_artifact.tar.xz.sha256sum").exists()
        )

        # Download to a new location
        dest_dir = Path(self.temp_dir) / "dest"
        dest_dir.mkdir()
        dest_file = dest_dir / "downloaded.tar.xz"

        self.backend.download_artifact("test_artifact.tar.xz", dest_file)

        # Verify content
        self.assertTrue(dest_file.exists())
        self.assertEqual(dest_file.read_text(), "test artifact content xz")
        # Verify sha256sum was also copied
        self.assertTrue((dest_dir / "test_artifact.tar.xz.sha256sum").exists())

    def test_upload_and_download_artifact_zst(self):
        """Test uploading and downloading a .tar.zst artifact."""
        # Create a source file to upload
        source_dir = Path(self.temp_dir) / "source_zst"
        source_dir.mkdir()
        source_file = source_dir / "test_artifact.tar.zst"
        source_file.write_text("test artifact content zst")

        # Also create a sha256sum file
        sha_file = source_dir / "test_artifact.tar.zst.sha256sum"
        sha_file.write_text("def456  test_artifact.tar.zst\n")

        # Upload with sha256sum_path
        self.backend.upload_artifact(
            source_file, "test_artifact.tar.zst", sha256sum_path=sha_file
        )

        # Verify it exists in the backend
        self.assertTrue(self.backend.artifact_exists("test_artifact.tar.zst"))
        self.assertTrue(
            (self.backend.base_path / "test_artifact.tar.zst.sha256sum").exists()
        )

        # Download to a new location
        dest_dir = Path(self.temp_dir) / "dest_zst"
        dest_dir.mkdir()
        dest_file = dest_dir / "downloaded.tar.zst"

        self.backend.download_artifact("test_artifact.tar.zst", dest_file)

        # Verify content
        self.assertTrue(dest_file.exists())
        self.assertEqual(dest_file.read_text(), "test artifact content zst")
        # Verify sha256sum was also copied
        self.assertTrue((dest_dir / "test_artifact.tar.zst.sha256sum").exists())

    def test_download_nonexistent_artifact(self):
        """Test that downloading a nonexistent artifact raises FileNotFoundError."""
        dest_file = Path(self.temp_dir) / "dest" / "nonexistent.tar.xz"
        with self.assertRaises(FileNotFoundError):
            self.backend.download_artifact("nonexistent.tar.xz", dest_file)

    def test_upload_nonexistent_source(self):
        """Test that uploading a nonexistent source raises FileNotFoundError."""
        nonexistent = Path(self.temp_dir) / "nonexistent.tar.xz"
        with self.assertRaises(FileNotFoundError):
            self.backend.upload_artifact(nonexistent, "test.tar.xz")

    def test_artifact_exists_xz(self):
        """Test artifact_exists method with .tar.xz."""
        self.assertFalse(self.backend.artifact_exists("nonexistent.tar.xz"))

        (self.backend.base_path / "exists.tar.xz").touch()
        self.assertTrue(self.backend.artifact_exists("exists.tar.xz"))

    def test_artifact_exists_zst(self):
        """Test artifact_exists method with .tar.zst."""
        self.assertFalse(self.backend.artifact_exists("nonexistent.tar.zst"))

        (self.backend.base_path / "exists.tar.zst").touch()
        self.assertTrue(self.backend.artifact_exists("exists.tar.zst"))

    def test_copy_artifact_between_local_backends(self):
        """Test copy_artifact copies files between two local backends."""
        source = LocalDirectoryBackend(
            staging_dir=Path(self.temp_dir),
            output_root=_make_local_root(run_id="source-run"),
        )
        dest = LocalDirectoryBackend(
            staging_dir=Path(self.temp_dir),
            output_root=_make_local_root(run_id="dest-run"),
        )

        # Create artifact and sha256sum in source
        artifact_key = "test_lib_generic.tar.zst"
        (source.base_path / artifact_key).write_bytes(b"test content")
        (source.base_path / f"{artifact_key}.sha256sum").write_text(
            "abc123  test_lib_generic.tar.zst\n"
        )

        # Copy to dest
        dest.copy_artifact(artifact_key, source)

        # Verify artifact and sha256sum were both copied
        self.assertTrue(dest.artifact_exists(artifact_key))
        self.assertEqual((dest.base_path / artifact_key).read_bytes(), b"test content")
        self.assertTrue((dest.base_path / f"{artifact_key}.sha256sum").exists())

    def test_copy_artifact_without_sha256sum(self):
        """Test copy_artifact works when no sha256sum file exists."""
        source = LocalDirectoryBackend(
            staging_dir=Path(self.temp_dir),
            output_root=_make_local_root(run_id="source-run"),
        )
        dest = LocalDirectoryBackend(
            staging_dir=Path(self.temp_dir),
            output_root=_make_local_root(run_id="dest-run"),
        )

        # Create artifact without sha256sum
        artifact_key = "test_lib_generic.tar.zst"
        (source.base_path / artifact_key).write_bytes(b"test content")

        # Copy should succeed without error
        dest.copy_artifact(artifact_key, source)

        self.assertTrue(dest.artifact_exists(artifact_key))
        self.assertFalse((dest.base_path / f"{artifact_key}.sha256sum").exists())

    def test_copy_artifact_nonexistent_raises(self):
        """Test copy_artifact raises FileNotFoundError for missing source artifact."""
        source = LocalDirectoryBackend(
            staging_dir=Path(self.temp_dir),
            output_root=_make_local_root(run_id="source-run"),
        )
        dest = LocalDirectoryBackend(
            staging_dir=Path(self.temp_dir),
            output_root=_make_local_root(run_id="dest-run"),
        )

        with self.assertRaises(FileNotFoundError):
            dest.copy_artifact("nonexistent.tar.zst", source)

    def test_copy_artifact_wrong_backend_type_raises(self):
        """Test copy_artifact raises TypeError when source is a different backend type."""
        s3_source = S3Backend(
            output_root=_make_s3_root(run_id="source-run"),
        )

        with self.assertRaises(TypeError):
            self.backend.copy_artifact("test.tar.zst", s3_source)


class TestS3Backend(unittest.TestCase):
    """Tests for S3Backend with mocked boto3 client."""

    def setUp(self):
        """Set up S3Backend with mocked client."""
        self.output_root = _make_s3_root()
        self.backend = S3Backend(output_root=self.output_root)

    def test_base_uri(self):
        """Test that base_uri returns the correct S3 URI."""
        expected = "s3://test-bucket/external/test-run-456-linux"
        self.assertEqual(self.backend.base_uri, expected)

    def test_s3_prefix(self):
        """Test that s3_prefix is constructed correctly."""
        self.assertEqual(self.backend.s3_prefix, "external/test-run-456-linux")

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_list_artifacts(self, mock_client_prop):
        """Test listing S3 artifacts (both zstd and xz)."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        # Mock paginator with both zstd and xz artifacts
        mock_paginator = mock.MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "external/test-run-456-linux/blas_lib_gfx94X.tar.zst"},
                    {"Key": "external/test-run-456-linux/blas_dev_gfx94X.tar.xz"},
                    {
                        "Key": "external/test-run-456-linux/blas_lib_gfx94X.tar.zst.sha256sum"
                    },
                ]
            }
        ]

        artifacts = self.backend.list_artifacts()
        self.assertEqual(len(artifacts), 2)
        self.assertIn("blas_lib_gfx94X.tar.zst", artifacts)
        self.assertIn("blas_dev_gfx94X.tar.xz", artifacts)

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_list_artifacts_with_filter(self, mock_client_prop):
        """Test listing S3 artifacts with name filter."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        mock_paginator = mock.MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "external/test-run-456-linux/blas_lib_gfx94X.tar.xz"},
                    {"Key": "external/test-run-456-linux/fft_lib_generic.tar.xz"},
                ]
            }
        ]

        artifacts = self.backend.list_artifacts(name_filter="blas")
        self.assertEqual(len(artifacts), 1)
        self.assertIn("blas_lib_gfx94X.tar.xz", artifacts)

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_download_artifact_xz(self, mock_client_prop):
        """Test downloading a .tar.xz artifact from S3."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = Path(temp_dir) / "downloaded.tar.xz"
            self.backend.download_artifact("test.tar.xz", dest_path)

            mock_client.download_file.assert_called_once_with(
                "test-bucket",
                "external/test-run-456-linux/test.tar.xz",
                str(dest_path),
            )

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_download_artifact_zst(self, mock_client_prop):
        """Test downloading a .tar.zst artifact from S3."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        with tempfile.TemporaryDirectory() as temp_dir:
            dest_path = Path(temp_dir) / "downloaded.tar.zst"
            self.backend.download_artifact("test.tar.zst", dest_path)

            mock_client.download_file.assert_called_once_with(
                "test-bucket",
                "external/test-run-456-linux/test.tar.zst",
                str(dest_path),
            )

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_upload_artifact_xz(self, mock_client_prop):
        """Test uploading a .tar.xz artifact to S3."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client
        # Artifact does not exist
        mock_client.head_object.side_effect = Exception("Not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "test.tar.xz"
            source_path.touch()

            self.backend.upload_artifact(source_path, "test.tar.xz")

            mock_client.upload_file.assert_called_once_with(
                str(source_path),
                "test-bucket",
                "external/test-run-456-linux/test.tar.xz",
            )

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_upload_artifact_zst(self, mock_client_prop):
        """Test uploading a .tar.zst artifact to S3."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client
        # Artifact does not exist
        mock_client.head_object.side_effect = Exception("Not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "test.tar.zst"
            source_path.touch()

            self.backend.upload_artifact(source_path, "test.tar.zst")

            mock_client.upload_file.assert_called_once_with(
                str(source_path),
                "test-bucket",
                "external/test-run-456-linux/test.tar.zst",
            )

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_artifact_exists_true(self, mock_client_prop):
        """Test artifact_exists when artifact exists."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        self.assertTrue(self.backend.artifact_exists("exists.tar.xz"))
        mock_client.head_object.assert_called_once()

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_artifact_exists_false(self, mock_client_prop):
        """Test artifact_exists when artifact does not exist."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client
        mock_client.head_object.side_effect = Exception("Not found")

        self.assertFalse(self.backend.artifact_exists("nonexistent.tar.xz"))

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_copy_artifact_same_bucket(self, mock_client_prop):
        """Test S3 server-side copy within the same bucket."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client
        # sha256sum exists in source
        mock_client.head_object.return_value = {}

        source = S3Backend(
            output_root=_make_s3_root(
                bucket="test-bucket", run_id="source-run", external_repo=""
            )
        )
        dest = S3Backend(
            output_root=_make_s3_root(
                bucket="test-bucket", run_id="dest-run", external_repo=""
            )
        )
        # Share the mock client
        source._s3_client = mock_client

        dest.copy_artifact("artifact_lib_generic.tar.zst", source)

        # Verify both artifact and sha256sum were copied
        self.assertEqual(mock_client.copy.call_count, 2)
        mock_client.copy.assert_any_call(
            {
                "Bucket": "test-bucket",
                "Key": "source-run-linux/artifact_lib_generic.tar.zst",
            },
            "test-bucket",
            "dest-run-linux/artifact_lib_generic.tar.zst",
        )
        mock_client.copy.assert_any_call(
            {
                "Bucket": "test-bucket",
                "Key": "source-run-linux/artifact_lib_generic.tar.zst.sha256sum",
            },
            "test-bucket",
            "dest-run-linux/artifact_lib_generic.tar.zst.sha256sum",
        )

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_copy_artifact_cross_bucket(self, mock_client_prop):
        """Test S3 server-side copy across different buckets."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client
        # sha256sum does not exist in source
        mock_client.head_object.side_effect = Exception("Not found")

        source = S3Backend(
            output_root=_make_s3_root(
                bucket="therock-ci-artifacts",
                run_id="source-run",
                external_repo="",
            )
        )
        dest = S3Backend(
            output_root=_make_s3_root(
                bucket="therock-ci-artifacts-external",
                run_id="dest-run",
                external_repo="ROCm-rocm-libraries/",
            )
        )
        source._s3_client = mock_client

        dest.copy_artifact("artifact_lib_generic.tar.zst", source)

        # Only artifact copied (no sha256sum in source)
        mock_client.copy.assert_called_once_with(
            {
                "Bucket": "therock-ci-artifacts",
                "Key": "source-run-linux/artifact_lib_generic.tar.zst",
            },
            "therock-ci-artifacts-external",
            "ROCm-rocm-libraries/dest-run-linux/artifact_lib_generic.tar.zst",
        )

    def test_copy_artifact_wrong_backend_type_raises(self):
        """Test copy_artifact raises TypeError when source is a different backend type."""
        import tempfile

        local_source = LocalDirectoryBackend(
            staging_dir=Path(tempfile.mkdtemp()),
            output_root=_make_local_root(run_id="source-run"),
        )

        with self.assertRaises(TypeError):
            self.backend.copy_artifact("test.tar.zst", local_source)


class TestS3BackendCredentials(unittest.TestCase):
    """Live tests for S3Backend credential resolution.

    These exercise the real boto3 credential chain (no mocks) to verify:
    - UNSIGNED fallback works for public bucket reads without credentials
    - AWS_SHARED_CREDENTIALS_FILE is used (so fork PRs can upload artifacts)
    """

    # Env vars that could provide credentials to boto3's default chain.
    _CREDENTIAL_ENV_VARS = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_PROFILE",
    ]

    def _make_backend(self):
        output_root = WorkflowOutputRoot(
            bucket="therock-ci-artifacts",
            external_repo="",
            run_id="placeholder",
            platform="linux",
        )
        return S3Backend(output_root=output_root)

    def test_list_objects_unsigned(self):
        """Listing objects in a public bucket must succeed without credentials."""
        # Point config/credentials files at nonexistent paths so that
        # ~/.aws/credentials on developer machines is not picked up.
        env_overrides = {
            "AWS_CONFIG_FILE": "/nonexistent/aws/config",
            "AWS_SHARED_CREDENTIALS_FILE": "/nonexistent/aws/credentials",
        }
        with mock.patch.dict(os.environ, env_overrides, clear=False):
            for var in self._CREDENTIAL_ENV_VARS:
                if var not in env_overrides:
                    os.environ.pop(var, None)

            backend = self._make_backend()

            config = backend.s3_client._client_config
            self.assertEqual(
                config.signature_version,
                UNSIGNED,
                "Client should be UNSIGNED when all credentials are cleared",
            )

            # A real list call against the public bucket should succeed
            response = backend.s3_client.list_objects_v2(
                Bucket="therock-ci-artifacts",
                MaxKeys=1,
            )
            self.assertIn("Contents", response)
            self.assertGreater(len(response["Contents"]), 0)

    def test_shared_credentials_file_produces_authenticated_client(self):
        """AWS_SHARED_CREDENTIALS_FILE must produce an authenticated client.

        Fork PRs provide credentials via a mounted credentials.ini file,
        not via AWS_ACCESS_KEY_ID/SECRET/TOKEN env vars. This test would
        fail with the old manual 3-env-var check that ignored
        AWS_SHARED_CREDENTIALS_FILE.
        """
        # Write a minimal credentials file. Values are intentionally not
        # real AWS keys — boto3 loads any syntactically valid values.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ini", delete=False
        ) as creds_file:
            creds_file.write(
                "[default]\n"
                "aws_access_key_id = test\n"
                "aws_secret_access_key = test\n"
            )
            creds_path = creds_file.name

        try:
            env_overrides = {
                "AWS_SHARED_CREDENTIALS_FILE": creds_path,
                "AWS_CONFIG_FILE": "/nonexistent/aws/config",
            }
            env_clear = [
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "AWS_PROFILE",
            ]
            with mock.patch.dict(os.environ, env_overrides, clear=False):
                for var in env_clear:
                    os.environ.pop(var, None)

                backend = self._make_backend()
                config = backend.s3_client._client_config

                self.assertNotEqual(
                    config.signature_version,
                    UNSIGNED,
                    "Client must NOT use UNSIGNED when credentials are "
                    "available via AWS_SHARED_CREDENTIALS_FILE. The old "
                    "manual env-var check ignored this credential source, "
                    "breaking uploads from fork PRs.",
                )
        finally:
            os.unlink(creds_path)


class TestCreateBackendFromEnv(unittest.TestCase):
    """Tests for create_backend_from_env factory function."""

    def test_local_backend_from_env(self):
        """Test that LocalDirectoryBackend is created when env var is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(
                os.environ,
                {
                    "THEROCK_LOCAL_STAGING_DIR": temp_dir,
                    "THEROCK_RUN_ID": "env-run-id",
                    "THEROCK_PLATFORM": "windows",
                },
            ):
                backend = create_backend_from_env()

                self.assertIsInstance(backend, LocalDirectoryBackend)
                self.assertIn("env-run-id", backend.base_uri)
                self.assertIn("windows", backend.base_uri)

    def test_local_backend_with_overrides(self):
        """Test LocalDirectoryBackend with explicit run_id and platform."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(
                os.environ,
                {
                    "THEROCK_LOCAL_STAGING_DIR": temp_dir,
                },
            ):
                backend = create_backend_from_env(
                    run_id="override-run", platform="override-platform"
                )

                self.assertIsInstance(backend, LocalDirectoryBackend)
                self.assertIn("override-run", backend.base_uri)
                self.assertIn("override-platform", backend.base_uri)

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def test_s3_backend_when_no_local_dir(self, mock_retrieve):
        """Test that S3Backend is created when THEROCK_LOCAL_STAGING_DIR is not set."""
        mock_retrieve.return_value = ("", "test-bucket")
        with mock.patch.dict(
            os.environ,
            {"THEROCK_RUN_ID": "s3-run-id"},
            clear=True,
        ):
            backend = create_backend_from_env()

            self.assertIsInstance(backend, S3Backend)
            self.assertEqual(backend.bucket, "test-bucket")
            self.assertIn("s3-run-id", backend.s3_prefix)


# ---------------------------------------------------------------------------
# Artifact conflict detection helpers
# ---------------------------------------------------------------------------


class TestParseSha256sumContent(unittest.TestCase):
    """Tests for _parse_sha256sum_content helper function."""

    def test_standard_format(self):
        """Test parsing standard sha256sum format."""
        content = "abc123def456  filename.tar.zst\n"
        self.assertEqual(_parse_sha256sum_content(content), "abc123def456")

    def test_no_trailing_newline(self):
        """Test parsing without trailing newline."""
        content = "abc123def456  filename.tar.zst"
        self.assertEqual(_parse_sha256sum_content(content), "abc123def456")

    def test_full_sha256_hash(self):
        """Test parsing a full 64-character SHA256 hash."""
        sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        content = f"{sha}  empty_file.tar.xz\n"
        self.assertEqual(_parse_sha256sum_content(content), sha)

    def test_empty_content_raises(self):
        """Test that empty content raises ValueError."""
        with self.assertRaises(ValueError):
            _parse_sha256sum_content("")

    def test_whitespace_only_raises(self):
        """Test that whitespace-only content raises ValueError."""
        with self.assertRaises(ValueError):
            _parse_sha256sum_content("   \n  ")


class TestReadLocalSha256sum(unittest.TestCase):
    """Tests for _read_local_sha256sum helper function."""

    def test_reads_existing_sha256sum(self):
        """Test reading an existing sha256sum file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "test.tar.zst"
            artifact_path.touch()
            sha_path = Path(temp_dir) / "test.tar.zst.sha256sum"
            sha_path.write_text("deadbeef12345678  test.tar.zst\n")

            result = _read_local_sha256sum(artifact_path)
            self.assertEqual(result, "deadbeef12345678")

    def test_returns_none_if_no_sha256sum(self):
        """Test that None is returned if sha256sum file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "test.tar.zst"
            artifact_path.touch()

            result = _read_local_sha256sum(artifact_path)
            self.assertIsNone(result)


# ---------------------------------------------------------------------------
# LocalDirectoryBackend conflict detection tests
# ---------------------------------------------------------------------------


class TestLocalDirectoryBackendConflict(unittest.TestCase):
    """Tests for artifact conflict detection in LocalDirectoryBackend."""

    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_root = _make_local_root()
        self.backend = LocalDirectoryBackend(
            staging_dir=Path(self.temp_dir),
            output_root=self.output_root,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_artifact_matching_sha_skips_upload(self):
        """Test that artifact with matching SHA skips upload."""
        # Create existing artifact with sha256sum
        artifact_key = "test_lib_generic.tar.zst"
        sha_hash = "abc123def456"
        (self.backend.base_path / artifact_key).write_bytes(b"existing content")
        (self.backend.base_path / f"{artifact_key}.sha256sum").write_text(
            f"{sha_hash}  {artifact_key}\n"
        )

        # Create source with same SHA
        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        source_file = source_dir / artifact_key
        source_file.write_bytes(b"new content")  # Content doesn't matter for test
        (source_dir / f"{artifact_key}.sha256sum").write_text(
            f"{sha_hash}  {artifact_key}\n"
        )

        # Upload should skip (not overwrite)
        original_content = (self.backend.base_path / artifact_key).read_bytes()
        self.backend.upload_artifact(source_file, artifact_key)

        # Content should be unchanged (skipped)
        self.assertEqual(
            (self.backend.base_path / artifact_key).read_bytes(), original_content
        )

    def test_artifact_differing_sha_raises_error(self):
        """Test that artifact with differing SHA raises error."""
        # Create existing artifact with sha256sum
        artifact_key = "test_lib_generic.tar.zst"
        (self.backend.base_path / artifact_key).write_bytes(b"existing content")
        (self.backend.base_path / f"{artifact_key}.sha256sum").write_text(
            f"existing_sha_123  {artifact_key}\n"
        )

        # Create source with different SHA
        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        source_file = source_dir / artifact_key
        source_file.write_bytes(b"new content")
        (source_dir / f"{artifact_key}.sha256sum").write_text(
            f"different_sha_456  {artifact_key}\n"
        )

        # Upload should raise error
        with self.assertRaises(ArtifactConflictError) as ctx:
            self.backend.upload_artifact(source_file, artifact_key)

        error_msg = str(ctx.exception)
        self.assertIn("existing_sha_123", error_msg)
        self.assertIn("different_sha_456", error_msg)
        self.assertIn("build system problem", error_msg)

    def test_artifact_no_existing_uploads_normally(self):
        """Test that artifact uploads normally if it doesn't exist."""
        artifact_key = "test_lib_generic.tar.zst"

        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        source_file = source_dir / artifact_key
        source_file.write_bytes(b"new content")
        (source_dir / f"{artifact_key}.sha256sum").write_text(
            f"sha123  {artifact_key}\n"
        )

        self.backend.upload_artifact(source_file, artifact_key)

        self.assertTrue(self.backend.artifact_exists(artifact_key))
        self.assertEqual(
            (self.backend.base_path / artifact_key).read_bytes(), b"new content"
        )

    def test_artifact_no_existing_sha256sum_computes_from_file(self):
        """Test that artifact computes SHA from file if no sha256sum exists."""
        artifact_key = "test_lib_generic.tar.zst"
        (self.backend.base_path / artifact_key).write_bytes(b"existing content")
        # No sha256sum file - will compute hash from file contents

        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        source_file = source_dir / artifact_key
        source_file.write_bytes(b"new content")
        (source_dir / f"{artifact_key}.sha256sum").write_text(
            f"sha123  {artifact_key}\n"
        )

        # Should raise error because file contents differ
        with self.assertRaises(ArtifactConflictError) as ctx:
            self.backend.upload_artifact(source_file, artifact_key)

        error_msg = str(ctx.exception)
        self.assertIn("build system problem", error_msg)

    def test_artifact_no_existing_sha256sum_identical_skips(self):
        """Test that identical artifact is skipped even without sha256sum file."""
        artifact_key = "test_lib_generic.tar.zst"
        content = b"identical content"
        (self.backend.base_path / artifact_key).write_bytes(content)
        # No sha256sum file - will compute hash from file contents

        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        source_file = source_dir / artifact_key
        source_file.write_bytes(content)  # Same content

        # Should skip (no error, no overwrite)
        self.backend.upload_artifact(source_file, artifact_key)

        self.assertEqual((self.backend.base_path / artifact_key).read_bytes(), content)

    def test_non_generic_artifact_also_gets_conflict_detection(self):
        """Test that non-generic artifacts also get conflict detection."""
        artifact_key = "test_lib_gfx94X.tar.zst"
        (self.backend.base_path / artifact_key).write_bytes(b"existing content")
        (self.backend.base_path / f"{artifact_key}.sha256sum").write_text(
            f"existing_sha_123  {artifact_key}\n"
        )

        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        source_file = source_dir / artifact_key
        source_file.write_bytes(b"new content")
        (source_dir / f"{artifact_key}.sha256sum").write_text(
            f"different_sha_456  {artifact_key}\n"
        )

        # Should raise error - conflict detection applies to ALL artifacts
        with self.assertRaises(ArtifactConflictError) as ctx:
            self.backend.upload_artifact(source_file, artifact_key)

        error_msg = str(ctx.exception)
        self.assertIn("existing_sha_123", error_msg)
        self.assertIn("different_sha_456", error_msg)

    def test_backwards_compat_alias(self):
        """Test that ArtifactConflictError is an alias for ArtifactConflictError."""
        self.assertIs(ArtifactConflictError, ArtifactConflictError)


# ---------------------------------------------------------------------------
# S3Backend conflict detection tests
# ---------------------------------------------------------------------------


class TestS3BackendConflict(unittest.TestCase):
    """Tests for artifact conflict detection in S3Backend."""

    def setUp(self):
        """Set up S3Backend with mocked client."""
        self.output_root = _make_s3_root()
        self.backend = S3Backend(output_root=self.output_root)

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_artifact_matching_sha_skips_upload(self, mock_client_prop):
        """Test that artifact with matching SHA skips upload."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        # Mock artifact exists
        mock_client.head_object.return_value = {}  # exists

        # Mock sha256sum download
        import io

        sha_hash = "abc123def456"

        def download_fileobj_side_effect(bucket, key, buffer):
            buffer.write(f"{sha_hash}  test_lib_generic.tar.zst\n".encode())

        mock_client.download_fileobj.side_effect = download_fileobj_side_effect

        # Create source file with matching SHA
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "test_lib_generic.tar.zst"
            source_path.touch()
            (Path(temp_dir) / "test_lib_generic.tar.zst.sha256sum").write_text(
                f"{sha_hash}  test_lib_generic.tar.zst\n"
            )

            self.backend.upload_artifact(source_path, "test_lib_generic.tar.zst")

        # upload_file should NOT be called (skipped)
        mock_client.upload_file.assert_not_called()

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_artifact_differing_sha_raises_error(self, mock_client_prop):
        """Test that artifact with differing SHA raises error."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        # Mock artifact exists
        mock_client.head_object.return_value = {}

        # Mock sha256sum download
        def download_fileobj_side_effect(bucket, key, buffer):
            buffer.write(b"existing_sha_123  test_lib_generic.tar.zst\n")

        mock_client.download_fileobj.side_effect = download_fileobj_side_effect

        # Create source file with different SHA
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "test_lib_generic.tar.zst"
            source_path.touch()
            (Path(temp_dir) / "test_lib_generic.tar.zst.sha256sum").write_text(
                "different_sha_456  test_lib_generic.tar.zst\n"
            )

            with self.assertRaises(ArtifactConflictError) as ctx:
                self.backend.upload_artifact(source_path, "test_lib_generic.tar.zst")

            error_msg = str(ctx.exception)
            self.assertIn("existing_sha_123", error_msg)
            self.assertIn("different_sha_456", error_msg)
            self.assertIn("build system problem", error_msg)

        mock_client.upload_file.assert_not_called()

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_artifact_no_existing_uploads_normally(self, mock_client_prop):
        """Test that artifact uploads normally if it doesn't exist."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        # Mock artifact doesn't exist
        mock_client.head_object.side_effect = Exception("Not found")

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "test_lib_generic.tar.zst"
            source_path.touch()

            self.backend.upload_artifact(source_path, "test_lib_generic.tar.zst")

        mock_client.upload_file.assert_called_once()

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_artifact_no_sha256sum_computes_from_file(self, mock_client_prop):
        """Test that artifact computes SHA from S3 file if no sha256sum exists."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        # Artifact exists, sha256sum doesn't
        def head_object_side_effect(Bucket, Key):
            if Key.endswith(".sha256sum"):
                raise Exception("Not found")
            return {}

        mock_client.head_object.side_effect = head_object_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "test_lib_generic.tar.zst"
            source_path.write_bytes(b"local content")

            # Mock downloading the S3 artifact (different content)
            def download_file_side_effect(bucket, key, path):
                Path(path).write_bytes(b"s3 content")

            mock_client.download_file.side_effect = download_file_side_effect

            # Should raise error because file contents differ
            with self.assertRaises(ArtifactConflictError) as ctx:
                self.backend.upload_artifact(source_path, "test_lib_generic.tar.zst")

            error_msg = str(ctx.exception)
            self.assertIn("build system problem", error_msg)

        mock_client.upload_file.assert_not_called()

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_artifact_no_sha256sum_identical_skips(self, mock_client_prop):
        """Test that identical artifact is skipped even without sha256sum file."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        # Artifact exists, sha256sum doesn't
        def head_object_side_effect(Bucket, Key):
            if Key.endswith(".sha256sum"):
                raise Exception("Not found")
            return {}

        mock_client.head_object.side_effect = head_object_side_effect

        content = b"identical content"
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "test_lib_generic.tar.zst"
            source_path.write_bytes(content)

            # Mock downloading the S3 artifact (same content)
            def download_file_side_effect(bucket, key, path):
                Path(path).write_bytes(content)

            mock_client.download_file.side_effect = download_file_side_effect

            # Should skip (no error, no upload)
            self.backend.upload_artifact(source_path, "test_lib_generic.tar.zst")

        mock_client.upload_file.assert_not_called()

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_non_generic_artifact_also_gets_conflict_detection(self, mock_client_prop):
        """Test that non-generic artifacts also get conflict detection."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        # Mock artifact exists
        mock_client.head_object.return_value = {}

        # Mock sha256sum download
        def download_fileobj_side_effect(bucket, key, buffer):
            buffer.write(b"existing_sha_123  test_lib_gfx94X.tar.zst\n")

        mock_client.download_fileobj.side_effect = download_fileobj_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "test_lib_gfx94X.tar.zst"
            source_path.touch()
            (Path(temp_dir) / "test_lib_gfx94X.tar.zst.sha256sum").write_text(
                "different_sha_456  test_lib_gfx94X.tar.zst\n"
            )

            # Should raise error - conflict detection applies to ALL artifacts
            with self.assertRaises(ArtifactConflictError) as ctx:
                self.backend.upload_artifact(source_path, "test_lib_gfx94X.tar.zst")

            error_msg = str(ctx.exception)
            self.assertIn("existing_sha_123", error_msg)
            self.assertIn("different_sha_456", error_msg)

        mock_client.upload_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
