#!/usr/bin/env python
"""Unit tests for artifact_backend.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifact_backend import (
    ArtifactBackend,
    LocalDirectoryBackend,
    S3Backend,
    create_backend_from_env,
)


class TestLocalDirectoryBackend(unittest.TestCase):
    """Tests for LocalDirectoryBackend."""

    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = LocalDirectoryBackend(
            staging_dir=Path(self.temp_dir),
            run_id="test-run-123",
            platform="linux",
        )

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_base_uri(self):
        """Test that base_uri returns the correct path."""
        expected = str(Path(self.temp_dir) / "run-test-run-123-linux")
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
        """Test listing artifacts with tar.xz files present."""
        # Create some test artifact files
        (self.backend.base_path / "blas_lib_gfx94X.tar.xz").touch()
        (self.backend.base_path / "blas_dev_gfx94X.tar.xz").touch()
        (self.backend.base_path / "fft_lib_generic.tar.xz").touch()
        # Create a sha256sum file (should be excluded)
        (self.backend.base_path / "blas_lib_gfx94X.tar.xz.sha256sum").touch()
        # Create a non-artifact file (should be excluded)
        (self.backend.base_path / "README.txt").touch()

        artifacts = self.backend.list_artifacts()
        self.assertEqual(len(artifacts), 3)
        self.assertIn("blas_lib_gfx94X.tar.xz", artifacts)
        self.assertIn("blas_dev_gfx94X.tar.xz", artifacts)
        self.assertIn("fft_lib_generic.tar.xz", artifacts)

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

    def test_upload_and_download_artifact(self):
        """Test uploading and downloading an artifact."""
        # Create a source file to upload
        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        source_file = source_dir / "test_artifact.tar.xz"
        source_file.write_text("test artifact content")

        # Also create a sha256sum file
        sha_file = source_dir / "test_artifact.tar.xz.sha256sum"
        sha_file.write_text("abc123  test_artifact.tar.xz\n")

        # Upload
        self.backend.upload_artifact(source_file, "test_artifact.tar.xz")

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
        self.assertEqual(dest_file.read_text(), "test artifact content")
        # Verify sha256sum was also copied
        self.assertTrue((dest_dir / "test_artifact.tar.xz.sha256sum").exists())

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

    def test_artifact_exists(self):
        """Test artifact_exists method."""
        self.assertFalse(self.backend.artifact_exists("nonexistent.tar.xz"))

        (self.backend.base_path / "exists.tar.xz").touch()
        self.assertTrue(self.backend.artifact_exists("exists.tar.xz"))


class TestS3Backend(unittest.TestCase):
    """Tests for S3Backend with mocked boto3 client."""

    def setUp(self):
        """Set up S3Backend with mocked client."""
        self.backend = S3Backend(
            bucket="test-bucket",
            run_id="test-run-456",
            platform="linux",
            external_repo="external/",
        )

    def test_base_uri(self):
        """Test that base_uri returns the correct S3 URI."""
        expected = "s3://test-bucket/external/test-run-456-linux"
        self.assertEqual(self.backend.base_uri, expected)

    def test_s3_prefix(self):
        """Test that s3_prefix is constructed correctly."""
        self.assertEqual(self.backend.s3_prefix, "external/test-run-456-linux")

    @mock.patch("boto3.client")
    def test_s3_client_with_credentials(self, mock_boto_client):
        """Test S3 client initialization with AWS credentials."""
        mock_client = mock.MagicMock()
        mock_boto_client.return_value = mock_client

        with mock.patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "test-key",
                "AWS_SECRET_ACCESS_KEY": "test-secret",
                "AWS_SESSION_TOKEN": "test-token",
            },
        ):
            backend = S3Backend(
                bucket="test-bucket",
                run_id="test-run",
                platform="linux",
            )
            # Access client to trigger initialization
            _ = backend.s3_client

        mock_boto_client.assert_called_once()
        call_kwargs = mock_boto_client.call_args[1]
        self.assertEqual(call_kwargs["aws_access_key_id"], "test-key")
        self.assertEqual(call_kwargs["aws_secret_access_key"], "test-secret")
        self.assertEqual(call_kwargs["aws_session_token"], "test-token")

    @mock.patch("boto3.client")
    def test_s3_client_without_credentials(self, mock_boto_client):
        """Test S3 client initialization without AWS credentials (unsigned)."""
        mock_client = mock.MagicMock()
        mock_boto_client.return_value = mock_client

        # Clear any existing credentials
        with mock.patch.dict(os.environ, {}, clear=True):
            backend = S3Backend(
                bucket="test-bucket",
                run_id="test-run",
                platform="linux",
            )
            # Access client to trigger initialization
            _ = backend.s3_client

        mock_boto_client.assert_called_once()
        call_kwargs = mock_boto_client.call_args[1]
        # Should use unsigned config
        self.assertIn("config", call_kwargs)

    @mock.patch.object(S3Backend, "s3_client", new_callable=mock.PropertyMock)
    def test_list_artifacts(self, mock_client_prop):
        """Test listing S3 artifacts."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

        # Mock paginator
        mock_paginator = mock.MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "external/test-run-456-linux/blas_lib_gfx94X.tar.xz"},
                    {"Key": "external/test-run-456-linux/blas_dev_gfx94X.tar.xz"},
                    {
                        "Key": "external/test-run-456-linux/blas_lib_gfx94X.tar.xz.sha256sum"
                    },
                ]
            }
        ]

        artifacts = self.backend.list_artifacts()
        self.assertEqual(len(artifacts), 2)
        self.assertIn("blas_lib_gfx94X.tar.xz", artifacts)
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
    def test_download_artifact(self, mock_client_prop):
        """Test downloading an artifact from S3."""
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
    def test_upload_artifact(self, mock_client_prop):
        """Test uploading an artifact to S3."""
        mock_client = mock.MagicMock()
        mock_client_prop.return_value = mock_client

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

    @mock.patch("_therock_utils.artifact_backend.S3Backend")
    def test_s3_backend_when_no_local_dir(self, mock_s3_backend):
        """Test that S3Backend is created when THEROCK_LOCAL_STAGING_DIR is not set."""
        with mock.patch.dict(
            os.environ,
            {
                "THEROCK_S3_BUCKET": "test-bucket",
                "THEROCK_RUN_ID": "s3-run-id",
            },
            clear=True,
        ):
            # Mock the github_actions_utils import to fail
            with mock.patch.dict(
                sys.modules, {"_therock_utils.github_actions_utils": None}
            ):
                backend = create_backend_from_env()

                mock_s3_backend.assert_called_once()
                call_kwargs = mock_s3_backend.call_args[1]
                self.assertEqual(call_kwargs["bucket"], "test-bucket")
                self.assertEqual(call_kwargs["run_id"], "s3-run-id")


if __name__ == "__main__":
    unittest.main()
