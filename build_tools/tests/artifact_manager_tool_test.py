#!/usr/bin/env python
"""Integration tests for artifact_manager.py CLI tool.

These tests verify end-to-end behavior of the artifact_manager push/fetch commands,
particularly error handling and exit codes.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifact_backend import ArtifactBackend, LocalDirectoryBackend

# Minimal topology TOML for testing push/fetch behavior.
# Defines two stages: upstream-stage produces artifacts, downstream-stage consumes them.
TEST_TOPOLOGY_TOML = """\
[metadata]
version = "2.0"
description = "Test topology for artifact_manager tests"

[build_stages.upstream-stage]
description = "Upstream stage that produces artifacts"
artifact_groups = ["upstream-group"]

[build_stages.downstream-stage]
description = "Downstream stage that consumes artifacts"
artifact_groups = ["downstream-group"]

[artifact_groups.upstream-group]
description = "Upstream artifact group"
type = "generic"

[artifact_groups.downstream-group]
description = "Downstream artifact group"
type = "generic"
artifact_group_deps = ["upstream-group"]

[artifacts.test-artifact]
artifact_group = "upstream-group"
type = "target-neutral"

[artifacts.downstream-artifact]
artifact_group = "downstream-group"
type = "target-neutral"
artifact_deps = ["test-artifact"]
"""

# Platform used consistently across all tests
TEST_PLATFORM = "linux"


class FailingBackend(ArtifactBackend):
    """Backend that fails operations after a configurable number of successes.

    Can be configured to fail uploads, downloads, or both.
    """

    def __init__(
        self,
        staging_dir: Optional[Path] = None,
        run_id: str = "local",
        platform: str = TEST_PLATFORM,
        fail_uploads_after: Optional[int] = None,
        fail_downloads_after: Optional[int] = None,
    ):
        """Initialize the failing backend.

        Args:
            staging_dir: Directory for successful operations (optional).
            run_id: Run ID for path construction.
            platform: Platform name for path construction.
            fail_uploads_after: Number of successful uploads before failing.
                               None means don't fail uploads.
            fail_downloads_after: Number of successful downloads before failing.
                                 None means don't fail downloads.
        """
        self.fail_uploads_after = fail_uploads_after
        self.fail_downloads_after = fail_downloads_after
        self.upload_count = 0
        self.download_count = 0
        self.run_id = run_id
        self.platform = platform

        # Use a real backend for successful operations
        if staging_dir:
            self._real_backend = LocalDirectoryBackend(
                staging_dir=staging_dir,
                run_id=run_id,
                platform=platform,
            )
        else:
            self._real_backend = None

    @property
    def base_uri(self) -> str:
        return f"failing://test-{self.run_id}-{self.platform}"

    def list_artifacts(self, name_filter=None):
        if self._real_backend:
            return self._real_backend.list_artifacts(name_filter)
        return []

    def download_artifact(self, artifact_key, dest_path):
        self.download_count += 1
        if (
            self.fail_downloads_after is not None
            and self.download_count > self.fail_downloads_after
        ):
            raise RuntimeError(
                f"Simulated download failure for {artifact_key} "
                f"(download #{self.download_count}, configured to fail after "
                f"{self.fail_downloads_after})"
            )
        if self._real_backend:
            return self._real_backend.download_artifact(artifact_key, dest_path)
        raise FileNotFoundError(f"No backend configured: {artifact_key}")

    def upload_artifact(self, source_path, artifact_key):
        self.upload_count += 1
        if (
            self.fail_uploads_after is not None
            and self.upload_count > self.fail_uploads_after
        ):
            raise RuntimeError(
                f"Simulated upload failure for {artifact_key} "
                f"(upload #{self.upload_count}, configured to fail after "
                f"{self.fail_uploads_after})"
            )
        if self._real_backend:
            return self._real_backend.upload_artifact(source_path, artifact_key)

    def artifact_exists(self, artifact_key):
        if self._real_backend:
            return self._real_backend.artifact_exists(artifact_key)
        return False


class ArtifactManagerTestBase(unittest.TestCase):
    """Base class for artifact_manager tests with common setup/teardown."""

    def setUp(self):
        """Create temporary directories and save environment."""
        # Save environment to restore later
        self._saved_environ = os.environ.copy()

        # Create temp directory (use system default, not hardcoded path)
        self.temp_dir = tempfile.mkdtemp(prefix="artifact_manager_test_")
        self.build_dir = Path(self.temp_dir) / "build"
        self.staging_dir = Path(self.temp_dir) / "staging"
        self.output_dir = Path(self.temp_dir) / "output"
        self.build_dir.mkdir(parents=True)
        self.staging_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)

        # Write test topology to a file
        self.topology_path = Path(self.temp_dir) / "BUILD_TOPOLOGY.toml"
        self.topology_path.write_text(TEST_TOPOLOGY_TOML)

    def tearDown(self):
        """Clean up temporary directories and restore environment."""
        # Restore environment
        os.environ.clear()
        os.environ.update(self._saved_environ)

        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_fake_precompressed_artifact(
        self, name: str, component: str, target_family: str
    ) -> Path:
        """Create a fake pre-compressed artifact tarball.

        Note: Content is intentionally invalid zstd - tests should not attempt
        to actually decompress these files.
        """
        artifacts_dir = self.build_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        archive_name = f"{name}_{component}_{target_family}.tar.zst"
        archive_path = artifacts_dir / archive_name
        archive_path.write_bytes(b"fake zstd archive content")

        # Also create sha256sum
        sha_path = artifacts_dir / f"{archive_name}.sha256sum"
        sha_path.write_text(f"abc123  {archive_name}\n")

        return archive_path

    def _create_fake_artifact_dir(
        self, name: str, component: str, target_family: str
    ) -> Path:
        """Create a fake artifact directory with minimal content."""
        artifacts_dir = self.build_dir / "artifacts"
        artifact_name = f"{name}_{component}_{target_family}"
        artifact_dir = artifacts_dir / artifact_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "dummy.txt").write_text(f"Artifact: {artifact_name}\n")
        return artifact_dir

    def _create_staged_artifact(
        self, name: str, component: str, target_family: str, run_id: str = "local"
    ) -> str:
        """Create a fake artifact in the staging directory."""
        backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            run_id=run_id,
            platform=TEST_PLATFORM,
        )

        archive_name = f"{name}_{component}_{target_family}.tar.zst"
        temp_archive = Path(self.temp_dir) / archive_name
        temp_archive.write_bytes(b"fake zstd archive content")

        backend.upload_artifact(temp_archive, archive_name)
        temp_archive.unlink()

        return archive_name


class TestPushFailureExitCode(ArtifactManagerTestBase):
    """Tests that push command exits with non-zero code on upload failures."""

    @mock.patch("artifact_manager._delay_for_retry")
    @mock.patch("artifact_manager.create_backend_from_env")
    def test_push_fails_when_all_uploads_fail(self, mock_backend_factory, mock_delay):
        """Test that push exits with code 1 when all uploads fail."""
        import artifact_manager

        failing_backend = FailingBackend(fail_uploads_after=0)
        mock_backend_factory.return_value = failing_backend

        self._create_fake_precompressed_artifact("test-artifact", "lib", "generic")

        argv = [
            "push",
            "--stage",
            "upstream-stage",
            "--build-dir",
            str(self.build_dir),
            "--topology",
            str(self.topology_path),
            "--local-staging-dir",
            str(self.staging_dir),
            "--platform",
            TEST_PLATFORM,
        ]

        with self.assertRaises(SystemExit) as ctx:
            artifact_manager.main(argv)

        self.assertEqual(ctx.exception.code, 1)
        mock_backend_factory.assert_called_once()

    @mock.patch("artifact_manager._delay_for_retry")
    @mock.patch("artifact_manager.create_backend_from_env")
    def test_push_fails_when_some_uploads_fail(self, mock_backend_factory, mock_delay):
        """Test that push exits with code 1 when some (but not all) uploads fail."""
        import artifact_manager

        failing_backend = FailingBackend(
            fail_uploads_after=1, staging_dir=self.staging_dir
        )
        mock_backend_factory.return_value = failing_backend

        self._create_fake_precompressed_artifact("test-artifact", "lib", "generic")
        self._create_fake_precompressed_artifact("test-artifact", "dev", "generic")
        self._create_fake_precompressed_artifact("test-artifact", "run", "generic")

        argv = [
            "push",
            "--stage",
            "upstream-stage",
            "--build-dir",
            str(self.build_dir),
            "--topology",
            str(self.topology_path),
            "--local-staging-dir",
            str(self.staging_dir),
            "--platform",
            TEST_PLATFORM,
        ]

        with self.assertRaises(SystemExit) as ctx:
            artifact_manager.main(argv)

        self.assertEqual(ctx.exception.code, 1)

    def test_push_succeeds_when_all_uploads_succeed(self):
        """Test that push exits normally (no exception) when all uploads succeed."""
        import artifact_manager

        self._create_fake_precompressed_artifact("test-artifact", "lib", "generic")

        argv = [
            "push",
            "--stage",
            "upstream-stage",
            "--build-dir",
            str(self.build_dir),
            "--topology",
            str(self.topology_path),
            "--local-staging-dir",
            str(self.staging_dir),
            "--platform",
            TEST_PLATFORM,
            "--run-id",
            "local",
        ]

        # Should complete without raising SystemExit
        artifact_manager.main(argv)

        # Verify artifacts were uploaded
        backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            run_id="local",
            platform=TEST_PLATFORM,
        )
        self.assertTrue(backend.artifact_exists("test-artifact_lib_generic.tar.zst"))

        # Verify sha256sum was also uploaded
        self.assertTrue(
            backend.artifact_exists("test-artifact_lib_generic.tar.zst.sha256sum")
        )


class TestPushCompressionFailure(ArtifactManagerTestBase):
    """Tests that push command handles compression failures correctly."""

    @mock.patch("artifact_manager.compress_artifact")
    def test_push_fails_when_compression_fails(self, mock_compress):
        """Test that push exits with code 1 when compression fails."""
        import artifact_manager

        mock_compress.return_value = None

        self._create_fake_artifact_dir("test-artifact", "lib", "generic")

        argv = [
            "push",
            "--stage",
            "upstream-stage",
            "--build-dir",
            str(self.build_dir),
            "--topology",
            str(self.topology_path),
            "--local-staging-dir",
            str(self.staging_dir),
            "--platform",
            TEST_PLATFORM,
        ]

        with self.assertRaises(SystemExit) as ctx:
            artifact_manager.main(argv)

        self.assertEqual(ctx.exception.code, 1)
        mock_compress.assert_called_once()


class TestFetchFailureExitCode(ArtifactManagerTestBase):
    """Tests that fetch command exits with non-zero code on download failures."""

    @mock.patch("artifact_manager._delay_for_retry")
    @mock.patch("artifact_manager.create_backend_from_env")
    def test_fetch_fails_when_download_fails(self, mock_backend_factory, mock_delay):
        """Test that fetch exits with code 1 when download fails."""
        import artifact_manager

        self._create_staged_artifact("test-artifact", "lib", "generic")

        failing_backend = FailingBackend(
            fail_downloads_after=0, staging_dir=self.staging_dir, run_id="local"
        )
        mock_backend_factory.return_value = failing_backend

        argv = [
            "fetch",
            "--stage",
            "downstream-stage",
            "--output-dir",
            str(self.output_dir),
            "--topology",
            str(self.topology_path),
            "--local-staging-dir",
            str(self.staging_dir),
            "--platform",
            TEST_PLATFORM,
        ]

        with self.assertRaises(SystemExit) as ctx:
            artifact_manager.main(argv)

        self.assertEqual(ctx.exception.code, 1)
        mock_backend_factory.assert_called_once()

    @mock.patch("artifact_manager.extract_artifact")
    def test_fetch_fails_when_extraction_fails(self, mock_extract):
        """Test that fetch exits with code 1 when extraction fails."""
        import artifact_manager

        self._create_staged_artifact("test-artifact", "lib", "generic")

        mock_extract.return_value = None

        argv = [
            "fetch",
            "--stage",
            "downstream-stage",
            "--output-dir",
            str(self.output_dir),
            "--topology",
            str(self.topology_path),
            "--local-staging-dir",
            str(self.staging_dir),
            "--platform",
            TEST_PLATFORM,
            "--run-id",
            "local",
        ]

        with self.assertRaises(SystemExit) as ctx:
            artifact_manager.main(argv)

        self.assertEqual(ctx.exception.code, 1)
        mock_extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
