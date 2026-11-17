#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for artifact_manager.py CLI tool.

These tests verify artifact_manager push, fetch, copy, and cache population,
particularly error handling and exit codes.
"""

import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import subprocess
import tarfile
import hashlib
import os
import platform
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock
from unittest.mock import patch
from urllib.error import HTTPError


def is_windows():
    return platform.system() == "Windows"


sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1] / "artifact_cache"))

from _therock_utils.archive_util import open_archive_for_write
from _therock_utils.artifact_backend import (
    ArtifactBackend,
    LocalDirectoryBackend,
    S3Backend,
    HTTPSBackend,
    create_backend_from_uri,
)
from _therock_utils.storage_location import StorageLocation
from _therock_utils.archive_util import open_archive_for_write
from _therock_utils.workflow_outputs import WorkflowOutputRoot

import artifact_cache as cache
import artifact_manager
import buildctl

# Minimal topology TOML for testing push/fetch behavior.
# Defines two stages: upstream-stage produces artifacts, downstream-stage consumes them.
TEST_TOPOLOGY_TOML = """\
[metadata]
version = "2.0"
description = "Test topology for artifact_manager tests"

[build_stages.upstream-stage]
description = "Upstream stage that produces artifacts"
artifact_groups = ["upstream-group"]

[build_stages.second-upstream-stage]
description = "Second upstream stage that produces different artifacts"
artifact_groups = ["second-upstream-group"]

[build_stages.downstream-stage]
description = "Downstream stage that consumes artifacts"
artifact_groups = ["downstream-group"]

[artifact_groups.upstream-group]
description = "Upstream artifact group"
type = "generic"

[artifact_groups.second-upstream-group]
description = "Second upstream artifact group"
type = "generic"

[artifact_groups.downstream-group]
description = "Downstream artifact group"
type = "generic"
artifact_group_deps = ["upstream-group", "second-upstream-group"]

[artifacts.test-artifact]
artifact_group = "upstream-group"
type = "target-neutral"

[artifacts.second-artifact]
artifact_group = "second-upstream-group"
type = "target-neutral"

[artifacts.downstream-artifact]
artifact_group = "downstream-group"
type = "target-neutral"
artifact_deps = ["test-artifact", "second-artifact"]
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
            output_root = WorkflowOutputRoot.for_local(run_id=run_id, platform=platform)
            self._real_backend = LocalDirectoryBackend(
                staging_dir=staging_dir,
                output_root=output_root,
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

    def copy_artifact(self, artifact_key, source_backend):
        if self._real_backend:
            return self._real_backend.copy_artifact(artifact_key, source_backend)
        raise RuntimeError(f"No backend configured for copy: {artifact_key}")

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
        archive_content = b"fake zstd archive content"
        archive_path.write_bytes(archive_content)

        # Also create sha256sum
        sha_path = artifacts_dir / f"{archive_name}.sha256sum"
        sha256 = hashlib.sha256(archive_content).hexdigest()
        sha_path.write_text(f"{sha256}  {archive_name}\n")

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
        output_root = WorkflowOutputRoot.for_local(
            run_id=run_id, platform=TEST_PLATFORM
        )
        backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=output_root,
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
        archive_path = self._create_fake_precompressed_artifact(
            "test-artifact", "lib", "generic"
        )
        sha_path = Path(f"{archive_path}.sha256sum")
        self.assertTrue(sha_path.exists())

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
        output_root = WorkflowOutputRoot.for_local(
            run_id="local", platform=TEST_PLATFORM
        )
        backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=output_root,
        )
        self.assertTrue(backend.artifact_exists("test-artifact_lib_generic.tar.zst"))

        # Verify sha256sum was also uploaded
        self.assertTrue(
            backend.artifact_exists("test-artifact_lib_generic.tar.zst.sha256sum")
        )

    def test_push_succeeds_when_precompressed_artifact_is_missing_sha256sum(self):
        """Test that pre-compressed artifacts do not require a sha256sum sidecar."""
        archive_path = self._create_fake_precompressed_artifact(
            "test-artifact", "lib", "generic"
        )
        Path(f"{archive_path}.sha256sum").unlink()

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

        artifact_manager.main(argv)

        output_root = WorkflowOutputRoot.for_local(
            run_id="local", platform=TEST_PLATFORM
        )
        backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=output_root,
        )
        self.assertTrue(backend.artifact_exists("test-artifact_lib_generic.tar.zst"))
        self.assertFalse(
            backend.artifact_exists("test-artifact_lib_generic.tar.zst.sha256sum")
        )


class TestPushCompressionFailure(ArtifactManagerTestBase):
    """Tests that push command handles compression failures correctly."""

    @mock.patch("artifact_manager.compress_artifact")
    def test_push_fails_when_compression_fails(self, mock_compress):
        """Test that push exits with code 1 when compression fails."""
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


class TestPushStageAll(ArtifactManagerTestBase):
    """Tests that push --stage all collects produced artifacts from all stages."""

    def test_push_all_stages_uploads_all_produced_artifacts(self):
        """Test that --stage all pushes artifacts from every stage."""
        self._create_fake_precompressed_artifact("test-artifact", "lib", "generic")
        self._create_fake_precompressed_artifact("second-artifact", "lib", "generic")
        self._create_fake_precompressed_artifact(
            "downstream-artifact", "lib", "generic"
        )

        argv = [
            "push",
            "--stage",
            "all",
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

        artifact_manager.main(argv)

        output_root = WorkflowOutputRoot.for_local(
            run_id="local", platform=TEST_PLATFORM
        )
        backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=output_root,
        )
        for name in ["test-artifact", "second-artifact", "downstream-artifact"]:
            self.assertTrue(
                backend.artifact_exists(f"{name}_lib_generic.tar.zst"),
                f"{name} should be pushed with --stage all",
            )


class TestUploadArtifactFingerprints(ArtifactManagerTestBase):
    """Verify per-artifact fingerprint ordering and failure handling."""

    def setUp(self):
        super().setUp()
        self.backend = mock.create_autospec(ArtifactBackend, instance=True)
        delay_patcher = mock.patch("artifact_manager._delay_for_retry")
        self.delay = delay_patcher.start()
        self.addCleanup(delay_patcher.stop)

    def _make_request(
        self,
        archive_directory="artifacts",
        extension=".tar.zst",
        target_family="generic",
        with_checksum=True,
        with_fingerprint=True,
    ):
        stem = f"test-artifact_lib_{target_family}"
        archive_dir = self.build_dir / archive_directory
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{stem}{extension}"
        content = b"fake archive content"
        archive_path.write_bytes(content)
        if with_checksum:
            Path(f"{archive_path}.sha256sum").write_text(
                f"{hashlib.sha256(content).hexdigest()}  {archive_path.name}\n"
            )

        fingerprint_path = self.build_dir / "artifacts" / f"{stem}.fprint"
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        if with_fingerprint:
            fingerprint_path.write_text("test-fingerprint\n")
        else:
            fingerprint_path.unlink(missing_ok=True)

        request = artifact_manager.UploadRequest(
            source_path=archive_path,
            artifact_key=archive_path.name,
            backend=self.backend,
        )
        return request, fingerprint_path

    def _expected_calls(self, request, fingerprint_path, with_checksum=True):
        calls = [mock.call(request.source_path, request.artifact_key)]
        if with_checksum:
            calls.append(
                mock.call(
                    Path(f"{request.source_path}.sha256sum"),
                    f"{request.artifact_key}.sha256sum",
                )
            )
        calls.append(mock.call(fingerprint_path, fingerprint_path.name))
        return calls

    def test_fingerprint_upload_follows_archive_and_optional_checksum(self):
        for directory in ("artifacts", ".upload_cache"):
            for extension in (".tar.zst", ".tar.xz"):
                for family in ("generic", "gfx94X-dcgpu"):
                    for with_checksum in (True, False):
                        with self.subTest(
                            directory=directory,
                            extension=extension,
                            family=family,
                            with_checksum=with_checksum,
                        ):
                            self.backend.reset_mock()
                            request, fingerprint = self._make_request(
                                archive_directory=directory,
                                extension=extension,
                                target_family=family,
                                with_checksum=with_checksum,
                            )
                            if not with_checksum:
                                Path(f"{request.source_path}.sha256sum").unlink(
                                    missing_ok=True
                                )
                            self.assertTrue(artifact_manager.upload_artifact(request))
                            self.assertEqual(
                                self.backend.upload_artifact.call_args_list,
                                self._expected_calls(
                                    request, fingerprint, with_checksum
                                ),
                            )
        self.delay.assert_not_called()

    def test_missing_fingerprint_is_skipped_without_retry(self):
        request, fingerprint = self._make_request(with_fingerprint=False)
        with mock.patch("artifact_manager.log") as log:
            self.assertTrue(artifact_manager.upload_artifact(request))
        self.assertEqual(
            self.backend.upload_artifact.call_args_list,
            self._expected_calls(request, fingerprint)[:-1],
        )
        log.assert_any_call(
            f"  == Skipping fingerprint: missing or not a regular file: "
            f"{fingerprint}"
        )
        self.delay.assert_not_called()

    def test_archive_failure_never_uploads_fingerprint(self):
        request, _ = self._make_request()
        self.backend.upload_artifact.side_effect = OSError("Archive upload failed")
        self.assertFalse(artifact_manager.upload_artifact(request))
        self.assertEqual(
            self.backend.upload_artifact.call_args_list,
            [mock.call(request.source_path, request.artifact_key)] * 3,
        )
        self.assertEqual(self.delay.call_count, 2)

    def test_checksum_failure_never_uploads_fingerprint(self):
        request, fingerprint = self._make_request()

        def upload(source_path, artifact_key):
            if artifact_key.endswith(".sha256sum"):
                raise OSError("Checksum upload failed")

        self.backend.upload_artifact.side_effect = upload
        self.assertFalse(artifact_manager.upload_artifact(request))
        self.assertEqual(
            self.backend.upload_artifact.call_args_list,
            self._expected_calls(request, fingerprint)[:-1] * 3,
        )
        self.assertEqual(self.delay.call_count, 2)

    def test_transient_fingerprint_failure_retries_then_succeeds(self):
        request, fingerprint = self._make_request()
        self.backend.upload_artifact.side_effect = [
            None,
            None,
            OSError("Fingerprint upload failed"),
            None,
        ]
        with mock.patch(
            "artifact_manager.calculate_hash", wraps=artifact_manager.calculate_hash
        ) as calculate_hash:
            self.assertTrue(artifact_manager.upload_artifact(request))
        calculate_hash.assert_called_once_with(request.source_path, "sha256")
        expected = self._expected_calls(request, fingerprint)
        self.assertEqual(
            self.backend.upload_artifact.call_args_list,
            expected + [expected[-1]],
        )
        self.delay.assert_called_once_with(2)

    def test_persistent_fingerprint_failure_raises(self):
        request, fingerprint = self._make_request()
        upload_error = OSError("Fingerprint upload failed")

        def upload(source_path, artifact_key):
            if artifact_key.endswith(".fprint"):
                raise upload_error

        self.backend.upload_artifact.side_effect = upload

        with mock.patch(
            "artifact_manager.calculate_hash",
            wraps=artifact_manager.calculate_hash,
        ) as calculate_hash:
            with self.assertRaises(RuntimeError) as ctx:
                artifact_manager.upload_artifact(request)

        self.assertEqual(
            str(ctx.exception),
            f"Failed to upload {fingerprint.name} after 3 attempts",
        )
        self.assertIs(ctx.exception.__cause__, upload_error)

        calculate_hash.assert_called_once_with(request.source_path, "sha256")

        expected = self._expected_calls(request, fingerprint)
        self.assertEqual(
            self.backend.upload_artifact.call_args_list,
            expected[:-1] + [expected[-1]] * 3,
        )
        self.assertEqual(
            self.delay.call_args_list,
            [mock.call(2), mock.call(4)],
        )


class TestFetchFailures(ArtifactManagerTestBase):
    """Tests that fetch fails when required artifacts cannot be retrieved."""

    @mock.patch("artifact_manager.create_backend_from_env")
    def test_fetch_fails_when_no_artifacts_match(self, mock_backend_factory):
        """Test that fetch raises when the backend has no matches."""
        import artifact_manager

        mock_backend_factory.return_value = FailingBackend()
        argv = [
            "fetch",
            "--stage",
            "downstream-stage",
            "--output-dir",
            str(self.output_dir),
            "--topology",
            str(self.topology_path),
            "--platform",
            TEST_PLATFORM,
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "No matching artifacts found in failing://test-local-linux",
        ):
            artifact_manager.main(argv)

    @mock.patch("artifact_manager._delay_for_retry")
    @mock.patch("artifact_manager.create_backend_from_env")
    def test_fetch_fails_when_download_fails(self, mock_backend_factory, mock_delay):
        """Test that fetch exits with code 1 when download fails."""
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


class TestFetchStageAll(ArtifactManagerTestBase):
    """Tests that fetch --stage all fetches all artifacts in the topology."""

    def test_fetch_all_fetches_every_artifact(self):
        """Test that --stage all fetches artifacts from every stage."""
        self._create_staged_artifact("test-artifact", "lib", "generic")
        self._create_staged_artifact("second-artifact", "lib", "generic")
        self._create_staged_artifact("downstream-artifact", "lib", "generic")

        extract_calls = []

        def mock_extract(request):
            extract_calls.append(request)
            return request.output_dir

        with mock.patch("artifact_manager.extract_artifact", mock_extract):
            argv = [
                "fetch",
                "--stage",
                "all",
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

            artifact_manager.main(argv)

        fetched_names = {c.archive_path.stem.split("_")[0] for c in extract_calls}
        for name in ["test-artifact", "second-artifact", "downstream-artifact"]:
            self.assertIn(
                name,
                fetched_names,
                f"{name} should be fetched with --stage all",
            )

    def test_fetch_exclude_components_skips_test_artifacts(self) -> None:
        """Test that --exclude-components filters artifact components including xnack variants."""
        self._create_staged_artifact("test-artifact", "lib", "generic")
        self._create_staged_artifact("test-artifact", "test", "generic")
        self._create_staged_artifact("test-artifact", "test", "gfx942")
        # xnack artifacts use ':' which is invalid in Windows filenames
        if not is_windows():
            self._create_staged_artifact("test-artifact", "test", "gfx942:xnack+")
        self._create_staged_artifact("second-artifact", "run", "generic")
        self._create_staged_artifact("second-artifact", "test", "generic")

        extract_calls: list[artifact_manager.ExtractRequest] = []

        def mock_extract(request: artifact_manager.ExtractRequest) -> Path:
            extract_calls.append(request)
            return request.output_dir

        with mock.patch("artifact_manager.extract_artifact", mock_extract):
            argv = [
                "fetch",
                "--stage",
                "all",
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
                "--amdgpu-targets",
                "gfx942",
                "--exclude-components",
                "test",
            ]

            artifact_manager.main(argv)

        fetched_keys = [c.archive_path.name for c in extract_calls]
        self.assertIn("test-artifact_lib_generic.tar.zst", fetched_keys)
        self.assertIn("second-artifact_run_generic.tar.zst", fetched_keys)
        self.assertFalse(
            any("_test_" in key for key in fetched_keys),
            f"Should not fetch test artifacts, got: {fetched_keys}",
        )

    def test_fetch_exclude_components_rejects_unknown_component(self) -> None:
        """Test that --exclude-components validates component names."""
        self._create_staged_artifact("test-artifact", "lib", "generic")

        argv = [
            "fetch",
            "--stage",
            "all",
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
            "--exclude-components",
            "unknown",
        ]

        with self.assertRaises(ValueError):
            artifact_manager.main(argv)

    def test_fetch_exclude_artifacts_skips_named_artifacts(self) -> None:
        """Test that --exclude-artifacts filters whole artifacts."""
        self._create_staged_artifact("test-artifact", "lib", "generic")
        self._create_staged_artifact("second-artifact", "lib", "generic")

        extract_calls: list[artifact_manager.ExtractRequest] = []

        def mock_extract(request: artifact_manager.ExtractRequest) -> Path:
            extract_calls.append(request)
            return request.output_dir

        with mock.patch("artifact_manager.extract_artifact", mock_extract):
            argv = [
                "fetch",
                "--stage",
                "all",
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
                "--exclude-artifacts",
                "second-artifact",
            ]

            artifact_manager.main(argv)

        fetched_keys = [c.archive_path.name for c in extract_calls]
        self.assertIn("test-artifact_lib_generic.tar.zst", fetched_keys)
        self.assertNotIn("second-artifact_lib_generic.tar.zst", fetched_keys)

    def test_fetch_exclude_artifacts_rejects_unknown_artifact(self) -> None:
        """Test that --exclude-artifacts validates artifact names."""
        argv = [
            "fetch",
            "--stage",
            "all",
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
            "--exclude-artifacts",
            "unknown-artifact",
        ]

        with self.assertRaises(ValueError):
            artifact_manager.main(argv)


class TestFetchAmdgpuTargets(ArtifactManagerTestBase):
    """Tests that fetch command correctly handles --amdgpu-targets for split artifacts."""

    def test_fetch_with_amdgpu_targets_finds_individual_target_archives(self):
        """Test that --amdgpu-targets matches individual-target and xnack-suffixed archives."""
        # Stage generic, per-target, and xnack-suffixed artifacts
        self._create_staged_artifact("test-artifact", "lib", "generic")
        self._create_staged_artifact("test-artifact", "lib", "gfx942")
        # xnack artifacts use ':' which is invalid in Windows filenames
        if not is_windows():
            self._create_staged_artifact("test-artifact", "lib", "gfx942:xnack+")

        extract_calls = []

        def mock_extract(request):
            extract_calls.append(request)
            return request.output_dir

        with mock.patch("artifact_manager.extract_artifact", mock_extract):
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
                "--amdgpu-targets",
                "gfx942",
            ]

            artifact_manager.main(argv)

        # Should have fetched generic, gfx942, and gfx942:xnack+ (non-Windows only)
        fetched_keys = [c.archive_path.name for c in extract_calls]
        self.assertTrue(
            any("generic" in k for k in fetched_keys),
            f"Should fetch generic artifact, got: {fetched_keys}",
        )
        self.assertTrue(
            any("gfx942.tar.zst" in k for k in fetched_keys),
            f"Should fetch gfx942 artifact, got: {fetched_keys}",
        )
        if not is_windows():
            self.assertTrue(
                any("gfx942:xnack+" in k for k in fetched_keys),
                f"Should fetch gfx942:xnack+ artifact, got: {fetched_keys}",
            )

    def test_fetch_with_amdgpu_targets_skips_other_targets(self):
        """Test that --amdgpu-targets doesn't fetch archives for other targets."""
        # Stage artifacts for two different targets
        self._create_staged_artifact("test-artifact", "lib", "generic")
        self._create_staged_artifact("test-artifact", "lib", "gfx942")
        self._create_staged_artifact("test-artifact", "lib", "gfx1100")

        extract_calls = []

        def mock_extract(request):
            extract_calls.append(request)
            return request.output_dir

        with mock.patch("artifact_manager.extract_artifact", mock_extract):
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
                "--amdgpu-targets",
                "gfx942",
            ]

            artifact_manager.main(argv)

        fetched_keys = [c.archive_path.name for c in extract_calls]
        self.assertFalse(
            any("gfx1100" in k for k in fetched_keys),
            f"Should NOT fetch gfx1100 artifact, got: {fetched_keys}",
        )

    def test_fetch_with_families_and_targets_is_inclusive(self):
        """Test that --amdgpu-families and --amdgpu-targets together fetch both."""
        # Stage family-named and target-named artifacts
        self._create_staged_artifact("test-artifact", "lib", "generic")
        self._create_staged_artifact("test-artifact", "lib", "gfx94X-dcgpu")
        self._create_staged_artifact("test-artifact", "lib", "gfx942")

        extract_calls = []

        def mock_extract(request):
            extract_calls.append(request)
            return request.output_dir

        with mock.patch("artifact_manager.extract_artifact", mock_extract):
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
                "--amdgpu-families",
                "gfx94X-dcgpu",
                "--amdgpu-targets",
                "gfx942",
            ]

            artifact_manager.main(argv)

        fetched_keys = [c.archive_path.name for c in extract_calls]
        self.assertTrue(
            any("generic" in k for k in fetched_keys),
            f"Should fetch generic, got: {fetched_keys}",
        )
        self.assertTrue(
            any("gfx94X-dcgpu" in k for k in fetched_keys),
            f"Should fetch family archive, got: {fetched_keys}",
        )
        self.assertTrue(
            any("gfx942" in k for k in fetched_keys),
            f"Should fetch target archive, got: {fetched_keys}",
        )

    def test_fetch_with_semicolon_separated_families(self):
        """Test that --amdgpu-families accepts semicolon-separated values."""
        # Stage artifacts for two families plus generic
        self._create_staged_artifact("test-artifact", "lib", "generic")
        self._create_staged_artifact("test-artifact", "lib", "gfx94X-dcgpu")
        self._create_staged_artifact("test-artifact", "lib", "gfx110X-all")

        extract_calls = []

        def mock_extract(request):
            extract_calls.append(request)
            return request.output_dir

        with mock.patch("artifact_manager.extract_artifact", mock_extract):
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
                "--amdgpu-families",
                "gfx94X-dcgpu;gfx110X-all",
            ]

            artifact_manager.main(argv)

        fetched_keys = [c.archive_path.name for c in extract_calls]
        self.assertTrue(
            any("generic" in k for k in fetched_keys),
            f"Should fetch generic, got: {fetched_keys}",
        )
        self.assertTrue(
            any("gfx94X-dcgpu" in k for k in fetched_keys),
            f"Should fetch gfx94X-dcgpu, got: {fetched_keys}",
        )
        self.assertTrue(
            any("gfx110X-all" in k for k in fetched_keys),
            f"Should fetch gfx110X-all, got: {fetched_keys}",
        )


class TestFetchFlatten(ArtifactManagerTestBase):
    """Tests that fetch command correctly flattens artifacts."""

    def test_fetch_flatten_merges_artifacts_into_single_directory(self):
        """Test that fetch --flatten merges all artifacts into a single directory."""
        # Create two staged artifacts
        self._create_staged_artifact("test-artifact", "lib", "generic")
        self._create_staged_artifact("test-artifact", "run", "generic")

        # Mock extract_artifact to just verify it was called with flatten=True
        extract_calls = []

        def mock_extract(request):
            extract_calls.append(request)
            # Return success
            return request.output_dir

        with mock.patch("artifact_manager.extract_artifact", mock_extract):
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
                "--flatten",
            ]

            artifact_manager.main(argv)

            # Verify extract was called with flatten=True
            self.assertGreater(
                len(extract_calls), 0, "extract_artifact should have been called"
            )
            for request in extract_calls:
                self.assertTrue(
                    request.flatten,
                    "With --flatten, ExtractRequest.flatten should be True",
                )
                self.assertEqual(
                    request.output_dir,
                    self.output_dir,
                    "With --flatten, output should be directly to output_dir (not artifacts/ subdirectory)",
                )

    def test_fetch_without_flatten_creates_artifact_subdirectories(self):
        """Test that fetch without --flatten creates separate artifact subdirectories."""
        self._create_staged_artifact("test-artifact", "lib", "generic")

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

        with mock.patch("artifact_manager.extract_artifact") as mock_extract:
            # Make extract_artifact return success
            mock_extract.return_value = (
                self.output_dir / "artifacts" / "test-artifact_lib_generic"
            )

            artifact_manager.main(argv)

            # Verify extract_artifact was called
            mock_extract.assert_called_once()

            # Verify the ExtractRequest had correct output_dir (artifacts subdirectory)
            call_args = mock_extract.call_args[0][0]
            self.assertEqual(
                call_args.output_dir,
                self.output_dir / "artifacts",
                "Without --flatten, artifacts should be extracted to 'artifacts/' subdirectory",
            )
            self.assertFalse(
                call_args.flatten,
                "Without --flatten, ExtractRequest.flatten should be False",
            )

    def test_flatten_and_bootstrap_are_mutually_exclusive(self):
        """Test that --flatten and --bootstrap cannot be used together."""
        self._create_staged_artifact("test-artifact", "lib", "generic")

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
            "--flatten",
            "--bootstrap",
        ]

        # argparse should reject this combination
        with self.assertRaises(SystemExit) as ctx:
            artifact_manager.main(argv)

        # Exit code 2 is used by argparse for command-line syntax errors
        self.assertEqual(ctx.exception.code, 2)


class TestFetchExtractionCache(ArtifactManagerTestBase):
    """Tests for reusing extracted artifacts in flattened output trees."""

    def _create_real_artifact(self) -> Path:
        artifact_dir = Path(self.temp_dir) / "artifact"
        stage_dir = artifact_dir / "component" / "stage"
        stage_dir.mkdir(parents=True)
        manifest_path = artifact_dir / "artifact_manifest.txt"
        manifest_path.write_text("component/stage\n")

        regular_file = stage_dir / "regular.txt"
        regular_file.write_text("artifact contents")
        executable_file = stage_dir / "executable"
        executable_file.write_text("#!/bin/sh\n")
        executable_file.chmod(0o755)
        os.link(regular_file, stage_dir / "regular-hardlink.txt")
        if not is_windows():
            os.symlink("regular.txt", stage_dir / "regular-symlink.txt")

        archive_path = Path(self.temp_dir) / "test-artifact_lib_generic.tar.zst"
        with open_archive_for_write(archive_path, "zstd") as archive:
            # ArtifactPopulator requires the manifest to be the first member.
            archive.add(manifest_path, arcname="artifact_manifest.txt")
            for filename in (
                "regular.txt",
                "regular-hardlink.txt",
                "executable",
                "regular-symlink.txt",
            ):
                path = stage_dir / filename
                if path.exists() or path.is_symlink():
                    archive.add(path, arcname=f"component/stage/{filename}")
        return archive_path

    def test_second_flatten_reuses_extracted_files(self):
        import artifact_manager

        archive_path = self._create_real_artifact()
        extraction_cache_dir = Path(self.temp_dir) / "extraction-cache"
        direct_output = Path(self.temp_dir) / "direct-output"
        first_output = Path(self.temp_dir) / "first-output"
        second_output = Path(self.temp_dir) / "second-output"

        direct_result = artifact_manager.extract_artifact(
            artifact_manager.ExtractRequest(
                archive_path=archive_path,
                output_dir=direct_output,
                delete_archive=False,
                flatten=True,
            )
        )
        self.assertEqual(direct_result, direct_output)

        first_result = artifact_manager.extract_artifact(
            artifact_manager.ExtractRequest(
                archive_path=archive_path,
                output_dir=first_output,
                delete_archive=False,
                flatten=True,
                extraction_cache_dir=extraction_cache_dir,
            )
        )
        self.assertEqual(first_result, first_output)

        # Prove the second flatten no longer needs the compressed archive.
        archive_path.unlink()
        second_result = artifact_manager.extract_artifact(
            artifact_manager.ExtractRequest(
                archive_path=archive_path,
                output_dir=second_output,
                delete_archive=False,
                flatten=True,
                extraction_cache_dir=extraction_cache_dir,
            )
        )
        self.assertEqual(second_result, second_output)

        cached_stage = extraction_cache_dir / archive_path.name / "component" / "stage"
        # Cached extraction must preserve direct extraction's mode.
        direct_executable_mode = stat.S_IMODE(
            (direct_output / "executable").stat().st_mode
        )
        for output_dir in (direct_output, first_output, second_output):
            self.assertEqual(
                (output_dir / "regular.txt").read_text(), "artifact contents"
            )
            self.assertEqual(
                stat.S_IMODE((output_dir / "executable").stat().st_mode),
                direct_executable_mode,
            )
            self.assertTrue(
                os.path.samefile(
                    output_dir / "regular.txt",
                    output_dir / "regular-hardlink.txt",
                )
            )
            if output_dir != direct_output:
                self.assertTrue(
                    os.path.samefile(
                        output_dir / "regular.txt", cached_stage / "regular.txt"
                    )
                )
            if not is_windows():
                self.assertEqual(
                    os.readlink(output_dir / "regular-symlink.txt"), "regular.txt"
                )

    def test_incomplete_cache_entry_is_rebuilt(self):
        """A cache entry without its completion manifest is replaced."""
        import artifact_manager

        archive_path = self._create_real_artifact()
        extraction_cache_dir = Path(self.temp_dir) / "extraction-cache"
        cached_artifact_dir = extraction_cache_dir / archive_path.name
        cached_artifact_dir.mkdir(parents=True)
        stale_file = cached_artifact_dir / "stale.txt"
        stale_file.write_text("incomplete extraction")
        output_dir = Path(self.temp_dir) / "output"

        result = artifact_manager.extract_artifact(
            artifact_manager.ExtractRequest(
                archive_path=archive_path,
                output_dir=output_dir,
                delete_archive=False,
                flatten=True,
                extraction_cache_dir=extraction_cache_dir,
            )
        )

        self.assertEqual(result, output_dir)
        self.assertFalse(stale_file.exists())
        self.assertEqual(
            (cached_artifact_dir / "artifact_manifest.txt").read_text(),
            "component/stage\n",
        )
        cached_regular_file = (
            cached_artifact_dir / "component" / "stage" / "regular.txt"
        )
        self.assertEqual(cached_regular_file.read_text(), "artifact contents")
        self.assertTrue(
            os.path.samefile(output_dir / "regular.txt", cached_regular_file)
        )


class TestFetchDownloadCache(ArtifactManagerTestBase):
    """Tests for --download-cache-dir behavior."""

    def test_shared_cache_skips_redownload(self):
        """Second fetch with same --download-cache-dir skips already-cached archives.

        Uses a FailingBackend for the second fetch — if the cache check in
        download_artifact() works, the backend's download method is never
        reached, so the failing backend won't cause errors. If the cache
        check is removed, the second fetch hits the FailingBackend and fails.
        """
        self._create_staged_artifact("test-artifact", "lib", "generic")

        cache_dir = Path(self.temp_dir) / "shared-cache"
        cache_dir.mkdir()

        def mock_extract(request):
            return request.output_dir

        base_argv = [
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
            "--flatten",
            "--download-cache-dir",
            str(cache_dir),
        ]

        # First fetch — downloads to cache
        with mock.patch("artifact_manager.extract_artifact", mock_extract):
            artifact_manager.main(base_argv)

        first_cached = list(cache_dir.glob("*.tar.zst"))
        self.assertGreater(len(first_cached), 0)

        # Second fetch with a FailingBackend. If the cache check works,
        # download_artifact() returns early before calling the backend,
        # so the failing backend is never hit.
        failing_backend = FailingBackend(fail_downloads_after=0)
        # Give it the same listing so artifact matching works
        failing_backend.list_artifacts = lambda name_filter=None: [
            f.name for f in first_cached
        ]

        second_output = Path(self.temp_dir) / "output2"
        second_output.mkdir()
        second_argv = base_argv.copy()
        idx = second_argv.index("--output-dir") + 1
        second_argv[idx] = str(second_output)

        with (
            mock.patch("artifact_manager.extract_artifact", mock_extract),
            mock.patch(
                "artifact_manager.create_backend_from_env",
                return_value=failing_backend,
            ),
        ):
            # Should succeed — cache hits bypass the failing backend
            artifact_manager.main(second_argv)


class TestCopy(ArtifactManagerTestBase):
    """Tests for the copy subcommand."""

    def _create_source_artifact(
        self, name: str, component: str, target_family: str, run_id: str = "source-run"
    ) -> str:
        """Create a fake artifact in the source run's staging area."""
        return self._create_staged_artifact(name, component, target_family, run_id)

    def _run_copy(
        self, extra_argv=None, source_run_id="source-run", dest_run_id="dest-run"
    ):
        """Run the copy command with standard arguments."""
        argv = [
            "copy",
            "--source-run-id",
            source_run_id,
            "--stage",
            "upstream-stage",
            "--topology",
            str(self.topology_path),
            "--local-staging-dir",
            str(self.staging_dir),
            "--platform",
            TEST_PLATFORM,
            "--run-id",
            dest_run_id,
        ]
        if extra_argv:
            argv.extend(extra_argv)

        artifact_manager.main(argv)

    @mock.patch("artifact_manager._delay_for_retry")
    def test_copy_transfers_artifacts_between_runs(self, mock_delay):
        """Test that copy moves produced artifacts from source to dest run."""
        self._create_source_artifact("test-artifact", "lib", "generic")

        self._run_copy()

        # Verify artifact exists in dest
        dest_backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=WorkflowOutputRoot.for_local(
                run_id="dest-run", platform=TEST_PLATFORM
            ),
        )
        self.assertTrue(
            dest_backend.artifact_exists("test-artifact_lib_generic.tar.zst")
        )

    @mock.patch("artifact_manager._delay_for_retry")
    def test_copy_transfers_sha256sum_files(self, mock_delay):
        """Test that copy also transfers sha256sum files (best-effort)."""
        self._create_source_artifact("test-artifact", "lib", "generic")

        # Also create a sha256sum file in the source
        source_backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=WorkflowOutputRoot.for_local(
                run_id="source-run", platform=TEST_PLATFORM
            ),
        )
        sha_path = (
            source_backend.base_path / "test-artifact_lib_generic.tar.zst.sha256sum"
        )
        sha_path.write_text("abc123  test-artifact_lib_generic.tar.zst\n")

        self._run_copy()

        dest_backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=WorkflowOutputRoot.for_local(
                run_id="dest-run", platform=TEST_PLATFORM
            ),
        )
        self.assertTrue(
            (
                dest_backend.base_path / "test-artifact_lib_generic.tar.zst.sha256sum"
            ).exists()
        )

    @mock.patch("artifact_manager._delay_for_retry")
    def test_copy_multiple_components(self, mock_delay):
        """Test that copy handles multiple components of the same artifact."""
        self._create_source_artifact("test-artifact", "lib", "generic")
        self._create_source_artifact("test-artifact", "dev", "generic")
        self._create_source_artifact("test-artifact", "run", "generic")

        self._run_copy()

        dest_backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=WorkflowOutputRoot.for_local(
                run_id="dest-run", platform=TEST_PLATFORM
            ),
        )
        for comp in ["lib", "dev", "run"]:
            self.assertTrue(
                dest_backend.artifact_exists(f"test-artifact_{comp}_generic.tar.zst"),
                f"Component {comp} should be copied",
            )

    @mock.patch("artifact_manager._delay_for_retry")
    def test_copy_only_copies_produced_artifacts(self, mock_delay):
        """Test that copy only copies artifacts produced by the specified stage."""
        # Stage artifacts for both upstream and downstream stages
        self._create_source_artifact("test-artifact", "lib", "generic")
        self._create_source_artifact("downstream-artifact", "lib", "generic")

        self._run_copy()

        dest_backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=WorkflowOutputRoot.for_local(
                run_id="dest-run", platform=TEST_PLATFORM
            ),
        )
        # upstream-stage produces test-artifact
        self.assertTrue(
            dest_backend.artifact_exists("test-artifact_lib_generic.tar.zst")
        )
        # upstream-stage does NOT produce downstream-artifact
        self.assertFalse(
            dest_backend.artifact_exists("downstream-artifact_lib_generic.tar.zst")
        )

    @mock.patch("artifact_manager._delay_for_retry")
    def test_copy_dry_run_does_not_copy(self, mock_delay):
        """Test that --dry-run lists artifacts without copying."""
        self._create_source_artifact("test-artifact", "lib", "generic")

        self._run_copy(extra_argv=["--dry-run"])

        dest_backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=WorkflowOutputRoot.for_local(
                run_id="dest-run", platform=TEST_PLATFORM
            ),
        )
        self.assertFalse(
            dest_backend.artifact_exists("test-artifact_lib_generic.tar.zst")
        )

    @mock.patch("artifact_manager._delay_for_retry")
    @mock.patch("artifact_manager._create_source_backend")
    @mock.patch("artifact_manager.create_backend_from_env")
    def test_copy_fails_when_copy_fails(
        self, mock_dest_factory, mock_source_factory, mock_delay
    ):
        """Test that copy exits with code 1 when artifact copy fails."""
        # Source backend has the artifact
        source_backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=WorkflowOutputRoot.for_local(
                run_id="source-run", platform=TEST_PLATFORM
            ),
        )
        self._create_source_artifact("test-artifact", "lib", "generic")
        mock_source_factory.return_value = source_backend

        # Dest backend will fail on copy
        failing_dest = FailingBackend(fail_uploads_after=0)
        # Override copy_artifact to always fail
        failing_dest.copy_artifact = mock.Mock(
            side_effect=RuntimeError("Simulated copy failure")
        )
        failing_dest.list_artifacts = source_backend.list_artifacts
        mock_dest_factory.return_value = failing_dest

        argv = [
            "copy",
            "--source-run-id",
            "source-run",
            "--stage",
            "upstream-stage",
            "--topology",
            str(self.topology_path),
            "--platform",
            TEST_PLATFORM,
            "--run-id",
            "dest-run",
        ]

        with self.assertRaises(SystemExit) as ctx:
            artifact_manager.main(argv)

        self.assertEqual(ctx.exception.code, 1)

    def test_copy_invalid_stage_exits_with_error(self):
        """Test that copy exits with code 1 for invalid stage name."""
        argv = [
            "copy",
            "--source-run-id",
            "source-run",
            "--stage",
            "nonexistent-stage",
            "--topology",
            str(self.topology_path),
            "--local-staging-dir",
            str(self.staging_dir),
            "--platform",
            TEST_PLATFORM,
            "--run-id",
            "dest-run",
        ]

        with self.assertRaises(SystemExit) as ctx:
            artifact_manager.main(argv)

        self.assertEqual(ctx.exception.code, 1)

    def test_copy_missing_source_run_id_exits_with_error(self):
        """Test that copy exits with code 2 (argparse) when --source-run-id is missing."""
        argv = [
            "copy",
            "--stage",
            "upstream-stage",
            "--topology",
            str(self.topology_path),
            "--local-staging-dir",
            str(self.staging_dir),
            "--platform",
            TEST_PLATFORM,
        ]

        with self.assertRaises(SystemExit) as ctx:
            artifact_manager.main(argv)

        self.assertEqual(ctx.exception.code, 2)

    @mock.patch("artifact_manager._delay_for_retry")
    def test_copy_multiple_stages(self, mock_delay):
        """Test that copy with comma-separated stages copies artifacts from all stages."""
        self._create_source_artifact("test-artifact", "lib", "generic")
        self._create_staged_artifact(
            "second-artifact", "lib", "generic", run_id="source-run"
        )

        artifact_manager.main(
            [
                "copy",
                "--source-run-id",
                "source-run",
                "--stage",
                "upstream-stage,second-upstream-stage",
                "--topology",
                str(self.topology_path),
                "--local-staging-dir",
                str(self.staging_dir),
                "--platform",
                TEST_PLATFORM,
                "--run-id",
                "dest-run",
            ]
        )

        dest_backend = LocalDirectoryBackend(
            staging_dir=self.staging_dir,
            output_root=WorkflowOutputRoot.for_local(
                run_id="dest-run", platform=TEST_PLATFORM
            ),
        )
        self.assertTrue(
            dest_backend.artifact_exists("test-artifact_lib_generic.tar.zst")
        )
        self.assertTrue(
            dest_backend.artifact_exists("second-artifact_lib_generic.tar.zst")
        )


class ParseTargetFamiliesTest(unittest.TestCase):
    """Unit tests for artifact_manager.parse_target_families."""

    def setUp(self):
        self.am = artifact_manager

    def _make_args(
        self,
        amdgpu_families: str = "",
        amdgpu_targets: str = "",
        generic_only: bool = False,
        expand_family_to_targets: bool = False,
    ):
        return argparse.Namespace(
            amdgpu_families=amdgpu_families,
            amdgpu_targets=amdgpu_targets,
            generic_only=generic_only,
            expand_family_to_targets=expand_family_to_targets,
        )

    def test_generic_only(self):
        args = self._make_args(generic_only=True)
        result = self.am.parse_target_families(args)
        self.assertEqual(result, ["generic"])

    def test_family_no_expansion(self):
        args = self._make_args(amdgpu_families="gfx110X-all")
        result = self.am.parse_target_families(args)
        self.assertEqual(result, ["generic", "gfx110X-all"])
        # No per-target entries without the flag.
        self.assertNotIn("gfx1100", result)

    def test_family_with_expansion_multi_target(self):
        fake_map = {"gfx110X-all": ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]}
        with mock.patch.object(self.am, "amdgpu_family_map", return_value=fake_map):
            args = self._make_args(
                amdgpu_families="gfx110X-all", expand_family_to_targets=True
            )
            result = self.am.parse_target_families(args)
        self.assertIn("generic", result)
        self.assertIn("gfx110X-all", result)
        for t in ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]:
            self.assertIn(t, result)

    def test_family_with_expansion_single_target(self):
        # A single-target family: the target should appear once even though
        # it also matches its own self-family name.
        fake_map = {"gfx942": ["gfx942"], "gfx94X-dcgpu": ["gfx942"]}
        with mock.patch.object(self.am, "amdgpu_family_map", return_value=fake_map):
            args = self._make_args(
                amdgpu_families="gfx94X-dcgpu", expand_family_to_targets=True
            )
            result = self.am.parse_target_families(args)
        self.assertEqual(result.count("gfx942"), 1)

    def test_unknown_family_skipped(self):
        # A family not in the cmake file should not crash; just the family name
        # stays in the list.
        fake_map = {}
        with mock.patch.object(self.am, "amdgpu_family_map", return_value=fake_map):
            args = self._make_args(
                amdgpu_families="custom-family", expand_family_to_targets=True
            )
            result = self.am.parse_target_families(args)
        self.assertIn("custom-family", result)

    def test_explicit_targets_still_added(self):
        args = self._make_args(
            amdgpu_families="gfx110X-all", amdgpu_targets="gfx1100,gfx1101"
        )
        result = self.am.parse_target_families(args)
        self.assertIn("gfx110X-all", result)
        self.assertIn("gfx1100", result)
        self.assertIn("gfx1101", result)


class FamilyArtifactSelectionTest(unittest.TestCase):
    """Exercise the family-expansion path used by packaging workflows."""

    def setUp(self):
        self.available = {
            "rand_lib_generic.tar.zst",
            "rand_lib_gfx1250.tar.zst",
            "rand_lib_gfx1250-strict.tar.zst",
            "rand_lib_gfx125X-all.tar.zst",
            "rand_lib_gfx942.tar.zst",
        }

    def _select(self, families: str) -> set[str]:
        args = argparse.Namespace(
            amdgpu_families=families,
            amdgpu_targets="",
            generic_only=False,
            expand_family_to_targets=True,
        )
        targets = artifact_manager.parse_target_families(args)
        return set(
            artifact_manager.find_available_artifacts({"rand"}, targets, self.available)
        )

    def test_gfx1250_does_not_select_gfx1250_strict(self):
        self.assertEqual(
            self._select("gfx1250"),
            {"rand_lib_generic.tar.zst", "rand_lib_gfx1250.tar.zst"},
        )

    def test_gfx1250_strict_does_not_select_gfx1250(self):
        self.assertEqual(
            self._select("gfx1250-strict"),
            {"rand_lib_generic.tar.zst", "rand_lib_gfx1250-strict.tar.zst"},
        )

    def test_combined_selection_preserves_both_targets(self):
        self.assertEqual(
            self._select("gfx1250;gfx1250-strict"),
            {
                "rand_lib_generic.tar.zst",
                "rand_lib_gfx1250.tar.zst",
                "rand_lib_gfx1250-strict.tar.zst",
            },
        )

    def test_broad_family_selects_family_and_enabled_target_artifacts(self):
        self.assertEqual(
            self._select("gfx125X-all"),
            {
                "rand_lib_generic.tar.zst",
                "rand_lib_gfx125X-all.tar.zst",
                "rand_lib_gfx1250.tar.zst",
                "rand_lib_gfx1250-strict.tar.zst",
            },
        )

    def test_explicit_target_can_supplement_broad_family(self):
        self.assertEqual(
            self._select("gfx125X-all;gfx942"),
            {
                "rand_lib_generic.tar.zst",
                "rand_lib_gfx125X-all.tar.zst",
                "rand_lib_gfx1250.tar.zst",
                "rand_lib_gfx1250-strict.tar.zst",
                "rand_lib_gfx942.tar.zst",
            },
        )


class BootstrapMarkerTest(unittest.TestCase):
    """End-to-end tests for where bootstrapping writes ".prebuilt" markers.

    Covers both BootstrappingPopulator implementations: artifact_manager.py
    (CI fetch) and buildctl.py (local bootstrap from an artifacts dir).
    """

    def setUp(self):
        self.populator_factories = {
            "artifact_manager": artifact_manager.BootstrappingPopulator,
            "buildctl": buildctl.BootstrappingPopulator,
        }
        self.temp_dir = Path(tempfile.mkdtemp(prefix="bootstrap_marker_test_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_archive(self, tag: str, basedir: str, relfiles) -> Path:
        """Write a minimal artifact archive declaring a single manifest basedir."""
        content_dir = self.temp_dir / "content" / tag
        for relfile in relfiles:
            p = content_dir / basedir / relfile
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(relfile)
        manifest = content_dir / "artifact_manifest.txt"
        manifest.write_text(basedir + "\n")

        archive_path = self.temp_dir / f"{tag}.tar.xz"
        with open_archive_for_write(archive_path, "xz") as tf:
            # artifact_manifest.txt must be the first member.
            tf.add(str(manifest), arcname="artifact_manifest.txt")
            for relfile in relfiles:
                tf.add(
                    str(content_dir / basedir / relfile),
                    arcname=f"{basedir}/{relfile}",
                )
        return archive_path

    def _bootstrap(self, impl: str, basedir: str, relfiles) -> Path:
        tag = f"{impl}_lib_generic"
        archive = self._make_archive(tag, basedir, relfiles)
        output_dir = self.temp_dir / f"build-{impl}"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.populator_factories[impl](output_path=output_dir)(archive)
        return output_dir

    def test_nested_basedir_marks_enclosing_stage_dir(self):
        """Regression test for a basedir nested below the subproject stage dir.

        `dctools/artifact-rdc.toml` declares
        "dctools/rdc/stage/portable-rdc". Naming the marker after that basedir
        yields "dctools/rdc/stage/portable-rdc.prebuilt", which
        therock_subproject.cmake never checks, so rdc is rebuilt from source
        despite its artifact having been fetched and extracted.
        """
        for impl in self.populator_factories:
            with self.subTest(impl=impl):
                output_dir = self._bootstrap(
                    impl,
                    "dctools/rdc/stage/portable-rdc",
                    ["bin/rdcd", "lib/librdc.so"],
                )
                stage_dir = output_dir / "dctools" / "rdc" / "stage"

                # The marker the build looks for: "${_stage_dir}.prebuilt".
                self.assertTrue(
                    (output_dir / "dctools" / "rdc" / "stage.prebuilt").exists(),
                    "Bootstrap marker not created beside the subproject stage dir",
                )
                self.assertFalse(
                    (stage_dir / "portable-rdc.prebuilt").exists(),
                    "Marker written at a path the build never checks",
                )
                # Extraction is unaffected: content still lands in the basedir.
                self.assertTrue((stage_dir / "portable-rdc" / "bin" / "rdcd").exists())
                self.assertTrue(
                    (stage_dir / "portable-rdc" / "lib" / "librdc.so").exists()
                )

    def test_stage_basedir_marker_unchanged(self):
        """The ~100 basedirs that are stage dirs keep their existing marker."""
        for impl in self.populator_factories:
            with self.subTest(impl=impl):
                output_dir = self._bootstrap(
                    impl, "core/ROCR-Runtime/stage", ["lib/libhsa-runtime64.so"]
                )
                subproject_dir = output_dir / "core" / "ROCR-Runtime"

                self.assertTrue((subproject_dir / "stage.prebuilt").exists())
                self.assertTrue(
                    (subproject_dir / "stage" / "lib" / "libhsa-runtime64.so").exists()
                )

    def test_basedir_without_stage_component_marker_unchanged(self):
        """A basedir with no "stage" component is left alone."""
        for impl in self.populator_factories:
            with self.subTest(impl=impl):
                output_dir = self._bootstrap(
                    impl, "math-libs/hipthreads/build", ["lib/libhipthreads.so"]
                )

                self.assertTrue(
                    (
                        output_dir / "math-libs" / "hipthreads" / "build.prebuilt"
                    ).exists()
                )


class TestPopulate(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="therock-population-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source with spaces"
        self.source.mkdir()
        self.staging = self.root / "outputs"
        self.destination = self.staging / "456-linux"
        environment = patch.dict(
            os.environ, {"THEROCK_LOCAL_STAGING_DIR": str(self.staging)}
        )
        environment.start()
        self.addCleanup(environment.stop)
        retry_delay = patch.object(artifact_manager, "_delay_for_retry")
        retry_delay.start()
        self.addCleanup(retry_delay.stop)
        self.plan_path = self.root / "plan.json"
        self.result_path = self.root / "result.json"
        self.fingerprint = "a" * 64
        for artifact in cache.LLVM_ARTIFACTS:
            archive = self.source / (artifact + ".tar.xz")
            payload = f"{artifact}: original payload\n".encode()
            with tarfile.open(archive, "w:xz") as output:
                info = tarfile.TarInfo("share/probe.txt")
                info.size = len(payload)
                output.addfile(info, io.BytesIO(payload))
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            (self.source / (archive.name + ".sha256sum")).write_text(
                f"{checksum}  {archive.name}\n", encoding="utf-8"
            )
            (self.source / (artifact + ".fprint")).write_text(
                self.fingerprint + "\n", encoding="utf-8"
            )
        self.index = cache.record_artifacts_from_ci_run(
            index=cache.CacheIndex(schema_version=1, entries=[]),
            metadata_dir=self.source,
            platform="linux",
            repository="ROCm/TheRock",
            run_id="123",
            root=self.source.as_uri(),
            archive_suffix=".tar.xz",
        )
        self.plan = cache.lookup_artifacts_by_fingerprint(
            self.index, self.source, "linux"
        )
        self.save_plan()

    def save_plan(self) -> None:
        cache.write_json(self.plan_path, self.plan)

    def cli(self, *extra: str) -> tuple[int, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                artifact_manager.main(
                    [
                        "populate",
                        "--plan",
                        str(self.plan_path),
                        "--output",
                        str(self.result_path),
                        "--platform",
                        "linux",
                        "--local-staging-dir",
                        str(self.staging),
                        "--run-id",
                        "456",
                        *extra,
                    ]
                )
                code = 0
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue() + stderr.getvalue()

    def assert_failure(self, expected: str) -> None:
        self.result_path.write_text(
            '{"status":"ready","action":"reuse"}', encoding="utf-8"
        )
        code, output = self.cli()
        self.assertEqual(code, 2, output)
        self.assertIn(expected, output)
        self.assertFalse(
            self.result_path.exists(), "Failure must remove an earlier ready result"
        )

    def assert_complete(self) -> None:
        result = json.loads(self.result_path.read_text())
        self.assertEqual((result["status"], result["action"]), ("ready", "reuse"))
        self.assertEqual(set(result["artifacts"]), set(cache.LLVM_ARTIFACTS))
        for request in self.plan.copy_requests:
            for filename in (
                request.source.archive,
                request.checksum_file,
                request.fingerprint_file,
            ):
                self.assertEqual(
                    (self.destination / filename).read_bytes(),
                    (self.source / filename).read_bytes(),
                )

    def test_local_copy_verifies_complete_bundle_and_repeats(self) -> None:
        original_plan = self.plan_path.read_bytes()
        self.destination.mkdir(parents=True)
        unrelated = self.destination / "other-project.tar.xz"
        unrelated.write_bytes(b"other project")
        for _ in range(2):
            code, output = self.cli()
            self.assertEqual(code, 0, output)
            self.assert_complete()
            self.assertEqual(self.plan_path.read_bytes(), original_plan)
            self.assertEqual(unrelated.read_bytes(), b"other project")

    def test_cli_runs_as_script_without_aws_dependencies(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                artifact_manager.__file__,
                "populate",
                "--plan",
                str(self.plan_path),
                "--local-staging-dir",
                str(self.staging),
                "--run-id",
                "456",
                "--platform",
                "linux",
                "--output",
                str(self.result_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_complete()

    def test_miss_has_no_io_and_clears_previous_ready_result(self) -> None:
        self.plan.status, self.plan.action, self.plan.copy_requests = (
            "miss",
            "build",
            [],
        )
        self.save_plan()
        self.result_path.write_text('{"status":"ready"}', encoding="utf-8")
        with patch.object(artifact_manager, "create_backend_from_uri") as download:
            code, output = self.cli()
        self.assertEqual(code, 0, output)
        download.assert_not_called()
        self.assertFalse(any(self.destination.glob("*")))
        result = json.loads(self.result_path.read_text())
        self.assertEqual(
            (result["status"], result["action"], result["artifacts"]),
            ("miss", "build", []),
        )

    def test_incomplete_duplicate_unsafe_and_incompatible_plans_fail_before_copy(
        self,
    ) -> None:
        original = cache.to_json(self.plan)
        changes = {
            "missing component": lambda data: data["copy_requests"].pop(),
            "duplicate component": lambda data: data["copy_requests"].__setitem__(
                0, data["copy_requests"][1]
            ),
            "mixed fingerprints": lambda data: data["copy_requests"][0].__setitem__(
                "fingerprint", "b" * 64
            ),
            "unsafe archive": lambda data: data["copy_requests"][0][
                "source"
            ].__setitem__("archive", "../outside.tar.xz"),
            "unsafe sidecar": lambda data: data["copy_requests"][0].__setitem__(
                "fingerprint_file", "../outside"
            ),
            "platform": lambda data: data.__setitem__("platform", "windows"),
            "schema": lambda data: data.__setitem__("schema_version", True),
            "action": lambda data: data.__setitem__("action", "build"),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                data = copy.deepcopy(original)
                change(data)
                self.plan_path.write_text(json.dumps(data))
                with patch.object(
                    artifact_manager, "create_backend_from_uri"
                ) as download:
                    self.assert_failure("ERROR:")
                    download.assert_not_called()
                self.assertFalse(any(self.destination.glob("*")))

    def test_missing_or_invalid_plan_removes_stale_result(self) -> None:
        for contents in (None, "not json"):
            with self.subTest(contents=contents):
                if contents is None:
                    self.plan_path.unlink(missing_ok=True)
                else:
                    self.plan_path.write_text(contents)
                self.assert_failure("ERROR:")

    def test_missing_or_mismatched_metadata_prevents_any_copy(self) -> None:
        request = self.plan.copy_requests[-1]
        for filename, contents in (
            (request.fingerprint_file, None),
            (request.fingerprint_file, b"b" * 64),
            (request.checksum_file, None),
            (request.checksum_file, b"b" * 64),
            (request.checksum_file, b""),
            (request.checksum_file, f"{request.source.sha256} wrong.tar.xz".encode()),
        ):
            with self.subTest(filename=filename, contents=contents):
                path = self.source / filename
                original = path.read_bytes()
                if contents is None:
                    path.unlink()
                else:
                    path.write_bytes(contents)
                try:
                    self.assert_failure("ERROR:")
                    self.assertFalse(any(self.destination.glob("*")))
                finally:
                    path.write_bytes(original)

    def test_missing_and_corrupt_archive_do_not_authorize_reuse(self) -> None:
        path = self.source / self.plan.copy_requests[1].source.archive
        original = path.read_bytes()
        for contents in (None, b"corrupt archive"):
            with self.subTest(contents=contents):
                if contents is None:
                    path.unlink()
                else:
                    path.write_bytes(contents)
                self.assert_failure("ERROR:")
                self.assertFalse(any(self.destination.glob("*.fprint")))
                path.write_bytes(original)
        code, output = self.cli()
        self.assertEqual(code, 0, output)
        self.assert_complete()

    def test_copy_failure_then_retry_verifies_all_components(self) -> None:
        original_copy = LocalDirectoryBackend.copy_artifact
        fail_name = self.plan.copy_requests[-1].fingerprint_file

        def fail_last(destination, key, source):
            if key == fail_name:
                raise OSError("simulated interrupted copy")
            return original_copy(destination, key, source)

        with patch.object(LocalDirectoryBackend, "copy_artifact", fail_last):
            self.assert_failure("simulated interrupted copy")
        code, output = self.cli()
        self.assertEqual(code, 0, output)
        self.assert_complete()

    def test_destination_corruption_is_detected_for_archives_and_sidecars(self) -> None:
        original_copy = LocalDirectoryBackend.copy_artifact
        request = self.plan.copy_requests[0]
        for corrupt_name in (
            request.source.archive,
            request.checksum_file,
            request.fingerprint_file,
        ):

            def corrupt(destination, key, source):
                original_copy(destination, key, source)
                if key == corrupt_name:
                    (destination.base_path / key).write_bytes(
                        b"corrupted at destination"
                    )

            with self.subTest(filename=corrupt_name):
                with patch.object(LocalDirectoryBackend, "copy_artifact", corrupt):
                    self.assert_failure("destination")

    def test_output_must_not_overwrite_input_plan_or_artifact(self) -> None:
        original = self.plan_path.read_bytes()
        code, output = self.cli("--output", str(self.plan_path))
        self.assertEqual(code, 2, output)
        self.assertEqual(self.plan_path.read_bytes(), original)
        request = self.plan.copy_requests[0]
        for filename in (request.fingerprint_file, request.source.archive):
            path = self.source / filename
            original = path.read_bytes()
            code, output = self.cli("--output", str(path))
            self.assertEqual(code, 2, output)
            self.assertEqual(path.read_bytes(), original)

    def test_destination_cannot_be_the_source(self) -> None:
        archive = self.source / self.plan.copy_requests[0].source.archive
        original = archive.read_bytes()
        with patch.object(
            artifact_manager,
            "create_backend_from_env",
            return_value=LocalDirectoryBackend(staging_dir=self.source),
        ):
            self.assert_failure("Destination must differ")
        self.assertEqual(archive.read_bytes(), original)

    def test_destination_readback_failure_removes_stale_result(self) -> None:
        original_download = LocalDirectoryBackend.download_artifact

        def fail_readback(backend, key, path):
            if backend.base_path == self.destination:
                raise OSError("cannot read copied archive")
            return original_download(backend, key, path)

        with patch.object(LocalDirectoryBackend, "download_artifact", fail_readback):
            self.assert_failure("cannot read copied archive")

    def test_digest_only_and_binary_checksum_sidecars(self) -> None:
        for position, request in enumerate(self.plan.copy_requests):
            suffix = f" *{request.source.archive}" if position % 2 else ""
            (self.source / request.checksum_file).write_text(
                request.source.sha256 + suffix + "\n", encoding="utf-8"
            )
        code, output = self.cli()
        self.assertEqual(code, 0, output)
        self.assert_complete()

    def test_each_component_uses_its_own_source(self) -> None:
        for position, request in enumerate(self.plan.copy_requests):
            component_source = self.root / f"run-{position}"
            component_source.mkdir()
            for name in (
                request.source.archive,
                request.checksum_file,
                request.fingerprint_file,
            ):
                shutil.copyfile(self.source / name, component_source / name)
            request.source.root = component_source.as_uri()
            request.producer.run_id = str(position)
        self.save_plan()
        code, output = self.cli()
        self.assertEqual(code, 0, output)
        self.assert_complete()

    def test_s3_copy_uses_recorded_locations_and_verifies_destination(self) -> None:
        objects = {}
        expected_copies = []
        for position, request in enumerate(self.plan.copy_requests):
            prefix = f"run-{position}-linux"
            request.source.root = f"https://cache.s3.amazonaws.com/{prefix}"
            for filename in (
                request.source.archive,
                request.checksum_file,
                request.fingerprint_file,
            ):
                objects[("cache", f"{prefix}/{filename}")] = (
                    self.source / filename
                ).read_bytes()
            expected_copies.append(
                {"Bucket": "cache", "Key": f"{prefix}/{request.source.archive}"}
            )
        self.save_plan()
        destination = S3Backend(StorageLocation("current-run-bucket", "456-linux"))
        client = mock.Mock()

        def download(bucket, key, path):
            Path(path).write_bytes(objects[(bucket, key)])

        def copy_object(source, bucket, key):
            objects[(bucket, key)] = objects[(source["Bucket"], source["Key"])]

        def head_object(Bucket, Key):
            if (Bucket, Key) not in objects:
                raise FileNotFoundError(Key)
            return {}

        client.download_file.side_effect = download
        client.copy.side_effect = copy_object
        client.head_object.side_effect = head_object
        with (
            patch.object(
                S3Backend,
                "s3_client",
                new_callable=mock.PropertyMock,
                return_value=client,
            ),
            patch.object(
                artifact_manager, "create_backend_from_env", return_value=destination
            ) as factory,
        ):
            code, output = self.cli("--run-github-repo", "ROCm/TheRock")
            self.assertEqual(code, 0, output)
            factory.assert_called_once_with(
                run_id="456", platform="linux", github_repository="ROCm/TheRock"
            )
            actual_copies = [
                call.args[0]
                for call in client.copy.call_args_list
                if call.args[0]["Key"].endswith(".tar.xz")
            ]
            self.assertEqual(actual_copies, expected_copies)
            for request in self.plan.copy_requests:
                for filename in (
                    request.source.archive,
                    request.checksum_file,
                    request.fingerprint_file,
                ):
                    self.assertEqual(
                        objects[(destination.bucket, f"456-linux/{filename}")],
                        (self.source / filename).read_bytes(),
                    )
            result = json.loads(self.result_path.read_text())
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["destination"], destination.base_uri)
            client.copy.side_effect = OSError("S3 copy failed")
            self.assert_failure("S3 copy failed")

    def test_https_sources_use_the_shared_copy_fallback(self) -> None:
        root = "https://artifacts.example.invalid/source"
        for request in self.plan.copy_requests:
            request.source.root = root
        self.save_plan()

        def response(request, timeout):
            url = request if isinstance(request, str) else request.full_url
            path = self.source / url.rsplit("/", 1)[-1]
            if not path.is_file():
                raise HTTPError(url, 404, "not found", {}, None)
            return io.BytesIO(path.read_bytes())

        with patch("_therock_utils.artifact_backend.urlopen", side_effect=response):
            code, output = self.cli()
        self.assertEqual(code, 0, output)
        self.assert_complete()


class TestRecordedArtifactBackends(unittest.TestCase):
    def test_recorded_roots_preserve_bucket_prefix_and_local_paths(self):
        for root in ("s3://bucket/prefix", "https://bucket.s3.amazonaws.com/prefix"):
            with self.subTest(root=root):
                backend = create_backend_from_uri(root)
                self.assertIsInstance(backend, S3Backend)
                self.assertEqual(
                    (backend.bucket, backend.s3_prefix), ("bucket", "prefix")
                )
        with tempfile.TemporaryDirectory(prefix="artifact source ") as temporary:
            path = Path(temporary).resolve()
            backend = create_backend_from_uri(path.as_uri())
            self.assertIsInstance(backend, LocalDirectoryBackend)
            self.assertEqual(backend.base_path, path)
        self.assertIsInstance(
            create_backend_from_uri("https://example.invalid/root"), HTTPSBackend
        )
        with self.assertRaises(ValueError):
            create_backend_from_uri("https://example.invalid/root?different=query")

    def test_http_optional_sidecar_only_treats_404_as_missing(self):
        source = HTTPSBackend("https://example.invalid")
        for status in (404, 403, 500):
            with (
                self.subTest(status=status),
                patch(
                    "_therock_utils.artifact_backend.urlopen",
                    side_effect=HTTPError(source.base_uri, status, "error", {}, None),
                ),
            ):
                if status == 404:
                    self.assertFalse(source.artifact_exists("archive.sha256sum"))
                else:
                    with self.assertRaises(HTTPError):
                        source.artifact_exists("archive.sha256sum")

    def test_local_to_s3_and_s3_to_local_copy_fallbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = LocalDirectoryBackend(directory / "source")
            local_destination = LocalDirectoryBackend(directory / "destination")
            remote = S3Backend(StorageLocation("bucket", "prefix"))
            client = mock.Mock()
            remote._s3_client = client
            objects = {}
            archive = "test_lib_generic.tar.xz"
            (source.base_path / archive).write_bytes(b"archive bytes")
            (source.base_path / (archive + ".sha256sum")).write_bytes(b"sidecar bytes")

            def upload(path, bucket, key):
                objects[(bucket, key)] = Path(path).read_bytes()

            def download(bucket, key, path):
                Path(path).write_bytes(objects[(bucket, key)])

            def head(Bucket, Key):
                if (Bucket, Key) not in objects:
                    raise FileNotFoundError(Key)
                return {}

            client.upload_file.side_effect = upload
            client.download_file.side_effect = download
            client.head_object.side_effect = head
            remote.copy_artifact(archive, source)
            local_destination.copy_artifact(archive, remote)
            for name in (archive, archive + ".sha256sum"):
                self.assertEqual(
                    (local_destination.base_path / name).read_bytes(),
                    (source.base_path / name).read_bytes(),
                )
            client.copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
