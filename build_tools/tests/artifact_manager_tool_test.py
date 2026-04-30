#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
from _therock_utils.workflow_outputs import WorkflowOutputRoot

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


class TestFetchAmdgpuTargets(ArtifactManagerTestBase):
    """Tests that fetch command correctly handles --amdgpu-targets for split artifacts."""

    def test_fetch_with_amdgpu_targets_finds_individual_target_archives(self):
        """Test that --amdgpu-targets matches individual-target split archives."""
        import artifact_manager

        # Stage a generic artifact and a per-target artifact
        self._create_staged_artifact("test-artifact", "lib", "generic")
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
                "--amdgpu-targets",
                "gfx942",
            ]

            artifact_manager.main(argv)

        # Should have fetched both generic and gfx942
        fetched_keys = [c.archive_path.name for c in extract_calls]
        self.assertTrue(
            any("generic" in k for k in fetched_keys),
            f"Should fetch generic artifact, got: {fetched_keys}",
        )
        self.assertTrue(
            any("gfx942" in k for k in fetched_keys),
            f"Should fetch gfx942 artifact, got: {fetched_keys}",
        )

    def test_fetch_with_amdgpu_targets_skips_other_targets(self):
        """Test that --amdgpu-targets doesn't fetch archives for other targets."""
        import artifact_manager

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
        import artifact_manager

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
        import artifact_manager

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
        import artifact_manager

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
        import artifact_manager

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
        import artifact_manager

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


class TestFetchDownloadCache(ArtifactManagerTestBase):
    """Tests for --download-cache-dir behavior."""

    def test_shared_cache_skips_redownload(self):
        """Second fetch with same --download-cache-dir skips already-cached archives.

        Uses a FailingBackend for the second fetch — if the cache check in
        download_artifact() works, the backend's download method is never
        reached, so the failing backend won't cause errors. If the cache
        check is removed, the second fetch hits the FailingBackend and fails.
        """
        import artifact_manager

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
        import artifact_manager

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
        import artifact_manager

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
        import artifact_manager

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
        import artifact_manager

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
        import artifact_manager

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
        # Import here so sys.path is already set up.
        import artifact_manager as am

        self.am = am

    def _make_args(
        self,
        amdgpu_families: str = "",
        amdgpu_targets: str = "",
        generic_only: bool = False,
        expand_family_to_targets: bool = False,
    ):
        import argparse

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


if __name__ == "__main__":
    unittest.main()
