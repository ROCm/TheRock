# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import unittest
from unittest import mock
from unittest.mock import MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import fetch_artifacts
from _therock_utils.artifact_backend import ArtifactBackend, HTTPBackend, S3Backend
from _therock_utils.s3_buckets import S3BucketConfig
from fetch_artifacts import (
    list_artifacts_for_group,
    filter_artifacts,
    _get_base_arch,
    _matches_target,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent.parent


class ArtifactsIndexPageTest(unittest.TestCase):
    def testListArtifactsForGroup_FiltersByArtifactGroup(self):
        # Test that filtering by artifact_group works correctly
        backend = MagicMock(spec=ArtifactBackend)
        backend.base_uri = "s3://therock-ci-artifacts/ROCm-TheRock/123-linux"
        backend.list_artifacts.return_value = [
            "rocblas_lib_gfx94X.tar.xz",  # matches gfx94X
            "rocblas_lib_gfx110X.tar.xz",  # doesn't match
            "amd-llvm_lib_generic.tar.xz",  # matches generic
            "hipblas_lib_gfx94X.tar.xz",  # matches gfx94X
        ]

        result = list_artifacts_for_group(backend, "gfx94X")

        self.assertEqual(len(result), 3)
        self.assertIn("rocblas_lib_gfx94X.tar.xz", result)
        self.assertIn("amd-llvm_lib_generic.tar.xz", result)
        self.assertIn("hipblas_lib_gfx94X.tar.xz", result)
        self.assertNotIn("rocblas_lib_gfx110X.tar.xz", result)

    def testListArtifactsForGroup_MatchesSplitTargetArchives(self):
        """Test that amdgpu_targets matches individual-target split archives."""
        backend = MagicMock(spec=ArtifactBackend)
        backend.base_uri = "s3://therock-ci-artifacts/123-linux"
        backend.list_artifacts.return_value = [
            "blas_lib_generic.tar.zst",
            "blas_lib_gfx942.tar.zst",
            "blas_lib_gfx1100.tar.zst",
            "blas_test_generic.tar.zst",
            "blas_test_gfx942.tar.zst",
        ]

        result = list_artifacts_for_group(
            backend, "gfx94X-dcgpu", amdgpu_targets=["gfx942"]
        )

        # Should match generic + gfx942, not gfx1100
        self.assertIn("blas_lib_generic.tar.zst", result)
        self.assertIn("blas_lib_gfx942.tar.zst", result)
        self.assertIn("blas_test_generic.tar.zst", result)
        self.assertIn("blas_test_gfx942.tar.zst", result)
        self.assertNotIn("blas_lib_gfx1100.tar.zst", result)

    def testListArtifactsForGroup_InclusiveMatchesBothFamilyAndTarget(self):
        """Test inclusive matching: accepts both family-named and target-named archives."""
        backend = MagicMock(spec=ArtifactBackend)
        backend.base_uri = "s3://therock-ci-artifacts/123-linux"
        # Mix of old (family-named) and new (target-named) archives
        backend.list_artifacts.return_value = [
            "blas_lib_gfx94X-dcgpu.tar.xz",  # old: family name
            "fft_lib_gfx942.tar.zst",  # new: individual target
            "amd-llvm_lib_generic.tar.xz",  # generic
            "rand_lib_gfx110X-all.tar.xz",  # different family
        ]

        result = list_artifacts_for_group(
            backend, "gfx94X-dcgpu", amdgpu_targets=["gfx942"]
        )

        self.assertIn("blas_lib_gfx94X-dcgpu.tar.xz", result)
        self.assertIn("fft_lib_gfx942.tar.zst", result)
        self.assertIn("amd-llvm_lib_generic.tar.xz", result)
        self.assertNotIn("rand_lib_gfx110X-all.tar.xz", result)

    def testListArtifactsForGroup_NoTargetsBackwardsCompat(self):
        """Test that omitting amdgpu_targets preserves old family-only matching."""
        backend = MagicMock(spec=ArtifactBackend)
        backend.base_uri = "s3://therock-ci-artifacts/123-linux"
        backend.list_artifacts.return_value = [
            "blas_lib_gfx94X-dcgpu.tar.xz",
            "blas_lib_gfx942.tar.zst",
            "amd-llvm_lib_generic.tar.xz",
        ]

        # No amdgpu_targets — should only match family name + generic
        result = list_artifacts_for_group(backend, "gfx94X-dcgpu")

        self.assertIn("blas_lib_gfx94X-dcgpu.tar.xz", result)
        self.assertIn("amd-llvm_lib_generic.tar.xz", result)
        self.assertNotIn("blas_lib_gfx942.tar.zst", result)

    def testListArtifactsForGroup_MultipleTargets(self):
        """Test fetching with multiple individual targets."""
        backend = MagicMock(spec=ArtifactBackend)
        backend.base_uri = "s3://therock-ci-artifacts/123-linux"
        backend.list_artifacts.return_value = [
            "blas_lib_generic.tar.zst",
            "blas_lib_gfx942.tar.zst",
            "blas_lib_gfx90a.tar.zst",
            "blas_lib_gfx1100.tar.zst",
        ]

        result = list_artifacts_for_group(
            backend, "gfx94X-dcgpu", amdgpu_targets=["gfx942", "gfx90a"]
        )

        self.assertIn("blas_lib_generic.tar.zst", result)
        self.assertIn("blas_lib_gfx942.tar.zst", result)
        self.assertIn("blas_lib_gfx90a.tar.zst", result)
        self.assertNotIn("blas_lib_gfx1100.tar.zst", result)

    def testListArtifactsForGroup_IgnoresNonArtifactFiles(self):
        """Test that files not matching ArtifactName pattern are skipped."""
        backend = MagicMock(spec=ArtifactBackend)
        backend.base_uri = "s3://therock-ci-artifacts/123-linux"
        backend.list_artifacts.return_value = [
            "blas_lib_generic.tar.zst",
            "README.md",
            "some_random_file.txt",
            "blas_lib_gfx942.tar.zst",
        ]

        result = list_artifacts_for_group(
            backend, "gfx94X-dcgpu", amdgpu_targets=["gfx942"]
        )

        self.assertEqual(len(result), 2)
        self.assertIn("blas_lib_generic.tar.zst", result)
        self.assertIn("blas_lib_gfx942.tar.zst", result)

    def testListArtifactsForGroup_MatchesXnackVariants(self):
        """Test that requesting base arch also matches xnack-suffixed variants."""
        backend = MagicMock(spec=ArtifactBackend)
        backend.base_uri = "s3://therock-ci-artifacts/123-linux"
        backend.list_artifacts.return_value = [
            "blas_lib_generic.tar.zst",
            "blas_lib_gfx942.tar.zst",
            "blas_test_gfx942.tar.zst",
            "rccl_test_gfx942:xnack+.tar.zst",  # xnack+ variant
            "rccl_lib_gfx942:xnack-.tar.zst",  # xnack- variant
            "blas_lib_gfx1100.tar.zst",  # different arch, should not match
        ]

        # Request base arch gfx942 - should also pull xnack variants
        result = list_artifacts_for_group(
            backend, artifact_group=None, amdgpu_targets=["gfx942"]
        )

        self.assertIn("blas_lib_generic.tar.zst", result)
        self.assertIn("blas_lib_gfx942.tar.zst", result)
        self.assertIn("blas_test_gfx942.tar.zst", result)
        self.assertIn("rccl_test_gfx942:xnack+.tar.zst", result)
        self.assertIn("rccl_lib_gfx942:xnack-.tar.zst", result)
        self.assertNotIn("blas_lib_gfx1100.tar.zst", result)

    def testListArtifactsForGroup_ExplicitXnackTargetMatchesBase(self):
        """Test that requesting xnack variant explicitly also matches base arch."""
        backend = MagicMock(spec=ArtifactBackend)
        backend.base_uri = "s3://therock-ci-artifacts/123-linux"
        backend.list_artifacts.return_value = [
            "blas_lib_generic.tar.zst",
            "blas_lib_gfx942.tar.zst",
            "rccl_test_gfx942:xnack+.tar.zst",
        ]

        # Request xnack+ variant explicitly - should also pull base arch
        result = list_artifacts_for_group(
            backend, artifact_group=None, amdgpu_targets=["gfx942:xnack+"]
        )

        self.assertIn("blas_lib_generic.tar.zst", result)
        self.assertIn("blas_lib_gfx942.tar.zst", result)
        self.assertIn("rccl_test_gfx942:xnack+.tar.zst", result)

    def testGetBaseArch_HandlesEdgeCases(self):
        """Test _get_base_arch with empty and garbage inputs."""
        self.assertEqual(_get_base_arch(""), "")
        self.assertEqual(_get_base_arch(":xnack+"), ":xnack+")
        self.assertEqual(_get_base_arch("gfx942"), "gfx942")
        self.assertEqual(_get_base_arch("gfx942:xnack+"), "gfx942")
        self.assertEqual(_get_base_arch("garbage-*&%^$"), "garbage-*&%^$")

    def testMatchesTarget_HandlesEdgeCases(self):
        """Test _matches_target with empty and garbage inputs."""
        requested = {"generic", "gfx942"}
        self.assertFalse(_matches_target("", requested))
        self.assertFalse(_matches_target("garbage-*&%^$", requested))
        self.assertTrue(_matches_target("gfx942", requested))
        self.assertTrue(_matches_target("gfx942:xnack+", requested))

    def testFilterArtifacts_NoIncludesOrExcludes(self):
        artifacts = {"foo_test", "foo_run", "bar_test", "bar_run"}

        filtered = filter_artifacts(artifacts, includes=[], excludes=[])
        # Include all by default.
        self.assertIn("foo_test", filtered)
        self.assertIn("foo_run", filtered)
        self.assertIn("bar_test", filtered)
        self.assertIn("bar_run", filtered)

    def testFilterArtifacts_OneInclude(self):
        artifacts = {"foo_test", "foo_run", "bar_test", "bar_run"}

        filtered = filter_artifacts(artifacts, includes=["foo"], excludes=[])
        self.assertIn("foo_test", filtered)
        self.assertIn("foo_run", filtered)
        self.assertNotIn("bar_test", filtered)
        self.assertNotIn("bar_run", filtered)

    def testFilterArtifacts_MultipleIncludes(self):
        artifacts = {"foo_test", "foo_run", "bar_test", "bar_run"}

        filtered = filter_artifacts(artifacts, includes=["foo", "test"], excludes=[])
        # Include if _any_ include matches.
        self.assertIn("foo_test", filtered)
        self.assertIn("foo_run", filtered)
        self.assertIn("bar_test", filtered)
        self.assertNotIn("bar_run", filtered)

    def testFilterArtifacts_OneExclude(self):
        artifacts = {"foo_test", "foo_run", "bar_test", "bar_run"}

        filtered = filter_artifacts(artifacts, includes=[], excludes=["foo"])
        self.assertNotIn("foo_test", filtered)
        self.assertNotIn("foo_run", filtered)
        self.assertIn("bar_test", filtered)
        self.assertIn("bar_run", filtered)

    def testFilterArtifacts_MultipleExcludes(self):
        artifacts = {"foo_test", "foo_run", "bar_test", "bar_run"}

        filtered = filter_artifacts(artifacts, includes=[], excludes=["foo", "test"])
        # Exclude if _any_ exclude matches.
        self.assertNotIn("foo_test", filtered)
        self.assertNotIn("foo_run", filtered)
        self.assertNotIn("bar_test", filtered)
        self.assertIn("bar_run", filtered)

    def testFilterArtifacts_IncludeAndExclude(self):
        artifacts = {"foo_test", "foo_run", "bar_test", "bar_run"}

        filtered = filter_artifacts(artifacts, includes=["foo"], excludes=["test"])
        # Must match at least one include and not match any exclude.
        self.assertNotIn("foo_test", filtered)
        self.assertIn("foo_run", filtered)
        self.assertNotIn("bar_test", filtered)
        self.assertNotIn("bar_run", filtered)


class TransportSelectionTest(unittest.TestCase):
    """Tests for --transport on fetch_artifacts.py."""

    def _argv(self, *extra):
        return ["--run-id", "12345", "--dry-run", *extra]

    def _parse(self, *extra):
        """Parse args without running, by intercepting run()."""
        with mock.patch.object(fetch_artifacts, "run") as mock_run:
            fetch_artifacts.main(self._argv(*extra))
        return mock_run.call_args[0][0]

    def test_default_transport_is_auto(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self._parse().transport, "auto")

    def test_env_var_supplies_the_default(self):
        with mock.patch.dict(
            os.environ, {"THEROCK_ARTIFACT_TRANSPORT": "http"}, clear=True
        ):
            self.assertEqual(self._parse().transport, "http")

    def test_flag_beats_the_env_var(self):
        with mock.patch.dict(
            os.environ, {"THEROCK_ARTIFACT_TRANSPORT": "http"}, clear=True
        ):
            self.assertEqual(self._parse("--transport", "s3").transport, "s3")

    def test_accepts_transports_it_cannot_serve(self):
        """'local' parses so that a shared env var does not break arg parsing.

        artifact_manager.py and this script read the same
        THEROCK_ARTIFACT_TRANSPORT; rejecting a value at parse time here would
        make setting it for one break the other. run() rejects it instead.
        """
        self.assertEqual(self._parse("--transport", "local").transport, "local")

    def test_local_transport_exits_with_a_pointer_to_artifact_manager(self):
        args = mock.MagicMock(transport="local", bucket_config_file=None)
        with self.assertRaises(SystemExit) as cm:
            fetch_artifacts.run(args)
        self.assertEqual(cm.exception.code, 1)

    @mock.patch("_therock_utils.workflow_outputs._retrieve_bucket_info")
    def _backend_for(self, transport, mock_retrieve):
        mock_retrieve.return_value = ("", S3BucketConfig(name="therock-ci-artifacts"))
        captured = {}

        def capture(backend, **kwargs):
            captured["backend"] = backend
            return set()

        with mock.patch.object(
            fetch_artifacts, "list_artifacts_for_group", side_effect=capture
        ):
            # Nothing to fetch, so run() exits 1 after building the backend.
            with self.assertRaises(SystemExit):
                fetch_artifacts.main(self._argv("--transport", transport))
        return captured["backend"]

    def test_http_transport_builds_an_http_backend(self):
        self.assertIsInstance(self._backend_for("http"), HTTPBackend)

    def test_s3_transport_builds_an_s3_backend(self):
        self.assertIsInstance(self._backend_for("s3"), S3Backend)


if __name__ == "__main__":
    unittest.main()
