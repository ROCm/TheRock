from pathlib import Path
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifact_backend import ArtifactBackend
from fetch_artifacts import (
    list_artifacts_for_group,
    filter_artifacts,
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


if __name__ == "__main__":
    unittest.main()
