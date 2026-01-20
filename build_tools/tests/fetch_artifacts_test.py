from botocore.exceptions import ClientError
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.artifact_backend import S3Backend
from fetch_artifacts import (
    list_s3_artifacts,
    filter_artifacts,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent.parent


class ArtifactsIndexPageTest(unittest.TestCase):
    def testListS3Artifacts_Found(self):
        # Create a mock backend that returns test artifacts
        backend = MagicMock(spec=S3Backend)
        backend.run_id = "123"
        backend.bucket = "therock-ci-artifacts"
        backend.s3_prefix = "ROCm-TheRock/123-linux"
        backend.list_artifacts.return_value = [
            "empty_1test.tar.xz",
            "empty_2test.tar.xz",
            "empty_3generic.tar.xz",
            "empty_4test.tar.xz",
        ]

        result = list_s3_artifacts(backend, "test")

        self.assertEqual(len(result), 4)
        self.assertTrue("empty_1test.tar.xz" in result)
        self.assertTrue("empty_2test.tar.xz" in result)
        self.assertTrue("empty_3generic.tar.xz" in result)
        self.assertTrue("empty_4test.tar.xz" in result)

    def testListS3Artifacts_NotFound(self):
        # Create a mock backend that raises ClientError
        backend = MagicMock(spec=S3Backend)
        backend.run_id = "123"
        backend.bucket = "therock-ci-artifacts"
        backend.s3_prefix = "ROCm-TheRock/123-linux"
        backend.list_artifacts.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Access Denied"}
            },
            operation_name="ListObjectsV2",
        )

        with self.assertRaises(ClientError) as context:
            list_s3_artifacts(backend, "test")

        self.assertEqual(context.exception.response["Error"]["Code"], "AccessDenied")

    def testListS3Artifacts_FiltersByArtifactGroup(self):
        # Test that filtering by artifact_group works correctly
        backend = MagicMock(spec=S3Backend)
        backend.run_id = "123"
        backend.bucket = "therock-ci-artifacts"
        backend.s3_prefix = "ROCm-TheRock/123-linux"
        backend.list_artifacts.return_value = [
            "rocblas_lib_gfx94X.tar.xz",  # matches gfx94X
            "rocblas_lib_gfx110X.tar.xz",  # doesn't match
            "amd-llvm_lib_generic.tar.xz",  # matches generic
            "hipblas_lib_gfx94X.tar.xz",  # matches gfx94X
        ]

        result = list_s3_artifacts(backend, "gfx94X")

        self.assertEqual(len(result), 3)
        self.assertIn("rocblas_lib_gfx94X.tar.xz", result)
        self.assertIn("amd-llvm_lib_generic.tar.xz", result)
        self.assertIn("hipblas_lib_gfx94X.tar.xz", result)
        self.assertNotIn("rocblas_lib_gfx110X.tar.xz", result)

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
