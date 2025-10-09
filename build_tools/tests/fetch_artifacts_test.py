from botocore.exceptions import ClientError
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from fetch_artifacts import (
    BucketMetadata,
    list_s3_artifacts,
    filter_artifacts,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent.parent


class ArtifactsIndexPageTest(unittest.TestCase):
    @patch("fetch_artifacts.paginator")
    def testListS3Artifacts_Found(self, mock_paginator):
        bucket_info = BucketMetadata(
            "ROCm-TheRock/", "therock-artifacts", "123", "linux"
        )
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "hello/empty_1test.tar.xz"},
                    {"Key": "hello/empty_2test.tar.xz"},
                ]
            },
            {"Contents": [{"Key": "test/empty_3generic.tar.xz"}]},
            {"Contents": [{"Key": "test/empty_3test.tar.xz.sha256sum"}]},
            {"Contents": [{"Key": "rocm-libraries/test/empty_4test.tar.xz"}]},
        ]

        result = list_s3_artifacts(bucket_info, "test")

        self.assertEqual(len(result), 4)
        self.assertTrue("empty_1test.tar.xz" in result)
        self.assertTrue("empty_2test.tar.xz" in result)
        self.assertTrue("empty_3generic.tar.xz" in result)
        self.assertTrue("empty_4test.tar.xz" in result)

    @patch("fetch_artifacts.paginator")
    def testListS3Artifacts_NotFound(self, mock_paginator):
        bucket_info = BucketMetadata(
            "ROCm-TheRock/", "therock-artifacts", "123", "linux"
        )
        mock_paginator.paginate.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Access Denied"}
            },
            operation_name="ListObjectsV2",
        )

        with self.assertRaises(ClientError) as context:
            list_s3_artifacts(bucket_info, "test")

        self.assertEqual(context.exception.response["Error"]["Code"], "AccessDenied")

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
