from botocore.exceptions import ClientError
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from fetch_artifacts import (
    BucketMetadata,
    retrieve_s3_artifacts,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent.parent


class ArtifactsIndexPageTest(unittest.TestCase):
    @patch("fetch_artifacts.paginator")
    def testRetrieveS3Artifacts(self, mock_paginator):
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

        result = retrieve_s3_artifacts(bucket_info, "test")

        self.assertEqual(len(result), 4)
        self.assertTrue("empty_1test.tar.xz" in result)
        self.assertTrue("empty_2test.tar.xz" in result)
        self.assertTrue("empty_3generic.tar.xz" in result)
        self.assertTrue("empty_4test.tar.xz" in result)

    @patch("fetch_artifacts.paginator")
    def testRetrieveS3ArtifactsNotFound(self, mock_paginator):
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
            retrieve_s3_artifacts(bucket_info, "test")

        self.assertEqual(context.exception.response["Error"]["Code"], "AccessDenied")


if __name__ == "__main__":
    unittest.main()
