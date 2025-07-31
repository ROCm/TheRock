from botocore.exceptions import ClientError
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from fetch_artifacts import (
    retrieve_s3_artifacts,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent.parent


class ArtifactsIndexPageTest(unittest.TestCase):
    @patch("fetch_artifacts.boto3.client")
    def testRetrieveS3Artifacts(self, mock_boto3_client):
        mock_paginator = MagicMock()
        mock_page_iterator = [
            {
                "Contents": [
                    {"Key": "hello/empty_1test.tar.xz"},
                    {"Key": "hello/empty_2test.tar.xz"},
                ]
            },
            {"Contents": [{"Key": "test/empty_3generic.tar.xz"}]},
            {"Contents": [{"Key": "test/empty_3test.tar.xz.sha256sum"}]},
        ]

        mock_client_instance = MagicMock()
        mock_client_instance.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = mock_page_iterator
        mock_boto3_client.return_value = mock_client_instance

        result = retrieve_s3_artifacts("123", "test")

        self.assertEqual(len(result), 3)
        self.assertTrue("empty_1test.tar.xz" in result)
        self.assertTrue("empty_2test.tar.xz" in result)
        self.assertTrue("empty_3generic.tar.xz" in result)

    @patch("fetch_artifacts.boto3.client")
    def testRetrieveS3ArtifactsNotFound(self, mock_boto3_client):
        mock_paginator = MagicMock()
        mock_page_iterator = [
            {
                "Contents": [
                    {"Key": "hello/empty_1test.tar.xz"},
                    {"Key": "hello/empty_2test.tar.xz"},
                ]
            },
            {"Contents": [{"Key": "test/empty_3generic.tar.xz"}]},
            {"Contents": [{"Key": "test/empty_3test.tar.xz.sha256sum"}]},
        ]

        mock_client_instance = MagicMock()
        mock_client_instance.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Access Denied"}
            },
            operation_name="ListObjectsV2",
        )
        mock_boto3_client.return_value = mock_client_instance

        with self.assertRaises(ClientError) as context:
            retrieve_s3_artifacts("123", "test")

        self.assertEqual(context.exception.response["Error"]["Code"], "AccessDenied")


if __name__ == "__main__":
    unittest.main()
