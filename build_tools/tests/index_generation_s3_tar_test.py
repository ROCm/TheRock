# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for build_tools/index_generation_s3_tar.py."""

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import index_generation_s3_tar


class IndexGenerationS3TarTest(unittest.TestCase):
    bucket_name = "tarball-index-test-bucket"
    prefix = "v5/rocm/extras/rvs/tarball"
    index_key = f"{prefix}/index.html"

    def setUp(self) -> None:
        self.s3_client = mock.MagicMock()
        self.paginator = self.s3_client.get_paginator.return_value
        self.s3_client.meta.region_name = "us-east-2"
        self._set_contents([])

    def _set_contents(self, contents: list[dict[str, object]]) -> None:
        self.paginator.paginate.return_value = [{"Contents": contents}]

    def _uploaded_html(self, call_index: int = -1) -> str:
        put_call = self.s3_client.put_object.call_args_list[call_index]
        body = put_call.kwargs["Body"]
        self.assertIsInstance(body, bytes)
        return body.decode("utf-8")

    def test_empty_prefix_raises_by_default(self) -> None:
        """Existing callers must continue to reject an empty prefix."""
        with self.assertRaises(FileNotFoundError) as raised:
            index_generation_s3_tar.generate_index_s3(
                s3_client=self.s3_client,
                bucket_name=self.bucket_name,
                prefix=self.prefix,
                upload=True,
            )

        self.assertEqual(
            str(raised.exception),
            f"No .tar.gz files found in bucket {self.bucket_name}.",
        )
        self.s3_client.put_object.assert_not_called()

    def test_deleting_last_tarball_uploads_empty_index_when_allowed(self) -> None:
        """Removing the last tarball must replace the stale index."""
        tarball_key = f"{self.prefix}/test-rvs.tar.gz"

        # Generate an index while the final tarball still exists.
        self._set_contents(
            [
                {
                    "Key": tarball_key,
                    "LastModified": datetime(
                        2026,
                        7,
                        31,
                        tzinfo=timezone.utc,
                    ),
                }
            ]
        )

        index_generation_s3_tar.generate_index_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            upload=True,
            allow_empty=True,
        )

        populated_html = self._uploaded_html()
        self.assertIn('"name": "test-rvs.tar.gz"', populated_html)

        # Simulate the Lambda running after that final tarball is deleted.
        self._set_contents([])

        result = index_generation_s3_tar.generate_index_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            upload=True,
            allow_empty=True,
        )

        self.assertEqual(
            result,
            (
                f"https://{self.bucket_name}.s3.amazonaws.com/"
                f"{self.index_key}"
            ),
        )
        self.assertEqual(self.s3_client.put_object.call_count, 2)

        empty_upload = self.s3_client.put_object.call_args_list[-1]
        self.assertEqual(empty_upload.kwargs["Bucket"], self.bucket_name)
        self.assertEqual(empty_upload.kwargs["Key"], self.index_key)
        self.assertEqual(empty_upload.kwargs["ContentType"], "text/html")

        empty_html = self._uploaded_html()
        self.assertIn("const files = [];", empty_html)
        self.assertIn("No tarballs available.", empty_html)
        self.assertNotIn('"name": "test-rvs.tar.gz"', empty_html)


if __name__ == "__main__":
    unittest.main()
