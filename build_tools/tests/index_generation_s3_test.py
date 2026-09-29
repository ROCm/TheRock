# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for build_tools/index_generation_s3.py."""

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import index_generation_s3


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
            index_generation_s3.generate_index_s3(
                s3_client=self.s3_client,
                bucket_name=self.bucket_name,
                prefix=self.prefix,
                upload=True,
            )

        self.assertEqual(
            str(raised.exception),
            f"No .tar.gz files found in bucket {self.bucket_name} "
            f"under prefix '{self.prefix}'.",
        )
        self.s3_client.put_object.assert_not_called()

    def test_index_offers_multiarch_filter(self) -> None:
        """Combined archives must be selectable separately from GPU archives."""
        self._set_contents(
            [
                {
                    "Key": f"{self.prefix}/therock-dist-linux-{target}-10.1.0.tar.gz",
                    "LastModified": datetime(2026, 8, 7, tzinfo=timezone.utc),
                }
                for target in ("multiarch", "gfx90a", "gfx1100")
            ]
        )

        index_generation_s3.generate_index_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            upload=True,
        )

        html = self._uploaded_html()
        self.assertIn('<option value="all">All</option>', html)
        self.assertIn('<option value="multiarch">Multiarch</option>', html)
        self.assertIn('<option value="gfx90a">gfx90a</option>', html)
        self.assertIn('<option value="gfx1100">gfx1100</option>', html)

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

        index_generation_s3.generate_index_s3(
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

        result = index_generation_s3.generate_index_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            upload=True,
            allow_empty=True,
        )

        self.assertEqual(
            result,
            (f"https://{self.bucket_name}.s3.amazonaws.com/" f"{self.index_key}"),
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


class IndexGenerationS3Test(unittest.TestCase):
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
            index_generation_s3.generate_index_s3(
                s3_client=self.s3_client,
                bucket_name=self.bucket_name,
                prefix=self.prefix,
                upload=True,
            )
        self.assertEqual(
            str(raised.exception),
            f"No .tar.gz files found in bucket {self.bucket_name} "
            f"under prefix '{self.prefix}'.",
        )
        self.s3_client.put_object.assert_not_called()

    def test_index_offers_multiarch_filter(self) -> None:
        """Combined archives must be selectable separately from GPU archives."""
        self._set_contents(
            [
                {
                    "Key": f"{self.prefix}/therock-dist-linux-{target}-10.1.0.tar.gz",
                    "LastModified": datetime(2026, 8, 7, tzinfo=timezone.utc),
                }
                for target in ("multiarch", "gfx90a", "gfx1100")
            ]
        )
        index_generation_s3.generate_index_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            upload=True,
        )
        html = self._uploaded_html()
        self.assertIn('<option value="all">All</option>', html)
        self.assertIn('<option value="multiarch">Multiarch</option>', html)
        self.assertIn('<option value="gfx90a">gfx90a</option>', html)
        self.assertIn('<option value="gfx1100">gfx1100</option>', html)

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
        index_generation_s3.generate_index_s3(
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
        result = index_generation_s3.generate_index_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            prefix=self.prefix,
            upload=True,
            allow_empty=True,
        )

        self.assertEqual(
            result,
            (f"https://{self.bucket_name}.s3.amazonaws.com/" f"{self.index_key}"),
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

    def test_index_supports_custom_artifact_options(self) -> None:
        """Custom artifact settings support non-tarball flat indexes."""
        windows_prefix = "v5/rocm/extras/rocoptiq/windows-installers/win11"
        msi_key = f"{windows_prefix}/roc-optiq-1.1.0.0-win64.msi"
        tarball_key = f"{windows_prefix}/ignored.tar.gz"

        self._set_contents(
            [
                {
                    "Key": msi_key,
                    "LastModified": datetime(2026, 9, 29, tzinfo=timezone.utc),
                },
                {
                    "Key": tarball_key,
                    "LastModified": datetime(2026, 9, 29, tzinfo=timezone.utc),
                },
            ]
        )

        index_generation_s3.generate_index_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            prefix=windows_prefix,
            upload=True,
            artifact_suffix=".msi",
            page_title="Windows Installers",
            show_filter=False,
        )

        html = self._uploaded_html()
        self.assertIn("<title>Windows Installers</title>", html)
        self.assertIn("<h1>Windows Installers</h1>", html)
        self.assertIn("roc-optiq-1.1.0.0-win64.msi", html)
        self.assertNotIn("ignored.tar.gz", html)
        self.assertNotIn('id="filter"', html)
        self.assertNotIn("Multiarch", html)

    def test_index_supports_custom_empty_message(self) -> None:
        """Custom empty messages are used for non-tarball indexes."""
        windows_prefix = "v5/rocm/extras/rocoptiq/windows-installers/win11"

        index_generation_s3.generate_index_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            prefix=windows_prefix,
            upload=True,
            allow_empty=True,
            artifact_suffix=".msi",
            page_title="Windows Installers",
            show_filter=False,
            empty_message="No Windows installers available.",
        )

        html = self._uploaded_html()
        self.assertIn(
            "No Windows installers available.",
            html,
        )

    def test_directory_index_lists_direct_child_directories(self) -> None:
        """Directory indexes list only direct child directories."""
        windows_prefix = "v5/rocm/extras/rocoptiq/windows-installers"

        self.paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": f"{windows_prefix}/win11/"},
                    {"Prefix": f"{windows_prefix}/win12/"},
                ]
            }
        ]

        result = index_generation_s3.generate_directory_index_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            prefix=windows_prefix,
            upload=True,
        )

        self.assertEqual(
            result,
            (
                f"https://{self.bucket_name}.s3.amazonaws.com/"
                f"{windows_prefix}/index.html"
            ),
        )

        upload = self.s3_client.put_object.call_args
        self.assertEqual(
            upload.kwargs["Bucket"],
            self.bucket_name,
        )
        self.assertEqual(
            upload.kwargs["Key"],
            f"{windows_prefix}/index.html",
        )
        self.assertEqual(
            upload.kwargs["ContentType"],
            "text/html",
        )

        html = self._uploaded_html()
        self.assertIn('href="win11/"', html)
        self.assertIn('href="win12/"', html)
        self.assertNotIn("nested/", html)


if __name__ == "__main__":
    unittest.main()
