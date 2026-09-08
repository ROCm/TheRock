# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import write_product_release_bucket_info


class WriteProductReleaseBucketInfoTest(unittest.TestCase):
    @mock.patch("write_product_release_bucket_info.gha_set_output")
    def test_writes_core_product_publication_outputs(self, mock_set_output):
        write_product_release_bucket_info.main(
            ["--release-type", "prerelease", "--product", "core"]
        )

        mock_set_output.assert_called_once_with(
            {
                "bucket": "therock-repo-amd-rc-core",
                "iam_role": "arn:aws:iam::324352301041:role/therock-repo-rc-core",
                "aws_region": "us-east-2",
                "package_index_url": "https://rc.repo.amd.com/rocm/whl-next/",
            }
        )

    @mock.patch("write_product_release_bucket_info.gha_set_output")
    def test_writes_bkc_framework_publication_outputs(self, mock_set_output):
        write_product_release_bucket_info.main(
            ["--release-type", "nightly-bkc", "--product", "pytorch"]
        )

        mock_set_output.assert_called_once_with(
            {
                "bucket": "therock-repo-amd-bkc-pytorch",
                "iam_role": "arn:aws:iam::324352301041:role/therock-repo-bkc-pytorch",
                "aws_region": "us-east-2",
                "package_index_url": "https://bkc.repo.amd.com/rocm/whl-next/",
            }
        )

    def test_rejects_invalid_product(self):
        with self.assertRaises(SystemExit):
            write_product_release_bucket_info.main(
                ["--release-type", "dev", "--product", "python"]
            )


if __name__ == "__main__":
    unittest.main()
