# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(THIS_DIR.parent.parent))
sys.path.insert(0, os.fspath(THIS_DIR.parent))

import write_artifacts_bucket_info as m


class WriteArtifactsBucketInfoTest(unittest.TestCase):
    def test_ci_release_type_writes_ci_bucket_info(self) -> None:
        with mock.patch.dict(os.environ, {"GITHUB_REPOSITORY": "ROCm/TheRock"}):
            with mock.patch.object(m, "gha_set_output") as set_output:
                m.main(["--release-type", "ci"])

        set_output.assert_called_once_with(
            {
                "bucket": "therock-ci-artifacts",
                "iam_role": "arn:aws:iam::692859939525:role/therock-ci",
                "aws_region": "us-east-2",
            }
        )

    def test_empty_release_type_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            m.main(["--release-type", ""])


if __name__ == "__main__":
    unittest.main()
