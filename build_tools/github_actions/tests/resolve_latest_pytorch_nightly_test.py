# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import datetime
import os
import sys
import unittest
from pathlib import Path

from packaging.tags import Tag
from packaging.version import Version

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import resolve_latest_pytorch_nightly as m

CP312_LINUX = Tag("cp312", "cp312", "linux_x86_64")
TODAY = datetime.date(2026, 11, 19)


def project_page(*filenames: str) -> str:
    links = "".join(
        f'<a href="{name}#sha256=00">{name}</a><br/>\n' for name in filenames
    )
    return f"<!DOCTYPE html><html><body>\n{links}</body></html>\n"


class WheelVersionsTest(unittest.TestCase):
    def test_keeps_only_wheels_for_the_requested_tag(self):
        page = project_page(
            "torch-2.16.0a0+rocm10.3.0a20261119-cp312-cp312-linux_x86_64.whl",
            "torch-2.16.0a0+rocm10.3.0a20261119-cp313-cp313-linux_x86_64.whl",
            "torch-2.16.0a0+rocm10.3.0a20261118-cp312-cp312-win_amd64.whl",
            "torch-2.16.0a0+rocm10.3.0a20261118.tar.gz",
        )

        self.assertEqual(
            m.wheel_versions(page, CP312_LINUX),
            {Version("2.16.0a0+rocm10.3.0a20261119")},
        )


class NewestNightlyTest(unittest.TestCase):
    def test_skips_release_lines_and_versions_without_a_device_wheel(self):
        torch = {
            Version("2.13.1+rocm10.3.0a20261119"),
            Version("2.16.0a0+rocm10.3.0a20261118"),
            Version("2.16.0a0+rocm10.3.0a20261119"),
        }
        # The 20261119 device wheel has not been uploaded yet.
        device = {
            Version("2.13.1+rocm10.3.0a20261119"),
            Version("2.16.0a0+rocm10.3.0a20261118"),
        }

        self.assertEqual(
            m.newest_nightly(torch, device, TODAY, max_age_days=3),
            Version("2.16.0a0+rocm10.3.0a20261118"),
        )

    def test_stale_nightly_fails(self):
        versions = {Version("2.16.0a0+rocm10.3.0a20261110")}

        with self.assertRaisesRegex(ValueError, "9 days old"):
            m.newest_nightly(versions, versions, TODAY, max_age_days=3)

    def test_no_nightly_fails(self):
        versions = {Version("2.15.0+rocm10.3.0a20261119")}

        with self.assertRaisesRegex(ValueError, "No nightly"):
            m.newest_nightly(versions, versions, TODAY, max_age_days=3)


if __name__ == "__main__":
    unittest.main()
