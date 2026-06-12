# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(THIS_DIR.parent))
sys.path.insert(0, os.fspath(THIS_DIR.parent.parent))

import multi_arch_release_urls as m


class MultiArchReleaseUrlsTest(unittest.TestCase):
    def test_known_release_types(self) -> None:
        self.assertEqual(
            m.get_index_url("dev"),
            "https://rocm.devreleases.amd.com/whl-multi-arch/",
        )
        self.assertEqual(
            m.get_index_url("nightly"),
            "https://rocm.nightlies.amd.com/whl-multi-arch/",
        )
        self.assertEqual(
            m.get_index_url("prerelease"),
            "https://rocm.prereleases.amd.com/whl-multi-arch/",
        )

    def test_unknown_release_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown release_type"):
            m.get_index_url("bogus")

    def test_main_sets_output(self) -> None:
        with mock.patch.object(m, "gha_set_output") as gha_set_output:
            m.main(["--release-type", "dev"])
        outputs = gha_set_output.call_args.args[0]
        self.assertEqual(
            outputs["package_index_url"],
            "https://rocm.devreleases.amd.com/whl-multi-arch/",
        )

    def test_publish_script_shares_mapping(self) -> None:
        import publish_pytorch_to_release_bucket as publish

        self.assertEqual(publish.MULTI_ARCH_INDEX_URLS, m.MULTI_ARCH_INDEX_URLS)


if __name__ == "__main__":
    unittest.main()
