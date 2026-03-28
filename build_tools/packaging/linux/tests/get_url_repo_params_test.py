# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Unit test coverage for get_url_repo_params.py:
#   get_base_url, get_repo_sub_folder, get_repo_url, and main() subcommands.

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import get_url_repo_params


class GetBaseUrlTest(unittest.TestCase):
    """Tests for get_base_url()."""

    def test_returns_scheme_and_netloc(self):
        # Test that get_base_url returns scheme and netloc only, stripping path.
        self.assertEqual(
            get_url_repo_params.get_base_url("https://example.com/v2/whl"),
            "https://example.com",
        )

    def test_strips_query_and_fragment(self):
        # Test that get_base_url strips query string and fragment.
        self.assertEqual(
            get_url_repo_params.get_base_url("https://example.com/path?q=1#anchor"),
            "https://example.com",
        )

    def test_http_url(self):
        # Test that get_base_url works with http.
        self.assertEqual(
            get_url_repo_params.get_base_url("http://repo.local/artifacts"),
            "http://repo.local",
        )

    def test_invalid_url_no_scheme_raises(self):
        # Test that get_base_url raises ValueError when URL has no scheme.
        with self.assertRaises(ValueError) as ctx:
            get_url_repo_params.get_base_url("not-a-url")
        self.assertIn("Invalid URL", str(ctx.exception))

    def test_invalid_url_empty_raises(self):
        # Test that get_base_url raises ValueError for empty or invalid URL.
        with self.assertRaises(ValueError):
            get_url_repo_params.get_base_url("")


class GetRepoSubFolderTest(unittest.TestCase):
    """Tests for get_repo_sub_folder()."""

    def test_returns_last_segment_when_yyyyMMdd_artifact(self):
        # Test that get_repo_sub_folder returns last segment when it matches YYYYMMDD-\d+.
        self.assertEqual(
            get_url_repo_params.get_repo_sub_folder("v3/packages/deb/20260204-12345"),
            "20260204-12345",
        )

    def test_returns_empty_when_last_segment_not_date_artifact(self):
        # Test that get_repo_sub_folder returns empty when last segment does not match pattern.
        self.assertEqual(
            get_url_repo_params.get_repo_sub_folder("v3/packages/deb/"),
            "",
        )
        self.assertEqual(
            get_url_repo_params.get_repo_sub_folder("v3/packages/deb/stable"),
            "",
        )

    def test_strips_slashes(self):
        # Test that leading/trailing slashes are stripped before splitting.
        self.assertEqual(
            get_url_repo_params.get_repo_sub_folder("/v3/deb/20260204-999/"),
            "20260204-999",
        )

    def test_empty_prefix_returns_empty(self):
        # Test that empty or slash-only prefix returns empty string.
        self.assertEqual(get_url_repo_params.get_repo_sub_folder(""), "")
        self.assertEqual(get_url_repo_params.get_repo_sub_folder("/"), "")


class GetRepoUrlTest(unittest.TestCase):
    """Tests for get_repo_url()."""

    def test_prerelease_deb(self):
        # Test that prerelease + deb yields base/os_profile.
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="prerelease",
                native_package_type="deb",
                repo_base_url="https://x.com",
                os_profile="ubuntu2404",
                repo_sub_folder="",
            ),
            "https://x.com/ubuntu2404",
        )

    def test_prerelease_rpm(self):
        # Test that prerelease + rpm yields base/os_profile/x86_64/
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="prerelease",
                native_package_type="rpm",
                repo_base_url="https://x.com",
                os_profile="rhel8",
                repo_sub_folder="",
            ),
            "https://x.com/rhel8/x86_64/",
        )

    def test_nightly_deb(self):
        # Test that non-prerelease + deb yields base/deb/repo_sub_folder/
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="deb",
                repo_base_url="https://x.com",
                os_profile="ubuntu2404",
                repo_sub_folder="20260204-12345",
            ),
            "https://x.com/deb/20260204-12345/",
        )

    def test_nightly_rpm(self):
        # Test that non-prerelease + rpm yields base/rpm/repo_sub_folder/x86_64/
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="rpm",
                repo_base_url="https://x.com",
                os_profile="rhel8",
                repo_sub_folder="20260204-12345",
            ),
            "https://x.com/rpm/20260204-12345/x86_64/",
        )

    def test_strips_trailing_slash_from_base(self):
        # Test that repo_base_url trailing slash is stripped.
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="prerelease",
                native_package_type="deb",
                repo_base_url="https://x.com/",
                os_profile="ubuntu2404",
                repo_sub_folder="",
            ),
            "https://x.com/ubuntu2404",
        )


class MainSubcommandsTest(unittest.TestCase):
    """Tests for main() subcommands (get-base-url, get-repo-sub-folder, get-repo-url)."""

    def test_get_base_url_success(self):
        # Test that get-base-url subcommand prints repo_base_url= and returns 0.
        with patch("sys.stdout") as mock_stdout:
            code = get_url_repo_params.main(
                ["get-base-url", "--from-url", "https://example.com/v2/whl"]
            )
        self.assertEqual(code, 0)
        mock_stdout.write.assert_called()
        output = "".join(c[0][0] for c in mock_stdout.write.call_args_list)
        self.assertIn("repo_base_url=https://example.com", output)

    def test_get_base_url_invalid_returns_one(self):
        # Test that get-base-url with invalid URL returns 1 and prints error.
        with patch("sys.stderr"):
            code = get_url_repo_params.main(["get-base-url", "--from-url", "not-a-url"])
        self.assertEqual(code, 1)

    def test_get_repo_sub_folder_success(self):
        # Test that get-repo-sub-folder prints repo_sub_folder= and returns 0.
        with patch("sys.stdout") as mock_stdout:
            code = get_url_repo_params.main(
                ["get-repo-sub-folder", "--from-s3-prefix", "v3/deb/20260204-12345"]
            )
        self.assertEqual(code, 0)
        output = "".join(c[0][0] for c in mock_stdout.write.call_args_list)
        self.assertIn("repo_sub_folder=20260204-12345", output)

    def test_get_repo_url_success(self):
        # Test that get-repo-url prints repo_url= and returns 0.
        with patch("sys.stdout") as mock_stdout:
            code = get_url_repo_params.main(
                [
                    "get-repo-url",
                    "--release-type",
                    "prerelease",
                    "--native-package-type",
                    "deb",
                    "--repo-base-url",
                    "https://x.com",
                    "--os-profile",
                    "ubuntu2404",
                    "--repo-sub-folder",
                    "",
                ]
            )
        self.assertEqual(code, 0)
        output = "".join(c[0][0] for c in mock_stdout.write.call_args_list)
        self.assertIn("repo_url=https://x.com/ubuntu2404", output)

    def test_get_repo_url_error_returns_one(self):
        # Test that get-repo-url returns 1 and prints error when get_repo_url raises.
        with patch(
            "get_url_repo_params.get_repo_url", side_effect=ValueError("bad params")
        ):
            with patch("sys.stderr"):
                code = get_url_repo_params.main(
                    [
                        "get-repo-url",
                        "--release-type",
                        "prerelease",
                        "--native-package-type",
                        "deb",
                        "--repo-base-url",
                        "https://x.com",
                        "--os-profile",
                        "ubuntu2404",
                        "--repo-sub-folder",
                        "",
                    ]
                )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
