#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Unit test coverage for get_url_repo_params.py:
#   get_base_url, get_gpg_key_url, gpg_key_url_needed_for_release_type, get_repo_sub_folder,
#   get_repo_url, extract_gfx_arch, and main() subcommands.

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent.parent))
import get_url_repo_params


def _run_main_with_output(argv: list[str]) -> tuple[int, str]:
    """Run main() with a temp GITHUB_OUTPUT file; return (exit_code, file_contents)."""
    with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as f:
        tmp_path = f.name
    try:
        with patch.dict(os.environ, {"GITHUB_OUTPUT": tmp_path}):
            code = get_url_repo_params.main(argv)
        contents = Path(tmp_path).read_text()
    finally:
        os.unlink(tmp_path)
    return code, contents


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


class GetGpgKeyUrlTest(unittest.TestCase):
    """Tests for get_gpg_key_url()."""

    def test_extracts_base_and_adds_gpg_path(self):
        # Test that get_gpg_key_url extracts base URL and appends /gpg/rocm.gpg.
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url(
                "https://rocm.prereleases.amd.com/packages/ubuntu2404"
            ),
            "https://rocm.prereleases.amd.com/gpg/rocm.gpg",
        )

    def test_strips_path_from_url(self):
        # Test that get_gpg_key_url strips path and query from URL.
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url(
                "https://repo.amd.com/rocm/packages/rhel10/x86_64/"
            ),
            "https://repo.amd.com/gpg/rocm.gpg",
        )

    def test_handles_nightly_url(self):
        # Test that get_gpg_key_url works with nightly URLs.
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url(
                "https://rocm.nightlies.amd.com/deb/20260204-12345/"
            ),
            "https://rocm.nightlies.amd.com/gpg/rocm.gpg",
        )


class GpgKeyUrlNeededForReleaseTypeTest(unittest.TestCase):
    """Tests for gpg_key_url_needed_for_release_type()."""

    def test_none_means_always_derive(self):
        self.assertTrue(get_url_repo_params.gpg_key_url_needed_for_release_type(None))

    def test_prerelease_and_release(self):
        self.assertTrue(
            get_url_repo_params.gpg_key_url_needed_for_release_type("prerelease")
        )
        self.assertTrue(
            get_url_repo_params.gpg_key_url_needed_for_release_type("release")
        )
        self.assertTrue(
            get_url_repo_params.gpg_key_url_needed_for_release_type("  Prerelease  ")
        )

    def test_dev_nightly_ci_empty(self):
        self.assertFalse(get_url_repo_params.gpg_key_url_needed_for_release_type("dev"))
        self.assertFalse(
            get_url_repo_params.gpg_key_url_needed_for_release_type("nightly")
        )
        self.assertFalse(get_url_repo_params.gpg_key_url_needed_for_release_type("ci"))
        self.assertFalse(get_url_repo_params.gpg_key_url_needed_for_release_type(""))


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


class ExtractGfxArchTest(unittest.TestCase):
    """Tests for extract_gfx_arch()."""

    def test_extracts_and_lowercases_gfx_arch(self):
        # Test that extract_gfx_arch returns the first segment lowercased.
        self.assertEqual(
            get_url_repo_params.extract_gfx_arch("gfx94X-dcgpu"),
            "gfx94x",
        )

    def test_handles_lowercase_input(self):
        # Test that extract_gfx_arch works with already-lowercase input.
        self.assertEqual(
            get_url_repo_params.extract_gfx_arch("gfx1100-consumer"),
            "gfx1100",
        )

    def test_handles_uppercase_prefix(self):
        # Test that extract_gfx_arch lowercases uppercase prefix.
        self.assertEqual(
            get_url_repo_params.extract_gfx_arch("GFX942-server"),
            "gfx942",
        )

    def test_handles_no_dash(self):
        # Test that extract_gfx_arch returns the whole string if no dash present.
        self.assertEqual(
            get_url_repo_params.extract_gfx_arch("gfx1100"),
            "gfx1100",
        )

    def test_empty_string_raises(self):
        # Test that extract_gfx_arch raises ValueError for empty string.
        with self.assertRaises(ValueError) as ctx:
            get_url_repo_params.extract_gfx_arch("")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_handles_multiple_dashes(self):
        # Test that extract_gfx_arch only takes first segment when multiple dashes.
        self.assertEqual(
            get_url_repo_params.extract_gfx_arch("gfx94X-dcgpu-extra-info"),
            "gfx94x",
        )

    def test_comma_separated_list(self):
        # Test that extract_gfx_arch handles comma-separated artifact groups.
        self.assertEqual(
            get_url_repo_params.extract_gfx_arch("gfx94X-dcgpu,gfx1100-consumer"),
            "gfx94x,gfx1100",
        )

    def test_semicolon_separated_list(self):
        # Test that extract_gfx_arch handles semicolon-separated artifact groups.
        self.assertEqual(
            get_url_repo_params.extract_gfx_arch("gfx94X-dcgpu;gfx1100-consumer"),
            "gfx94x,gfx1100",
        )

    def test_mixed_case_list(self):
        # Test that extract_gfx_arch lowercases all items in list.
        self.assertEqual(
            get_url_repo_params.extract_gfx_arch("GFX942-server,GFX1100-consumer"),
            "gfx942,gfx1100",
        )

    def test_list_with_spaces(self):
        # Test that extract_gfx_arch strips whitespace from list items.
        self.assertEqual(
            get_url_repo_params.extract_gfx_arch("gfx94X-dcgpu , gfx1100-consumer"),
            "gfx94x,gfx1100",
        )


class MainSubcommandsTest(unittest.TestCase):
    """Tests for main() subcommands (get-base-url, get-repo-sub-folder, get-repo-url)."""

    def test_get_base_url_success(self):
        # Test that get-base-url subcommand writes repo_base_url= to GITHUB_OUTPUT.
        code, output = _run_main_with_output(
            ["get-base-url", "--from-url", "https://example.com/v2/whl"]
        )
        self.assertEqual(code, 0)
        self.assertIn("repo_base_url=https://example.com", output)

    def test_get_base_url_invalid_returns_one(self):
        # Test that get-base-url with invalid URL returns 1 and prints error.
        with patch("sys.stderr"):
            code = get_url_repo_params.main(["get-base-url", "--from-url", "not-a-url"])
        self.assertEqual(code, 1)

    def test_get_repo_sub_folder_success(self):
        # Test that get-repo-sub-folder writes repo_sub_folder= to GITHUB_OUTPUT.
        code, output = _run_main_with_output(
            ["get-repo-sub-folder", "--from-s3-prefix", "v3/deb/20260204-12345"]
        )
        self.assertEqual(code, 0)
        self.assertIn("repo_sub_folder=20260204-12345", output)

    def test_get_repo_url_success(self):
        # Test that get-repo-url writes repo_url= to GITHUB_OUTPUT.
        code, output = _run_main_with_output(
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

    def test_extract_gfx_arch_success(self):
        # Test that extract-gfx-arch writes gfx_arch= to GITHUB_OUTPUT.
        code, output = _run_main_with_output(
            ["extract-gfx-arch", "--artifact-group", "gfx94X-dcgpu"]
        )
        self.assertEqual(code, 0)
        self.assertIn("gfx_arch=gfx94x", output)

    def test_extract_gfx_arch_lowercase(self):
        # Test that extract-gfx-arch handles already-lowercase input.
        code, output = _run_main_with_output(
            ["extract-gfx-arch", "--artifact-group", "gfx1100-consumer"]
        )
        self.assertEqual(code, 0)
        self.assertIn("gfx_arch=gfx1100", output)

    def test_extract_gfx_arch_empty_returns_one(self):
        # Test that extract-gfx-arch with empty artifact_group returns 1.
        with patch("sys.stderr"):
            code = get_url_repo_params.main(
                ["extract-gfx-arch", "--artifact-group", ""]
            )
        self.assertEqual(code, 1)

    def test_extract_gfx_arch_comma_list(self):
        # Test that extract-gfx-arch handles comma-separated list.
        code, output = _run_main_with_output(
            ["extract-gfx-arch", "--artifact-group", "gfx94X-dcgpu,gfx1100-consumer"]
        )
        self.assertEqual(code, 0)
        self.assertIn("gfx_arch=gfx94x,gfx1100", output)

    def test_extract_gfx_arch_semicolon_list(self):
        # Test that extract-gfx-arch handles semicolon-separated list.
        code, output = _run_main_with_output(
            ["extract-gfx-arch", "--artifact-group", "gfx94X-dcgpu;gfx1100-consumer"]
        )
        self.assertEqual(code, 0)
        self.assertIn("gfx_arch=gfx94x,gfx1100", output)

    def test_get_gpg_url_success(self):
        # Test that get-gpg-url writes gpg_key_url= to GITHUB_OUTPUT.
        code, output = _run_main_with_output(
            [
                "get-gpg-url",
                "--from-url",
                "https://rocm.prereleases.amd.com/packages/ubuntu2404",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "gpg_key_url=https://rocm.prereleases.amd.com/gpg/rocm.gpg", output
        )

    def test_get_gpg_url_with_release_type_dev_emits_empty(self):
        code, output = _run_main_with_output(
            [
                "get-gpg-url",
                "--release-type",
                "dev",
                "--from-url",
                "https://rocm.prereleases.amd.com/packages/ubuntu2404",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("gpg_key_url=", output)
        self.assertNotIn("rocm.gpg", output)

    def test_get_gpg_url_with_release_type_dev_ignores_invalid_url(self):
        code, output = _run_main_with_output(
            [
                "get-gpg-url",
                "--release-type",
                "nightly",
                "--from-url",
                "not-a-valid-url",
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(output.strip(), "gpg_key_url=")

    def test_get_gpg_url_with_release_type_prerelease(self):
        code, output = _run_main_with_output(
            [
                "get-gpg-url",
                "--release-type",
                "prerelease",
                "--from-url",
                "https://rocm.prereleases.amd.com/packages/ubuntu2404",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "gpg_key_url=https://rocm.prereleases.amd.com/gpg/rocm.gpg", output
        )

    def test_get_gpg_url_with_release_type_release(self):
        code, output = _run_main_with_output(
            [
                "get-gpg-url",
                "--release-type",
                "release",
                "--from-url",
                "https://repo.amd.com/rocm/packages/rhel10/x86_64/",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("gpg_key_url=https://repo.amd.com/gpg/rocm.gpg", output)


class GetContainerImageTest(unittest.TestCase):
    """Tests for get_container_image()."""

    def test_ubuntu_returns_ubuntu_image(self):
        self.assertEqual(
            get_url_repo_params.get_container_image("ubuntu2404"),
            "ubuntu:24.04",
        )

    def test_debian_returns_ubuntu_image(self):
        self.assertEqual(
            get_url_repo_params.get_container_image("debian12"),
            "ubuntu:24.04",
        )

    def test_sles_returns_bci_image(self):
        self.assertEqual(
            get_url_repo_params.get_container_image("sles16"),
            "registry.suse.com/bci/bci-base:16.0",
        )

    def test_rhel_returns_ubi_image(self):
        self.assertEqual(
            get_url_repo_params.get_container_image("rhel10"),
            "registry.access.redhat.com/ubi10/ubi:10.1",
        )

    def test_get_container_image_subcommand(self):
        # Test that get-container-image writes container_image= to GITHUB_OUTPUT.
        code, output = _run_main_with_output(
            ["get-container-image", "--os-profile", "ubuntu2404"]
        )
        self.assertEqual(code, 0)
        self.assertIn("container_image=ubuntu:24.04", output)


if __name__ == "__main__":
    unittest.main()
