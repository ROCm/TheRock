#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Unit test coverage for get_url_repo_params.py:
#   get_base_url, get_base_url_from_release_type, get_gpg_key_url,
#   get_gpg_key_url_from_release_type, derive_gpg_key_url_for_repo_outputs,
#   gpg_key_url_needed_for_release_type,
#   get_repo_sub_folder,
#   get_repo_url, get_repo_url_per_family, get_repo_url_multi_arch, normalize_layout,
#   get_native_package_type_from_os_profile, extract_gfx_arch,
#   ContractLegacyAndDerivedTest (explicit vs derived/minimal parity),
#   and main() subcommands.

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


class GetBaseUrlFromReleaseTypeTest(unittest.TestCase):
    """Tests for get_base_url_from_release_type()."""

    def test_known_release_lines(self):
        self.assertEqual(
            get_url_repo_params.get_base_url_from_release_type("prerelease"),
            "https://rocm.prereleases.amd.com",
        )
        self.assertEqual(
            get_url_repo_params.get_base_url_from_release_type("Prereleases"),
            "https://rocm.prereleases.amd.com",
        )
        self.assertEqual(
            get_url_repo_params.get_base_url_from_release_type("release"),
            "https://repo.amd.com",
        )
        self.assertEqual(
            get_url_repo_params.get_base_url_from_release_type("stable"),
            "https://repo.amd.com",
        )
        self.assertEqual(
            get_url_repo_params.get_base_url_from_release_type("nightly"),
            "https://rocm.nightlies.amd.com",
        )
        self.assertEqual(
            get_url_repo_params.get_base_url_from_release_type("nightlies"),
            "https://rocm.nightlies.amd.com",
        )
        self.assertEqual(
            get_url_repo_params.get_base_url_from_release_type("dev"),
            "https://rocm.devreleases.amd.com",
        )

    def test_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_url_repo_params.get_base_url_from_release_type("ci")
        self.assertIn("Unknown release_type", str(ctx.exception))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            get_url_repo_params.get_base_url_from_release_type("")
        with self.assertRaises(ValueError):
            get_url_repo_params.get_base_url_from_release_type("   ")


class GetGpgKeyUrlTest(unittest.TestCase):
    """Tests for get_gpg_key_url()."""

    def test_extracts_base_and_adds_gpg_path(self):
        # Test that get_gpg_key_url keeps the /packages prefix and appends gpg/rocm.gpg.
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url(
                "https://rocm.prereleases.amd.com/packages/ubuntu2404"
            ),
            "https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg",
        )

    def test_strips_path_from_url(self):
        # Test that get_gpg_key_url keeps /rocm/packages and appends gpg/rocm.gpg.
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url(
                "https://repo.amd.com/rocm/packages/rhel10/x86_64/"
            ),
            "https://repo.amd.com/rocm/packages/gpg/rocm.gpg",
        )

    def test_handles_nightly_url(self):
        # No /packages/ in path: fall back to .../packages/gpg/rocm.gpg on the host.
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url(
                "https://rocm.nightlies.amd.com/deb/20260204-12345/"
            ),
            "https://rocm.nightlies.amd.com/packages/gpg/rocm.gpg",
        )

    def test_handles_multi_arch_repo_url(self):
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url(
                "https://rocm.nightlies.amd.com/packages-multi-arch/deb/20260204-12345/"
            ),
            "https://rocm.nightlies.amd.com/packages-multi-arch/gpg/rocm.gpg",
        )

    def test_repo_amd_com_without_packages_segment(self):
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url("https://repo.amd.com/"),
            "https://repo.amd.com/rocm/packages/gpg/rocm.gpg",
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
            get_url_repo_params.gpg_key_url_needed_for_release_type("prereleases")
        )
        self.assertTrue(
            get_url_repo_params.gpg_key_url_needed_for_release_type("release")
        )
        self.assertTrue(
            get_url_repo_params.gpg_key_url_needed_for_release_type("stable")
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


class GetGpgKeyUrlFromReleaseTypeTest(unittest.TestCase):
    """Tests for get_gpg_key_url_from_release_type()."""

    def test_prerelease_and_release_hosts(self):
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url_from_release_type("prerelease"),
            "https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg",
        )
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url_from_release_type("stable"),
            "https://repo.amd.com/rocm/packages/gpg/rocm.gpg",
        )

    def test_multi_arch_layout_hosts(self):
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url_from_release_type(
                "prerelease", layout="multi_arch"
            ),
            "https://rocm.prereleases.amd.com/packages-multi-arch/gpg/rocm.gpg",
        )
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url_from_release_type(
                "stable", layout="multiarch"
            ),
            "https://repo.amd.com/packages-multi-arch/gpg/rocm.gpg",
        )

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_url_repo_params.get_gpg_key_url_from_release_type("ci")


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

    def test_finds_release_id_in_multi_arch_s3_prefix(self):
        self.assertEqual(
            get_url_repo_params.get_repo_sub_folder(
                "12345678-linux/packages/deb/20260204-12345"
            ),
            "20260204-12345",
        )


class NormalizeLayoutTest(unittest.TestCase):
    """Tests for normalize_layout()."""

    def test_defaults_to_per_family(self):
        self.assertEqual(
            get_url_repo_params.normalize_layout(None),
            get_url_repo_params.LAYOUT_PER_FAMILY,
        )
        self.assertEqual(
            get_url_repo_params.normalize_layout(""),
            get_url_repo_params.LAYOUT_PER_FAMILY,
        )

    def test_aliases(self):
        self.assertEqual(
            get_url_repo_params.normalize_layout("legacy"),
            get_url_repo_params.LAYOUT_PER_FAMILY,
        )
        self.assertEqual(
            get_url_repo_params.normalize_layout("multiarch"),
            get_url_repo_params.LAYOUT_MULTI_ARCH,
        )

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_url_repo_params.normalize_layout("unknown")


class GetNativePackageTypeFromOsProfileTest(unittest.TestCase):
    """Tests for get_native_package_type_from_os_profile()."""

    def test_ubuntu_debian_deb(self):
        self.assertEqual(
            get_url_repo_params.get_native_package_type_from_os_profile("ubuntu2404"),
            "deb",
        )
        self.assertEqual(
            get_url_repo_params.get_native_package_type_from_os_profile("debian12"),
            "deb",
        )

    def test_rhel_sles_rpm(self):
        self.assertEqual(
            get_url_repo_params.get_native_package_type_from_os_profile("rhel10"),
            "rpm",
        )
        self.assertEqual(
            get_url_repo_params.get_native_package_type_from_os_profile("sles16"),
            "rpm",
        )

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            get_url_repo_params.get_native_package_type_from_os_profile("")


class GetRepoUrlTest(unittest.TestCase):
    """Tests for get_repo_url() default (per_family) layout."""

    def test_prereleases_alias_matches_prerelease(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="prereleases",
                native_package_type="deb",
                repo_base_url="https://repo_url.com",
                os_profile="ubuntu2404",
                repo_sub_folder="",
            ),
            "https://repo_url.com/packages/ubuntu2404",
        )

    def test_prerelease_deb(self):
        # native_packaging.md: .../packages/ubuntu2404
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="prerelease",
                native_package_type="deb",
                repo_base_url="https://repo_url.com",
                os_profile="ubuntu2404",
                repo_sub_folder="",
            ),
            "https://repo_url.com/packages/ubuntu2404",
        )

    def test_prerelease_rpm(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="prerelease",
                native_package_type="rpm",
                repo_base_url="https://repo_url.com",
                os_profile="rhel8",
                repo_sub_folder="",
            ),
            "https://repo_url.com/packages/rhel8/x86_64/",
        )

    def test_release_deb_matches_native_packaging_doc(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="release",
                native_package_type="deb",
                repo_base_url="https://repo.amd.com",
                os_profile="ubuntu2404",
                repo_sub_folder="",
            ),
            "https://repo.amd.com/rocm/packages/ubuntu2404",
        )

    def test_stable_rpm_matches_native_packaging_doc(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="stable",
                native_package_type="rpm",
                repo_base_url="https://repo.amd.com",
                os_profile="rhel10",
                repo_sub_folder="",
            ),
            "https://repo.amd.com/rocm/packages/rhel10/x86_64/",
        )

    def test_nightly_deb(self):
        # Test that non-prerelease + deb yields base/deb/repo_sub_folder/
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="deb",
                repo_base_url="https://repo_url.com",
                os_profile="ubuntu2404",
                repo_sub_folder="20260204-12345",
            ),
            "https://repo_url.com/deb/20260204-12345/",
        )

    def test_nightly_rpm(self):
        # Test that non-prerelease + rpm yields base/rpm/repo_sub_folder/x86_64/
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="rpm",
                repo_base_url="https://repo_url.com",
                os_profile="rhel8",
                repo_sub_folder="20260204-12345",
            ),
            "https://repo_url.com/rpm/20260204-12345/x86_64/",
        )

    def test_non_prerelease_empty_repo_subfolder_no_double_slash_deb(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="release",
                native_package_type="deb",
                repo_base_url="https://repo.amd.com",
                os_profile="ubuntu2404",
                repo_sub_folder="",
            ),
            "https://repo.amd.com/rocm/packages/ubuntu2404",
        )

    def test_non_prerelease_empty_repo_subfolder_no_double_slash_rpm(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="rpm",
                repo_base_url="https://repo.amd.com",
                os_profile="rhel10",
                repo_sub_folder="",
            ),
            "https://repo.amd.com/rpm/x86_64/",
        )

    def test_strips_trailing_slash_from_base(self):
        # Test that repo_base_url trailing slash is stripped.
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="prerelease",
                native_package_type="deb",
                repo_base_url="https://repo_url.com/",
                os_profile="ubuntu2404",
                repo_sub_folder="",
            ),
            "https://repo_url.com/packages/ubuntu2404",
        )

    def test_explicit_per_family_layout_matches_default(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="deb",
                repo_base_url="https://repo_url.com",
                os_profile="ubuntu2404",
                repo_sub_folder="20260204-12345",
                layout="per_family",
            ),
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="deb",
                repo_base_url="https://repo_url.com",
                os_profile="ubuntu2404",
                repo_sub_folder="20260204-12345",
            ),
        )


class GetRepoUrlMultiArchTest(unittest.TestCase):
    """Tests for get_repo_url(..., layout=multi_arch)."""

    def test_nightly_deb_matches_releases_doc(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="deb",
                repo_base_url="https://rocm.nightlies.amd.com",
                os_profile="ubuntu2404",
                repo_sub_folder="20260501-25200531110",
                layout="multi_arch",
            ),
            "https://rocm.nightlies.amd.com/packages-multi-arch/deb/20260501-25200531110",
        )

    def test_nightly_rpm_matches_releases_doc(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="rpm",
                repo_base_url="https://rocm.nightlies.amd.com",
                os_profile="rhel10",
                repo_sub_folder="20260501-25200531110",
                layout="multi_arch",
            ),
            "https://rocm.nightlies.amd.com/packages-multi-arch/rpm/20260501-25200531110/x86_64",
        )

    def test_os_profile_ignored_for_multi_arch(self):
        deb_url = get_url_repo_params.get_repo_url_multi_arch(
            repo_base_url="https://repo_url.com",
            native_package_type="deb",
            repo_sub_folder="20260204-1",
        )
        self.assertEqual(
            get_url_repo_params.get_repo_url(
                release_type="nightly",
                native_package_type="deb",
                repo_base_url="https://repo_url.com",
                os_profile="rhel10",
                repo_sub_folder="20260204-1",
                layout="multi_arch",
            ),
            deb_url,
        )

    def test_empty_release_id_index_urls(self):
        self.assertEqual(
            get_url_repo_params.get_repo_url_multi_arch("https://repo_url.com", "deb", ""),
            "https://repo_url.com/packages-multi-arch/deb",
        )
        self.assertEqual(
            get_url_repo_params.get_repo_url_multi_arch("https://repo_url.com/", "rpm", ""),
            "https://repo_url.com/packages-multi-arch/rpm/x86_64",
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

    def test_get_base_url_from_release_type_success(self):
        code, output = _run_main_with_output(
            ["get-base-url", "--release-type", "nightly"]
        )
        self.assertEqual(code, 0)
        self.assertIn("repo_base_url=https://rocm.nightlies.amd.com", output)

    def test_get_base_url_invalid_returns_one(self):
        # Test that get-base-url with invalid URL returns 1 and prints error.
        with patch("sys.stderr"):
            code = get_url_repo_params.main(["get-base-url", "--from-url", "not-a-url"])
        self.assertEqual(code, 1)

    def test_get_base_url_unknown_release_type_returns_one(self):
        with patch("sys.stderr"):
            code = get_url_repo_params.main(
                ["get-base-url", "--release-type", "unknown-channel"]
            )
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
                "https://repo_url.com",
                "--os-profile",
                "ubuntu2404",
                "--repo-sub-folder",
                "",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("repo_url=https://repo_url.com/packages/ubuntu2404", output)
        self.assertIn(
            "gpg_key_url=https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg",
            output,
        )

    def test_get_repo_url_minimal_prerelease_ubuntu(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "prerelease",
                "--os-profile",
                "ubuntu2404",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "repo_url=https://rocm.prereleases.amd.com/packages/ubuntu2404",
            output,
        )
        self.assertIn(
            "gpg_key_url=https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg",
            output,
        )

    def test_get_repo_url_minimal_nightly_with_subfolder(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "nightly",
                "--os-profile",
                "ubuntu2404",
                "--repo-sub-folder",
                "20260204-12345",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "repo_url=https://rocm.nightlies.amd.com/deb/20260204-12345/",
            output,
        )
        self.assertIn("gpg_key_url=", output)
        self.assertNotIn("rocm.gpg", output)

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
                        "https://repo_url.com",
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

    def test_get_repo_url_with_from_url_override_prerelease_gpg(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "prerelease",
                "--os-profile",
                "ubuntu2404",
                "--from-url",
                "https://rocm.prereleases.amd.com/packages/ubuntu2404",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "gpg_key_url=https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg", output
        )

    def test_get_repo_url_release_with_from_url_gpg(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "release",
                "--os-profile",
                "rhel10",
                "--native-package-type",
                "rpm",
                "--from-url",
                "https://repo.amd.com/rocm/packages/rhel10/x86_64/",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "gpg_key_url=https://repo.amd.com/rocm/packages/gpg/rocm.gpg", output
        )

    def test_get_repo_url_release_minimal_gpg_only_release_host(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "release",
                "--os-profile",
                "ubuntu2404",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "gpg_key_url=https://repo.amd.com/rocm/packages/gpg/rocm.gpg", output
        )

    def test_get_repo_url_dev_unsigned_empty_gpg(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "dev",
                "--os-profile",
                "ubuntu2404",
                "--from-url",
                "https://rocm.prereleases.amd.com/packages/ubuntu2404",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("gpg_key_url=", output)
        self.assertNotIn("rocm.gpg", output)

    def test_get_repo_url_nightly_invalid_from_url_still_unsigned(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "nightly",
                "--os-profile",
                "ubuntu2404",
                "--from-url",
                "not-a-valid-url",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("gpg_key_url=", output)
        self.assertNotIn("rocm.gpg", output)

    def test_get_repo_url_multi_arch_layout_cli(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--layout",
                "multi_arch",
                "--release-type",
                "nightly",
                "--os-profile",
                "ubuntu2404",
                "--repo-sub-folder",
                "20260501-25200531110",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "repo_url=https://rocm.nightlies.amd.com/packages-multi-arch/deb/20260501-25200531110",
            output,
        )
        self.assertIn("gpg_key_url=", output)
        self.assertNotIn("rocm.gpg", output)

    def test_get_repo_url_multi_arch_prerelease_signed_gpg(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--layout",
                "multi_arch",
                "--release-type",
                "prerelease",
                "--os-profile",
                "ubuntu2404",
                "--repo-sub-folder",
                "20260204-12345",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "repo_url=https://rocm.prereleases.amd.com/packages-multi-arch/deb/20260204-12345",
            output,
        )
        self.assertIn(
            "gpg_key_url=https://rocm.prereleases.amd.com/packages-multi-arch/gpg/rocm.gpg",
            output,
        )


class ContractLegacyAndDerivedTest(unittest.TestCase):
    """Explicit-input ('legacy') paths vs derived defaults must agree where intended."""

    def test_get_base_url_from_url_same_host_as_release_type_prerelease(self):
        self.assertEqual(
            get_url_repo_params.get_base_url(
                "https://rocm.prereleases.amd.com/packages/ubuntu2404"
            ),
            get_url_repo_params.get_base_url_from_release_type("prerelease"),
        )

    def test_get_gpg_key_url_from_sample_url_matches_release_type_only_prerelease(self):
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url(
                "https://rocm.prereleases.amd.com/packages/ubuntu2404"
            ),
            get_url_repo_params.get_gpg_key_url_from_release_type("prerelease"),
        )

    def test_get_gpg_key_url_sample_url_matches_release_type_release(self):
        self.assertEqual(
            get_url_repo_params.get_gpg_key_url(
                "https://repo.amd.com/rocm/packages/rhel10/x86_64/"
            ),
            get_url_repo_params.get_gpg_key_url_from_release_type("release"),
        )

    def test_get_repo_url_derived_components_equals_explicit_prerelease_ubuntu(self):
        explicit = get_url_repo_params.get_repo_url_per_family(
            release_type="prerelease",
            native_package_type="deb",
            repo_base_url="https://rocm.prereleases.amd.com",
            os_profile="ubuntu2404",
            repo_sub_folder="",
        )
        derived = get_url_repo_params.get_repo_url(
            release_type="prerelease",
            native_package_type=get_url_repo_params.get_native_package_type_from_os_profile(
                "ubuntu2404"
            ),
            repo_base_url=get_url_repo_params.get_base_url_from_release_type(
                "prerelease"
            ),
            os_profile="ubuntu2404",
            repo_sub_folder="",
        )
        self.assertEqual(explicit, derived)

    def test_cli_get_repo_url_minimal_matches_explicit_canonical_host(self):
        code_m, out_m = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "prerelease",
                "--os-profile",
                "ubuntu2404",
            ]
        )
        code_x, out_x = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "prerelease",
                "--native-package-type",
                "deb",
                "--repo-base-url",
                "https://rocm.prereleases.amd.com",
                "--os-profile",
                "ubuntu2404",
                "--repo-sub-folder",
                "",
            ]
        )
        self.assertEqual(code_m, 0)
        self.assertEqual(code_x, 0)
        self.assertEqual(out_m, out_x)

    def test_cli_get_repo_url_minimal_nightly_matches_explicit_same_parts(self):
        code_m, out_m = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "nightly",
                "--os-profile",
                "rhel10",
                "--repo-sub-folder",
                "20260204-9",
            ]
        )
        code_x, out_x = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "nightly",
                "--native-package-type",
                "rpm",
                "--repo-base-url",
                "https://rocm.nightlies.amd.com",
                "--os-profile",
                "rhel10",
                "--repo-sub-folder",
                "20260204-9",
            ]
        )
        self.assertEqual(out_m, out_x)

    def test_cli_get_repo_url_explicit_custom_base_differs_from_derived_minimal(self):
        _, out_derived = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "prerelease",
                "--os-profile",
                "ubuntu2404",
            ]
        )
        _, out_custom = _run_main_with_output(
            [
                "get-repo-url",
                "--release-type",
                "prerelease",
                "--native-package-type",
                "deb",
                "--repo-base-url",
                "https://internal.example.com",
                "--os-profile",
                "ubuntu2404",
                "--repo-sub-folder",
                "",
            ]
        )
        self.assertIn("rocm.prereleases.amd.com", out_derived)
        self.assertIn("internal.example.com", out_custom)
        self.assertIn("/packages/ubuntu2404", out_custom)
        self.assertNotEqual(out_derived, out_custom)


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

    def test_rhel8_returns_ubi8_image(self):
        self.assertEqual(
            get_url_repo_params.get_container_image("rhel8"),
            "registry.access.redhat.com/ubi8/ubi:8.10",
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
