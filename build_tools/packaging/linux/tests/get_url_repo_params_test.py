#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for ``build_tools/packaging/linux/get_url_repo_params.py``.

Canonical spec coverage for native Linux install-test parameters (repo URLs,
GPG paths, GPU arch tokens, container images). Exercises helpers and CLI
subcommands that write ``KEY=value`` lines for ``$GITHUB_OUTPUT``.

Coverage (trimmed suite, table-driven via ``subTest``):

  - P0 per-family repo URLs (``…/packages/``, ``…/rocm/packages/``, nightly deb|rpm)
  - P1 multi-arch layout (``packages-multi-arch/…``)
  - GPG beside packages tree + signed-line hosts + derivation policy
  - ``normalize_layout``, fail-fast ``release_type`` / unknown layout
  - Wired CI today: ``extract-gfx-arch``, ``get-container-image`` (+ CLI smoke each)
  - One happy-path CLI smoke per remaining subcommand

Not covered here (P2 / separate PRs): ``upload_package_repo._package_install_url``,
full CLI error-matrix.

Prerequisites:

  - Python 3.10 or newer
  - Run from TheROCK repository root
  - Stdlib only; ``$GITHUB_OUTPUT`` is mocked to a temp file

Run::

  python3 -m unittest \\
    build_tools.packaging.linux.tests.get_url_repo_params_test -v

  python3.12 build_tools/packaging/linux/tests/get_url_repo_params_test.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

# Resolve packaging modules from linux/ and build_tools/ (style guide).
_LINUX_DIR = Path(__file__).resolve().parent.parent
_BUILD_TOOLS_DIR = _LINUX_DIR.parent.parent
for _path in (_BUILD_TOOLS_DIR, _LINUX_DIR):
    _path_str = os.fspath(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import get_url_repo_params  # noqa: E402

_EXAMPLE = get_url_repo_params.EXAMPLE_CDN_BASE


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


def _repo_url(**kwargs: str) -> str:
    """Shorthand for get_repo_url with common defaults."""
    defaults = {
        "release_type": "prerelease",
        "native_package_type": "deb",
        "repo_base_url": _EXAMPLE,
        "os_profile": "ubuntu2404",
        "repo_sub_folder": "",
    }
    defaults.update(kwargs)
    layout = defaults.pop("layout", None)
    return get_url_repo_params.get_repo_url(**defaults, layout=layout)


class GetBaseUrlTest(unittest.TestCase):
    """Tests for ``get_base_url()``."""

    def test_strips_to_scheme_and_netloc(self):
        self.assertEqual(
            get_url_repo_params.get_base_url(f"{_EXAMPLE}/v2/whl?q=1#x"), _EXAMPLE
        )

    def test_invalid_url_raises(self):
        with self.assertRaises(ValueError):
            get_url_repo_params.get_base_url("not-a-url")


class GetGpgKeyUrlTest(unittest.TestCase):
    """Tests for ``get_gpg_key_url()``."""

    def test_gpg_paths_beside_packages_tree(self):
        cases = [
            (
                f"{_EXAMPLE}/packages/ubuntu2404",
                f"{_EXAMPLE}/packages/gpg/rocm.gpg",
            ),
            (
                f"{_EXAMPLE}/rocm/packages/rhel10/x86_64/",
                f"{_EXAMPLE}/rocm/packages/gpg/rocm.gpg",
            ),
            (
                f"{_EXAMPLE}/packages-multi-arch/deb/20260204-12345/",
                f"{_EXAMPLE}/packages-multi-arch/gpg/rocm.gpg",
            ),
            (
                f"{_EXAMPLE}/rocm/packages-multi-arch/ubuntu2404",
                f"{_EXAMPLE}/rocm/packages-multi-arch/gpg/rocm.gpg",
            ),
            (
                "https://repo.amd.com/",
                "https://repo.amd.com/rocm/packages/gpg/rocm.gpg",
            ),
        ]
        for repo_url, gpg_url in cases:
            with self.subTest(repo_url=repo_url):
                self.assertEqual(get_url_repo_params.get_gpg_key_url(repo_url), gpg_url)


class GetGpgKeyUrlFromReleaseTypeTest(unittest.TestCase):
    """Tests for ``get_gpg_key_url_from_release_type()``."""

    def test_signed_release_hosts(self):
        cases = [
            (
                "prerelease",
                None,
                "https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg",
            ),
            ("stable", None, "https://repo.amd.com/rocm/packages/gpg/rocm.gpg"),
            (
                "prerelease",
                "multi_arch",
                "https://rocm.prereleases.amd.com/packages-multi-arch/gpg/rocm.gpg",
            ),
            (
                "stable",
                "multiarch",
                "https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg",
            ),
        ]
        for release_type, layout, expected in cases:
            with self.subTest(release_type=release_type, layout=layout):
                self.assertEqual(
                    get_url_repo_params.get_gpg_key_url_from_release_type(
                        release_type, layout=layout
                    ),
                    expected,
                )
        with self.assertRaises(ValueError):
            get_url_repo_params.get_gpg_key_url_from_release_type("ci")


class NormalizeLayoutTest(unittest.TestCase):
    """Tests for ``normalize_layout()``."""

    def test_normalize_layout(self):
        per_family = get_url_repo_params.LAYOUT_PER_FAMILY
        multi_arch = get_url_repo_params.LAYOUT_MULTI_ARCH
        for layout, expected in [
            (None, per_family),
            ("", per_family),
            ("legacy", per_family),
            ("multiarch", multi_arch),
        ]:
            with self.subTest(layout=layout):
                self.assertEqual(get_url_repo_params.normalize_layout(layout), expected)
        with self.assertRaises(ValueError):
            get_url_repo_params.normalize_layout("unknown")


class GpgKeyUrlNeededForReleaseTypeTest(unittest.TestCase):
    """Tests for ``gpg_key_url_needed_for_release_type()``."""

    def test_derivation_policy(self):
        self.assertTrue(get_url_repo_params.gpg_key_url_needed_for_release_type(None))
        for signed in ("prerelease", "prereleases", "release", "stable"):
            with self.subTest(release_type=signed):
                self.assertTrue(
                    get_url_repo_params.gpg_key_url_needed_for_release_type(signed)
                )
        for unsigned in ("dev", "nightly", "ci", ""):
            with self.subTest(release_type=unsigned):
                self.assertFalse(
                    get_url_repo_params.gpg_key_url_needed_for_release_type(unsigned)
                )


class GetRepoSubFolderTest(unittest.TestCase):
    """Tests for ``get_repo_sub_folder()``."""

    def test_extracts_date_artifact_from_last_segment(self):
        self.assertEqual(
            get_url_repo_params.get_repo_sub_folder("v3/packages/deb/20260204-12345"),
            "20260204-12345",
        )

    def test_non_matching_last_segment_returns_empty(self):
        self.assertEqual(
            get_url_repo_params.get_repo_sub_folder("v3/packages/deb/"), ""
        )


class GetRepoUrlPerFamilyTest(unittest.TestCase):
    """Tests for ``get_repo_url()`` per-family layout."""

    def test_url_shapes(self):
        cases = [
            ("prereleases", "deb", "ubuntu2404", "", f"{_EXAMPLE}/packages/ubuntu2404"),
            ("prerelease", "rpm", "rhel8", "", f"{_EXAMPLE}/packages/rhel8/x86_64/"),
            (
                "release",
                "deb",
                "ubuntu2404",
                "",
                f"{_EXAMPLE}/rocm/packages/ubuntu2404",
            ),
            ("stable", "rpm", "rhel10", "", f"{_EXAMPLE}/rocm/packages/rhel10/x86_64/"),
            (
                "nightly",
                "deb",
                "ubuntu2404",
                "20260204-12345",
                f"{_EXAMPLE}/deb/20260204-12345/",
            ),
            (
                "nightly",
                "rpm",
                "rhel8",
                "20260204-12345",
                f"{_EXAMPLE}/rpm/20260204-12345/x86_64/",
            ),
        ]
        for release_type, pkg_type, os_profile, sub_folder, expected in cases:
            with self.subTest(release_type=release_type, pkg_type=pkg_type):
                self.assertEqual(
                    _repo_url(
                        release_type=release_type,
                        native_package_type=pkg_type,
                        repo_sub_folder=sub_folder,
                        os_profile=os_profile,
                    ),
                    expected,
                )

    def test_fail_fast_on_bad_release_type(self):
        for release_type, msg in [
            ("typo-channel", "Unknown release_type"),
            ("", "cannot be empty"),
        ]:
            with self.subTest(release_type=release_type):
                with self.assertRaises(ValueError) as ctx:
                    _repo_url(release_type=release_type)
                self.assertIn(msg, str(ctx.exception))


class GetRepoUrlMultiArchTest(unittest.TestCase):
    """Tests for current repo.amd.com multi-arch Core package URLs."""

    def test_url_shapes(self):
        cases = [
            (
                "stable",
                "deb",
                "ubuntu2404",
                "",
                f"{_EXAMPLE}/rocm/core/packages/ubuntu2404",
            ),
            (
                "prerelease",
                "deb",
                "ubuntu2404",
                "",
                f"{_EXAMPLE}/rocm/core/packages/ubuntu2404",
            ),
            (
                "nightly",
                "deb",
                "ubuntu2404",
                "20260501-25200531110",
                f"{_EXAMPLE}/rocm/core/packages/deb/20260501-25200531110",
            ),
            (
                "nightly",
                "rpm",
                "rhel10",
                "20260501-25200531110",
                f"{_EXAMPLE}/rocm/core/packages/rpm/20260501-25200531110/x86_64",
            ),
        ]
        for release_type, pkg_type, os_profile, sub_folder, expected in cases:
            with self.subTest(release_type=release_type, pkg_type=pkg_type):
                self.assertEqual(
                    _repo_url(
                        release_type=release_type,
                        native_package_type=pkg_type,
                        repo_sub_folder=sub_folder,
                        os_profile=os_profile,
                        layout="multi_arch",
                    ),
                    expected,
                )


class ExtractGfxArchTest(unittest.TestCase):
    """Tests for ``extract_gfx_arch()``."""

    def test_single_artifact_group(self):
        self.assertEqual(get_url_repo_params.extract_gfx_arch("gfx94X-dcgpu"), "gfx94x")

    def test_list_artifact_groups(self):
        for groups, expected in [
            ("gfx94X-dcgpu,gfx1100-consumer", "gfx94x,gfx1100"),
            ("gfx94X-dcgpu;gfx1100-consumer", "gfx94x,gfx1100"),
        ]:
            with self.subTest(groups=groups):
                self.assertEqual(get_url_repo_params.extract_gfx_arch(groups), expected)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            get_url_repo_params.extract_gfx_arch("")


class GetContainerImageTest(unittest.TestCase):
    """Tests for ``get_container_image()``."""

    # test_native_linux_packages_install.yml feeds this value straight into its
    # container image, so each profile is pinned to the exact string rather than
    # compared against another call to the function under test.
    _EXPECTED_IMAGES = [
        ("ubuntu2404", "ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest"),
        ("debian12", "ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest"),
        ("sles16", "registry.suse.com/bci/bci-base:16.0"),
        ("rhel8", "registry.access.redhat.com/ubi8/ubi:8.10"),
        ("rhel10", "registry.access.redhat.com/ubi10/ubi:10.1"),
    ]

    def test_profile_mapping(self):
        for os_profile, expected in self._EXPECTED_IMAGES:
            with self.subTest(os_profile=os_profile):
                self.assertEqual(
                    get_url_repo_params.get_container_image(os_profile), expected
                )


class MainSubcommandsTest(unittest.TestCase):
    """One happy-path CLI smoke per subcommand (GITHUB_OUTPUT wiring)."""

    def test_get_base_url_cli(self):
        code, output = _run_main_with_output(
            ["get-base-url", "--from-url", f"{_EXAMPLE}/v2/whl"]
        )
        self.assertEqual(code, 0)
        self.assertIn(f"repo_base_url={_EXAMPLE}", output)

    def test_get_repo_sub_folder_cli(self):
        code, output = _run_main_with_output(
            ["get-repo-sub-folder", "--from-s3-prefix", "v3/deb/20260204-12345"]
        )
        self.assertEqual(code, 0)
        self.assertIn("repo_sub_folder=20260204-12345", output)

    def test_get_repo_url_cli(self):
        code, output = _run_main_with_output(
            [
                "get-repo-url",
                "--layout",
                "multi_arch",
                "--release-type",
                "stable",
                "--native-package-type",
                "deb",
                "--repo-base-url",
                _EXAMPLE,
                "--os-profile",
                "ubuntu2404",
                "--repo-sub-folder",
                "",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn(f"repo_url={_EXAMPLE}/rocm/core/packages/ubuntu2404", output)

    def test_get_gpg_url_cli(self):
        code, output = _run_main_with_output(
            [
                "get-gpg-url",
                "--release-type",
                "dev",
                "--from-url",
                f"{_EXAMPLE}/packages/ubuntu2404",
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(output.strip(), "gpg_key_url=")

    def test_extract_gfx_arch_cli(self):
        code, output = _run_main_with_output(
            ["extract-gfx-arch", "--artifact-group", "gfx94X-dcgpu,gfx1100-consumer"]
        )
        self.assertEqual(code, 0)
        self.assertIn("gfx_arch=gfx94x,gfx1100", output)

    def test_get_container_image_cli(self):
        code, output = _run_main_with_output(
            ["get-container-image", "--os-profile", "ubuntu2404"]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "container_image=ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest", output
        )


class GetPublicRepoBaseUrlTest(unittest.TestCase):
    """Tests for the RFC0012 stream mapping and its subcommand."""

    def test_release_maps_to_the_stable_stream(self):
        self.assertEqual(
            get_url_repo_params.get_public_repo_base_url("release"),
            "https://stable.repo.amd.com/rocm/core/packages",
        )

    def test_nightly_maps_to_the_nightly_stream(self):
        self.assertEqual(
            get_url_repo_params.get_public_repo_base_url("nightly"),
            "https://nightly.repo.amd.com/rocm/core/packages",
        )

    def test_prerelease_has_no_stream_yet(self):
        # The documented mapping is prerelease -> rc, but rc.repo.amd.com serves
        # nothing on any distro, so a package pointing there could not refresh.
        # Empty means "no amdrocm-repo package for this line", which is how the
        # workflow skips it. Restoring it is one entry plus a test.
        self.assertEqual(get_url_repo_params.get_public_repo_base_url("prerelease"), "")

    def test_ci_and_dev_are_empty(self):
        self.assertEqual(get_url_repo_params.get_public_repo_base_url("ci"), "")
        self.assertEqual(get_url_repo_params.get_public_repo_base_url("dev"), "")

    def test_unknown_line_is_empty(self):
        self.assertEqual(get_url_repo_params.get_public_repo_base_url("bogus"), "")

    def test_case_insensitive(self):
        self.assertEqual(
            get_url_repo_params.get_public_repo_base_url("Release"),
            "https://stable.repo.amd.com/rocm/core/packages",
        )

    def test_key_url_is_outside_the_packages_base(self):
        # packages live at <root>/core/packages/, the key at <root>/gpg/, so
        # neither URL can be reached from the other by appending or trimming one
        # segment. The builder takes the key URL whole for that reason.
        base = get_url_repo_params.get_public_repo_base_url("release")
        key = get_url_repo_params.get_public_repo_gpg_key_url("release")
        self.assertEqual(key, "https://stable.repo.amd.com/rocm/gpg/packages.gpg")
        self.assertFalse(key.startswith(base))
        self.assertFalse(base.startswith(key))

    def test_unsigned_stream_has_no_key_url(self):
        # nightly serves no InRelease, no Release.gpg and no repomd.xml.asc, and
        # its gpg/packages.gpg is a 404, so it must not advertise a key.
        self.assertNotEqual(get_url_repo_params.get_public_repo_base_url("nightly"), "")
        self.assertEqual(get_url_repo_params.get_public_repo_gpg_key_url("nightly"), "")

    def test_release_line_maps_to_a_differently_named_stream(self):
        # The vocabularies differ: the "release" build line configures the
        # "stable" stream. Pin it so the two are not conflated.
        self.assertEqual(
            get_url_repo_params.get_public_repo_stream("release"), "stable"
        )
        self.assertEqual(
            get_url_repo_params.get_public_repo_stream("nightly"), "nightly"
        )
        self.assertEqual(get_url_repo_params.get_public_repo_stream("prerelease"), "")

    def test_subcommand_emits_url_and_key(self):
        code, output = _run_main_with_output(
            ["get-public-repo-base-url", "--release-type", "release"]
        )
        self.assertEqual(code, 0)
        self.assertIn(
            "repo_base_url=https://stable.repo.amd.com/rocm/core/packages",
            output,
        )
        self.assertIn("gpg_key_url=https://stable.repo.amd.com/rocm", output)
        self.assertIn("stream=stable", output)

    def test_subcommand_emits_empty_for_ci(self):
        code, output = _run_main_with_output(
            ["get-public-repo-base-url", "--release-type", "ci"]
        )
        self.assertEqual(code, 0)
        self.assertIn("repo_base_url=", output)
        self.assertNotIn("packages-multi-arch", output)


class NightlySubFolderTest(unittest.TestCase):
    """Tests for nightly_sub_folder() and its subcommand."""

    def test_formats_with_injected_date(self):
        self.assertEqual(
            get_url_repo_params.nightly_sub_folder("12345", today=date(2026, 7, 16)),
            "20260716-12345",
        )

    def test_zero_padded_month_and_day(self):
        self.assertEqual(
            get_url_repo_params.nightly_sub_folder("7", today=date(2026, 1, 3)),
            "20260103-7",
        )

    def test_rejects_non_numeric_run_id(self):
        # The value is written to $GITHUB_OUTPUT, and gha_set_output uses a
        # heredoc whose delimiter it does not verify, so an embedded newline
        # could terminate it early and inject further step outputs.
        for bad in ["1\nEOF_mag1c\ninjected=yes", "a/b", "12345\n", "", "abc"]:
            with self.subTest(run_id=bad):
                with self.assertRaises(ValueError):
                    get_url_repo_params.nightly_sub_folder(bad)

    def test_subcommand_writes_exactly_one_output_for_hostile_input(self):
        code, output = _run_main_with_output(
            [
                "get-nightly-sub-folder",
                "--release-type",
                "nightly",
                "--run-id",
                "1\nEOF_mag1c\ninjected=yes",
            ]
        )
        self.assertNotEqual(code, 0)
        self.assertNotIn("injected=yes", output)

    def test_subcommand_emits_sub_folder(self):
        code, output = _run_main_with_output(
            ["get-nightly-sub-folder", "--run-id", "12345"]
        )
        self.assertEqual(code, 0)
        # The date is the current UTC date; assert the structure, not the value.
        self.assertRegex(output, r"repo_sub_folder=\d{8}-12345")

    def test_subcommand_emits_sub_folder_for_nightly(self):
        code, output = _run_main_with_output(
            ["get-nightly-sub-folder", "--release-type", "nightly", "--run-id", "12345"]
        )
        self.assertEqual(code, 0)
        self.assertRegex(output, r"repo_sub_folder=\d{8}-12345")

    def test_subcommand_emits_empty_for_other_lines(self):
        # Only the nightly line publishes into a dated sub-folder; the other
        # lines publish at the repository root.
        for release_type in ("release", "prerelease", "ci"):
            with self.subTest(release_type=release_type):
                code, output = _run_main_with_output(
                    [
                        "get-nightly-sub-folder",
                        "--release-type",
                        release_type,
                        "--run-id",
                        "12345",
                    ]
                )
                self.assertEqual(code, 0)
                self.assertIn("repo_sub_folder=", output)
                self.assertNotRegex(output, r"repo_sub_folder=\S")


class GetContainerImageMapTest(unittest.TestCase):
    """Tests for get_container_image_map() and its subcommand."""

    def test_maps_rpm_profiles(self):
        self.assertEqual(
            get_url_repo_params.get_container_image_map(["rhel8", "sles16"]),
            {
                "rhel8": "registry.access.redhat.com/ubi8/ubi:8.10",
                "sles16": "registry.suse.com/bci/bci-base:16.0",
            },
        )

    def test_subcommand_emits_json_map(self):
        code, output = _run_main_with_output(
            ["get-container-image-map", "--os-profiles", '["ubuntu2404"]']
        )
        self.assertEqual(code, 0)
        # Recover the JSON value written after ``images=``.
        line = next(l for l in output.splitlines() if l.startswith("images="))
        images = json.loads(line[len("images=") :])
        self.assertEqual(
            images, {"ubuntu2404": "ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest"}
        )

    def test_subcommand_rejects_invalid_json(self):
        code, _ = _run_main_with_output(
            ["get-container-image-map", "--os-profiles", "not-json"]
        )
        self.assertEqual(code, 1)

    def test_subcommand_rejects_non_list_json(self):
        code, _ = _run_main_with_output(
            ["get-container-image-map", "--os-profiles", '{"a": 1}']
        )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
