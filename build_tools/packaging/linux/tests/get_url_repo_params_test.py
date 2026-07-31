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
#7004 container-image string assertions, full CLI error-matrix.

Prerequisites:

  - Python 3.10 or newer
  - Run from TheROCK repository root
  - Stdlib only; ``$GITHUB_OUTPUT`` is mocked to a temp file

Run::

  python3 -m unittest \\
    build_tools.packaging.linux.tests.get_url_repo_params_test -v

  python3.12 build_tools/packaging/linux/tests/get_url_repo_params_test.py -v
"""

import os
import sys
import tempfile
import unittest
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
    """Tests for ``get_repo_url(..., layout=multi_arch)``."""

    def test_url_shapes(self):
        cases = [
            (
                "stable",
                "deb",
                "ubuntu2404",
                "",
                f"{_EXAMPLE}/rocm/packages-multi-arch/ubuntu2404",
            ),
            (
                "prerelease",
                "deb",
                "ubuntu2404",
                "",
                f"{_EXAMPLE}/packages-multi-arch/ubuntu2404",
            ),
            (
                "nightly",
                "deb",
                "ubuntu2404",
                "20260501-25200531110",
                f"{_EXAMPLE}/packages-multi-arch/deb/20260501-25200531110",
            ),
            (
                "nightly",
                "rpm",
                "rhel10",
                "20260501-25200531110",
                f"{_EXAMPLE}/packages-multi-arch/rpm/20260501-25200531110/x86_64",
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
    """Tests for ``get_container_image()`` (asserts against implementation output)."""

    def test_profile_mapping(self):
        # Ubuntu/debian share one image; sles/rhel use distinct registry images.
        ubuntu = get_url_repo_params.get_container_image("ubuntu2404")
        self.assertEqual(get_url_repo_params.get_container_image("debian12"), ubuntu)
        self.assertEqual(
            get_url_repo_params.get_container_image("sles16"),
            "registry.suse.com/bci/bci-base:16.0",
        )
        self.assertEqual(
            get_url_repo_params.get_container_image("rhel10"),
            "registry.access.redhat.com/ubi10/ubi:10.1",
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
        self.assertIn(
            f"repo_url={_EXAMPLE}/rocm/packages-multi-arch/ubuntu2404", output
        )

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
        expected = get_url_repo_params.get_container_image("ubuntu2404")
        code, output = _run_main_with_output(
            ["get-container-image", "--os-profile", "ubuntu2404"]
        )
        self.assertEqual(code, 0)
        self.assertIn(f"container_image={expected}", output)


if __name__ == "__main__":
    unittest.main()
