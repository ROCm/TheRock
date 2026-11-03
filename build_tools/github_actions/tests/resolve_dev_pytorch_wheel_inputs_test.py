# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import resolve_dev_pytorch_wheel_inputs as m
from _therock_utils.cmake_amdgpu_targets import amdgpu_family_map

STABLE = "https://stable.repo.amd.com/rocm/whl-next/"

BUILD_SH = """
case "$tag" in
  pytorch-linux-jammy-py3.14-clang21)
    ANACONDA_PYTHON_VERSION=3.14
    ;;
  pytorch-linux-jammy-rocm-n-py3 | pytorch-linux-noble-rocm-n-py3.11)
    ANACONDA_PYTHON_VERSION=3.11
    ROCM_VERSION=10.1
    THEROCK_INDEX_URL="https://stable.repo.amd.com/rocm/whl-next/"
    ;;
  pytorch-linux-noble-rocm-preview-py3.11)
    ROCM_VERSION=$(tr -d '[:space:]' < rocm-preview.txt)
    ;;
esac
"""

STABLE_PAGE = """
<a href="rocm-10.0.0.tar.gz#sha256=aa">rocm-10.0.0.tar.gz</a><br/>
<a href="rocm-10.1.0.tar.gz#sha256=bb">rocm-10.1.0.tar.gz</a><br/>
<a href="rocm_sdk_core-10.2.0-py3-none-linux_x86_64.whl">x</a><br/>
"""

NIGHTLY_PAGE = """
<a href="rocm-10.3.0a20261102.tar.gz">rocm-10.3.0a20261102.tar.gz</a>
<a href="rocm-10.3.0a20261103.tar.gz">rocm-10.3.0a20261103.tar.gz</a>
"""


def _families(**overrides):
    kwargs = dict(
        build_linux=True,
        build_windows=False,
        datacenter_instinct_gpus=False,
        consumer_radeon_gpus=False,
        extra_amdgpu_families="",
        known_families=set(amdgpu_family_map()),
    )
    kwargs.update(overrides)
    return m.resolve_families(**kwargs)


class ResolveFamiliesTest(unittest.TestCase):
    def test_checkbox_families_exist_in_cmake(self):
        known = set(amdgpu_family_map())
        for family in m.DATACENTER_INSTINCT_FAMILIES + m.CONSUMER_RADEON_FAMILIES:
            self.assertIn(family, known)

    def test_datacenter_is_linux_only(self):
        families = _families(build_windows=True, datacenter_instinct_gpus=True)
        self.assertEqual(families["linux"], list(m.DATACENTER_INSTINCT_FAMILIES))
        self.assertEqual(families["windows"], [])

    def test_consumer_builds_on_both_platforms(self):
        families = _families(build_windows=True, consumer_radeon_gpus=True)
        self.assertEqual(families["linux"], list(m.CONSUMER_RADEON_FAMILIES))
        self.assertEqual(families["windows"], list(m.CONSUMER_RADEON_FAMILIES))

    def test_extras_are_appended_and_deduplicated(self):
        families = _families(
            consumer_radeon_gpus=True,
            extra_amdgpu_families="gfx1151; gfx1150,gfx94X-dcgpu",
        )
        self.assertEqual(
            families["linux"],
            [*m.CONSUMER_RADEON_FAMILIES, "gfx1150", "gfx94X-dcgpu"],
        )

    def test_unselected_platform_is_empty(self):
        families = _families(datacenter_instinct_gpus=True)
        self.assertEqual(families["windows"], [])

    def test_unknown_extra_family_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gfx999"):
            _families(extra_amdgpu_families="gfx999")

    def test_nothing_selected_is_rejected(self):
        with self.assertRaises(ValueError):
            _families()
        with self.assertRaises(ValueError):
            _families(
                build_linux=False, build_windows=True, datacenter_instinct_gpus=True
            )


class UpstreamRocmVersionTest(unittest.TestCase):
    def test_reads_the_trunk_image_branch(self):
        self.assertEqual(m.parse_upstream_rocm_version(BUILD_SH), "10.1")

    def test_missing_image_is_an_error(self):
        with self.assertRaises(ValueError):
            m.parse_upstream_rocm_version(BUILD_SH, image="pytorch-linux-unknown")

    def test_non_literal_version_is_an_error(self):
        with self.assertRaises(ValueError):
            m.parse_upstream_rocm_version(
                BUILD_SH, image="pytorch-linux-noble-rocm-preview-py3.11"
            )


class ResolveRocmVersionTest(unittest.TestCase):
    def _resolve(self, requested, page=STABLE_PAGE):
        return m.resolve_rocm_version(
            requested,
            STABLE,
            fetch_url=lambda url: page,
            fetch_upstream_build_sh=lambda: BUILD_SH,
        )

    def test_index_parsing_ignores_other_packages(self):
        self.assertEqual(m.parse_index_versions(STABLE_PAGE), ["10.0.0", "10.1.0"])

    def test_trunk_resolves_to_the_newest_matching_release(self):
        self.assertEqual(self._resolve("trunk")[0], "10.1.0")

    def test_trunk_without_a_match_points_at_latest(self):
        with self.assertRaisesRegex(ValueError, "latest"):
            self._resolve("trunk", page=NIGHTLY_PAGE)

    def test_latest_picks_the_newest_version(self):
        self.assertEqual(
            self._resolve("latest", page=NIGHTLY_PAGE)[0], "10.3.0a20261103"
        )

    def test_exact_and_series_versions(self):
        self.assertEqual(self._resolve("10.0.0"), ("10.0.0", "exact"))
        self.assertEqual(self._resolve("10.1")[0], "10.1.0")

    def test_unavailable_or_invalid_versions_are_rejected(self):
        with self.assertRaises(ValueError):
            self._resolve("9.9.9")
        with self.assertRaises(ValueError):
            self._resolve("not-a-version")


class MainTest(unittest.TestCase):
    def test_main_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "github_output"
            with mock.patch.dict(
                os.environ, {"GITHUB_OUTPUT": os.fspath(output)}
            ), mock.patch.object(
                m, "_fetch_url", return_value=STABLE_PAGE
            ), mock.patch.object(
                m, "gha_fetch_text_file_contents", return_value=BUILD_SH
            ):
                m.main(
                    [
                        "--build-windows",
                        "true",
                        "--consumer-radeon-gpus",
                        "true",
                        "--rocm-package-index-url",
                        STABLE,
                    ]
                )
            families = ";".join(m.CONSUMER_RADEON_FAMILIES)
            self.assertEqual(
                output.read_text().splitlines(),
                [
                    f"linux_amdgpu_families={families}",
                    f"windows_amdgpu_families={families}",
                    "rocm_version=10.1.0",
                ],
            )


if __name__ == "__main__":
    unittest.main()
