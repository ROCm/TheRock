# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the keep-list (--multi-arch-targets) logic in
promote_packages.py.

These tests exercise the pure-function helpers (multi-arch detection, keep-list
application to METADATA / requires.txt / _dist_info.py, filename-based per-target
arch extraction, version string promotion) on synthetic inputs. They run
without network access and complete in well under a second.

The end-to-end integration tests that download real RC packages live in
promote_packages_test.py and are not exercised here.
"""

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import promote_packages as ptf


def _write_tmp(suffix: str, body: str) -> Path:
    """Write `body` to a fresh temp file with `suffix` and return its path.
    Caller is responsible for unlinking.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(textwrap.dedent(body))
        return Path(f.name)


def get_multi_arch_metadata_torch_archs_body() -> str:
    return """\
Metadata-Version: 2.2
Name: torch
Version: 2.8.0+rocm7.13.0a20260505
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Dynamic: requires-dist
Dynamic: requires-python
Dynamic: summary
Requires-Dist: rocm-bootstrap
Provides-Extra: device-gfx1010
Requires-Dist: amd-torch-device-gfx1010 == 2.8.0+rocm7.13.0a20260505; extra == "device-gfx1010"
Provides-Extra: device-gfx1011
Requires-Dist: amd-torch-device-gfx1011 == 2.8.0+rocm7.13.0a20260505; extra == "device-gfx1011"
Provides-Extra: device-gfx1012
Requires-Dist: amd-torch-device-gfx1012 == 2.8.0+rocm7.13.0a20260505; extra == "device-gfx1012"
Provides-Extra: device-gfx1030
Requires-Dist: amd-torch-device-gfx1030 == 2.8.0+rocm7.13.0a20260505; extra == "device-gfx1030"
Provides-Extra: device-all
Requires-Dist: amd-torch-device-gfx1010 == 2.8.0+rocm7.13.0a20260505; extra == "device-all"
Requires-Dist: amd-torch-device-gfx1011 == 2.8.0+rocm7.13.0a20260505; extra == "device-all"
Requires-Dist: amd-torch-device-gfx1012 == 2.8.0+rocm7.13.0a20260505; extra == "device-all"
Requires-Dist: amd-torch-device-gfx1030 == 2.8.0+rocm7.13.0a20260505; extra == "device-all"

PyTorch is a Python package that provides two high-level features:
- Tensor computation (like NumPy) with strong GPU acceleration
"""


def get_multi_arch_metadata_archs_body() -> str:
    return """\
Metadata-Version: 2.4
Name: rocm
Version: 7.13.0a20260505
Requires-Dist: rocm==7.13.0a20260505
Requires-Dist: rocm-sdk-core==7.13.0a20260505
Provides-Extra: libraries
Requires-Dist: rocm-sdk-libraries==7.13.0a20260505; extra == "libraries"
Provides-Extra: device
Requires-Dist: rocm-sdk-device-gfx1010==7.13.0a20260505; extra == "device"
Provides-Extra: devel
Requires-Dist: rocm-sdk-devel==7.13.0a20260505; extra == "devel"
Provides-Extra: profiler
Requires-Dist: rocm-profiler==7.13.0a20260505; extra == "profiler"
Provides-Extra: device-gfx1010
Requires-Dist: rocm-sdk-device-gfx1010==7.13.0a20260505; extra == "device-gfx1010"
Provides-Extra: device-gfx1011
Requires-Dist: rocm-sdk-device-gfx1011==7.13.0a20260505; extra == "device-gfx1011"
Provides-Extra: device-gfx1012
Requires-Dist: rocm-sdk-device-gfx1012==7.13.0a20260505; extra == "device-gfx1012"
Provides-Extra: device-all
Requires-Dist: rocm-sdk-device-gfx1010==7.13.0a20260505; extra == "device-all"
Requires-Dist: rocm-sdk-device-gfx1011==7.13.0a20260505; extra == "device-all"
Requires-Dist: rocm-sdk-device-gfx1012==7.13.0a20260505; extra == "device-all"
Dynamic: provides-extra
Dynamic: requires-dist
"""


def get_multi_arch_metadata_no_archs_body() -> str:
    return """\
Metadata-Version: 2.4
Name: rocm-sdk-device-gfx1153
Version: 7.13.0a20260505
Requires-Dist: rocm-sdk-libraries==7.13.0a20260505
Dynamic: requires-dist
"""


def get_single_arch_metadata_body() -> str:
    return """\
Metadata-Version: 2.4
Name: rocm
Version: 7.13.0a20260505
Requires-Dist: rocm==7.13.0a20260505
Requires-Dist: rocm-sdk-core==7.13.0a20260505
Provides-Extra: libraries
Requires-Dist: rocm-sdk-libraries-gfx94X-dcgpu==7.13.0a20260505; extra == "libraries"
Provides-Extra: device
Requires-Dist: rocm-sdk-device-gfx94X-dcgpu==7.13.0a20260505; extra == "device"
Provides-Extra: devel
Requires-Dist: rocm-sdk-devel==7.13.0a20260505; extra == "devel"
Provides-Extra: profiler
Requires-Dist: rocm-profiler==7.13.0a20260505; extra == "profiler"
Dynamic: provides-extra
Dynamic: requires-dist
"""


class ScanMultiarchMetadataTest(unittest.TestCase):
    """Detection of multi-arch vs single-arch in METADATA / PKG-INFO."""

    def _scan(self, body: str) -> set[str]:
        path = _write_tmp(".METADATA", body)
        try:
            return ptf._scan_multiarch_metadata(path)
        finally:
            path.unlink()

    def test_torch_aggregator_with_extras_is_multiarch(self):
        archs = self._scan(get_multi_arch_metadata_torch_archs_body())
        self.assertEqual(archs, {"gfx1010", "gfx1011", "gfx1012", "gfx1030"})

    def test_rocm_aggregator_with_extras_is_multiarch(self):
        archs = self._scan(get_multi_arch_metadata_archs_body())
        self.assertEqual(archs, {"gfx1010", "gfx1011", "gfx1012"})

    def test_per_target_leaf_metadata_is_single_arch(self):
        # The leaf wheel METADATA names a `*-gfx<N>` dep but has no device-*
        # extras markers — must be treated as single-arch (no-op).
        archs = self._scan(get_single_arch_metadata_body())
        self.assertEqual(archs, set())


class ApplyKeepListToMetadataTest(unittest.TestCase):
    def _apply(self, body: str, keep: list[str]) -> str:
        path = _write_tmp(".METADATA", body)
        try:
            ptf._apply_keep_arch_list_to_metadata(path, keep)
            return path.read_text(encoding="utf-8")
        finally:
            path.unlink()

    def test_drops_non_kept_extras_and_requires(self):
        # Torch body: multi-arch with no bare `extra == "device"` line — pure
        # drop, no repoint.
        result = self._apply(get_multi_arch_metadata_torch_archs_body(), ["gfx1011"])
        self.assertNotIn("gfx1010", result)
        self.assertNotIn("gfx1012", result)
        self.assertNotIn("gfx1030", result)
        self.assertIn("Provides-Extra: device-all", result)
        self.assertIn("Provides-Extra: device-gfx1011", result)
        self.assertIn("amd-torch-device-gfx1011", result)
        # Non-arch deps are preserved.
        self.assertIn("Requires-Dist: rocm-bootstrap", result)

    def test_repoints_device_extra_when_default_arch_dropped(self):
        # rocm body: the `; extra == "device"` line names the package's default
        # arch (gfx1010). If that arch is dropped, the line must be repointed
        # at the keep[0] package — NOT removed (so the [device] extra still
        # has a dep).
        result = self._apply(
            get_multi_arch_metadata_archs_body(), ["gfx1011", "gfx1012"]
        )
        self.assertNotIn("gfx1010", result)
        self.assertIn(
            'Requires-Dist: rocm-sdk-device-gfx1011==7.13.0a20260505; extra == "device"',
            result,
        )

    def test_preserves_device_extra_when_default_arch_kept(self):
        # If the `extra == "device"` arch is in the keep list, leave it alone.
        result = self._apply(get_multi_arch_metadata_archs_body(), ["gfx1010"])
        self.assertNotIn("gfx1011", result)
        self.assertNotIn("gfx1012", result)
        self.assertIn(
            'Requires-Dist: rocm-sdk-device-gfx1010==7.13.0a20260505; extra == "device"',
            result,
        )

    def test_single_arch_metadata_is_left_unchanged(self):
        body = get_single_arch_metadata_body()
        result = self._apply(body, ["gfx950"])
        self.assertEqual(result, body)

    def test_empty_intersection_raises(self):
        path = _write_tmp(".METADATA", get_multi_arch_metadata_archs_body())
        try:
            with self.assertRaises(ValueError) as cm:
                ptf._apply_keep_arch_list_to_metadata(path, ["gfx9999"])
            self.assertIn("no overlap", str(cm.exception))
        finally:
            path.unlink()


class UpdateRunpathVersionTest(unittest.TestCase):
    def test_updates_rocm_version_in_rocm_package_runpaths(self):
        runpath = (
            "$ORIGIN/../../_rocm_sdk_core_rocm7.13.0a20260505/lib:"
            "$ORIGIN/../../_rocm_sdk_libraries_rocm7.13.0a20260505/lib"
        )

        self.assertEqual(
            ptf._update_runpath_version(runpath, "7.13.0a20260505", "7.13.0"),
            (
                "$ORIGIN/../../_rocm_sdk_core_rocm7.13.0/lib:"
                "$ORIGIN/../../_rocm_sdk_libraries_rocm7.13.0/lib"
            ),
        )

    def test_leaves_unversioned_runpath_unchanged(self):
        runpath = "$ORIGIN:$ORIGIN/../../_rocm_sdk_core/lib"

        self.assertEqual(
            ptf._update_runpath_version(runpath, "7.13.0a20260505", "7.13.0"),
            runpath,
        )


class ApplyKeepListToRequiresTxtTest(unittest.TestCase):
    @staticmethod
    def _get_requires_txt_multi_arch_body() -> str:
        return """\
rocm==7.13.0a20260505
rocm-sdk-core==7.13.0a20260505

[devel]
rocm-sdk-devel==7.13.0a20260505

[device]
rocm-sdk-device-gfx1010==7.13.0a20260505

[device-all]
rocm-sdk-device-gfx1010==7.13.0a20260505
rocm-sdk-device-gfx1011==7.13.0a20260505

[device-gfx1010]
rocm-sdk-device-gfx1010==7.13.0a20260505

[device-gfx1011]
rocm-sdk-device-gfx1011==7.13.0a20260505

[libraries]
rocm-sdk-libraries==7.13.0a20260505
"""

    @staticmethod
    def _get_requires_txt_single_arch_body() -> str:
        return """\
rocm==7.13.0a20260505
rocm-sdk-core==7.13.0a20260505

[devel]
rocm-sdk-devel==7.13.0a20260505

[device]
rocm-sdk-device-gfx94X-dcgpu==7.13.0a20260505

[libraries]
rocm-sdk-libraries-gfx94X-dcgpu==7.13.0a20260505
"""

    def _apply(self, body: str, keep: list[str]) -> str:
        path = _write_tmp(".txt", body)
        try:
            ptf._apply_keep_list_to_requires_txt(path, keep)
            return path.read_text(encoding="utf-8")
        finally:
            path.unlink()

    def test_drops_non_kept_section_and_filters_device_all(self):
        result = self._apply(self._get_requires_txt_multi_arch_body(), ["gfx1011"])
        # The dropped section header is gone entirely.
        self.assertNotIn("[device-gfx1010]", result)
        # Within [device-all], only the kept arch survives.
        self.assertNotIn("rocm-sdk-device-gfx1010", result)
        self.assertIn("[device-all]", result)
        self.assertIn("[device-gfx1011]", result)
        self.assertIn("rocm-sdk-device-gfx1011", result)
        # [device] body is repointed at the keep[0] body (not emptied).
        device_section = result.split("[device]", 1)[1].split("[", 1)[0]
        self.assertIn("rocm-sdk-device-gfx1011", device_section)

    def test_preserves_device_section_when_default_arch_kept(self):
        # If the `[device]` body's arch is already in the keep list, leave it alone.
        result = self._apply(self._get_requires_txt_multi_arch_body(), ["gfx1010"])
        self.assertNotIn("gfx1011", result)
        device_section = result.split("[device]", 1)[1].split("[", 1)[0]
        self.assertIn("rocm-sdk-device-gfx1010", device_section)

    def test_single_arch_requires_txt_is_left_unchanged(self):
        result = self._apply(self._get_requires_txt_single_arch_body(), ["gfx94X"])
        self.assertEqual(result, self._get_requires_txt_single_arch_body())


class ApplyKeepListToDistInfoPyTest(unittest.TestCase):
    def _dist_info_multi_arch_body(self) -> str:
        return """\
AVAILABLE_TARGET_FAMILIES: list[str] = []
DEFAULT_TARGET_FAMILY = 'gfx1010'
AVAILABLE_TARGET_FAMILIES.append('gfx1010')
AVAILABLE_TARGET_FAMILIES.append('gfx906')
AVAILABLE_TARGET_FAMILIES.append('gfx1030')
AVAILABLE_TARGET_FAMILIES.append('gfx1151')
AVAILABLE_TARGET_FAMILIES.append('gfx11')
AVAILABLE_TARGET_FAMILIES.append('gfx1201')
"""

    def _dist_info_single_arch_body(self) -> str:
        return """\
AVAILABLE_TARGET_FAMILIES: list[str] = []
DEFAULT_TARGET_FAMILY = 'gfx94X-dcgpu'
AVAILABLE_TARGET_FAMILIES.append('gfx94X-dcgpu')
"""

    def _apply(self, body: str, keep: list[str]) -> str:
        path = _write_tmp(".py", body)
        try:
            ptf._apply_keep_list_to_dist_info_py(path, keep)
            return path.read_text(encoding="utf-8")
        finally:
            path.unlink()

    def test_drops_non_kept_archs_and_repoints_default(self):
        # Default arch (gfx1010) is dropped → DEFAULT_TARGET_FAMILY repoints
        # at keep[0] (gfx906). Note the function always rewrites the default
        # line with double quotes, even if the source used single quotes.
        result = self._apply(self._dist_info_multi_arch_body(), ["gfx906", "gfx1030"])
        self.assertNotIn("gfx1010", result)
        self.assertNotIn("gfx1151", result)
        self.assertIn("AVAILABLE_TARGET_FAMILIES.append('gfx906')", result)
        self.assertIn("AVAILABLE_TARGET_FAMILIES.append('gfx1030')", result)
        self.assertIn('DEFAULT_TARGET_FAMILY = "gfx906"', result)

    def test_repoint_prefers_specific_arch_over_family_prefix(self):
        # Keep list mixes a family prefix (gfx11, 2 digits) and a specific
        # arch (gfx1201, 4 digits). When the default arch (gfx1010) is dropped,
        # `_repoint_priority` must prefer the specific arch — so the repoint
        # target is 'gfx1201', not 'gfx11', regardless of keep-list order.
        result = self._apply(self._dist_info_multi_arch_body(), ["gfx11", "gfx1201"])
        self.assertIn('DEFAULT_TARGET_FAMILY = "gfx1201"', result)

    def test_default_pointing_at_kept_arch_is_preserved(self):
        # Default arch (gfx1010) is in the keep list → line stays verbatim
        # (single-quoted, as authored).
        result = self._apply(self._dist_info_multi_arch_body(), ["gfx1010"])
        self.assertIn("DEFAULT_TARGET_FAMILY = 'gfx1010'", result)

    def test_single_arch_dist_info_is_left_unchanged(self):
        body = self._dist_info_single_arch_body()
        result = self._apply(body, ["gfx950"])
        self.assertEqual(result, body)

    def test_empty_intersection_raises(self):
        path = _write_tmp(".py", self._dist_info_multi_arch_body())
        try:
            with self.assertRaises(ValueError):
                ptf._apply_keep_list_to_dist_info_py(path, ["gfx9999"])
        finally:
            path.unlink()


class ComputeNewVersionStrTest(unittest.TestCase):
    """The four common transitions parameterised by (src_version_type, dest)."""

    def test_rc_to_release(self):
        self.assertEqual(
            ptf.compute_new_version_str("7.10.0rc1", "rc", "release"), "7.10.0"
        )

    def test_a_to_release_strips_full_date(self):
        self.assertEqual(
            ptf.compute_new_version_str("7.13.0a20260501", "a", "release"), "7.13.0"
        )

    def test_a_to_rc_replaces_segment(self):
        self.assertEqual(
            ptf.compute_new_version_str("7.13.0a20260501", "a", "rc1"), "7.13.0rc1"
        )

    def test_local_segment_is_rewritten_too(self):
        # torch packages embed the rocm version in the PEP 440 local segment.
        # `compute_new_version_str` is called separately on public + local.
        self.assertEqual(
            ptf.compute_new_version_str("rocm7.13.0a20260501", "a", "release"),
            "rocm7.13.0",
        )


class ParseArgumentsMutexTest(unittest.TestCase):
    """`--skip-version-promotion` is mutex with the version flags; sentinel
    defaults distinguish 'unset' from 'explicitly set'."""

    def _parse(self, argv: list[str]):
        return ptf.parse_arguments(argv)

    def test_defaults_resolve_after_mutex_check(self):
        ns = self._parse(["--input-dir", "/tmp"])
        self.assertEqual(ns.dest_version, "release")
        self.assertEqual(ns.src_version_type, "rc")

    def test_skip_version_promotion_requires_multi_arch_targets(self):
        with self.assertRaises(SystemExit):
            self._parse(["--input-dir", "/tmp", "--skip-version-promotion"])

    def test_skip_version_promotion_rejects_dest_version(self):
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--input-dir",
                    "/tmp",
                    "--skip-version-promotion",
                    "--multi-arch-targets",
                    "gfx950",
                    "--dest-version",
                    "release",
                ]
            )

    def test_skip_version_promotion_rejects_src_version_type(self):
        with self.assertRaises(SystemExit):
            self._parse(
                [
                    "--input-dir",
                    "/tmp",
                    "--skip-version-promotion",
                    "--multi-arch-targets",
                    "gfx950",
                    "--src-version-type",
                    "rc",
                ]
            )

    def test_dest_version_must_be_recognised(self):
        with self.assertRaises(SystemExit):
            self._parse(["--input-dir", "/tmp", "--dest-version", "garbage"])

    def test_dest_version_a_must_be_valid_calendar_date(self):
        with self.assertRaises(SystemExit):
            # Feb 30 doesn't exist.
            self._parse(["--input-dir", "/tmp", "--dest-version", "a20260230"])

    def test_multi_arch_targets_passthrough(self):
        ns = self._parse(
            ["--input-dir", "/tmp", "--multi-arch-targets", "gfx1010,gfx1201"]
        )
        self.assertEqual(ns.multi_arch_targets, "gfx1010,gfx1201")


if __name__ == "__main__":
    unittest.main()
