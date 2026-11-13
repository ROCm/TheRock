# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the GPU target discovery helpers in rocm_sdk._dist_info."""

from contextlib import AbstractContextManager
import os
import subprocess
import sys
import unittest
import unittest.mock
from pathlib import Path

# Import the runtime module straight from the package template source tree.
ROCM_SDK_SRC = (
    Path(__file__).resolve().parent.parent
    / "packaging"
    / "python"
    / "templates"
    / "rocm"
    / "src"
)
sys.path.insert(0, os.fspath(ROCM_SDK_SRC))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from rocm_sdk import _dist_info as di  # noqa: E402

MULTI_ARCH_FAMILIES = ["gfx1100", "gfx1101", "gfx1151", "gfx1250", "gfx1250-strict"]


class _FakeDist:
    def __init__(self, name: str | None) -> None:
        self.metadata = {"Name": name}


def _fake_distributions(
    *names: str | None,
) -> AbstractContextManager[unittest.mock.MagicMock]:
    """Patches the installed distributions to ones with the given names."""
    return unittest.mock.patch.object(
        di.importlib.metadata,
        "distributions",
        return_value=[_FakeDist(n) for n in names],
    )


def _fake_offload_arch(output: str) -> AbstractContextManager[unittest.mock.MagicMock]:
    """Patches the `offload-arch` query to return the given output."""
    return unittest.mock.patch.object(
        di.subprocess, "check_output", return_value=output
    )


class DiscoverSystemGpuTargetsTest(unittest.TestCase):
    def test_deduplicates_in_reported_order(self):
        with _fake_offload_arch("gfx1151\ngfx1100\n\ngfx1151\n"):
            self.assertEqual(di.discover_system_gpu_targets(), ["gfx1151", "gfx1100"])

    def test_empty_output(self):
        with _fake_offload_arch(""):
            self.assertEqual(di.discover_system_gpu_targets(), [])

    def test_tool_failure(self):
        error = subprocess.CalledProcessError(1, ["offload-arch"], output="")
        with unittest.mock.patch.object(
            di.subprocess, "check_output", side_effect=error
        ):
            self.assertIsNone(di.discover_system_gpu_targets())

    def test_tool_missing(self):
        with unittest.mock.patch.object(
            di.subprocess, "check_output", side_effect=FileNotFoundError()
        ):
            self.assertIsNone(di.discover_system_gpu_targets())


class DiscoverCurrentTargetFamilyTest(unittest.TestCase):
    def test_selects_first_available_target(self):
        with unittest.mock.patch.object(
            di, "AVAILABLE_TARGET_FAMILIES", MULTI_ARCH_FAMILIES
        ), _fake_offload_arch("gfx1036\ngfx1100\n"):
            self.assertEqual(di.discover_current_target_family(), "gfx1100")

    def test_selects_family(self):
        with unittest.mock.patch.object(
            di, "AVAILABLE_TARGET_FAMILIES", ["gfx110X-all"]
        ), _fake_offload_arch("gfx1101\n"):
            self.assertEqual(di.discover_current_target_family(), "gfx110X-all")

    def test_no_match(self):
        with unittest.mock.patch.object(
            di, "AVAILABLE_TARGET_FAMILIES", MULTI_ARCH_FAMILIES
        ), _fake_offload_arch("gfx942\n"):
            self.assertIsNone(di.discover_current_target_family())


class InstalledDeviceTargetsTest(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch.object(
            di, "AVAILABLE_TARGET_FAMILIES", MULTI_ARCH_FAMILIES
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_is_multi_arch(self):
        self.assertTrue(di.is_multi_arch_distribution())

    def test_single_owner_is_not_multi_arch(self):
        for families in (["gfx110X-all"], ["gfx1250", "gfx1250-strict"]):
            with self.subTest(families=families), unittest.mock.patch.object(
                di, "AVAILABLE_TARGET_FAMILIES", families
            ):
                self.assertFalse(di.is_multi_arch_distribution())

    def test_no_device_packages(self):
        with _fake_distributions("rocm", "rocm-sdk-core", "rocm-sdk-libraries"):
            self.assertEqual(di.get_installed_device_targets(), [])

    def test_installed_device_packages(self):
        with _fake_distributions(
            "rocm-sdk-device-gfx1151",
            "rocm_sdk_device_gfx1100",  # Non-normalized name.
            "rocm-sdk-core",
            None,
        ):
            self.assertEqual(di.get_installed_device_targets(), ["gfx1100", "gfx1151"])

    def test_unknown_device_package_ignored(self):
        with _fake_distributions("rocm-sdk-device-gfx942", "rocm-sdk-devel"):
            self.assertEqual(di.get_installed_device_targets(), [])

    def test_shared_owner_reported_once(self):
        with _fake_distributions("rocm-sdk-device-gfx1250"):
            self.assertEqual(di.get_installed_device_targets(), ["gfx1250"])

    def test_installed_gpu_targets_multi_arch(self):
        with _fake_distributions("rocm-sdk-device-gfx1101"):
            self.assertEqual(di.get_installed_gpu_targets(), ["gfx1101"])

    def test_installed_gpu_targets_single_family(self):
        with unittest.mock.patch.object(
            di, "AVAILABLE_TARGET_FAMILIES", ["gfx110X-all"]
        ), unittest.mock.patch.object(
            di,
            "get_dist_amdgpu_targets",
            return_value=["gfx1100", "gfx1101", "gfx1102", "gfx1103"],
        ):
            self.assertEqual(
                di.get_installed_gpu_targets(),
                ["gfx1100", "gfx1101", "gfx1102", "gfx1103"],
            )


if __name__ == "__main__":
    unittest.main()
