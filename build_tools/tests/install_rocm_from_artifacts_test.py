#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for install_rocm_from_artifacts.py."""

import argparse
from datetime import datetime
from pathlib import Path
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import install_rocm_from_artifacts as mod


class TestRetrieveArtifactsByRunId(unittest.TestCase):
    """Exercises how retrieve_artifacts_by_run_id() builds fetch_artifacts argv."""

    def _run_main(self, extra_args):
        """Run main() with fetch_artifacts mocked, returning the captured argv."""
        captured = {}

        def fake_fetch(argv):
            captured["argv"] = argv

        with mock.patch.object(mod, "fetch_artifacts_main", fake_fetch):
            mod.main(
                [
                    "--run-id",
                    "12345",
                    "--artifact-group",
                    "gfx942",
                    "--amdgpu-targets",
                    "gfx942",
                    "--dry-run",
                ]
                + extra_args
            )
        return captured["argv"]

    def test_core_arguments_forwarded(self):
        argv = self._run_main([])
        self.assertIn("--run-id", argv)
        self.assertIn("12345", argv)
        self.assertIn("--artifact-group", argv)
        self.assertIn("gfx942", argv)
        self.assertIn("--dry-run", argv)

    def test_artifact_flag_adds_lib_pattern_without_test(self):
        argv = self._run_main(["--blas"])
        self.assertIn("blas_lib", argv)
        self.assertNotIn("blas_test", argv)

    def test_tests_flag_adds_test_pattern(self):
        argv = self._run_main(["--blas", "--tests"])
        self.assertIn("blas_lib", argv)
        self.assertIn("blas_test", argv)

    def test_unselected_artifact_is_excluded(self):
        argv = self._run_main(["--blas"])
        self.assertNotIn("mirage_run", argv)

    def test_mirage_flag_includes_mirage_run(self):
        argv = self._run_main(["--mirage"])
        self.assertIn("mirage_run", argv)


class TestReleaseDiscovery(unittest.TestCase):
    def test_extract_version_ignores_test_tarball(self) -> None:
        self.assertIsNone(
            mod.extract_version_from_asset_name(
                "therock-dist-linux-gfx94X-dcgpu-tests-7.13.0.tar.gz",
                "gfx94X-dcgpu",
                "linux",
            )
        )

    def test_fetch_and_sort_nightly_releases_ignores_test_tarballs(self) -> None:
        paginator = mock.Mock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": (
                            "therock-dist-linux-gfx94X-dcgpu-tests-"
                            "7.13.0a20260102.tar.gz"
                        ),
                        "LastModified": datetime(2026, 1, 2),
                        "Size": 20,
                    },
                    {
                        "Key": "therock-dist-linux-gfx94X-dcgpu-7.13.0a20260101.tar.gz",
                        "LastModified": datetime(2026, 1, 1),
                        "Size": 10,
                    },
                ]
            }
        ]
        s3_client = mock.Mock()
        s3_client.get_paginator.return_value = paginator

        with mock.patch.object(mod, "s3_client", s3_client):
            releases = mod._fetch_and_sort_nightly_releases("gfx94X-dcgpu", "linux")

        self.assertEqual(
            [release["asset_name"] for release in releases],
            ["therock-dist-linux-gfx94X-dcgpu-7.13.0a20260101.tar.gz"],
        )

    def test_list_available_nightly_gpu_families_ignores_test_tarballs(self) -> None:
        paginator = mock.Mock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz"},
                    {"Key": ("therock-dist-linux-gfx94X-dcgpu-tests-" "7.13.0.tar.gz")},
                ]
            }
        ]
        s3_client = mock.Mock()
        s3_client.get_paginator.return_value = paginator

        with mock.patch.object(mod, "s3_client", s3_client):
            families = mod.list_available_nightly_gpu_families("linux")

        self.assertEqual(families, {"gfx94X-dcgpu"})


def _make_run_id_args(**overrides) -> argparse.Namespace:
    """Return a minimal args namespace suitable for retrieve_artifacts_by_run_id."""
    defaults = dict(
        run_id="12345",
        artifact_group="gfx110X-all",
        output_dir=Path("/tmp/therock-test"),
        # Non-empty amdgpu_targets skips the expand_families call.
        amdgpu_targets="gfx1100",
        dry_run=False,
        run_github_repo=None,
        base_only=False,
        aqlprofile=False,
        blas=False,
        debug_tools=False,
        fft=False,
        hipdnn=False,
        hipdnn_integration_tests=False,
        hipdnn_samples=False,
        hipfile=False,
        miopen=False,
        miopenprovider=False,
        hipkernelprovider=False,
        hiptensor=False,
        hipblasltprovider=False,
        prim=False,
        rand=False,
        rccl=False,
        rocshmem=False,
        mpi=False,
        rocdecode=False,
        rocjpeg=False,
        rocjitsu=False,
        mirage=False,
        rocprofiler_compute=False,
        rocprofiler_sdk=False,
        rocprofiler_systems=False,
        rocprofiler_systems_examples=False,
        rocrtst=False,
        rocalution=False,
        rocwmma=False,
        libhipcxx=False,
        tests=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _captured_fetch_argv(args: argparse.Namespace) -> list[str]:
    """Run retrieve_artifacts_by_run_id and return the argv passed to fetch_artifacts_main."""
    with mock.patch.object(mod, "fetch_artifacts_main") as mock_fetch:
        mod.retrieve_artifacts_by_run_id(args)
        (argv,), _ = mock_fetch.call_args
    return argv


class TestDebugToolsAmdLlvmDev(unittest.TestCase):
    """Tests that --debug-tools pulls amd-llvm_dev (required for rocgdb testing)."""

    def test_debug_tools_includes_amd_llvm_dev(self) -> None:
        argv = _captured_fetch_argv(_make_run_id_args(debug_tools=True))
        self.assertIn("amd-llvm_dev", argv)


if __name__ == "__main__":
    unittest.main()
