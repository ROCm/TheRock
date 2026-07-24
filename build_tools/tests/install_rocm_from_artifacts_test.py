#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for install_rocm_from_artifacts.py."""

import argparse
from datetime import datetime
from pathlib import Path
import os
import sys
import tempfile
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

    def test_extract_version_ignores_hpc_tarball(self) -> None:
        self.assertIsNone(
            mod.extract_version_from_asset_name(
                "therock-dist-linux-gfx94X-dcgpu-hpc-7.13.0.tar.gz",
                "gfx94X-dcgpu",
                "linux",
            )
        )

    def test_fetch_and_sort_nightly_releases_ignores_hpc_tarballs(self) -> None:
        paginator = mock.Mock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": (
                            "therock-dist-linux-gfx94X-dcgpu-hpc-"
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

        # Only the default tarball is returned; the HPC tarball (which is newer)
        # must not be selected as a release.
        self.assertEqual(
            [release["asset_name"] for release in releases],
            ["therock-dist-linux-gfx94X-dcgpu-7.13.0a20260101.tar.gz"],
        )

    def test_list_available_nightly_gpu_families_ignores_hpc_tarballs(self) -> None:
        paginator = mock.Mock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz"},
                    {"Key": ("therock-dist-linux-gfx94X-dcgpu-hpc-" "7.13.0.tar.gz")},
                ]
            }
        ]
        s3_client = mock.Mock()
        s3_client.get_paginator.return_value = paginator

        with mock.patch.object(mod, "s3_client", s3_client):
            families = mod.list_available_nightly_gpu_families("linux")

        self.assertEqual(families, {"gfx94X-dcgpu"})


class TestHpcReleaseInstall(unittest.TestCase):
    """Exercises the opt-in --hpc release install path."""

    def test_release_asset_name_default(self) -> None:
        self.assertEqual(
            mod._release_asset_name("gfx94X-dcgpu", "7.13.0"),
            f"therock-dist-{mod.PLATFORM}-gfx94X-dcgpu-7.13.0.tar.gz",
        )

    def test_release_asset_name_hpc(self) -> None:
        self.assertEqual(
            mod._release_asset_name("gfx94X-dcgpu", "7.13.0", hpc=True),
            f"therock-dist-{mod.PLATFORM}-gfx94X-dcgpu-hpc-7.13.0.tar.gz",
        )

    def test_retrieve_release_assets_without_hpc_downloads_only_default(self) -> None:
        s3_client = mock.Mock()
        downloaded: list[str] = []
        s3_client.download_fileobj.side_effect = (
            lambda bucket, key, fileobj: downloaded.append(key)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(mod, "s3_client", s3_client), mock.patch.object(
                mod, "_untar_files"
            ):
                mod._retrieve_s3_release_assets(
                    "therock-nightly-tarball",
                    "gfx94X-dcgpu",
                    "7.13.0",
                    Path(tmpdir),
                    hpc=False,
                )
        self.assertEqual(
            downloaded,
            [f"therock-dist-{mod.PLATFORM}-gfx94X-dcgpu-7.13.0.tar.gz"],
        )

    def test_retrieve_release_assets_with_hpc_downloads_both(self) -> None:
        s3_client = mock.Mock()
        downloaded: list[str] = []
        s3_client.download_fileobj.side_effect = (
            lambda bucket, key, fileobj: downloaded.append(key)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(mod, "s3_client", s3_client), mock.patch.object(
                mod, "_untar_files"
            ):
                mod._retrieve_s3_release_assets(
                    "therock-nightly-tarball",
                    "gfx94X-dcgpu",
                    "7.13.0",
                    Path(tmpdir),
                    hpc=True,
                )
        self.assertEqual(
            downloaded,
            [
                f"therock-dist-{mod.PLATFORM}-gfx94X-dcgpu-7.13.0.tar.gz",
                f"therock-dist-{mod.PLATFORM}-gfx94X-dcgpu-hpc-7.13.0.tar.gz",
            ],
        )

    def test_release_dry_run_with_hpc_lists_both(self) -> None:
        args = argparse.Namespace(
            output_dir=Path("/tmp/out"),
            artifact_group="gfx94X-dcgpu",
            release="7.13.0a20260101",
            dry_run=True,
            hpc=True,
        )
        logs: list[str] = []
        with mock.patch.object(mod, "log", lambda m: logs.append(m)):
            mod.retrieve_artifacts_by_release(args)
        would_download = [m for m in logs if "Would download" in m]
        self.assertEqual(len(would_download), 2)
        self.assertTrue(any("-hpc-" in m for m in would_download))

    def test_retrieve_release_assets_missing_hpc_tarball_is_graceful(self) -> None:
        """A missing HPC tarball must not fail the (already done) default install."""
        from botocore.exceptions import ClientError

        default_name = f"therock-dist-{mod.PLATFORM}-gfx94X-dcgpu-7.13.0.tar.gz"
        hpc_name = f"therock-dist-{mod.PLATFORM}-gfx94X-dcgpu-hpc-7.13.0.tar.gz"

        downloaded: list[str] = []

        def fake_download(bucket, key, fileobj):
            downloaded.append(key)
            if key == hpc_name:
                raise ClientError(
                    {"Error": {"Code": "404", "Message": "Not Found"}},
                    "GetObject",
                )

        s3_client = mock.Mock()
        s3_client.download_fileobj.side_effect = fake_download

        logs: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(mod, "s3_client", s3_client), mock.patch.object(
                mod, "_untar_files"
            ), mock.patch.object(mod, "log", lambda m: logs.append(m)):
                # Should not raise even though the HPC object is missing.
                mod._retrieve_s3_release_assets(
                    "therock-nightly-tarball",
                    "gfx94X-dcgpu",
                    "7.13.0",
                    Path(tmpdir),
                    hpc=True,
                )
            # The default tarball was fully downloaded; the empty HPC partial
            # file was cleaned up.
            self.assertFalse((Path(tmpdir) / hpc_name).exists())

        # Both downloads were attempted, default first then HPC.
        self.assertEqual(downloaded, [default_name, hpc_name])
        # A clear warning was emitted for the missing HPC tarball.
        self.assertTrue(
            any("not found" in m and "HPC" in m for m in logs),
            f"expected a missing-HPC warning, got: {logs}",
        )

    def test_retrieve_release_assets_hpc_non_404_error_propagates(self) -> None:
        """Non-missing S3 errors on the HPC tarball must not be swallowed."""
        from botocore.exceptions import ClientError

        hpc_name = f"therock-dist-{mod.PLATFORM}-gfx94X-dcgpu-hpc-7.13.0.tar.gz"

        def fake_download(bucket, key, fileobj):
            if key == hpc_name:
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "Denied"}},
                    "GetObject",
                )

        s3_client = mock.Mock()
        s3_client.download_fileobj.side_effect = fake_download

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(mod, "s3_client", s3_client), mock.patch.object(
                mod, "_untar_files"
            ):
                with self.assertRaises(ClientError):
                    mod._retrieve_s3_release_assets(
                        "therock-nightly-tarball",
                        "gfx94X-dcgpu",
                        "7.13.0",
                        Path(tmpdir),
                        hpc=True,
                    )


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
