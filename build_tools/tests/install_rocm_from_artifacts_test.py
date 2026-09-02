#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for install_rocm_from_artifacts.py."""

import argparse
import io
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

    def test_base_only_includes_rocjitsu_hotswap(self):
        argv = self._run_main(["--base-only"])
        self.assertIn("rocjitsu-hotswap_lib", argv)

    def test_hipdnn_integration_tests_includes_rocrand(self):
        # The hipdnn_gpu_ref_tests binary links librocrand for GPU tensor data
        # generation, so the test runners must fetch the rand artifact even
        # though --rand was not requested.
        argv = self._run_main(["--hipdnn-integration-tests"])
        self.assertIn("hipdnn-integration-tests_run", argv)
        self.assertIn("rand_lib", argv)


def _tarball_name(platform: str, artifact_group: str, version: str) -> str:
    """Return a tarball name matching the platform under test."""
    return f"therock-dist-{platform}-{artifact_group}-{version}.tar.gz"


def _s3_object(
    platform: str,
    artifact_group: str,
    version: str,
    *,
    last_modified: datetime,
    size: int = 0,
) -> dict:
    """Return a tarball object in the published multi-arch S3 layout."""
    return {
        "Key": mod._multiarch_tarball_s3_key(
            _tarball_name(platform, artifact_group, version)
        ),
        "LastModified": last_modified,
        "Size": size,
    }


class TestMultiarchTarballNamePattern(unittest.TestCase):
    def test_extracts_named_filename_parts(self) -> None:
        test_cases = [
            ("linux", "gfx94X-dcgpu", "7.15.0a20260722"),
            ("windows", "gfx110X-all", "7.15.0rc20260722"),
            ("linux", "gfx90a", "7.15.0.dev0+deadbeef"),
            ("windows", "multiarch", "7.15.0"),
        ]

        for platform, artifact_group, version in test_cases:
            with self.subTest(platform=platform, artifact_group=artifact_group):
                match = mod.MULTIARCH_TARBALL_NAME_PATTERN.fullmatch(
                    _tarball_name(platform, artifact_group, version)
                )

                self.assertIsNotNone(match)
                self.assertEqual(
                    match.groupdict(),
                    {
                        "platform": platform,
                        "artifact_group": artifact_group,
                        "version": version,
                    },
                )

    def test_rejects_unversioned_filename(self) -> None:
        self.assertIsNone(
            mod.MULTIARCH_TARBALL_NAME_PATTERN.fullmatch(
                "therock-dist-linux-gfx94X-dcgpu-not-a-version.tar.gz"
            )
        )


class TestReleaseDiscovery(unittest.TestCase):
    @staticmethod
    def _paginator(*objects: dict) -> mock.Mock:
        paginator = mock.Mock()
        paginator.paginate.return_value = [{"Contents": objects}]
        return paginator

    def test_latest_release_dry_run_discovers_non_test_tarball(self) -> None:
        platform = mod.PLATFORM
        paginator = self._paginator(
            _s3_object(
                platform,
                "gfx94X-dcgpu-tests",
                "7.15.0a20260723",
                last_modified=datetime(2026, 7, 23),
            ),
            _s3_object(
                platform,
                "gfx94X-dcgpu",
                "7.15.0a20260722",
                last_modified=datetime(2026, 7, 22),
            ),
            _s3_object(
                platform,
                "gfx110X-all",
                "7.15.0a20260723",
                last_modified=datetime(2026, 7, 23),
            ),
        )
        output = io.StringIO()

        with (
            mock.patch.object(
                mod.s3_client, "get_paginator", return_value=paginator
            ) as get_paginator,
            mock.patch("sys.stdout", output),
        ):
            mod.main(
                [
                    "--latest-release",
                    "--artifact-group",
                    "gfx94X-dcgpu",
                    "--dry-run",
                ]
            )

        asset_name = _tarball_name(platform, "gfx94X-dcgpu", "7.15.0a20260722")
        self.assertIn("Found latest release: 7.15.0a20260722", output.getvalue())
        self.assertIn(f"Would download: {asset_name}", output.getvalue())
        get_paginator.assert_called_once_with("list_objects_v2")
        paginator.paginate.assert_called_once_with(
            Bucket=mod.NIGHTLY_TARBALL_BUCKET.name,
            Prefix=mod._multiarch_tarball_s3_key(f"therock-dist-{platform}-"),
        )

    def test_discovery_supports_linux_and_windows_tarballs(self) -> None:
        version = "7.15.0a20260722"
        for platform in ("linux", "windows"):
            asset_name = _tarball_name(platform, "gfx94X-dcgpu", version)
            paginator = self._paginator(
                _s3_object(
                    platform,
                    "gfx94X-dcgpu",
                    version,
                    last_modified=datetime(2026, 7, 22),
                )
            )

            with mock.patch.object(
                mod.s3_client, "get_paginator", return_value=paginator
            ):
                result = mod.discover_latest_release("gfx94X-dcgpu", platform)

            self.assertEqual(result, (version, asset_name))

    def test_nightly_release_dry_run_reports_s3_location_and_asset(self) -> None:
        version = "7.15.0a20260722"
        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", version)
        expected_s3_uri = (
            f"s3://{mod.NIGHTLY_TARBALL_BUCKET.name}/"
            f"{mod._multiarch_tarball_s3_key(asset_name)}"
        )
        output = io.StringIO()

        with mock.patch("sys.stdout", output):
            mod.main(
                [
                    "--release",
                    version,
                    "--artifact-group",
                    "gfx94X-dcgpu",
                    "--dry-run",
                ]
            )

        self.assertIn(f"Would download: {expected_s3_uri}", output.getvalue())
        self.assertIn(f"asset {asset_name}", output.getvalue())

    def test_multiarch_tarball_downloads_selected_s3_object(self) -> None:
        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", "7.15.0a20260722")
        expected_key = mod._multiarch_tarball_s3_key(asset_name)

        def write_tarball(_, __, file) -> None:
            file.write(b"tarball contents")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with (
                mock.patch.object(
                    mod.s3_client, "download_fileobj", side_effect=write_tarball
                ) as download_fileobj,
                mock.patch.object(mod, "_untar_files") as untar_files,
            ):
                mod._retrieve_multiarch_tarball(
                    mod.NIGHTLY_TARBALL_BUCKET.name,
                    asset_name,
                    output_dir,
                )

            self.assertEqual(
                (output_dir / asset_name).read_bytes(), b"tarball contents"
            )
            bucket, key, _ = download_fileobj.call_args.args
            self.assertEqual(bucket, mod.NIGHTLY_TARBALL_BUCKET.name)
            self.assertEqual(key, expected_key)
            untar_files.assert_called_once_with(output_dir, output_dir / asset_name)

    def test_dev_release_uses_dev_multiarch_tarball_bucket(self) -> None:
        version = "7.15.0.dev0+deadbeef"
        output_dir = Path("/tmp/therock-test")
        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", version)
        args = argparse.Namespace(
            artifact_group="gfx94X-dcgpu",
            output_dir=output_dir,
            release=version,
            dry_run=False,
        )
        output = io.StringIO()

        with (
            mock.patch.object(mod, "_retrieve_multiarch_tarball") as retrieve_tarball,
            mock.patch("sys.stdout", output),
        ):
            mod.retrieve_artifacts_by_release(args)

        retrieve_tarball.assert_called_once_with(
            mod.DEV_TARBALL_BUCKET.name,
            asset_name,
            output_dir,
        )
        self.assertIn(
            f"Retrieving dev multi-arch artifacts from "
            f"s3://{mod.DEV_TARBALL_BUCKET.name}/{mod.MULTIARCH_TARBALL_S3_PREFIX}/",
            output.getvalue(),
        )

    def test_extract_version_ignores_test_tarball(self) -> None:
        self.assertIsNone(
            mod.extract_version_from_asset_name(
                _tarball_name(mod.PLATFORM, "gfx94X-dcgpu-tests", "7.15.0a20260723"),
                "gfx94X-dcgpu-tests",
                mod.PLATFORM,
            )
        )

    def test_list_available_nightly_gpu_families_ignores_test_tarballs(
        self,
    ) -> None:
        for platform in ("linux", "windows"):
            paginator = self._paginator(
                _s3_object(
                    platform,
                    "gfx94X-dcgpu",
                    "7.15.0a20260723",
                    last_modified=datetime(2026, 7, 23),
                ),
                _s3_object(
                    platform,
                    "gfx94X-dcgpu-tests",
                    "7.15.0a20260723",
                    last_modified=datetime(2026, 7, 23),
                ),
                _s3_object(
                    platform,
                    "multiarch",
                    "7.15.0a20260723",
                    last_modified=datetime(2026, 7, 23),
                ),
            )
            with mock.patch.object(
                mod.s3_client, "get_paginator", return_value=paginator
            ):
                families = mod.list_available_nightly_gpu_families(platform)

            self.assertEqual(families, {"gfx94X-dcgpu", "multiarch"})

    def test_stable_release_uses_last_modified_for_ordering(self) -> None:
        paginator = self._paginator(
            _s3_object(
                mod.PLATFORM,
                "gfx94X-dcgpu",
                "7.15.0",
                last_modified=datetime(2026, 7, 22),
            ),
            _s3_object(
                mod.PLATFORM,
                "gfx94X-dcgpu",
                "7.16.0",
                last_modified=datetime(2026, 7, 23),
            ),
        )

        with mock.patch.object(mod.s3_client, "get_paginator", return_value=paginator):
            releases = mod._fetch_and_sort_nightly_releases("gfx94X-dcgpu")

        self.assertEqual(
            [release["version"] for release in releases],
            ["7.16.0", "7.15.0"],
        )
        self.assertEqual(releases[0]["last_modified"], datetime(2026, 7, 23))


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
        kfdtest=False,
        rocwmma=False,
        rpp=False,
        solver=False,
        sparse=False,
        libhipcxx=False,
        hipthreads=False,
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
