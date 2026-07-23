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


def _tarball_name(platform: str, artifact_group: str, version: str) -> str:
    """Return a tarball name matching the platform under test."""
    return f"therock-dist-{platform}-{artifact_group}-{version}.tar.gz"


class TestReleaseDiscovery(unittest.TestCase):
    def test_latest_release_dry_run_discovers_non_test_tarball(self) -> None:
        index_html = f"""
            <a href="{_tarball_name(mod.PLATFORM, 'gfx94X-dcgpu', 'tests-7.15.0a20260723')}">
            test tarball
            </a>
            <a href="{_tarball_name(mod.PLATFORM, 'gfx94X-dcgpu', '7.15.0a20260722')}">
            release tarball
            </a>
            <a href="{_tarball_name(mod.PLATFORM, 'gfx110X-all', '7.15.0a20260723')}">
            other artifact group
            </a>
        """
        output = io.StringIO()

        with (
            mock.patch.object(
                mod,
                "urlopen",
                side_effect=lambda _: io.BytesIO(index_html.encode()),
            ) as urlopen,
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

        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", "7.15.0a20260722")
        self.assertIn("Found latest release: 7.15.0a20260722", output.getvalue())
        self.assertIn(f"Would download: {asset_name}", output.getvalue())
        urlopen.assert_called_once_with(mod.NIGHTLY_TARBALL_INDEX_URL)

    def test_discovery_supports_linux_and_windows_tarballs(self) -> None:
        version = "7.15.0a20260722"
        for platform in ("linux", "windows"):
            asset_name = _tarball_name(platform, "gfx94X-dcgpu", version)
            index_html = f'const files = [{{"name": "{asset_name}", "mtime": 1.0}}];'

            with mock.patch.object(
                mod,
                "urlopen",
                side_effect=lambda _: io.BytesIO(index_html.encode()),
            ):
                result = mod.discover_latest_release("gfx94X-dcgpu", platform)

            self.assertEqual(result, (version, asset_name))

    def test_latest_release_dry_run_reads_embedded_file_data(self) -> None:
        index_html = f"""
            <script>
                const files = [
                    {{
                        "name": "{_tarball_name(mod.PLATFORM, 'gfx94X-dcgpu', 'tests-7.15.0a20260723')}",
                        "mtime": 1784764800.0
                    }},
                    {{
                        "name": "{_tarball_name(mod.PLATFORM, 'gfx94X-dcgpu', '7.15.0a20260722')}",
                        "mtime": 1784678400.0
                    }}
                ];
            </script>
        """
        output = io.StringIO()

        with (
            mock.patch.object(
                mod,
                "urlopen",
                side_effect=lambda _: io.BytesIO(index_html.encode()),
            ) as urlopen,
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

        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", "7.15.0a20260722")
        self.assertIn("Found latest release: 7.15.0a20260722", output.getvalue())
        self.assertIn(f"Would download: {asset_name}", output.getvalue())
        urlopen.assert_called_once_with(mod.NIGHTLY_TARBALL_INDEX_URL)

    def test_nightly_release_dry_run_reports_multiarch_url_and_asset(self) -> None:
        version = "7.15.0a20260722"
        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", version)
        expected_url = f"{mod.NIGHTLY_TARBALL_INDEX_URL}{asset_name}"
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

        self.assertIn(f"Would download: {expected_url}", output.getvalue())
        self.assertIn(f"asset {asset_name}", output.getvalue())

    def test_multiarch_tarball_download_streams_selected_asset(self) -> None:
        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", "7.15.0a20260722")
        expected_url = f"{mod.NIGHTLY_TARBALL_INDEX_URL}{asset_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with (
                mock.patch.object(
                    mod, "urlopen", return_value=io.BytesIO(b"tarball contents")
                ) as urlopen,
                mock.patch.object(mod, "_untar_files") as untar_files,
            ):
                mod._retrieve_multiarch_tarball(asset_name, output_dir)

            self.assertEqual(
                (output_dir / asset_name).read_bytes(), b"tarball contents"
            )
            urlopen.assert_called_once_with(expected_url)
            untar_files.assert_called_once_with(output_dir, output_dir / asset_name)

    def test_dev_release_uses_dev_multiarch_tarball(self) -> None:
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
            asset_name, output_dir, mod.DEV_TARBALL_INDEX_URL
        )
        self.assertIn(
            f"Retrieving dev artifacts from multi-arch tarball feed "
            f"{mod.DEV_TARBALL_INDEX_URL}",
            output.getvalue(),
        )
        self.assertEqual(
            mod._tarball_url(mod.DEV_TARBALL_INDEX_URL, asset_name),
            f"{mod.DEV_TARBALL_INDEX_URL}{asset_name.replace('+', '%2B')}",
        )

    def test_extract_version_ignores_test_tarball(self) -> None:
        self.assertIsNone(
            mod.extract_version_from_asset_name(
                _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", "tests-7.15.0a20260723"),
                "gfx94X-dcgpu",
                mod.PLATFORM,
            )
        )

    def test_list_available_nightly_gpu_families_ignores_test_tarballs(
        self,
    ) -> None:
        for platform in ("linux", "windows"):
            asset_names = {
                _tarball_name(platform, "gfx94X-dcgpu", "7.15.0a20260723"),
                _tarball_name(platform, "gfx94X-dcgpu", "tests-7.15.0a20260723"),
                _tarball_name(platform, "multiarch", "7.15.0a20260723"),
            }
            with mock.patch.object(
                mod, "_fetch_multiarch_tarball_asset_names", return_value=asset_names
            ):
                families = mod.list_available_nightly_gpu_families(platform)

            self.assertEqual(families, {"gfx94X-dcgpu", "multiarch"})

    def test_unparseable_release_uses_last_modified_for_ordering(self) -> None:
        index_html = f"""
            <script>
                const files = [
                    {{
                        "name": "{_tarball_name(mod.PLATFORM, 'gfx94X-dcgpu', 'legacy-one')}",
                        "mtime": 100.0
                    }},
                    {{
                        "name": "{_tarball_name(mod.PLATFORM, 'gfx94X-dcgpu', 'legacy-two')}",
                        "mtime": 200.0
                    }}
                ];
            </script>
        """

        with mock.patch.object(
            mod,
            "urlopen",
            side_effect=lambda _: io.BytesIO(index_html.encode()),
        ):
            releases = mod._fetch_and_sort_nightly_releases("gfx94X-dcgpu")

        self.assertEqual(
            [release["version"] for release in releases],
            ["legacy-two", "legacy-one"],
        )
        self.assertEqual(releases[0]["last_modified"], datetime.fromtimestamp(200))


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
