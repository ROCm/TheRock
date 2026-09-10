#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for install_rocm_from_artifacts.py."""

import argparse
import io
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import index_generation_s3_tar
import install_rocm_from_artifacts as mod
from _therock_utils.s3_buckets import get_release_tarball_index_url

# A published tarball platform. The host platform is not usable here: tarballs
# are only published for linux and windows, so a test running on any other host
# would build names that no published tarball can match.
PUBLISHED_PLATFORM = "linux"


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


def _index_entry(
    platform: str,
    artifact_group: str,
    version: str,
    *,
    last_modified: datetime,
) -> dict:
    """Return a tarball entry as it appears in the published CDN index."""
    return {
        "name": _tarball_name(platform, artifact_group, version),
        "mtime": last_modified.replace(tzinfo=timezone.utc).timestamp(),
    }


def _generated_index_html(entries: list[dict], bucket_name: str) -> str:
    """Return an index page built by the generator that publishes the real one.

    Building the fixture with index_generation_s3_tar keeps the parser under
    test tied to the page format actually published to the CDN.
    """
    paginator = mock.Mock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {
                    "Key": f"v4/tarball/{entry['name']}",
                    "LastModified": datetime.fromtimestamp(
                        entry["mtime"], tz=timezone.utc
                    ),
                }
                for entry in entries
            ]
        }
    ]
    s3_client = mock.Mock()
    s3_client.get_paginator.return_value = paginator
    index_generation_s3_tar.generate_index_s3(
        s3_client, bucket_name, "v4/tarball", upload=True, allow_empty=True
    )
    return s3_client.put_object.call_args.kwargs["Body"].decode("utf-8")


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


class TestTarballIndexParsing(unittest.TestCase):
    """Exercises reading the published tarball list from a CDN index page."""

    def test_parses_entries_from_generated_index_page(self) -> None:
        entries = [
            _index_entry(
                PUBLISHED_PLATFORM,
                "gfx94X-dcgpu",
                "7.15.0a20260722",
                last_modified=datetime(2026, 7, 22),
            ),
            _index_entry(
                PUBLISHED_PLATFORM,
                "gfx110X-all",
                "7.15.0a20260723",
                last_modified=datetime(2026, 7, 23),
            ),
        ]
        index_html = _generated_index_html(entries, "therock-nightly-tarball")

        with mock.patch.object(
            mod, "_read_url", return_value=index_html.encode("utf-8")
        ) as read_url:
            parsed = mod._fetch_multiarch_tarball_index(mod.NIGHTLY_TARBALL_INDEX_URL)

        self.assertEqual(parsed, entries)
        read_url.assert_called_once_with(f"{mod.NIGHTLY_TARBALL_INDEX_URL}/index.html")

    def test_reports_index_url_when_file_list_is_missing(self) -> None:
        with mock.patch.object(mod, "_read_url", return_value=b"<html></html>"):
            with self.assertRaises(RuntimeError) as raised:
                mod._fetch_multiarch_tarball_index(mod.NIGHTLY_TARBALL_INDEX_URL)

        self.assertIn(
            f"{mod.NIGHTLY_TARBALL_INDEX_URL}/index.html", str(raised.exception)
        )

    def test_reports_index_url_when_file_list_is_malformed(self) -> None:
        with mock.patch.object(
            mod, "_read_url", return_value=b"<script>const files = [{,}];</script>"
        ):
            with self.assertRaises(RuntimeError) as raised:
                mod._fetch_multiarch_tarball_index(mod.NIGHTLY_TARBALL_INDEX_URL)

        self.assertIn(
            f"{mod.NIGHTLY_TARBALL_INDEX_URL}/index.html", str(raised.exception)
        )

    def test_read_url_reports_the_failing_url(self) -> None:
        url = "https://rocm.nightlies.amd.com/tarball-multi-arch/index.html"

        with mock.patch.object(
            mod.urllib.request,
            "urlopen",
            side_effect=urllib.error.HTTPError(url, 403, "Forbidden", {}, None),
        ):
            with self.assertRaises(RuntimeError) as raised:
                mod._read_url(url)

        self.assertIn(url, str(raised.exception))


class TestReleaseVersionClassification(unittest.TestCase):
    def test_classifies_each_published_release_kind(self) -> None:
        cases = {
            "7.15.0a20260722": "nightly",
            "6.4.0rc20250416": "nightly",
            "7.13.0rc2": "prerelease",
            "10.0.0rc0": "prerelease",
            "7.15.0.dev0+deadbeef": "dev",
        }
        for version, expected_kind in cases.items():
            with self.subTest(version=version):
                self.assertEqual(mod.classify_release_version(version), expected_kind)

    def test_rejects_an_unrecognised_version(self) -> None:
        for version in ("7.15.0", "nonsense", "7.15.0rc"):
            with self.subTest(version=version):
                self.assertIsNone(mod.classify_release_version(version))


class TestReleaseVersionSourceLines(unittest.TestCase):
    """The '--release' help text derives its URLs from the index resolver."""

    def test_covers_every_release_kind_that_publishes_tarballs(self) -> None:
        lines = mod.release_version_source_lines()
        for release_kind in mod.RELEASE_KIND_VERSION_PATTERNS:
            with self.subTest(release_kind=release_kind):
                self.assertTrue(any(f"({release_kind} example" in l for l in lines))

    def test_each_example_is_listed_under_the_index_it_resolves_to(self) -> None:
        lines = mod.release_version_source_lines()
        for release_kind, examples in mod.RELEASE_KIND_VERSION_EXAMPLES.items():
            for version in examples:
                with self.subTest(release_kind=release_kind, version=version):
                    index_url = get_release_tarball_index_url(release_kind, version)
                    self.assertTrue(
                        any(
                            line.startswith(f"\t - {index_url}/ ") and version in line
                            for line in lines
                        )
                    )

    def test_quoted_examples_classify_as_the_kind_they_are_listed_under(self) -> None:
        for release_kind, examples in mod.RELEASE_KIND_VERSION_EXAMPLES.items():
            for version in examples:
                with self.subTest(release_kind=release_kind, version=version):
                    self.assertEqual(
                        mod.classify_release_version(version), release_kind
                    )


class TestReleaseDiscovery(unittest.TestCase):
    @staticmethod
    def _index(*entries: dict) -> mock.Mock:
        return mock.Mock(return_value=list(entries))

    def test_latest_release_dry_run_discovers_non_test_tarball(self) -> None:
        platform = PUBLISHED_PLATFORM
        index = self._index(
            _index_entry(
                platform,
                "gfx94X-dcgpu-tests",
                "7.15.0a20260723",
                last_modified=datetime(2026, 7, 23),
            ),
            _index_entry(
                platform,
                "gfx94X-dcgpu",
                "7.15.0a20260722",
                last_modified=datetime(2026, 7, 22),
            ),
            _index_entry(
                platform,
                "gfx110X-all",
                "7.15.0a20260723",
                last_modified=datetime(2026, 7, 23),
            ),
        )
        output = io.StringIO()

        with (
            mock.patch.object(mod, "_fetch_multiarch_tarball_index", index),
            mock.patch.object(mod, "PLATFORM", platform),
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
        index.assert_called_with(mod.NIGHTLY_TARBALL_INDEX_URL)

    def test_discovery_supports_linux_and_windows_tarballs(self) -> None:
        version = "7.15.0a20260722"
        for platform in ("linux", "windows"):
            asset_name = _tarball_name(platform, "gfx94X-dcgpu", version)
            index = self._index(
                _index_entry(
                    platform,
                    "gfx94X-dcgpu",
                    version,
                    last_modified=datetime(2026, 7, 22),
                )
            )

            with mock.patch.object(mod, "_fetch_multiarch_tarball_index", index):
                result = mod.discover_latest_release("gfx94X-dcgpu", platform)

            self.assertEqual(result, (version, asset_name))

    def test_discovery_ignores_the_other_platform_tarballs(self) -> None:
        index = self._index(
            _index_entry(
                "windows",
                "gfx94X-dcgpu",
                "7.15.0a20260722",
                last_modified=datetime(2026, 7, 22),
            )
        )

        with mock.patch.object(mod, "_fetch_multiarch_tarball_index", index):
            self.assertIsNone(mod.discover_latest_release("gfx94X-dcgpu", "linux"))

    def test_nightly_release_dry_run_reports_cdn_url_and_asset(self) -> None:
        version = "10.1.0a20260901"
        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", version)
        expected_url = f"{mod.NIGHTLY_TARBALL_INDEX_URL}/{asset_name}"
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

    def test_multiarch_tarball_downloads_over_https(self) -> None:
        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", "7.15.0a20260722")
        expected_url = f"{mod.NIGHTLY_TARBALL_INDEX_URL}/{asset_name}"
        response = io.BytesIO(b"tarball contents")
        response.__enter__ = lambda self=response: self
        response.__exit__ = lambda *args: None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with (
                mock.patch.object(
                    mod.urllib.request, "urlopen", return_value=response
                ) as urlopen,
                mock.patch.object(mod, "_untar_files") as untar_files,
            ):
                mod._retrieve_multiarch_tarball(
                    mod.NIGHTLY_TARBALL_INDEX_URL,
                    asset_name,
                    output_dir,
                )

            self.assertEqual(
                (output_dir / asset_name).read_bytes(), b"tarball contents"
            )
            urlopen.assert_called_once_with(expected_url)
            untar_files.assert_called_once_with(output_dir, output_dir / asset_name)

    def test_download_failure_reports_the_url(self) -> None:
        asset_name = _tarball_name(mod.PLATFORM, "gfx94X-dcgpu", "7.15.0a20260722")
        expected_url = f"{mod.NIGHTLY_TARBALL_INDEX_URL}/{asset_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                mod.urllib.request,
                "urlopen",
                side_effect=urllib.error.HTTPError(
                    expected_url, 404, "Not Found", {}, None
                ),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    mod._retrieve_multiarch_tarball(
                        mod.NIGHTLY_TARBALL_INDEX_URL, asset_name, Path(temp_dir)
                    )

        self.assertIn(expected_url, str(raised.exception))

    def test_dev_tarball_url_percent_encodes_the_commit_hash_separator(self) -> None:
        """A raw '+' in the path is a 404 on the CDN; '%2B' is the tarball."""
        asset_name = _tarball_name(
            PUBLISHED_PLATFORM, "gfx110X-all", "10.1.0.dev0+0d83284"
        )
        index_url = "https://dev.repo.amd.com/rocm/core/tarball"

        url = mod._multiarch_tarball_url(index_url, asset_name)

        self.assertEqual(
            url,
            f"{index_url}/therock-dist-{PUBLISHED_PLATFORM}-gfx110X-all-"
            "10.1.0.dev0%2B0d83284.tar.gz",
        )
        self.assertNotIn("+", url)

    def test_dev_release_uses_the_dev_product_layout_index(self) -> None:
        self._assert_release_uses_index_url(
            "10.1.0.dev0+deadbeef", "https://dev.repo.amd.com/rocm/core/tarball"
        )

    def test_prerelease_before_the_product_layout_uses_the_legacy_index(self) -> None:
        self._assert_release_uses_index_url(
            "10.0.0rc4", "https://rocm.prereleases.amd.com/tarball-multi-arch"
        )

    def test_prerelease_from_the_product_layout_uses_the_rc_index(self) -> None:
        self._assert_release_uses_index_url(
            "10.1.0rc0", "https://rc.repo.amd.com/rocm/core/tarball"
        )

    def test_nightly_only_in_the_frozen_legacy_index_is_refused(self) -> None:
        version = "7.15.0a20260722"
        args = argparse.Namespace(
            artifact_group="gfx94X-dcgpu",
            output_dir=Path("/tmp/therock-test"),
            release=version,
            dry_run=False,
        )
        output = io.StringIO()

        with (
            mock.patch.object(mod, "_retrieve_multiarch_tarball") as retrieve_tarball,
            mock.patch("sys.stdout", output),
        ):
            mod.retrieve_artifacts_by_release(args)

        retrieve_tarball.assert_not_called()
        self.assertIn("legacy_multi_arch_releases.md", output.getvalue())

    def _assert_release_uses_index_url(self, version: str, index_url: str) -> None:
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

        retrieve_tarball.assert_called_once_with(index_url, asset_name, output_dir)
        self.assertIn(f"{index_url}/", output.getvalue())

    def test_extract_version_ignores_test_tarball(self) -> None:
        self.assertIsNone(
            mod.extract_version_from_asset_name(
                _tarball_name(
                    PUBLISHED_PLATFORM, "gfx94X-dcgpu-tests", "7.15.0a20260723"
                ),
                "gfx94X-dcgpu-tests",
                PUBLISHED_PLATFORM,
            )
        )

    def test_list_available_nightly_gpu_families_ignores_test_tarballs(
        self,
    ) -> None:
        for platform in ("linux", "windows"):
            index = self._index(
                _index_entry(
                    platform,
                    "gfx94X-dcgpu",
                    "7.15.0a20260723",
                    last_modified=datetime(2026, 7, 23),
                ),
                _index_entry(
                    platform,
                    "gfx94X-dcgpu-tests",
                    "7.15.0a20260723",
                    last_modified=datetime(2026, 7, 23),
                ),
                _index_entry(
                    platform,
                    "multiarch",
                    "7.15.0a20260723",
                    last_modified=datetime(2026, 7, 23),
                ),
            )
            with mock.patch.object(mod, "_fetch_multiarch_tarball_index", index):
                families = mod.list_available_nightly_gpu_families(platform)

            self.assertEqual(families, {"gfx94X-dcgpu", "multiarch"})

    def test_stable_release_uses_last_modified_for_ordering(self) -> None:
        index = self._index(
            _index_entry(
                PUBLISHED_PLATFORM,
                "gfx94X-dcgpu",
                "7.15.0",
                last_modified=datetime(2026, 7, 22),
            ),
            _index_entry(
                PUBLISHED_PLATFORM,
                "gfx94X-dcgpu",
                "7.16.0",
                last_modified=datetime(2026, 7, 23),
            ),
        )

        with mock.patch.object(mod, "_fetch_multiarch_tarball_index", index):
            releases = mod._fetch_and_sort_nightly_releases(
                "gfx94X-dcgpu", PUBLISHED_PLATFORM
            )

        self.assertEqual(
            [release["version"] for release in releases],
            ["7.16.0", "7.15.0"],
        )
        self.assertEqual(
            releases[0]["last_modified"],
            datetime(2026, 7, 23, tzinfo=timezone.utc),
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
