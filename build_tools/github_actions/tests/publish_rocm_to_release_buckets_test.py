#!/usr/bin/env python
"""Unit tests for publish_rocm_to_release_buckets.py."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parents[2] / "packaging" / "linux"))

from get_url_repo_params import nightly_sub_folder, parse_build_date
from github_actions.publish_rocm_to_release_buckets import main
from _therock_utils.storage_backend import LocalStorageBackend
from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot


class TestPublishRocmToReleaseBuckets(unittest.TestCase):
    """Tests for the main() CLI entry point."""

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_dev_linux_copies_tarballs_and_python(self, mock_copy):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--skip-native-packages",
                "--dry-run",
            ]
        )

        # Calls: tarballs, python -> v4/whl
        self.assertEqual(mock_copy.call_count, 2)
        # First call: tarballs
        tarball_source, tarball_dest = mock_copy.call_args_list[0].args
        self.assertEqual(tarball_source.bucket, "therock-dev-artifacts")
        self.assertEqual(tarball_source.relative_path, "123-linux/tarballs")
        self.assertEqual(tarball_dest.bucket, "therock-repo-amd-dev-core")
        self.assertEqual(tarball_dest.relative_path, "v5/rocm/core/tarball")
        # Python packages publish directly
        python_source, python_dest = mock_copy.call_args_list[1].args
        self.assertEqual(python_source.bucket, "therock-dev-artifacts")
        self.assertEqual(python_source.relative_path, "123-linux/python")
        self.assertEqual(python_dest.bucket, "therock-dev-python")
        self.assertEqual(python_dest.relative_path, "v4/whl")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_nightly_windows_copies_to_correct_buckets(self, mock_copy):
        mock_copy.return_value = 1
        main(
            [
                "--run-id",
                "99",
                "--platform",
                "windows",
                "--release-type",
                "nightly",
                "--dry-run",
            ]
        )

        tarball_source, tarball_dest = mock_copy.call_args_list[0].args
        self.assertEqual(tarball_source.bucket, "therock-nightly-artifacts")
        self.assertEqual(tarball_source.relative_path, "99-windows/tarballs")
        self.assertEqual(tarball_dest.bucket, "therock-repo-amd-nightly-core")
        self.assertEqual(tarball_dest.relative_path, "v5/rocm/core/tarball")

        python_source, python_dest = mock_copy.call_args_list[1].args
        self.assertEqual(python_source.bucket, "therock-nightly-artifacts")
        self.assertEqual(python_source.relative_path, "99-windows/python")
        self.assertEqual(python_dest.bucket, "therock-nightly-python")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_dev_linux_copies_native_packages(self, mock_copy):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--dry-run",
            ]
        )

        # Calls: tarballs, python -> v4/whl, deb, rpm, amd-repo-setup packages
        self.assertEqual(mock_copy.call_count, 5)
        # deb packages
        deb_source, deb_dest = mock_copy.call_args_list[2].args
        self.assertEqual(deb_source.bucket, "therock-dev-artifacts")
        self.assertEqual(deb_source.relative_path, "123-linux/packages/deb")
        self.assertEqual(deb_dest.bucket, "therock-repo-amd-dev-core")
        self.assertRegex(
            deb_dest.relative_path, r"^v5/rocm/core/packages/deb/\d{8}-123$"
        )
        # rpm packages
        rpm_source, rpm_dest = mock_copy.call_args_list[3].args
        self.assertEqual(rpm_source.bucket, "therock-dev-artifacts")
        self.assertEqual(rpm_source.relative_path, "123-linux/packages/rpm")
        self.assertEqual(rpm_dest.bucket, "therock-repo-amd-dev-core")
        self.assertRegex(
            rpm_dest.relative_path, r"^v5/rocm/core/packages/rpm/\d{8}-123$"
        )

    def _native_package_folders(self, mock_copy, *extra_args):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "nightly",
                "--dry-run",
                *extra_args,
            ]
        )
        # Calls: tarballs, python, deb, rpm, amd-repo-setup packages
        return [mock_copy.call_args_list[i].args[1].relative_path for i in (2, 3)]

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_build_date_names_the_native_package_folders(self, mock_copy):
        # A run that crosses midnight: the setup package was built with the
        # run's build date, so the repository must be published under the same
        # folder, not under the date at publish time.
        folders = self._native_package_folders(mock_copy, "--build-date", "20200101")

        setup_package_folder = nightly_sub_folder(
            "123", today=parse_build_date("20200101")
        )
        self.assertEqual(
            folders,
            [
                f"v5/rocm/core/packages/deb/{setup_package_folder}",
                f"v5/rocm/core/packages/rpm/{setup_package_folder}",
            ],
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_empty_build_date_means_today(self, mock_copy):
        folders = self._native_package_folders(mock_copy, "--build-date", "")

        for folder in folders:
            self.assertRegex(folder, r"/\d{8}-123$")

    def test_rejects_a_malformed_build_date(self):
        for bad in ("2020010", "20201332", "x"):
            with self.subTest(build_date=bad):
                with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    main(
                        [
                            "--run-id",
                            "123",
                            "--release-type",
                            "nightly",
                            "--build-date",
                            bad,
                            "--dry-run",
                        ]
                    )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_repo_setup_packages_go_to_the_stream_core_bucket(self, mock_copy):
        mock_copy.return_value = 2
        cases = [
            ("dev", "therock-dev-artifacts", "therock-repo-amd-dev-core"),
            ("nightly", "therock-nightly-artifacts", "therock-repo-amd-nightly-core"),
            ("prerelease", "therock-prerelease-artifacts", "therock-repo-amd-rc-core"),
        ]
        for release_type, source_bucket, dest_bucket in cases:
            with self.subTest(release_type=release_type):
                mock_copy.reset_mock()
                main(
                    [
                        "--run-id",
                        "123",
                        "--platform",
                        "linux",
                        "--release-type",
                        release_type,
                        "--dry-run",
                    ]
                )

                # Last call, after the package repositories.
                source, dest = mock_copy.call_args_list[-1].args
                self.assertEqual(source.bucket, source_bucket)
                self.assertEqual(source.relative_path, "123-linux/amd-repo-setup")
                self.assertEqual(dest.bucket, dest_bucket)
                self.assertEqual(dest.relative_path, "v5/amd-repo-setup")
                # A published package is never replaced by default.
                self.assertEqual(
                    mock_copy.call_args_list[-1].kwargs, {"skip_existing": True}
                )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_windows_copies_native_msi_packages(self, mock_copy):
        mock_copy.return_value = 1
        main(
            [
                "--run-id",
                "99",
                "--platform",
                "windows",
                "--release-type",
                "nightly",
                "--dry-run",
            ]
        )
        # Calls: tarballs, python, msi (3 calls)
        self.assertEqual(mock_copy.call_count, 3)
        # msi packages are the last call
        msi_source, msi_dest = mock_copy.call_args_list[2].args
        self.assertEqual(msi_source.bucket, "therock-nightly-artifacts")
        self.assertEqual(msi_source.relative_path, "99-windows/packages/msi")
        self.assertEqual(msi_dest.bucket, "therock-repo-amd-nightly-core")
        self.assertRegex(
            msi_dest.relative_path, r"^v5/rocm/core/windows-installers/\d{8}-99$"
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_raises_when_no_tarballs_found(self, mock_copy):
        mock_copy.return_value = 0
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--run-id",
                    "123",
                    "--platform",
                    "linux",
                    "--release-type",
                    "dev",
                    "--dry-run",
                ]
            )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_asan_skips_python_packages(self, mock_copy):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--build-variant",
                "asan",
                "--skip-native-packages",
                "--dry-run",
            ]
        )

        # Only tarballs should be copied (python packages skipped for ASAN)
        self.assertEqual(mock_copy.call_count, 1)
        tarball_source, tarball_dest = mock_copy.call_args_list[0].args
        self.assertEqual(tarball_source.relative_path, "123-linux/tarballs")
        # ASAN tarballs go to separate folder
        self.assertEqual(tarball_dest.bucket, "therock-repo-amd-dev-core")
        self.assertEqual(tarball_dest.relative_path, "v5/rocm/core/tarball-asan")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_file")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.list_files")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_structured_places_rocm_packages_in_package_dirs(
        self, mock_copy_dir, mock_list, mock_copy_file
    ):
        # Structured multi-arch: python packages go into per-package dirs via
        # list_files + copy_file, not the flat copy_directory.
        mock_copy_dir.return_value = 2  # tarballs still use copy_directory
        mock_list.return_value = [
            StorageLocation(
                "therock-dev-artifacts",
                "123-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
            ),
            StorageLocation(
                "therock-dev-artifacts", "123-linux/python/rocm-7.13.0.tar.gz"
            ),
            StorageLocation(
                "therock-dev-artifacts",
                "123-linux/python/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl",
            ),
            # Non-accepted artifact in the listing must be ignored.
            StorageLocation("therock-dev-artifacts", "123-linux/python/index.html"),
        ]
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--structured",
                "--skip-native-packages",
                "--dry-run",
            ]
        )

        dest_by_src = {
            call.args[0].relative_path: call.args[1].relative_path
            for call in mock_copy_file.call_args_list
        }
        self.assertNotIn("123-linux/python/index.html", dest_by_src)
        self.assertEqual(
            dest_by_src[
                "123-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl"
            ],
            "v5/rocm/core/whl-next/rocm-sdk-core/"
            "rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
        )
        self.assertEqual(
            dest_by_src["123-linux/python/rocm-7.13.0.tar.gz"],
            "v5/rocm/core/whl-next/rocm/rocm-7.13.0.tar.gz",
        )
        self.assertEqual(
            dest_by_src[
                "123-linux/python/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl"
            ],
            "v5/rocm/core/whl-next/rocm-sdk-device-gfx1100/"
            "rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl",
        )
        # Destination bucket is the Core product bucket.
        for call in mock_copy_file.call_args_list:
            self.assertEqual(call.args[1].bucket, "therock-repo-amd-dev-core")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_file")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.list_files")
    def test_structured_python_index_override(
        self, mock_list, mock_copy_file, mock_copy_dir
    ):
        mock_copy_dir.return_value = 2  # tarballs
        mock_list.return_value = [
            StorageLocation(
                "therock-dev-artifacts",
                "123-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
            ),
        ]
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--structured",
                "--python-index",
                "whl",
                "--skip-native-packages",
                "--dry-run",
            ]
        )
        _, dest = mock_copy_file.call_args_list[0].args
        self.assertEqual(
            dest.relative_path,
            "v5/rocm/core/whl/rocm-sdk-core/"
            "rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_file")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.list_files")
    def test_structured_raises_when_no_python_packages(
        self, mock_list, mock_copy_file, mock_copy_dir
    ):
        mock_copy_dir.return_value = 2  # tarballs succeed
        mock_list.return_value = []
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--run-id",
                    "123",
                    "--platform",
                    "linux",
                    "--release-type",
                    "dev",
                    "--structured",
                    "--skip-native-packages",
                    "--dry-run",
                ]
            )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_asan_native_packages_use_separate_path(self, mock_copy):
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--build-variant",
                "asan",
                "--dry-run",
            ]
        )

        # Calls: tarballs, deb, rpm (no python for ASAN)
        self.assertEqual(mock_copy.call_count, 3)
        # deb packages go to packages-asan path
        deb_source, deb_dest = mock_copy.call_args_list[1].args
        self.assertEqual(deb_source.relative_path, "123-linux/packages/deb")
        self.assertEqual(deb_dest.bucket, "therock-repo-amd-dev-core")
        self.assertRegex(
            deb_dest.relative_path,
            r"^v5/rocm/core/packages-asan/deb/\d{8}-123$",
        )
        # rpm packages go to packages-asan path
        rpm_source, rpm_dest = mock_copy.call_args_list[2].args
        self.assertEqual(rpm_source.relative_path, "123-linux/packages/rpm")
        self.assertEqual(rpm_dest.bucket, "therock-repo-amd-dev-core")
        self.assertRegex(
            rpm_dest.relative_path,
            r"^v5/rocm/core/packages-asan/rpm/\d{8}-123$",
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_asan_debug_publishes_same_as_asan(self, mock_copy):
        # 'asan-debug' is a build_variant_suffix alias for 'asan' (see
        # amdgpu_family_matrix.py) — it must publish to the same tarball-asan
        # path and skip python packages, exactly like 'asan'.
        mock_copy.return_value = 2
        main(
            [
                "--run-id",
                "123",
                "--platform",
                "linux",
                "--release-type",
                "dev",
                "--build-variant",
                "asan-debug",
                "--skip-native-packages",
                "--dry-run",
            ]
        )

        self.assertEqual(mock_copy.call_count, 1)
        tarball_source, tarball_dest = mock_copy.call_args_list[0].args
        self.assertEqual(tarball_source.relative_path, "123-linux/tarballs")
        self.assertEqual(tarball_dest.bucket, "therock-repo-amd-dev-core")
        self.assertEqual(tarball_dest.relative_path, "v5/rocm/core/tarball-asan")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.copy_directory")
    def test_host_asan_variants_publish_same_as_asan(self, mock_copy):
        # host-asan and host-asan-debug are also ASAN-family variants and are
        # assumed to publish to the same "-asan" paths as full ASAN (see the
        # ASSUMPTION note on _is_asan_variant — unverified against a real
        # host-asan publish run, since nothing calls this script with
        # "host-asan" today).
        for build_variant in ("host-asan", "host-asan-debug"):
            with self.subTest(build_variant=build_variant):
                mock_copy.reset_mock()
                mock_copy.return_value = 2
                main(
                    [
                        "--run-id",
                        "123",
                        "--platform",
                        "linux",
                        "--release-type",
                        "dev",
                        "--build-variant",
                        build_variant,
                        "--skip-native-packages",
                        "--dry-run",
                    ]
                )

                self.assertEqual(mock_copy.call_count, 1)
                tarball_source, tarball_dest = mock_copy.call_args_list[0].args
                self.assertEqual(tarball_source.relative_path, "123-linux/tarballs")
                self.assertEqual(
                    tarball_dest.relative_path, "v5/rocm/core/tarball-asan"
                )


class TestRepoSetupPackagesArePublished(unittest.TestCase):
    """The amd-repo-setup packages are published to ``v5/amd-repo-setup/``.

    These tests drive main() with a real ``LocalStorageBackend`` and assert the
    files on disk; the rest of this module mocks the copy. The local backend
    ignores the bucket, which ``test_repo_setup_packages_go_to_the_stream_core_bucket``
    covers.
    """

    # One package per distro, under the builder's file names.
    NIGHTLY_111 = (
        "ubuntu2404/amd-repo-setup_10.1.0~20260101-111~ubuntu2404_all.deb",
        "rhel10/amd-repo-setup-10.1.0~20260101-111.nightly.el10.noarch.rpm",
    )
    NIGHTLY_222 = (
        "ubuntu2404/amd-repo-setup_10.1.0~20260101-222~ubuntu2404_all.deb",
        "rhel10/amd-repo-setup-10.1.0~20260101-222.nightly.el10.noarch.rpm",
    )
    RC = (
        "ubuntu2404/amd-repo-setup_10.1.0~pre3-1~ubuntu2404_all.deb",
        "rhel10/amd-repo-setup-10.1.0~rc3-1.rc.el10.noarch.rpm",
    )

    def setUp(self):
        self.staging = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.staging, ignore_errors=True)

    def _write(self, rel_path, contents=b"CONTENTS"):
        path = self.staging / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    def _publish(self, release_type, run_id, packages, *extra_args, contents=b"X"):
        """Seed one run's outputs, run the real publish, return what it wrote.

        Returns the published paths below ``v5/``, relative to the staging
        root; the local backend maps every bucket onto it.
        """
        root = WorkflowOutputRoot("unused", "", run_id, "linux")
        # main() fails if a run has no tarball or Python package.
        self._write(
            root.tarball("therock-dist-linux-gfx94X-dcgpu-10.1.0.tar.gz").relative_path
        )
        self._write(
            f"{root.python_packages().relative_path}/rocm_sdk_core-10.1.0-py3-none-any.whl"
        )
        self._write(
            f"{root.native_linux_packages('deb').relative_path}/dists/stable/Release"
        )
        self._write(
            f"{root.native_linux_packages('rpm').relative_path}/x86_64/repodata/repomd.xml"
        )
        for package in packages:
            self._write(
                f"{root.repo_setup_packages().relative_path}/{package}", contents
            )

        with mock.patch(
            "github_actions.publish_rocm_to_release_buckets.create_storage_backend",
            return_value=LocalStorageBackend(staging_dir=self.staging),
        ):
            main(
                [
                    "--run-id",
                    run_id,
                    "--platform",
                    "linux",
                    "--release-type",
                    release_type,
                    "--build-date",
                    "20260101",
                    *extra_args,
                ]
            )
        return {
            p.relative_to(self.staging).as_posix()
            for p in (self.staging / "v5").rglob("*")
            if p.is_file()
        }

    @staticmethod
    def _below(written, prefix):
        return {p for p in written if p.startswith(prefix)}

    def test_publishes_the_packages_beside_the_repositories(self):
        cases = {
            "nightly": (self.NIGHTLY_111, "20260101-111/"),
            "prerelease": (self.RC, ""),
        }
        for release_type, (packages, build_folder) in cases.items():
            with self.subTest(release_type=release_type):
                self.setUp()
                written = self._publish(release_type, "111", packages)

                self.assertEqual(
                    self._below(written, "v5/amd-repo-setup/"),
                    {f"v5/amd-repo-setup/{p}" for p in packages},
                )
                self.assertEqual(
                    self._below(written, "v5/rocm/core/packages/"),
                    {
                        f"v5/rocm/core/packages/deb/{build_folder}dists/stable/Release",
                        f"v5/rocm/core/packages/rpm/{build_folder}x86_64/repodata/repomd.xml",
                    },
                )

    def test_each_nightly_build_adds_its_own_package(self):
        self._publish("nightly", "111", self.NIGHTLY_111, contents=b"RUN-1")
        written = self._publish("nightly", "222", self.NIGHTLY_222, contents=b"RUN-2")

        self.assertEqual(
            self._below(written, "v5/amd-repo-setup/"),
            {f"v5/amd-repo-setup/{p}" for p in self.NIGHTLY_111 + self.NIGHTLY_222},
        )
        for package in self.NIGHTLY_111:
            path = self.staging / "v5/amd-repo-setup" / package
            self.assertEqual(path.read_bytes(), b"RUN-1")

    def _republish_the_same_rc(self, *extra_args):
        """Publish an rc, then a rebuild of it with the same names; return bytes."""
        self._publish("prerelease", "111", self.RC, contents=b"RUN-1")
        with self.assertLogs(level="INFO") as logs:
            self._publish("prerelease", "222", self.RC, *extra_args, contents=b"RUN-2")
        published = {
            (self.staging / "v5/amd-repo-setup" / p).read_bytes() for p in self.RC
        }
        return published, "\n".join(logs.output)

    def test_a_published_package_is_not_replaced(self):
        published, logs = self._republish_the_same_rc()

        self.assertEqual(published, {b"RUN-1"})
        self.assertIn("already exists, not replaced", logs)

    def test_overwrite_replaces_a_published_package(self):
        published, _ = self._republish_the_same_rc("--overwrite-repo-setup-packages")

        self.assertEqual(published, {b"RUN-2"})

    def test_publishes_none_when_there_are_none_to_publish(self):
        cases = {
            "the run uploaded none": (),
            "asan": ("--build-variant", "asan"),
            "--skip-native-packages": ("--skip-native-packages",),
        }
        for case, extra_args in cases.items():
            with self.subTest(case=case):
                self.setUp()
                packages = () if case == "the run uploaded none" else self.NIGHTLY_111
                written = self._publish("nightly", "111", packages, *extra_args)

                self.assertEqual(self._below(written, "v5/amd-repo-setup/"), set())


if __name__ == "__main__":
    unittest.main()
