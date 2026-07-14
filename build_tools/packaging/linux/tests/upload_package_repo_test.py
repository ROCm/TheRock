#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for build_tools/packaging/linux/upload_package_repo.py"""

import gzip
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_DIR.parent
sys.path.insert(0, os.fspath(LINUX_DIR))

import upload_package_repo as upload_repo


class FakeS3:
    def __init__(
        self,
        rpm_keys: list[str],
        downloads: dict[str, bytes] | None = None,
    ) -> None:
        self._rpm_keys = rpm_keys
        self._downloads = downloads or {}
        self.download_calls: list[tuple[str, str, str]] = []

    def get_paginator(self, op_name: str) -> object:
        assert op_name == "list_objects_v2"
        s3 = self

        class Dispatcher:
            def paginate(self, **kwargs: Any):
                prefix = kwargs.get("Prefix", "")
                if prefix.endswith("/x86_64/"):
                    yield {
                        "Contents": [{"Key": key} for key in s3._rpm_keys],
                    }
                else:
                    yield {}

        return Dispatcher()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.download_calls.append((bucket, key, filename))
        Path(filename).write_bytes(self._downloads.get(key, b"fake-rpm"))


class UploadPackageRepoTest(unittest.TestCase):
    def test_list_s3_rpm_keys_filters_non_rpms(self) -> None:
        s3 = FakeS3(
            rpm_keys=[
                "run-linux/packages/rpm/x86_64/pkg-a.rpm",
                "run-linux/packages/rpm/x86_64/repodata/primary.xml.gz",
                "run-linux/packages/rpm/x86_64/pkg-b.rpm",
            ]
        )
        keys = upload_repo._list_s3_rpm_keys(
            s3, "bucket", "run-linux/packages/rpm"
        )
        self.assertEqual(
            keys,
            [
                "run-linux/packages/rpm/x86_64/pkg-a.rpm",
                "run-linux/packages/rpm/x86_64/pkg-b.rpm",
            ],
        )

    def test_prepare_rpm_arch_dir_uses_local_and_downloads_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "packages"
            arch_dir = package_dir / "x86_64"
            arch_dir.mkdir(parents=True)
            (arch_dir / "local-only.rpm").write_bytes(b"local")
            (arch_dir / "shared.rpm").write_bytes(b"shared-local")

            s3 = FakeS3(
                rpm_keys=[
                    "prefix/x86_64/shared.rpm",
                    "prefix/x86_64/s3-only.rpm",
                ],
                downloads={
                    "prefix/x86_64/s3-only.rpm": b"from-s3",
                },
            )

            work_dir, local_count, downloaded_count = (
                upload_repo._prepare_rpm_arch_dir_for_repodata(
                    s3,
                    "bucket",
                    "prefix",
                    package_dir,
                    Path(temp_dir) / "work",
                )
            )

            self.assertEqual(local_count, 2)
            self.assertEqual(downloaded_count, 1)
            self.assertTrue((work_dir / "local-only.rpm").exists())
            self.assertTrue((work_dir / "shared.rpm").exists())
            self.assertTrue((work_dir / "s3-only.rpm").exists())
            self.assertEqual(len(s3.download_calls), 1)
            self.assertEqual(s3.download_calls[0][1], "prefix/x86_64/s3-only.rpm")

    def test_validate_rpm_repodata_raises_on_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            arch_dir = Path(temp_dir) / "x86_64"
            arch_dir.mkdir(parents=True)
            (arch_dir / "pkg-a.rpm").write_bytes(b"a")
            (arch_dir / "pkg-b.rpm").write_bytes(b"b")

            repodata_dir = arch_dir / "repodata"
            repodata_dir.mkdir(parents=True)
            primary_xml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<metadata xmlns="http://linux.duke.edu/metadata/common" '
                'xmlns:rpm="http://linux.duke.edu/metadata/rpm" packages="1">'
                '<package type="rpm">'
                "<name>pkg-a</name>"
                "</package>"
                "</metadata>"
            )
            with gzip.open(repodata_dir / "primary.xml.gz", "wb") as handle:
                handle.write(primary_xml.encode("utf-8"))

            with self.assertRaisesRegex(RuntimeError, "RPM repodata is incomplete"):
                upload_repo._validate_rpm_repodata(arch_dir)


if __name__ == "__main__":
    unittest.main()
