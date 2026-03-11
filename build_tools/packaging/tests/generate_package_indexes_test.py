#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for build_tools/packaging/linux/generate_package_indexes.py

USAGE:
pytest build_tools/packaging/tests/generate_package_indexes_test.py -v

"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
LINUX_DIR = THIS_DIR.parent / "linux"
sys.path.insert(0, os.fspath(LINUX_DIR))

import generate_package_indexes as generate_package


class FakeS3:
    """Minimal fake for boto3 S3 client used by generate_*_from_s3.

    Simulates:
      - get_paginator("list_objects_v2").paginate(...)
      - put_object(...)
    """

    def __init__(
        self,
        list_pages_by_call: dict[tuple[str, str, str | None], list[dict[str, Any]]],
    ) -> None:
        # dict[(op_name, Prefix, Delimiter)] -> list[page_dict]
        self._list_pages_by_call = list_pages_by_call
        self.put_calls: list[dict[str, str]] = []

    def get_paginator(self, op_name: str) -> object:
        assert op_name == "list_objects_v2"
        s3 = self

        class Dispatcher:
            def paginate(self, **kwargs: Any):
                prefix = kwargs.get("Prefix")
                delim = kwargs.get("Delimiter", None)
                key = (op_name, prefix, delim)
                pages = s3._list_pages_by_call.get(key)
                if pages is None:
                    raise AssertionError(f"No fake pages configured for call {key}")
                for p in pages:
                    yield p

        return Dispatcher()

    def put_object(
        self, Bucket: str, Key: str, Body: bytes | bytearray | str, ContentType: str
    ) -> None:
        if isinstance(Body, (bytes, bytearray)):
            body_text = Body.decode("utf-8")
        else:
            body_text = str(Body)

        self.put_calls.append(
            {
                "Bucket": Bucket,
                "Key": Key,
                "Body": body_text,
                "ContentType": ContentType,
            }
        )

    def put_keys(self) -> list[str]:
        return [c["Key"] for c in self.put_calls]

    def put_body_for(self, key: str) -> str:
        for c in self.put_calls:
            if c["Key"] == key:
                return c["Body"]
        raise KeyError(key)


class GeneratePackageIndexesTest(unittest.TestCase):
    def test_generate_index_from_s3_creates_indexes_and_links(self) -> None:
        """Ensure recursive index generation builds all directories and links correctly.

        Verifies:
        - indexes are created for root, architecture dir, and repodata dir
        - files appear as links
        - directories appear as folder links
        - existing index.html objects are ignored
        """
        bucket = "b"
        prefix = "rpm/20260224-123"

        pages: list[dict[str, Any]] = [
            {
                "Contents": [
                    {"Key": f"{prefix}/x86_64/a.rpm"},
                    {"Key": f"{prefix}/x86_64/b.rpm"},
                    {"Key": f"{prefix}/x86_64/repodata/repomd.xml"},
                    {"Key": f"{prefix}/x86_64/repodata/index.html"},
                ]
            }
        ]

        s3 = FakeS3(
            list_pages_by_call={
                ("list_objects_v2", prefix, None): pages,
            }
        )

        generate_package.generate_index_from_s3(s3, bucket, prefix)

        self.assertIn(f"{prefix}/index.html", s3.put_keys())
        self.assertIn(f"{prefix}/x86_64/index.html", s3.put_keys())
        self.assertIn(f"{prefix}/x86_64/repodata/index.html", s3.put_keys())

        root_html = s3.put_body_for(f"{prefix}/index.html")
        self.assertIn('href="x86_64/index.html"', root_html)

        x86_html = s3.put_body_for(f"{prefix}/x86_64/index.html")
        self.assertIn('href="a.rpm"', x86_html)
        self.assertIn('href="repodata/index.html"', x86_html)

        repo_html = s3.put_body_for(f"{prefix}/x86_64/repodata/index.html")
        self.assertIn('href="repomd.xml"', repo_html)
        self.assertNotIn('href="index.html"', repo_html)

    def test_generate_index_from_s3_respects_max_depth(self) -> None:
        """Verify max_depth limits recursion depth of generated indexes.

        With max_depth=0:
        - root index must exist
        - first-level directory indexes allowed
        - deeper nested directories must NOT get indexes
        """
        bucket = "b"
        prefix = "rpm/20260224-123"

        pages: list[dict[str, Any]] = [
            {
                "Contents": [
                    {"Key": f"{prefix}/x86_64/a.rpm"},
                    {"Key": f"{prefix}/x86_64/repodata/repomd.xml"},
                ]
            }
        ]

        s3 = FakeS3(
            list_pages_by_call={
                ("list_objects_v2", prefix, None): pages,
            }
        )

        generate_package.generate_index_from_s3(s3, bucket, prefix, max_depth=0)

        keys = s3.put_keys()
        self.assertIn(f"{prefix}/index.html", keys)
        self.assertIn(f"{prefix}/x86_64/index.html", keys)
        self.assertNotIn(f"{prefix}/x86_64/repodata/index.html", keys)

    def test_generate_top_index_from_s3_lists_subfolders(self) -> None:
        """Ensure top-level index lists child prefixes and files correctly.

        Verifies:
        - subfolders returned via CommonPrefixes are included
        - top-level files are included
        - index.html itself is excluded
        """
        bucket = "b"
        top_prefix = "rpm"

        pages: list[dict[str, Any]] = [
            {
                "CommonPrefixes": [
                    {"Prefix": "rpm/20260224-111/"},
                    {"Prefix": "rpm/20260225-222/"},
                ],
                "Contents": [
                    {"Key": "rpm/somefile.txt"},
                    {"Key": "rpm/index.html"},
                ],
            }
        ]

        s3 = FakeS3(
            list_pages_by_call={
                ("list_objects_v2", f"{top_prefix}/", "/"): pages,
            }
        )

        generate_package.generate_top_index_from_s3(s3, bucket, top_prefix)

        self.assertIn(f"{top_prefix}/index.html", s3.put_keys())
        html = s3.put_body_for(f"{top_prefix}/index.html")

        self.assertIn('href="20260224-111/index.html"', html)
        self.assertIn('href="somefile.txt"', html)
        self.assertNotIn('href="index.html"', html)

    def test_generate_index_html_skips_dotfiles(self) -> None:
        """Ensure local filesystem index generation ignores dotfiles.

        Verifies:
        - visible files appear in the index
        - hidden files (starting with '.') are excluded
        """
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "repo"
            d.mkdir(parents=True, exist_ok=True)

            (d / ".hidden").write_text("x", encoding="utf-8")
            (d / "visible").write_text("y", encoding="utf-8")

            generate_package.generate_index_html(str(d))

            html = (d / "index.html").read_text(encoding="utf-8")
            self.assertIn("visible", html)
            self.assertNotIn(".hidden", html)


if __name__ == "__main__":
    unittest.main()
