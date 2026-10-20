#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the S3 therock-dist rename in promote_packages.py."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from botocore.exceptions import ClientError

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from promote_packages import (  # noqa: E402
    parse_arguments,
    promote_s3_tarballs,
    promoted_tarball_name,
    select_tarball_copies,
)

RELEASE_PREFIX = "v5/rocm/core/tarball/"
ASAN_PREFIX = "v5/rocm/core/tarball-asan/"


class FakeClient:
    def __init__(self, objects, existing=None):
        self.objects = list(objects)
        self.existing = set(existing or [])

    def get_paginator(self, name):
        if name != "list_objects_v2":
            raise AssertionError(name)
        return self

    def paginate(self, **kwargs):
        prefix = kwargs["Prefix"]
        contents = [
            {"Key": key, "Size": size}
            for key, size in self.objects
            if key.startswith(prefix)
        ]
        return [{"Contents": contents}]

    def head_object(self, Bucket, Key):
        if Key not in self.existing:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        return {}


def _args(**overrides):
    values = {
        "version": "7.13.0rc1",
        "source-bucket": "src-bucket",
        "dest-bucket": "dst-bucket",
    }
    values.update(overrides)
    argv = []
    for key, value in values.items():
        if value is True:
            argv.append(f"--{key}")
        elif value is not False:
            argv.append(f"--{key}={value}")
    return parse_arguments(argv)


class PromotedNameTest(unittest.TestCase):
    def test_rc_release_renames_version_token(self):
        self.assertEqual(
            promoted_tarball_name(
                "therock-dist-linux-gfx950-7.13.0rc1.tar.gz", "rc", "release"
            ),
            "therock-dist-linux-gfx950-7.13.0.tar.gz",
        )

    def test_final_version_is_unchanged(self):
        self.assertIsNone(
            promoted_tarball_name(
                "therock-dist-linux-gfx950-7.13.0.tar.gz", "rc", "release"
            )
        )


class SelectCopiesTest(unittest.TestCase):
    def test_selects_matching_variant_only(self):
        objects = [
            (RELEASE_PREFIX + "therock-dist-linux-gfx950-7.13.0rc1.tar.gz", 10),
            (ASAN_PREFIX + "therock-dist-linux-gfx950-7.13.0rc1.tar.gz", 20),
            (RELEASE_PREFIX + "therock-dist-linux-gfx950-7.12.0rc1.tar.gz", 30),
            (RELEASE_PREFIX + "rocm-7.13.0rc1.tar.gz", 40),
            (
                RELEASE_PREFIX + "nested/therock-dist-linux-gfx950-7.13.0rc1.tar.gz",
                50,
            ),
        ]
        planned, unchanged = select_tarball_copies(
            objects, RELEASE_PREFIX, RELEASE_PREFIX, "7.13.0rc1", "rc", "release"
        )
        self.assertEqual(unchanged, [])
        self.assertEqual(
            [dest for _, dest, _ in planned],
            [RELEASE_PREFIX + "therock-dist-linux-gfx950-7.13.0.tar.gz"],
        )
        self.assertEqual(planned[0][2], 10)


class RunTest(unittest.TestCase):
    def _objects(self):
        return [
            (RELEASE_PREFIX + "therock-dist-linux-gfx950-7.13.0rc1.tar.gz", 10),
            (ASAN_PREFIX + "therock-dist-linux-gfx950-7.13.0rc1.tar.gz", 99),
        ]

    def test_variant_sets_prefix(self):
        args = _args(**{"tarball-variant": "asan"})
        self.assertEqual(args.source_prefix, ASAN_PREFIX)
        self.assertEqual(args.dest_prefix, ASAN_PREFIX)

    def test_dry_run_does_not_copy(self):
        client = FakeClient(self._objects())
        copied = []
        output = io.StringIO()
        with redirect_stdout(output):
            code = promote_s3_tarballs(
                _args(),
                client=client,
                copy_fn=lambda src, dst: copied.append((src, dst)),
            )
        self.assertEqual(code, 0)
        self.assertEqual(copied, [])
        self.assertIn("[DRY-RUN] Would copy:", output.getvalue())
        self.assertIn("therock-dist-linux-gfx950-7.13.0.tar.gz", output.getvalue())
        self.assertNotIn("tarball-asan", output.getvalue())

    def test_execute_copies_renamed_key(self):
        client = FakeClient(self._objects())
        copied = []
        with redirect_stdout(io.StringIO()):
            code = promote_s3_tarballs(
                _args(execute=True),
                client=client,
                copy_fn=lambda src, dst: copied.append((src, dst)),
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            copied,
            [
                (
                    RELEASE_PREFIX + "therock-dist-linux-gfx950-7.13.0rc1.tar.gz",
                    RELEASE_PREFIX + "therock-dist-linux-gfx950-7.13.0.tar.gz",
                )
            ],
        )

    def test_existing_destination_is_skipped(self):
        dest = RELEASE_PREFIX + "therock-dist-linux-gfx950-7.13.0.tar.gz"
        client = FakeClient(self._objects(), existing={dest})
        copied = []
        output = io.StringIO()
        with redirect_stdout(output):
            code = promote_s3_tarballs(
                _args(execute=True),
                client=client,
                copy_fn=lambda src, dst: copied.append((src, dst)),
            )
        self.assertEqual(code, 0)
        self.assertEqual(copied, [])
        self.assertIn("[SKIP] Already exists:", output.getvalue())

    def test_allow_overwrite_copies_existing_destination(self):
        dest = RELEASE_PREFIX + "therock-dist-linux-gfx950-7.13.0.tar.gz"
        client = FakeClient(self._objects(), existing={dest})
        copied = []
        with redirect_stdout(io.StringIO()):
            code = promote_s3_tarballs(
                _args(execute=True, **{"allow-overwrite": True}),
                client=client,
                copy_fn=lambda src, dst: copied.append(dst),
            )
        self.assertEqual(code, 0)
        self.assertEqual(copied, [dest])

    def test_missing_version_fails(self):
        client = FakeClient(self._objects())
        output = io.StringIO()
        with redirect_stdout(output):
            code = promote_s3_tarballs(
                _args(version="7.99.0rc1"), client=client
            )
        self.assertEqual(code, 1)
        self.assertIn("[ERROR]:", output.getvalue())

    def test_local_promotion_still_requires_input_dir(self):
        with self.assertRaises(SystemExit):
            parse_arguments([])


if __name__ == "__main__":
    unittest.main()
