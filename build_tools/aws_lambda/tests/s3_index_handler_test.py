#!/usr/bin/env python
"""Unit tests for s3_index_handler.py."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# Add build_tools to path so _therock_utils and generate_s3_index are importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
# Add aws_lambda to path so s3_index_handler is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import s3_index_handler


def _make_s3_record(bucket: str, key: str) -> dict:
    return {"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}


def _make_direct_event(bucket: str, key: str) -> dict:
    return {"Records": [_make_s3_record(bucket, key)]}


def _make_sqs_event(bucket: str, key: str) -> dict:
    """Wrap an S3 event record in an SQS message, as Lambda receives it."""
    s3_event_body = json.dumps({"Records": [_make_s3_record(bucket, key)]})
    return {"Records": [{"body": s3_event_body}]}


class TestGetDirPrefix(unittest.TestCase):
    """Tests for _get_dir_prefix()."""

    def test_file_in_logs_group_dir(self):
        result = s3_index_handler._get_dir_prefix(
            "12345678901-linux/logs/gfx94X-dcgpu/build.log"
        )
        self.assertEqual(result, "12345678901-linux/logs/gfx94X-dcgpu")

    def test_file_in_manifests_group_dir(self):
        result = s3_index_handler._get_dir_prefix(
            "12345678901-linux/manifests/gfx94X-dcgpu/therock_manifest.json"
        )
        self.assertEqual(result, "12345678901-linux/manifests/gfx94X-dcgpu")

    def test_file_at_run_root(self):
        result = s3_index_handler._get_dir_prefix(
            "12345678901-linux/core_lib_gfx94X.tar.xz"
        )
        self.assertEqual(result, "12345678901-linux")

    def test_external_repo_prefix(self):
        result = s3_index_handler._get_dir_prefix(
            "Fork-TheRock/12345678901-linux/logs/gfx94X-dcgpu/build.log"
        )
        self.assertEqual(result, "Fork-TheRock/12345678901-linux/logs/gfx94X-dcgpu")

    def test_windows_platform(self):
        result = s3_index_handler._get_dir_prefix(
            "12345678901-windows/logs/gfx94X-dcgpu/build.log"
        )
        self.assertEqual(result, "12345678901-windows/logs/gfx94X-dcgpu")

    def test_index_html_returns_none(self):
        result = s3_index_handler._get_dir_prefix(
            "12345678901-linux/logs/gfx94X-dcgpu/index.html"
        )
        self.assertIsNone(result)

    def test_root_index_html_returns_none(self):
        result = s3_index_handler._get_dir_prefix(
            "12345678901-linux/index.html"
        )
        self.assertIsNone(result)

    def test_bucket_root_file_returns_none(self):
        result = s3_index_handler._get_dir_prefix("standalone_file.txt")
        self.assertIsNone(result)

    def test_python_dir_returns_none(self):
        result = s3_index_handler._get_dir_prefix(
            "12345678901-linux/python/foo-1.0-py3-none-any.whl"
        )
        self.assertIsNone(result)

    def test_python_subdir_returns_none(self):
        result = s3_index_handler._get_dir_prefix(
            "ROCm-TheRock/12345678901-linux/python/subdir/foo.whl"
        )
        self.assertIsNone(result)

    def test_therock_build_prof_returns_none(self):
        result = s3_index_handler._get_dir_prefix(
            "12345678901-linux/logs/gfx120X-all/therock-build-prof/profile.json"
        )
        self.assertIsNone(result)

    def test_therock_build_prof_external_returns_none(self):
        result = s3_index_handler._get_dir_prefix(
            "ROCm-TheRock/12345678901-linux/logs/gfx120X-all/therock-build-prof/profile.json"
        )
        self.assertIsNone(result)

    def test_deeply_nested_file_indexes_immediate_parent(self):
        """A deeply nested file indexes its own directory, not a higher level."""
        result = s3_index_handler._get_dir_prefix(
            "12345678901-linux/logs/math-libs/gfx1151/build.log"
        )
        self.assertEqual(result, "12345678901-linux/logs/math-libs/gfx1151")


class TestExtractS3Records(unittest.TestCase):
    """Tests for _extract_s3_records()."""

    def test_direct_s3_event_passthrough(self):
        event = _make_direct_event("therock-ci-artifacts", "12345-linux/file.log")
        records = s3_index_handler._extract_s3_records(event)
        self.assertEqual(len(records), 1)
        self.assertIn("s3", records[0])

    def test_sqs_wrapped_event_unwrapped(self):
        event = _make_sqs_event("therock-ci-artifacts", "12345-linux/file.log")
        records = s3_index_handler._extract_s3_records(event)
        self.assertEqual(len(records), 1)
        self.assertIn("s3", records[0])

    def test_empty_event_returns_empty(self):
        self.assertEqual(s3_index_handler._extract_s3_records({}), [])
        self.assertEqual(s3_index_handler._extract_s3_records({"Records": []}), [])


class TestCollectDirsToIndex(unittest.TestCase):
    """Tests for _collect_dirs_to_index()."""

    def test_single_record_leaf_and_ancestor(self):
        records = [_make_s3_record(
            "therock-ci-artifacts",
            "12345-linux/logs/gfx94X/build.log",
        )]
        dirs = s3_index_handler._collect_dirs_to_index(records)
        self.assertEqual(
            dirs["therock-ci-artifacts"],
            {"12345-linux/logs/gfx94X", "12345-linux/logs"},
        )

    def test_deduplication_same_dir(self):
        """50 files in the same directory produce one entry."""
        records = [
            _make_s3_record("therock-ci-artifacts", f"12345-linux/logs/gfx94X/file{i}.log")
            for i in range(50)
        ]
        dirs = s3_index_handler._collect_dirs_to_index(records)
        self.assertEqual(
            dirs["therock-ci-artifacts"],
            {"12345-linux/logs/gfx94X", "12345-linux/logs"},
        )

    def test_deduplication_ancestor(self):
        """Files in sibling dirs share a deduplicated ancestor entry."""
        records = [
            _make_s3_record("therock-ci-artifacts", "12345-linux/logs/gfx94X/a.log"),
            _make_s3_record("therock-ci-artifacts", "12345-linux/logs/gfx120X/b.log"),
        ]
        dirs = s3_index_handler._collect_dirs_to_index(records)
        self.assertEqual(
            dirs["therock-ci-artifacts"],
            {"12345-linux/logs/gfx94X", "12345-linux/logs/gfx120X", "12345-linux/logs"},
        )

    def test_run_root_not_walked_to(self):
        """Ancestor walk stops at the run prefix (depth 1 for standard bucket)."""
        records = [_make_s3_record(
            "therock-ci-artifacts",
            "12345-linux/logs/gfx94X/build.log",
        )]
        dirs = s3_index_handler._collect_dirs_to_index(records)
        self.assertNotIn("12345-linux", dirs["therock-ci-artifacts"])

    def test_external_bucket_run_prefix_depth(self):
        """External bucket stops ancestor walk at depth 2."""
        records = [_make_s3_record(
            "therock-ci-artifacts-external",
            "ROCm-TheRock/12345-linux/logs/gfx94X/build.log",
        )]
        dirs = s3_index_handler._collect_dirs_to_index(records)
        self.assertIn("ROCm-TheRock/12345-linux/logs/gfx94X", dirs["therock-ci-artifacts-external"])
        self.assertIn("ROCm-TheRock/12345-linux/logs", dirs["therock-ci-artifacts-external"])
        self.assertNotIn("ROCm-TheRock/12345-linux", dirs["therock-ci-artifacts-external"])
        self.assertNotIn("ROCm-TheRock", dirs["therock-ci-artifacts-external"])

    def test_index_html_skipped(self):
        records = [_make_s3_record(
            "therock-ci-artifacts",
            "12345-linux/logs/gfx94X/index.html",
        )]
        dirs = s3_index_handler._collect_dirs_to_index(records)
        self.assertEqual(dirs, {})

    def test_file_at_run_root_no_ancestor_walk(self):
        records = [_make_s3_record(
            "therock-ci-artifacts",
            "12345-linux/core_lib.tar.xz",
        )]
        dirs = s3_index_handler._collect_dirs_to_index(records)
        self.assertEqual(dirs["therock-ci-artifacts"], {"12345-linux"})


class TestHandler(unittest.TestCase):
    """Tests for the Lambda handler() entry point."""

    @mock.patch("s3_index_handler.generate_s3_index.generate_index_for_directory")
    @mock.patch("boto3.client")
    def test_log_file_indexes_leaf_and_ancestor(self, mock_boto3, mock_gen):
        event = _make_direct_event(
            "therock-ci-artifacts",
            "12345678901-linux/logs/gfx94X-dcgpu/build.log",
        )
        result = s3_index_handler.lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["indexed"], 2)
        prefixes = {c.kwargs["dir_prefix"] for c in mock_gen.call_args_list}
        self.assertEqual(prefixes, {
            "12345678901-linux/logs/gfx94X-dcgpu",
            "12345678901-linux/logs",
        })

    @mock.patch("s3_index_handler.generate_s3_index.generate_index_for_directory")
    @mock.patch("boto3.client")
    def test_artifact_file_indexes_run_root_only(self, mock_boto3, mock_gen):
        event = _make_direct_event(
            "therock-ci-artifacts",
            "12345678901-linux/core_lib_gfx94X.tar.xz",
        )
        result = s3_index_handler.lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 200)
        mock_gen.assert_called_once()
        self.assertEqual(mock_gen.call_args.kwargs["dir_prefix"], "12345678901-linux")

    @mock.patch("s3_index_handler.generate_s3_index.generate_index_for_directory")
    @mock.patch("boto3.client")
    def test_fifty_files_same_dir_indexed_once(self, mock_boto3, mock_gen):
        """50 uploads to the same directory deduplicate to 2 index calls."""
        records = [
            _make_s3_record("therock-ci-artifacts", f"12345-linux/logs/gfx94X/file{i}.log")
            for i in range(50)
        ]
        event = {"Records": records}
        result = s3_index_handler.lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["indexed"], 2)  # leaf + logs/
        prefixes = {c.kwargs["dir_prefix"] for c in mock_gen.call_args_list}
        self.assertEqual(prefixes, {"12345-linux/logs/gfx94X", "12345-linux/logs"})

    @mock.patch("s3_index_handler.generate_s3_index.generate_index_for_directory")
    @mock.patch("boto3.client")
    def test_sqs_event_unwrapped(self, mock_boto3, mock_gen):
        event = _make_sqs_event(
            "therock-ci-artifacts",
            "12345678901-linux/logs/gfx94X-dcgpu/build.log",
        )
        result = s3_index_handler.lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["indexed"], 2)

    @mock.patch("s3_index_handler.generate_s3_index.generate_index_for_directory")
    @mock.patch("boto3.client")
    def test_index_html_upload_is_skipped(self, mock_boto3, mock_gen):
        event = _make_direct_event(
            "therock-ci-artifacts",
            "12345678901-linux/logs/gfx94X-dcgpu/index.html",
        )
        result = s3_index_handler.lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["indexed"], 0)
        mock_gen.assert_not_called()

    @mock.patch("s3_index_handler.generate_s3_index.generate_index_for_directory")
    @mock.patch("boto3.client")
    def test_bucket_root_file_is_skipped(self, mock_boto3, mock_gen):
        event = _make_direct_event("therock-ci-artifacts", "standalone_file.txt")
        result = s3_index_handler.lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["indexed"], 0)
        mock_gen.assert_not_called()

    @mock.patch("s3_index_handler.generate_s3_index.generate_index_for_directory")
    @mock.patch("boto3.client")
    def test_failed_index_raises(self, mock_boto3, mock_gen):
        mock_gen.side_effect = RuntimeError("S3 error")
        event = _make_direct_event(
            "therock-ci-artifacts",
            "12345678901-linux/logs/gfx94X-dcgpu/build.log",
        )
        with self.assertRaises(RuntimeError):
            s3_index_handler.lambda_handler(event, None)

    @mock.patch("s3_index_handler.generate_s3_index.generate_index_for_directory")
    @mock.patch("boto3.client")
    def test_empty_records_returns_ok(self, mock_boto3, mock_gen):
        result = s3_index_handler.lambda_handler({"Records": []}, None)
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["indexed"], 0)
        mock_gen.assert_not_called()

    @mock.patch("s3_index_handler.generate_s3_index.generate_index_for_directory")
    @mock.patch("boto3.client")
    def test_external_bucket_correct_dirs(self, mock_boto3, mock_gen):
        event = _make_direct_event(
            "therock-ci-artifacts-external",
            "ROCm-TheRock/12345678901-linux/logs/gfx94X-dcgpu/build.log",
        )
        result = s3_index_handler.lambda_handler(event, None)

        self.assertEqual(result["statusCode"], 200)
        prefixes = {c.kwargs["dir_prefix"] for c in mock_gen.call_args_list}
        self.assertEqual(prefixes, {
            "ROCm-TheRock/12345678901-linux/logs/gfx94X-dcgpu",
            "ROCm-TheRock/12345678901-linux/logs",
        })
        buckets = {c.kwargs["bucket"] for c in mock_gen.call_args_list}
        self.assertEqual(buckets, {"therock-ci-artifacts-external"})


if __name__ == "__main__":
    unittest.main()
