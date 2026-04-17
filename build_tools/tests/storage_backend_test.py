#!/usr/bin/env python
"""Unit tests for storage_backend.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_location import StorageLocation
from _therock_utils.storage_backend import (
    LocalStorageBackend,
    S3StorageBackend,
    StorageBackend,
    create_storage_backend,
    infer_content_type,
)


# ---------------------------------------------------------------------------
# Content-type inference
# ---------------------------------------------------------------------------


class TestInferContentType(unittest.TestCase):
    def test_html(self):
        self.assertEqual(infer_content_type(Path("index.html")), "text/html")

    def test_json(self):
        self.assertEqual(infer_content_type(Path("manifest.json")), "application/json")

    def test_log(self):
        self.assertEqual(infer_content_type(Path("build.log")), "text/plain")

    def test_markdown(self):
        self.assertEqual(infer_content_type(Path("summary.md")), "text/plain")

    def test_gzip(self):
        self.assertEqual(infer_content_type(Path("ninja.tar.gz")), "application/gzip")

    def test_xz(self):
        self.assertEqual(infer_content_type(Path("core.tar.xz")), "application/x-xz")

    def test_zstd(self):
        self.assertEqual(infer_content_type(Path("core.tar.zst")), "application/zstd")

    def test_whl(self):
        ct = infer_content_type(Path("rocm-1.0-py3-none-any.whl"))
        self.assertIn(ct, ("application/zip", "application/octet-stream"))

    def test_unknown_extension(self):
        self.assertEqual(
            infer_content_type(Path("data.xyz123")), "application/octet-stream"
        )

    def test_sha256sum(self):
        # .sha256sum is not a known extension - falls to default.
        ct = infer_content_type(Path("core.tar.xz.sha256sum"))
        # The suffix is ".sha256sum", not ".xz".
        self.assertEqual(ct, "application/octet-stream")

    def test_case_insensitive(self):
        self.assertEqual(infer_content_type(Path("page.HTML")), "text/html")


# ---------------------------------------------------------------------------
# WorkflowOutputRoot.root()
# ---------------------------------------------------------------------------


class TestWorkflowOutputRootRoot(unittest.TestCase):
    def test_root_returns_output_location(self):
        rr = WorkflowOutputRoot(
            bucket="my-bucket", external_repo="", run_id="123", platform="linux"
        )
        loc = rr.root()
        self.assertIsInstance(loc, StorageLocation)
        self.assertEqual(loc.bucket, "my-bucket")
        self.assertEqual(loc.relative_path, "123-linux")

    def test_root_with_external_repo(self):
        rr = WorkflowOutputRoot(
            bucket="b", external_repo="owner-repo/", run_id="99", platform="windows"
        )
        loc = rr.root()
        self.assertEqual(loc.relative_path, "owner-repo/99-windows")


# ---------------------------------------------------------------------------
# LocalStorageBackend
# ---------------------------------------------------------------------------


class TestLocalStorageBackendUploadFile(unittest.TestCase):
    def test_copies_file(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            src_dir = Path(src)

            source = src_dir / "hello.txt"
            source.write_text("content")

            dest = StorageLocation("bucket", "run-1/hello.txt")
            backend = LocalStorageBackend(staging_dir)
            backend.upload_file(source, dest)

            target = staging_dir / "run-1" / "hello.txt"
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_text(), "content")

    def test_creates_parent_dirs(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            src_dir = Path(src)

            source = src_dir / "data.json"
            source.write_text("{}")

            dest = StorageLocation("bucket", "run-1/deep/nested/data.json")
            backend = LocalStorageBackend(staging_dir)
            backend.upload_file(source, dest)

            target = staging_dir / "run-1" / "deep" / "nested" / "data.json"
            self.assertTrue(target.is_file())

    def test_dry_run_does_not_copy(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            src_dir = Path(src)

            source = src_dir / "hello.txt"
            source.write_text("content")

            dest = StorageLocation("bucket", "run-1/hello.txt")
            backend = LocalStorageBackend(staging_dir, dry_run=True)
            backend.upload_file(source, dest)

            target = staging_dir / "run-1" / "hello.txt"
            self.assertFalse(target.exists())


class TestLocalStorageBackendUploadDirectory(unittest.TestCase):
    def _make_tree(self, base: Path):
        """Create a test directory tree:

        base/
            file1.tar.xz
            file1.tar.xz.sha256sum
            file2.log
            sub/
                nested.html
                deep/
                    deep.txt
        """
        base.mkdir(parents=True, exist_ok=True)
        (base / "file1.tar.xz").write_bytes(b"xz-data")
        (base / "file1.tar.xz.sha256sum").write_text("abc123")
        (base / "file2.log").write_text("log line")
        sub = base / "sub"
        sub.mkdir()
        (sub / "nested.html").write_text("<html/>")
        deep = sub / "deep"
        deep.mkdir()
        (deep / "deep.txt").write_text("deep content")

    def test_upload_all_files(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            self._make_tree(source_dir)

            dest = StorageLocation("bucket", "run-1")
            backend = LocalStorageBackend(staging_dir)
            count = backend.upload_directory(source_dir, dest)

            self.assertEqual(count, 5)
            self.assertTrue((staging_dir / "run-1" / "file1.tar.xz").is_file())
            self.assertTrue(
                (staging_dir / "run-1" / "file1.tar.xz.sha256sum").is_file()
            )
            self.assertTrue((staging_dir / "run-1" / "file2.log").is_file())
            self.assertTrue((staging_dir / "run-1" / "sub" / "nested.html").is_file())

    def test_upload_with_include_filter(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            self._make_tree(source_dir)

            dest = StorageLocation("bucket", "run-1")
            backend = LocalStorageBackend(staging_dir)
            count = backend.upload_directory(source_dir, dest, include=["*.tar.xz*"])

            # Only .tar.xz and .tar.xz.sha256sum should match
            self.assertEqual(count, 2)
            self.assertTrue((staging_dir / "run-1" / "file1.tar.xz").is_file())
            self.assertTrue(
                (staging_dir / "run-1" / "file1.tar.xz.sha256sum").is_file()
            )
            self.assertFalse((staging_dir / "run-1" / "file2.log").exists())
            self.assertFalse((staging_dir / "run-1" / "sub" / "nested.html").exists())

    def test_preserves_subdirectory_structure(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "logs"
            self._make_tree(source_dir)

            dest = StorageLocation("bucket", "run-1/logs/gfx94X")
            backend = LocalStorageBackend(staging_dir)
            backend.upload_directory(source_dir, dest)

            self.assertTrue(
                (
                    staging_dir / "run-1" / "logs" / "gfx94X" / "sub" / "nested.html"
                ).is_file()
            )

    def test_skips_symlinks(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            source_dir.mkdir()
            (source_dir / "real.tar.xz").write_bytes(b"data")
            try:
                (source_dir / "link.tar.xz").symlink_to(source_dir / "real.tar.xz")
            except OSError:
                self.skipTest("Cannot create symlinks on this platform")

            dest = StorageLocation("bucket", "run-1")
            backend = LocalStorageBackend(staging_dir)
            count = backend.upload_directory(source_dir, dest)

            self.assertEqual(count, 1)
            self.assertTrue((staging_dir / "run-1" / "real.tar.xz").is_file())
            self.assertFalse((staging_dir / "run-1" / "link.tar.xz").exists())

    def test_nonexistent_source_raises(self):

        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            dest = StorageLocation("bucket", "run-1")
            backend = LocalStorageBackend(staging_dir)

            with self.assertRaises(FileNotFoundError):
                backend.upload_directory(Path("/nonexistent"), dest)

    def test_dry_run_does_not_copy(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            self._make_tree(source_dir)

            dest = StorageLocation("bucket", "run-1")
            backend = LocalStorageBackend(staging_dir, dry_run=True)
            count = backend.upload_directory(source_dir, dest)

            # Count should reflect files that would be uploaded.
            self.assertEqual(count, 5)
            # But nothing should actually be written.
            self.assertFalse((staging_dir / "run-1").exists())

    def test_exclude_direct_children_only(self):
        """sub/* excludes direct children but not deeply nested files."""
        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            self._make_tree(source_dir)

            dest = StorageLocation("bucket", "run-1")
            backend = LocalStorageBackend(staging_dir)
            count = backend.upload_directory(source_dir, dest, exclude=["sub/*"])

            # sub/nested.html excluded, but sub/deep/deep.txt still uploaded.
            self.assertEqual(count, 4)
            self.assertTrue((staging_dir / "run-1" / "file1.tar.xz").is_file())
            self.assertFalse((staging_dir / "run-1" / "sub" / "nested.html").exists())
            self.assertTrue(
                (staging_dir / "run-1" / "sub" / "deep" / "deep.txt").is_file()
            )

    def test_exclude_recursive(self):
        """sub/**/* excludes files at all depths."""
        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            self._make_tree(source_dir)

            dest = StorageLocation("bucket", "run-1")
            backend = LocalStorageBackend(staging_dir)
            count = backend.upload_directory(source_dir, dest, exclude=["sub/**/*"])

            # sub/nested.html and sub/deep/deep.txt both excluded.
            self.assertEqual(count, 3)
            self.assertTrue((staging_dir / "run-1" / "file1.tar.xz").is_file())
            self.assertTrue((staging_dir / "run-1" / "file2.log").is_file())
            self.assertFalse((staging_dir / "run-1" / "sub" / "nested.html").exists())
            self.assertFalse(
                (staging_dir / "run-1" / "sub" / "deep" / "deep.txt").exists()
            )

    def test_exclude_with_include(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            self._make_tree(source_dir)

            dest = StorageLocation("bucket", "run-1")
            backend = LocalStorageBackend(staging_dir)
            # Include all files, but exclude .log files.
            count = backend.upload_directory(source_dir, dest, exclude=["*.log"])

            self.assertEqual(count, 4)
            self.assertTrue((staging_dir / "run-1" / "file1.tar.xz").is_file())
            self.assertTrue((staging_dir / "run-1" / "sub" / "nested.html").is_file())
            self.assertFalse((staging_dir / "run-1" / "file2.log").exists())

    def test_empty_directory_returns_zero(self):

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "empty"
            source_dir.mkdir()

            dest = StorageLocation("bucket", "run-1")
            backend = LocalStorageBackend(staging_dir)
            count = backend.upload_directory(source_dir, dest)
            self.assertEqual(count, 0)


class TestLocalStorageBackendCopyFile(unittest.TestCase):
    def test_copies_between_locations(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)

            # Create source file
            src_path = staging_dir / "run-1" / "file.whl"
            src_path.parent.mkdir(parents=True)
            src_path.write_text("wheel-data")

            source = StorageLocation("bucket", "run-1/file.whl")
            dest = StorageLocation("bucket", "release/file.whl")

            backend = LocalStorageBackend(staging_dir)
            backend.copy_file(source, dest)

            dst_path = staging_dir / "release" / "file.whl"
            self.assertTrue(dst_path.is_file())
            self.assertEqual(dst_path.read_text(), "wheel-data")

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)

            src_path = staging_dir / "a" / "b.txt"
            src_path.parent.mkdir(parents=True)
            src_path.write_text("data")

            source = StorageLocation("bucket", "a/b.txt")
            dest = StorageLocation("bucket", "x/y/z/b.txt")

            backend = LocalStorageBackend(staging_dir)
            backend.copy_file(source, dest)

            self.assertTrue((staging_dir / "x" / "y" / "z" / "b.txt").is_file())

    def test_dry_run_does_not_copy(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)

            src_path = staging_dir / "run-1" / "file.whl"
            src_path.parent.mkdir(parents=True)
            src_path.write_text("wheel-data")

            source = StorageLocation("bucket", "run-1/file.whl")
            dest = StorageLocation("bucket", "release/file.whl")

            backend = LocalStorageBackend(staging_dir, dry_run=True)
            backend.copy_file(source, dest)

            self.assertFalse((staging_dir / "release" / "file.whl").exists())


# ---------------------------------------------------------------------------
# S3StorageBackend
# ---------------------------------------------------------------------------


class TestS3StorageBackendUploadFile(unittest.TestCase):
    def test_calls_boto3_upload_file(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        source = Path("/tmp/build.log")
        dest = StorageLocation("my-bucket", "run-1/logs/build.log")
        backend.upload_file(source, dest)

        mock_client.upload_file.assert_called_once_with(
            str(source),
            "my-bucket",
            "run-1/logs/build.log",
            ExtraArgs={"ContentType": "text/plain"},
        )

    def test_content_type_for_html(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        source = Path("/tmp/index.html")
        dest = StorageLocation("my-bucket", "run-1/index.html")
        backend.upload_file(source, dest)

        mock_client.upload_file.assert_called_once_with(
            str(source),
            "my-bucket",
            "run-1/index.html",
            ExtraArgs={"ContentType": "text/html"},
        )

    def test_retries_on_failure(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        mock_client.upload_file.side_effect = [
            Exception("transient"),
            None,  # succeeds on second attempt
        ]
        backend._s3_client = mock_client

        source = Path("/tmp/data.json")
        dest = StorageLocation("bucket", "run-1/data.json")

        with mock.patch("_therock_utils.storage_backend.time.sleep"):
            backend.upload_file(source, dest)

        self.assertEqual(mock_client.upload_file.call_count, 2)

    def test_raises_after_max_retries(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        mock_client.upload_file.side_effect = Exception("persistent")
        backend._s3_client = mock_client

        source = Path("/tmp/data.json")
        dest = StorageLocation("bucket", "run-1/data.json")

        with mock.patch("_therock_utils.storage_backend.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                backend.upload_file(source, dest)

        self.assertIn("3 attempts", str(ctx.exception))
        self.assertEqual(mock_client.upload_file.call_count, 3)

    def test_dry_run_does_not_call_boto3(self):
        backend = S3StorageBackend(dry_run=True)
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        source = Path("/tmp/build.log")
        dest = StorageLocation("bucket", "run-1/build.log")
        backend.upload_file(source, dest)

        mock_client.upload_file.assert_not_called()


class TestS3StorageBackendCopyFile(unittest.TestCase):
    def test_calls_copy_object(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        source = StorageLocation("src-bucket", "run-1/file.whl")
        dest = StorageLocation("dest-bucket", "release/file.whl")
        backend.copy_file(source, dest)

        mock_client.copy_object.assert_called_once_with(
            Bucket="dest-bucket",
            Key="release/file.whl",
            CopySource={"Bucket": "src-bucket", "Key": "run-1/file.whl"},
        )

    def test_same_bucket_copy(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        source = StorageLocation("bucket", "staging/file.whl")
        dest = StorageLocation("bucket", "release/file.whl")
        backend.copy_file(source, dest)

        mock_client.copy_object.assert_called_once_with(
            Bucket="bucket",
            Key="release/file.whl",
            CopySource={"Bucket": "bucket", "Key": "staging/file.whl"},
        )

    def test_retries_on_failure(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        mock_client.copy_object.side_effect = [
            Exception("transient"),
            None,
        ]
        backend._s3_client = mock_client

        source = StorageLocation("src", "a.whl")
        dest = StorageLocation("dst", "b.whl")

        with mock.patch("_therock_utils.storage_backend.time.sleep"):
            backend.copy_file(source, dest)

        self.assertEqual(mock_client.copy_object.call_count, 2)

    def test_dry_run_does_not_call_boto3(self):
        backend = S3StorageBackend(dry_run=True)
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        source = StorageLocation("src", "a.whl")
        dest = StorageLocation("dst", "b.whl")
        backend.copy_file(source, dest)

        mock_client.copy_object.assert_not_called()


class TestLocalStorageBackendUploadFiles(unittest.TestCase):
    def test_uploads_all_files(self):
        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            src_dir = Path(src)

            (src_dir / "a.txt").write_text("aaa")
            (src_dir / "b.txt").write_text("bbb")

            files = [
                (src_dir / "a.txt", StorageLocation("bucket", "run-1/a.txt")),
                (src_dir / "b.txt", StorageLocation("bucket", "run-1/b.txt")),
            ]
            backend = LocalStorageBackend(staging_dir)
            count = backend.upload_files(files)

            self.assertEqual(count, 2)
            self.assertEqual((staging_dir / "run-1" / "a.txt").read_text(), "aaa")
            self.assertEqual((staging_dir / "run-1" / "b.txt").read_text(), "bbb")

    def test_empty_list_returns_zero(self):
        with tempfile.TemporaryDirectory() as staging:
            backend = LocalStorageBackend(Path(staging))
            self.assertEqual(backend.upload_files([]), 0)


class TestS3StorageBackendUploadFiles(unittest.TestCase):
    def test_uploads_all_files_in_parallel(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        files = [
            (Path("/tmp/a.log"), StorageLocation("bucket", "run-1/a.log")),
            (Path("/tmp/b.log"), StorageLocation("bucket", "run-1/b.log")),
            (Path("/tmp/c.log"), StorageLocation("bucket", "run-1/c.log")),
        ]
        count = backend.upload_files(files)

        self.assertEqual(count, 3)
        self.assertEqual(mock_client.upload_file.call_count, 3)

    def test_empty_list_returns_zero(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        count = backend.upload_files([])
        self.assertEqual(count, 0)
        mock_client.upload_file.assert_not_called()

    def test_single_file_skips_thread_pool(self):
        """A single file should not use a thread pool (no overhead)."""
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        files = [
            (Path("/tmp/only.log"), StorageLocation("bucket", "run-1/only.log")),
        ]

        with mock.patch(
            "_therock_utils.storage_backend.concurrent.futures.ThreadPoolExecutor"
        ) as mock_pool:
            count = backend.upload_files(files)

        self.assertEqual(count, 1)
        mock_pool.assert_not_called()
        mock_client.upload_file.assert_called_once()

    def test_dry_run_skips_thread_pool(self):
        backend = S3StorageBackend(dry_run=True)
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        files = [
            (Path("/tmp/a.log"), StorageLocation("bucket", "run-1/a.log")),
            (Path("/tmp/b.log"), StorageLocation("bucket", "run-1/b.log")),
        ]

        with mock.patch(
            "_therock_utils.storage_backend.concurrent.futures.ThreadPoolExecutor"
        ) as mock_pool:
            count = backend.upload_files(files)

        self.assertEqual(count, 2)
        mock_pool.assert_not_called()
        # Dry run: no actual boto3 calls
        mock_client.upload_file.assert_not_called()

    def test_error_propagation(self):
        """If a file fails after retries, upload_files raises RuntimeError."""
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()

        # Second file always fails (after _s3_retry exhausts retries)
        def upload_side_effect(filename, bucket, key, **kwargs):
            if "bad" in key:
                raise Exception("persistent failure")

        mock_client.upload_file.side_effect = upload_side_effect
        backend._s3_client = mock_client

        files = [
            (Path("/tmp/good.log"), StorageLocation("bucket", "run-1/good.log")),
            (Path("/tmp/bad.log"), StorageLocation("bucket", "run-1/bad.log")),
        ]

        with mock.patch("_therock_utils.storage_backend.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                backend.upload_files(files)

        self.assertIn("bad.log", str(ctx.exception))

    def test_concurrency_param_wired_through(self):
        backend = S3StorageBackend(upload_concurrency=5)
        self.assertEqual(backend._upload_concurrency, 5)


class TestS3StorageBackendMaxPoolConnections(unittest.TestCase):
    def test_default_pool_connections_match_concurrency(self):
        with mock.patch("boto3.client") as mock_boto3:
            backend = S3StorageBackend()
            _ = backend.s3_client

            mock_boto3.assert_called_once()
            config = mock_boto3.call_args.kwargs.get("config")
            self.assertIsNotNone(config)
            self.assertEqual(config.max_pool_connections, 10)

    def test_custom_concurrency_sets_pool_connections(self):
        with mock.patch("boto3.client") as mock_boto3:
            backend = S3StorageBackend(upload_concurrency=20)
            _ = backend.s3_client

            config = mock_boto3.call_args.kwargs.get("config")
            self.assertEqual(config.max_pool_connections, 20)


# ---------------------------------------------------------------------------
# LocalStorageBackend.list_files
# ---------------------------------------------------------------------------


class TestLocalStorageBackendListFiles(unittest.TestCase):
    def test_lists_all_files(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            d = staging_dir / "run-1" / "tarballs"
            d.mkdir(parents=True)
            (d / "a.tar.gz").write_bytes(b"a")
            (d / "b.tar.gz").write_bytes(b"b")
            (d / "c.log").write_text("log")

            loc = StorageLocation("bucket", "run-1/tarballs")
            backend = LocalStorageBackend(staging_dir)
            files = backend.list_files(loc)

            names = [f.relative_path.rsplit("/", 1)[-1] for f in files]
            self.assertEqual(names, ["a.tar.gz", "b.tar.gz", "c.log"])

    def test_include_filter(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            d = staging_dir / "run-1" / "tarballs"
            d.mkdir(parents=True)
            (d / "a.tar.gz").write_bytes(b"a")
            (d / "b.tar.gz").write_bytes(b"b")
            (d / "c.log").write_text("log")

            loc = StorageLocation("bucket", "run-1/tarballs")
            backend = LocalStorageBackend(staging_dir)
            files = backend.list_files(loc, include=["*.tar.gz"])

            names = [f.relative_path.rsplit("/", 1)[-1] for f in files]
            self.assertEqual(names, ["a.tar.gz", "b.tar.gz"])

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            d = staging_dir / "run-1" / "tarballs"
            d.mkdir(parents=True)

            loc = StorageLocation("bucket", "run-1/tarballs")
            backend = LocalStorageBackend(staging_dir)
            files = backend.list_files(loc)
            self.assertEqual(files, [])

    def test_nonexistent_directory(self):
        with tempfile.TemporaryDirectory() as staging:
            loc = StorageLocation("bucket", "run-1/tarballs")
            backend = LocalStorageBackend(Path(staging))
            files = backend.list_files(loc)
            self.assertEqual(files, [])

    def test_includes_nested_files(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            d = staging_dir / "run-1" / "tarballs"
            d.mkdir(parents=True)
            (d / "a.tar.gz").write_bytes(b"a")
            sub = d / "subdir"
            sub.mkdir()
            (sub / "nested.tar.gz").write_bytes(b"nested")

            loc = StorageLocation("bucket", "run-1/tarballs")
            backend = LocalStorageBackend(staging_dir)
            files = backend.list_files(loc)

            paths = [f.relative_path for f in files]
            self.assertEqual(
                paths,
                [
                    "run-1/tarballs/a.tar.gz",
                    "run-1/tarballs/subdir/nested.tar.gz",
                ],
            )


# ---------------------------------------------------------------------------
# S3StorageBackend.list_files
# ---------------------------------------------------------------------------


class TestS3StorageBackendListFiles(unittest.TestCase):
    def _make_backend_with_pages(self, pages):
        """Create an S3StorageBackend with a mock paginator returning *pages*."""
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        mock_paginator = mock.MagicMock()
        mock_paginator.paginate.return_value = pages
        mock_client.get_paginator.return_value = mock_paginator
        backend._s3_client = mock_client
        return backend

    def test_lists_all_files(self):
        pages = [
            {
                "Contents": [
                    {"Key": "run-1/tarballs/a.tar.gz"},
                    {"Key": "run-1/tarballs/b.tar.gz"},
                ]
            }
        ]
        backend = self._make_backend_with_pages(pages)
        loc = StorageLocation("bucket", "run-1/tarballs")
        files = backend.list_files(loc)

        self.assertEqual(len(files), 2)
        self.assertEqual(files[0].relative_path, "run-1/tarballs/a.tar.gz")
        self.assertEqual(files[1].relative_path, "run-1/tarballs/b.tar.gz")

    def test_include_filter(self):
        pages = [
            {
                "Contents": [
                    {"Key": "run-1/tarballs/a.tar.gz"},
                    {"Key": "run-1/tarballs/index.html"},
                ]
            }
        ]
        backend = self._make_backend_with_pages(pages)
        loc = StorageLocation("bucket", "run-1/tarballs")
        files = backend.list_files(loc, include=["*.tar.gz"])

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].relative_path, "run-1/tarballs/a.tar.gz")

    def test_empty_contents(self):
        pages = [{"Contents": []}]
        backend = self._make_backend_with_pages(pages)
        loc = StorageLocation("bucket", "run-1/tarballs")
        files = backend.list_files(loc)
        self.assertEqual(files, [])

    def test_no_contents_key(self):
        pages = [{}]
        backend = self._make_backend_with_pages(pages)
        loc = StorageLocation("bucket", "run-1/tarballs")
        files = backend.list_files(loc)
        self.assertEqual(files, [])

    def test_skips_prefix_directory_marker(self):
        pages = [
            {
                "Contents": [
                    {"Key": "run-1/tarballs/"},
                    {"Key": "run-1/tarballs/a.tar.gz"},
                ]
            }
        ]
        backend = self._make_backend_with_pages(pages)
        loc = StorageLocation("bucket", "run-1/tarballs")
        files = backend.list_files(loc)

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].relative_path, "run-1/tarballs/a.tar.gz")

    def test_multiple_pages(self):
        pages = [
            {"Contents": [{"Key": "run-1/tarballs/a.tar.gz"}]},
            {"Contents": [{"Key": "run-1/tarballs/b.tar.gz"}]},
        ]
        backend = self._make_backend_with_pages(pages)
        loc = StorageLocation("bucket", "run-1/tarballs")
        files = backend.list_files(loc)

        self.assertEqual(len(files), 2)

    def test_paginate_uses_trailing_slash(self):
        """Prefix should end with / to list directory contents."""
        backend = self._make_backend_with_pages([{}])
        mock_paginator = backend._s3_client.get_paginator.return_value

        loc = StorageLocation("bucket", "run-1/tarballs")
        backend.list_files(loc)

        mock_paginator.paginate.assert_called_once_with(
            Bucket="bucket", Prefix="run-1/tarballs/"
        )


# ---------------------------------------------------------------------------
# LocalStorageBackend.copy_directory
# ---------------------------------------------------------------------------


class TestLocalStorageBackendCopyDirectory(unittest.TestCase):
    def test_copies_filtered_files(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            src_dir = staging_dir / "run-1" / "tarballs"
            src_dir.mkdir(parents=True)
            (src_dir / "a.tar.gz").write_bytes(b"tarball-a")
            (src_dir / "b.tar.gz").write_bytes(b"tarball-b")
            (src_dir / "index.html").write_text("<html/>")

            source = StorageLocation("src-bucket", "run-1/tarballs")
            dest = StorageLocation("dst-bucket", "v3")
            backend = LocalStorageBackend(staging_dir)
            count = backend.copy_directory(source, dest, include=["*.tar.gz"])

            self.assertEqual(count, 2)
            self.assertEqual(
                (staging_dir / "v3" / "a.tar.gz").read_bytes(), b"tarball-a"
            )
            self.assertEqual(
                (staging_dir / "v3" / "b.tar.gz").read_bytes(), b"tarball-b"
            )
            self.assertFalse((staging_dir / "v3" / "index.html").exists())

    def test_copies_all_without_include(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            src_dir = staging_dir / "run-1" / "tarballs"
            src_dir.mkdir(parents=True)
            (src_dir / "a.tar.gz").write_bytes(b"data")
            (src_dir / "index.html").write_text("<html/>")

            source = StorageLocation("bucket", "run-1/tarballs")
            dest = StorageLocation("bucket", "v3")
            backend = LocalStorageBackend(staging_dir)
            count = backend.copy_directory(source, dest)

            self.assertEqual(count, 2)

    def test_empty_source_copies_nothing(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            src_dir = staging_dir / "run-1" / "tarballs"
            src_dir.mkdir(parents=True)

            source = StorageLocation("bucket", "run-1/tarballs")
            dest = StorageLocation("bucket", "v3")
            backend = LocalStorageBackend(staging_dir)
            count = backend.copy_directory(source, dest)

            self.assertEqual(count, 0)

    def test_cross_bucket_locations(self):
        """Source and dest can have different bucket names."""
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            src_dir = staging_dir / "run-1" / "tarballs"
            src_dir.mkdir(parents=True)
            (src_dir / "a.tar.gz").write_bytes(b"data")

            source = StorageLocation("therock-dev-artifacts", "run-1/tarballs")
            dest = StorageLocation("therock-dev-tarball", "v3")
            backend = LocalStorageBackend(staging_dir)
            count = backend.copy_directory(source, dest, include=["*.tar.gz"])

            self.assertEqual(count, 1)
            # Local backend ignores bucket names for file paths
            self.assertTrue((staging_dir / "v3" / "a.tar.gz").is_file())

    def test_preserves_subdirectory_structure(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            src_dir = staging_dir / "run-1" / "packages"
            src_dir.mkdir(parents=True)
            sub = src_dir / "gfx94X-dcgpu"
            sub.mkdir()
            (sub / "rocm_sdk-7.10.0.whl").write_bytes(b"wheel")
            (src_dir / "top.whl").write_bytes(b"top")

            source = StorageLocation("bucket", "run-1/packages")
            dest = StorageLocation("bucket", "v3")
            backend = LocalStorageBackend(staging_dir)
            count = backend.copy_directory(source, dest, include=["*.whl"])

            self.assertEqual(count, 2)
            self.assertTrue(
                (staging_dir / "v3" / "gfx94X-dcgpu" / "rocm_sdk-7.10.0.whl").is_file()
            )
            self.assertTrue((staging_dir / "v3" / "top.whl").is_file())

    def test_dry_run_does_not_copy(self):
        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            src_dir = staging_dir / "run-1" / "tarballs"
            src_dir.mkdir(parents=True)
            (src_dir / "a.tar.gz").write_bytes(b"data")

            source = StorageLocation("bucket", "run-1/tarballs")
            dest = StorageLocation("bucket", "v3")
            backend = LocalStorageBackend(staging_dir, dry_run=True)
            count = backend.copy_directory(source, dest, include=["*.tar.gz"])

            self.assertEqual(count, 1)
            self.assertFalse((staging_dir / "v3").exists())


# ---------------------------------------------------------------------------
# S3StorageBackend.copy_files
# ---------------------------------------------------------------------------


class TestS3StorageBackendCopyFiles(unittest.TestCase):
    def test_copies_all_files_in_parallel(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        files = [
            (
                StorageLocation("src", "run-1/a.tar.gz"),
                StorageLocation("dst", "v3/a.tar.gz"),
            ),
            (
                StorageLocation("src", "run-1/b.tar.gz"),
                StorageLocation("dst", "v3/b.tar.gz"),
            ),
            (
                StorageLocation("src", "run-1/c.tar.gz"),
                StorageLocation("dst", "v3/c.tar.gz"),
            ),
        ]
        count = backend.copy_files(files)

        self.assertEqual(count, 3)
        self.assertEqual(mock_client.copy_object.call_count, 3)

    def test_empty_list_returns_zero(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        count = backend.copy_files([])
        self.assertEqual(count, 0)
        mock_client.copy_object.assert_not_called()

    def test_single_file_skips_thread_pool(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        files = [
            (
                StorageLocation("src", "run-1/a.tar.gz"),
                StorageLocation("dst", "v3/a.tar.gz"),
            ),
        ]

        with mock.patch(
            "_therock_utils.storage_backend.concurrent.futures.ThreadPoolExecutor"
        ) as mock_pool:
            count = backend.copy_files(files)

        self.assertEqual(count, 1)
        mock_pool.assert_not_called()
        mock_client.copy_object.assert_called_once()

    def test_dry_run_does_not_call_boto3(self):
        backend = S3StorageBackend(dry_run=True)
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        files = [
            (
                StorageLocation("src", "run-1/a.tar.gz"),
                StorageLocation("dst", "v3/a.tar.gz"),
            ),
            (
                StorageLocation("src", "run-1/b.tar.gz"),
                StorageLocation("dst", "v3/b.tar.gz"),
            ),
        ]
        count = backend.copy_files(files)

        self.assertEqual(count, 2)
        mock_client.copy_object.assert_not_called()

    def test_error_propagation(self):
        backend = S3StorageBackend()
        mock_client = mock.MagicMock()

        def copy_side_effect(**kwargs):
            if "bad" in kwargs["Key"]:
                raise Exception("persistent failure")

        mock_client.copy_object.side_effect = copy_side_effect
        backend._s3_client = mock_client

        files = [
            (
                StorageLocation("src", "run-1/good.tar.gz"),
                StorageLocation("dst", "v3/good.tar.gz"),
            ),
            (
                StorageLocation("src", "run-1/bad.tar.gz"),
                StorageLocation("dst", "v3/bad.tar.gz"),
            ),
        ]

        with mock.patch("_therock_utils.storage_backend.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                backend.copy_files(files)

        self.assertIn("bad.tar.gz", str(ctx.exception))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateStorageBackend(unittest.TestCase):
    def test_returns_s3_backend_by_default(self):
        backend = create_storage_backend()
        self.assertIsInstance(backend, S3StorageBackend)

    def test_returns_local_backend_with_staging_dir(self):
        backend = create_storage_backend(staging_dir=Path("/tmp/staging"))
        self.assertIsInstance(backend, LocalStorageBackend)

    def test_dry_run_passed_through(self):
        backend = create_storage_backend(dry_run=True)
        self.assertIsInstance(backend, S3StorageBackend)
        self.assertTrue(backend._dry_run)

    def test_local_dry_run_passed_through(self):
        backend = create_storage_backend(staging_dir=Path("/tmp/staging"), dry_run=True)
        self.assertIsInstance(backend, LocalStorageBackend)
        self.assertTrue(backend._dry_run)

    def test_upload_concurrency_passed_to_s3_backend(self):
        backend = create_storage_backend(upload_concurrency=25)
        self.assertIsInstance(backend, S3StorageBackend)
        self.assertEqual(backend._upload_concurrency, 25)

    def test_upload_concurrency_ignored_for_local(self):
        backend = create_storage_backend(
            staging_dir=Path("/tmp/staging"), upload_concurrency=25
        )
        self.assertIsInstance(backend, LocalStorageBackend)


if __name__ == "__main__":
    unittest.main()
