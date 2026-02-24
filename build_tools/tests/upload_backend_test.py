#!/usr/bin/env python
"""Unit tests for upload_backend.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.run_outputs import OutputLocation, RunOutputRoot
from _therock_utils.upload_backend import (
    LocalUploadBackend,
    S3UploadBackend,
    UploadBackend,
    create_upload_backend,
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
        # .sha256sum is not a known extension — falls to default.
        ct = infer_content_type(Path("core.tar.xz.sha256sum"))
        # The suffix is ".sha256sum", not ".xz".
        self.assertEqual(ct, "application/octet-stream")

    def test_case_insensitive(self):
        self.assertEqual(infer_content_type(Path("page.HTML")), "text/html")


# ---------------------------------------------------------------------------
# RunOutputRoot.root()
# ---------------------------------------------------------------------------


class TestRunOutputRootRoot(unittest.TestCase):
    def test_root_returns_output_location(self):
        rr = RunOutputRoot(
            bucket="my-bucket", external_repo="", run_id="123", platform="linux"
        )
        loc = rr.root()
        self.assertIsInstance(loc, OutputLocation)
        self.assertEqual(loc.bucket, "my-bucket")
        self.assertEqual(loc.relative_path, "123-linux")

    def test_root_with_external_repo(self):
        rr = RunOutputRoot(
            bucket="b", external_repo="owner-repo/", run_id="99", platform="windows"
        )
        loc = rr.root()
        self.assertEqual(loc.relative_path, "owner-repo/99-windows")


# ---------------------------------------------------------------------------
# LocalUploadBackend
# ---------------------------------------------------------------------------


class TestLocalUploadBackendUploadFile(unittest.TestCase):
    def test_copies_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            src_dir = Path(src)

            source = src_dir / "hello.txt"
            source.write_text("content")

            dest = OutputLocation("bucket", "run-1/hello.txt")
            backend = LocalUploadBackend(staging_dir)
            backend.upload_file(source, dest)

            target = staging_dir / "run-1" / "hello.txt"
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_text(), "content")

    def test_creates_parent_dirs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            src_dir = Path(src)

            source = src_dir / "data.json"
            source.write_text("{}")

            dest = OutputLocation("bucket", "run-1/deep/nested/data.json")
            backend = LocalUploadBackend(staging_dir)
            backend.upload_file(source, dest)

            target = staging_dir / "run-1" / "deep" / "nested" / "data.json"
            self.assertTrue(target.is_file())

    def test_dry_run_does_not_copy(self):
        import tempfile

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            src_dir = Path(src)

            source = src_dir / "hello.txt"
            source.write_text("content")

            dest = OutputLocation("bucket", "run-1/hello.txt")
            backend = LocalUploadBackend(staging_dir, dry_run=True)
            backend.upload_file(source, dest)

            target = staging_dir / "run-1" / "hello.txt"
            self.assertFalse(target.exists())


class TestLocalUploadBackendUploadDirectory(unittest.TestCase):
    def _make_tree(self, base: Path):
        """Create a test directory tree:

        base/
            file1.tar.xz
            file1.tar.xz.sha256sum
            file2.log
            sub/
                nested.html
        """
        base.mkdir(parents=True, exist_ok=True)
        (base / "file1.tar.xz").write_bytes(b"xz-data")
        (base / "file1.tar.xz.sha256sum").write_text("abc123")
        (base / "file2.log").write_text("log line")
        sub = base / "sub"
        sub.mkdir()
        (sub / "nested.html").write_text("<html/>")

    def test_upload_all_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            self._make_tree(source_dir)

            dest = OutputLocation("bucket", "run-1")
            backend = LocalUploadBackend(staging_dir)
            count = backend.upload_directory(source_dir, dest)

            self.assertEqual(count, 4)
            self.assertTrue((staging_dir / "run-1" / "file1.tar.xz").is_file())
            self.assertTrue(
                (staging_dir / "run-1" / "file1.tar.xz.sha256sum").is_file()
            )
            self.assertTrue((staging_dir / "run-1" / "file2.log").is_file())
            self.assertTrue((staging_dir / "run-1" / "sub" / "nested.html").is_file())

    def test_upload_with_include_filter(self):
        import tempfile

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            self._make_tree(source_dir)

            dest = OutputLocation("bucket", "run-1")
            backend = LocalUploadBackend(staging_dir)
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
        import tempfile

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "logs"
            self._make_tree(source_dir)

            dest = OutputLocation("bucket", "run-1/logs/gfx94X")
            backend = LocalUploadBackend(staging_dir)
            backend.upload_directory(source_dir, dest)

            self.assertTrue(
                (
                    staging_dir / "run-1" / "logs" / "gfx94X" / "sub" / "nested.html"
                ).is_file()
            )

    def test_skips_symlinks(self):
        import tempfile

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            source_dir.mkdir()
            (source_dir / "real.tar.xz").write_bytes(b"data")
            try:
                (source_dir / "link.tar.xz").symlink_to(source_dir / "real.tar.xz")
            except OSError:
                self.skipTest("Cannot create symlinks on this platform")

            dest = OutputLocation("bucket", "run-1")
            backend = LocalUploadBackend(staging_dir)
            count = backend.upload_directory(source_dir, dest)

            self.assertEqual(count, 1)
            self.assertTrue((staging_dir / "run-1" / "real.tar.xz").is_file())
            self.assertFalse((staging_dir / "run-1" / "link.tar.xz").exists())

    def test_nonexistent_source_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as staging:
            staging_dir = Path(staging)
            dest = OutputLocation("bucket", "run-1")
            backend = LocalUploadBackend(staging_dir)

            with self.assertRaises(FileNotFoundError):
                backend.upload_directory(Path("/nonexistent"), dest)

    def test_dry_run_does_not_copy(self):
        import tempfile

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "artifacts"
            self._make_tree(source_dir)

            dest = OutputLocation("bucket", "run-1")
            backend = LocalUploadBackend(staging_dir, dry_run=True)
            count = backend.upload_directory(source_dir, dest)

            # Count should reflect files that would be uploaded.
            self.assertEqual(count, 4)
            # But nothing should actually be written.
            self.assertFalse((staging_dir / "run-1").exists())

    def test_empty_directory_returns_zero(self):
        import tempfile

        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as src:
            staging_dir = Path(staging)
            source_dir = Path(src) / "empty"
            source_dir.mkdir()

            dest = OutputLocation("bucket", "run-1")
            backend = LocalUploadBackend(staging_dir)
            count = backend.upload_directory(source_dir, dest)
            self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# S3UploadBackend
# ---------------------------------------------------------------------------


class TestS3UploadBackendUploadFile(unittest.TestCase):
    def test_calls_boto3_upload_file(self):
        backend = S3UploadBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        source = Path("/tmp/build.log")
        dest = OutputLocation("my-bucket", "run-1/logs/build.log")
        backend.upload_file(source, dest)

        mock_client.upload_file.assert_called_once_with(
            str(source),
            "my-bucket",
            "run-1/logs/build.log",
            ExtraArgs={"ContentType": "text/plain"},
        )

    def test_content_type_for_html(self):
        backend = S3UploadBackend()
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        source = Path("/tmp/index.html")
        dest = OutputLocation("my-bucket", "run-1/index.html")
        backend.upload_file(source, dest)

        mock_client.upload_file.assert_called_once_with(
            str(source),
            "my-bucket",
            "run-1/index.html",
            ExtraArgs={"ContentType": "text/html"},
        )

    def test_retries_on_failure(self):
        backend = S3UploadBackend()
        mock_client = mock.MagicMock()
        mock_client.upload_file.side_effect = [
            Exception("transient"),
            None,  # succeeds on second attempt
        ]
        backend._s3_client = mock_client

        source = Path("/tmp/data.json")
        dest = OutputLocation("bucket", "run-1/data.json")

        with mock.patch("_therock_utils.upload_backend.time.sleep"):
            backend.upload_file(source, dest)

        self.assertEqual(mock_client.upload_file.call_count, 2)

    def test_raises_after_max_retries(self):
        backend = S3UploadBackend()
        mock_client = mock.MagicMock()
        mock_client.upload_file.side_effect = Exception("persistent")
        backend._s3_client = mock_client

        source = Path("/tmp/data.json")
        dest = OutputLocation("bucket", "run-1/data.json")

        with mock.patch("_therock_utils.upload_backend.time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                backend.upload_file(source, dest)

        self.assertIn("3 attempts", str(ctx.exception))
        self.assertEqual(mock_client.upload_file.call_count, 3)

    def test_dry_run_does_not_call_boto3(self):
        backend = S3UploadBackend(dry_run=True)
        mock_client = mock.MagicMock()
        backend._s3_client = mock_client

        source = Path("/tmp/build.log")
        dest = OutputLocation("bucket", "run-1/build.log")
        backend.upload_file(source, dest)

        mock_client.upload_file.assert_not_called()


class TestS3UploadBackendClientInit(unittest.TestCase):
    def test_authenticated_client_when_env_vars_set(self):
        env = {
            "AWS_ACCESS_KEY_ID": "key",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_SESSION_TOKEN": "token",
        }
        with mock.patch.dict(os.environ, env):
            with mock.patch("boto3.client") as mock_boto3:
                backend = S3UploadBackend()
                _ = backend.s3_client

                mock_boto3.assert_called_once()
                call_kwargs = mock_boto3.call_args
                self.assertEqual(call_kwargs.kwargs["aws_access_key_id"], "key")

    def test_unsigned_client_when_env_vars_missing(self):
        env = {
            "AWS_ACCESS_KEY_ID": "",
            "AWS_SECRET_ACCESS_KEY": "",
            "AWS_SESSION_TOKEN": "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            # Ensure the vars are effectively missing (empty string != None).
            # We need to remove them entirely.
            for k in env:
                os.environ.pop(k, None)
            with mock.patch("boto3.client") as mock_boto3:
                backend = S3UploadBackend()
                _ = backend.s3_client

                mock_boto3.assert_called_once()
                call_kwargs = mock_boto3.call_args
                self.assertIn("config", call_kwargs.kwargs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateUploadBackend(unittest.TestCase):
    def test_returns_s3_backend_by_default(self):
        backend = create_upload_backend()
        self.assertIsInstance(backend, S3UploadBackend)

    def test_returns_local_backend_with_staging_dir(self):
        backend = create_upload_backend(staging_dir=Path("/tmp/staging"))
        self.assertIsInstance(backend, LocalUploadBackend)

    def test_dry_run_passed_through(self):
        backend = create_upload_backend(dry_run=True)
        self.assertIsInstance(backend, S3UploadBackend)
        self.assertTrue(backend._dry_run)

    def test_local_dry_run_passed_through(self):
        backend = create_upload_backend(staging_dir=Path("/tmp/staging"), dry_run=True)
        self.assertIsInstance(backend, LocalUploadBackend)
        self.assertTrue(backend._dry_run)


if __name__ == "__main__":
    unittest.main()
