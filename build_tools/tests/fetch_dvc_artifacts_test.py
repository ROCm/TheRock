#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for fetch_dvc_artifacts.py.

These cover only logic we wrote. boto3-owned behavior (HTTP retry, multipart
download, status-code handling) is the vendor's responsibility and not retested
here. The mocked-boto3 cases verify only that our orchestration glue around
download_fileobj preserves atomic-write and MD5-verification invariants.
"""

import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add build_tools to path so fetch_dvc_artifacts is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import fetch_dvc_artifacts as fda


def _md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


class TestParseDvcPointer(unittest.TestCase):
    """Tests for the .dvc YAML subset parser."""

    def _write(self, tmp: Path, content: str) -> Path:
        p = tmp / "x.dvc"
        p.write_text(content)
        return p

    def test_real_world_example(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n"
                "- md5: 35b9082b72e661dc5e37f14de1d9d4ed\n"
                "  size: 109795654\n"
                "  hash: md5\n"
                "  path: gfx908.kdb.bz2\n",
            )
            outs = fda._parse_dvc_pointer(p)
            self.assertEqual(len(outs), 1)
            self.assertEqual(outs[0].md5, "35b9082b72e661dc5e37f14de1d9d4ed")
            self.assertEqual(outs[0].size, 109795654)
            self.assertEqual(outs[0].path, "gfx908.kdb.bz2")
            self.assertFalse(outs[0].is_dir)

    def test_missing_outs_section(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(Path(t), "wmd: foo\n")
            with self.assertRaisesRegex(fda.FetchError, "no 'outs:' entries"):
                fda._parse_dvc_pointer(p)

    def test_missing_md5_field(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- size: 100\n  hash: md5\n  path: foo.bin\n",
            )
            with self.assertRaisesRegex(fda.FetchError, "missing required field"):
                fda._parse_dvc_pointer(p)

    def test_missing_path_field(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- md5: " + "a" * 32 + "\n  size: 100\n  hash: md5\n",
            )
            with self.assertRaisesRegex(fda.FetchError, "missing required field"):
                fda._parse_dvc_pointer(p)

    def test_bad_size(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- md5: " + "a" * 32 + "\n  size: huge\n  path: f\n",
            )
            with self.assertRaisesRegex(fda.FetchError, "bad size"):
                fda._parse_dvc_pointer(p)

    def test_invalid_md5_chars(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- md5: " + "z" * 32 + "\n  size: 1\n  path: f\n",
            )
            with self.assertRaisesRegex(fda.FetchError, "invalid md5"):
                fda._parse_dvc_pointer(p)

    def test_invalid_md5_length(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- md5: deadbeef\n  size: 1\n  path: f\n",
            )
            with self.assertRaisesRegex(fda.FetchError, "invalid md5"):
                fda._parse_dvc_pointer(p)

    def test_hash_field_optional_defaults_to_md5(self):
        # Real .dvc files always include `hash: md5`, but the field is redundant
        # and we accept its absence.
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- md5: " + "a" * 32 + "\n  size: 1\n  path: f\n",
            )
            outs = fda._parse_dvc_pointer(p)
            self.assertEqual(len(outs), 1)

    def test_unsupported_hash_algorithm(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- md5: "
                + "a" * 32
                + "\n  size: 1\n  hash: sha256\n  path: f\n",
            )
            with self.assertRaisesRegex(fda.FetchError, "unsupported hash"):
                fda._parse_dvc_pointer(p)

    def test_dir_suffix_recognized(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- md5: " + "a" * 32 + ".dir\n  size: 1\n  path: d\n",
            )
            outs = fda._parse_dvc_pointer(p)
            self.assertTrue(outs[0].is_dir)
            self.assertEqual(outs[0].bare_md5, "a" * 32)

    def test_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "# comment\n\n"
                "outs:\n"
                "# another comment\n"
                "- md5: " + "a" * 32 + "\n"
                "  size: 1\n"
                "  path: f\n",
            )
            outs = fda._parse_dvc_pointer(p)
            self.assertEqual(len(outs), 1)


class TestParseDvcConfig(unittest.TestCase):
    """Tests for the .dvc/config INI parser."""

    def _write(self, tmp: Path, content: str) -> Path:
        p = tmp / "config"
        p.write_text(content)
        return p

    def test_real_world_example(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "[core]\n"
                "    remote = storage\n"
                "    autostage = true\n"
                "['remote \"storage\"']\n"
                "    url = s3://therock-dvc/rocm-libraries\n"
                "    allow_anonymous_login = true\n",
            )
            r = fda._parse_dvc_config(p)
            self.assertEqual(r.bucket, "therock-dvc")
            self.assertEqual(r.prefix, "rocm-libraries")
            self.assertTrue(r.anonymous)

    def test_missing_core_section(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(Path(t), "['remote \"x\"']\n    url = s3://b/p\n")
            with self.assertRaisesRegex(fda.FetchError, "missing \\[core\\]"):
                fda._parse_dvc_config(p)

    def test_missing_remote_section(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(Path(t), "[core]\n    remote = ghost\n")
            with self.assertRaisesRegex(fda.FetchError, "missing.*ghost"):
                fda._parse_dvc_config(p)

    def test_missing_url(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "[core]\n    remote = storage\n['remote \"storage\"']\n",
            )
            with self.assertRaisesRegex(fda.FetchError, "missing 'url'"):
                fda._parse_dvc_config(p)

    def test_non_s3_remote_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "[core]\n    remote = storage\n"
                "['remote \"storage\"']\n    url = gs://bucket/prefix\n",
            )
            with self.assertRaisesRegex(fda.FetchError, "only s3:// remotes"):
                fda._parse_dvc_config(p)

    def test_anonymous_default_false(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "[core]\n    remote = storage\n"
                "['remote \"storage\"']\n    url = s3://b/p\n",
            )
            r = fda._parse_dvc_config(p)
            self.assertFalse(r.anonymous)


class TestS3Key(unittest.TestCase):
    """Tests for the S3 key builder (dvc 3.x layout: prefix/files/md5/...)."""

    def test_with_prefix(self):
        r = fda._Remote(bucket="b", prefix="rocm-libraries", anonymous=True)
        self.assertEqual(
            fda._s3_key(r, "35b9082b72e661dc5e37f14de1d9d4ed"),
            "rocm-libraries/files/md5/35/b9082b72e661dc5e37f14de1d9d4ed",
        )

    def test_without_prefix(self):
        r = fda._Remote(bucket="b", prefix="", anonymous=True)
        self.assertEqual(
            fda._s3_key(r, "35b9082b72e661dc5e37f14de1d9d4ed"),
            "files/md5/35/b9082b72e661dc5e37f14de1d9d4ed",
        )

    def test_dir_suffix_preserved(self):
        r = fda._Remote(bucket="b", prefix="p", anonymous=True)
        self.assertEqual(
            fda._s3_key(r, "a" * 32 + ".dir"),
            "p/files/md5/aa/" + "a" * 30 + ".dir",
        )

    def test_invalid_length_rejected(self):
        r = fda._Remote(bucket="b", prefix="p", anonymous=True)
        with self.assertRaises(fda.FetchError):
            fda._s3_key(r, "deadbeef")


class TestCachePath(unittest.TestCase):
    def test_layout(self):
        cache = Path("/cache")
        self.assertEqual(
            fda._cache_path(cache, "35b9082b72e661dc5e37f14de1d9d4ed"),
            Path("/cache/35/b9082b72e661dc5e37f14de1d9d4ed"),
        )

    def test_dir_suffix(self):
        cache = Path("/cache")
        self.assertEqual(
            fda._cache_path(cache, "a" * 32 + ".dir"),
            Path("/cache/aa/" + "a" * 30 + ".dir"),
        )


class TestMd5Of(unittest.TestCase):
    def test_streaming_hash_matches_one_shot(self):
        with tempfile.TemporaryDirectory() as t:
            data = b"x" * (3 << 20) + b"tail"  # > 1 chunk
            p = Path(t) / "f"
            p.write_bytes(data)
            self.assertEqual(fda._md5_of(p), _md5_hex(data))


class TestCacheMaterialize(unittest.TestCase):
    def test_hardlink_when_same_filesystem(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = tmp / "src.bin"
            src.write_bytes(b"abc")
            dest = tmp / "subdir" / "dest.bin"
            fda._materialize_from_cache(src, dest)
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), b"abc")
            # On the same filesystem we get a hardlink, so inode should match.
            self.assertEqual(src.stat().st_ino, dest.stat().st_ino)

    def test_replaces_existing_dest(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = tmp / "src.bin"
            src.write_bytes(b"new")
            dest = tmp / "dest.bin"
            dest.write_bytes(b"stale")
            fda._materialize_from_cache(src, dest)
            self.assertEqual(dest.read_bytes(), b"new")

    def test_falls_back_to_copy_when_link_fails(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = tmp / "src.bin"
            src.write_bytes(b"abc")
            dest = tmp / "dest.bin"
            with mock.patch("os.link", side_effect=OSError("xdev")):
                fda._materialize_from_cache(src, dest)
            self.assertEqual(dest.read_bytes(), b"abc")

    def test_store_in_cache_creates_entry(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = tmp / "src.bin"
            src.write_bytes(b"hello")
            cache_file = tmp / "cache" / "ab" / "cdef"
            fda._store_in_cache(src, cache_file)
            self.assertTrue(cache_file.exists())
            self.assertEqual(cache_file.read_bytes(), b"hello")

    def test_store_in_cache_skips_if_already_present(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            src = tmp / "src.bin"
            src.write_bytes(b"new")
            cache_file = tmp / "ab" / "cdef"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"existing")
            fda._store_in_cache(src, cache_file)
            # Existing entry preserved; cache is content-addressed so we trust it.
            self.assertEqual(cache_file.read_bytes(), b"existing")


# --------------------------------------------------------------------------
# Orchestration glue with mocked boto3.
# --------------------------------------------------------------------------


class _FakeS3:
    """Minimal stand-in for boto3.client('s3') that serves preconfigured bytes.

    We're not testing boto3 here - we're testing that our wrapper around its
    download_fileobj preserves atomic-write and MD5-verification invariants.
    """

    def __init__(self, payloads: dict[tuple[str, str], bytes]):
        self._payloads = payloads
        self.calls: list[tuple[str, str]] = []

    def download_fileobj(self, Bucket: str, Key: str, Fileobj) -> None:
        self.calls.append((Bucket, Key))
        try:
            Fileobj.write(self._payloads[(Bucket, Key)])
        except KeyError:
            raise RuntimeError(f"FakeS3: no payload for {Bucket}/{Key}")


class TestDownloadBlob(unittest.TestCase):
    """Tests for _download_blob's atomic-write + verify invariants."""

    def _remote(self) -> fda._Remote:
        return fda._Remote(bucket="bkt", prefix="p", anonymous=True)

    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            data = b"contents of the file"
            md5 = _md5_hex(data)
            remote = self._remote()
            key = fda._s3_key(remote, md5)
            s3 = _FakeS3({(remote.bucket, key): data})
            dest = tmp / "out.bin"
            fda._download_blob(s3, remote, md5, dest, expected_size=len(data))
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), data)

    def test_md5_mismatch_leaves_no_dest(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            real_data = b"corrupted bytes"
            wrong_md5 = "0" * 32  # not the actual md5 of real_data
            remote = self._remote()
            key = fda._s3_key(remote, wrong_md5)
            s3 = _FakeS3({(remote.bucket, key): real_data})
            dest = tmp / "out.bin"
            with self.assertRaisesRegex(fda.FetchError, "md5 mismatch"):
                fda._download_blob(
                    s3, remote, wrong_md5, dest, expected_size=len(real_data)
                )
            # Atomic write invariant: no destination file, no .tmp file.
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_suffix(dest.suffix + ".tmp").exists())

    def test_size_mismatch_leaves_no_dest(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            data = b"shorter than expected"
            md5 = _md5_hex(data)
            remote = self._remote()
            key = fda._s3_key(remote, md5)
            s3 = _FakeS3({(remote.bucket, key): data})
            dest = tmp / "out.bin"
            with self.assertRaisesRegex(fda.FetchError, "size mismatch"):
                fda._download_blob(s3, remote, md5, dest, expected_size=len(data) + 100)
            self.assertFalse(dest.exists())


class TestMaterializeFile(unittest.TestCase):
    """Tests for the cache fast paths and download fallback."""

    def _remote(self) -> fda._Remote:
        return fda._Remote(bucket="bkt", prefix="p", anonymous=True)

    def test_skips_when_destination_already_correct(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            data = b"already here"
            md5 = _md5_hex(data)
            dest = tmp / "out.bin"
            dest.write_bytes(data)
            s3 = _FakeS3({})  # would raise if called
            cache_dir = tmp / "cache"
            result = fda._materialize_file(
                s3,
                self._remote(),
                md5=md5,
                size=len(data),
                dest=dest,
                cache_dir=cache_dir,
                log=lambda _: None,
            )
            self.assertEqual(result.skipped, 1)
            self.assertEqual(result.fetched, 0)
            self.assertEqual(s3.calls, [])

    def test_uses_cache_when_destination_missing(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            data = b"from cache"
            md5 = _md5_hex(data)
            dest = tmp / "out.bin"
            cache_dir = tmp / "cache"
            cache_file = fda._cache_path(cache_dir, md5)
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(data)
            s3 = _FakeS3({})
            result = fda._materialize_file(
                s3,
                self._remote(),
                md5=md5,
                size=len(data),
                dest=dest,
                cache_dir=cache_dir,
                log=lambda _: None,
            )
            self.assertEqual(result.cached, 1)
            self.assertEqual(s3.calls, [])
            self.assertEqual(dest.read_bytes(), data)

    def test_poisoned_cache_is_discarded_and_redownloaded(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            data = b"good bytes"
            md5 = _md5_hex(data)
            dest = tmp / "out.bin"
            cache_dir = tmp / "cache"
            cache_file = fda._cache_path(cache_dir, md5)
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"X" * len(data))  # same size, wrong content
            remote = self._remote()
            key = fda._s3_key(remote, md5)
            s3 = _FakeS3({(remote.bucket, key): data})
            result = fda._materialize_file(
                s3,
                remote,
                md5=md5,
                size=len(data),
                dest=dest,
                cache_dir=cache_dir,
                log=lambda _: None,
            )
            self.assertEqual(result.fetched, 1)
            self.assertEqual(s3.calls, [(remote.bucket, key)])
            self.assertEqual(dest.read_bytes(), data)
            # Cache repopulated with good bytes.
            self.assertEqual(cache_file.read_bytes(), data)

    def test_downloads_and_populates_cache(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            data = b"fresh download"
            md5 = _md5_hex(data)
            dest = tmp / "out.bin"
            cache_dir = tmp / "cache"
            remote = self._remote()
            key = fda._s3_key(remote, md5)
            s3 = _FakeS3({(remote.bucket, key): data})
            result = fda._materialize_file(
                s3,
                remote,
                md5=md5,
                size=len(data),
                dest=dest,
                cache_dir=cache_dir,
                log=lambda _: None,
            )
            self.assertEqual(result.fetched, 1)
            self.assertEqual(dest.read_bytes(), data)
            cache_file = fda._cache_path(cache_dir, md5)
            self.assertTrue(cache_file.exists())
            self.assertEqual(cache_file.read_bytes(), data)


class TestPullEndToEnd(unittest.TestCase):
    """End-to-end pull() with mocked boto3."""

    def test_pull_two_files(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            project = tmp / "proj"
            (project / ".dvc").mkdir(parents=True)
            (project / ".dvc" / "config").write_text(
                "[core]\n    remote = storage\n"
                "['remote \"storage\"']\n    url = s3://b/pfx\n"
                "    allow_anonymous_login = true\n",
            )
            data_a = b"file A contents"
            data_b = b"file B contents"
            md5_a = _md5_hex(data_a)
            md5_b = _md5_hex(data_b)
            (project / "a.bin.dvc").write_text(
                f"outs:\n- md5: {md5_a}\n  size: {len(data_a)}\n"
                f"  hash: md5\n  path: a.bin\n"
            )
            (project / "sub").mkdir()
            (project / "sub" / "b.bin.dvc").write_text(
                f"outs:\n- md5: {md5_b}\n  size: {len(data_b)}\n"
                f"  hash: md5\n  path: b.bin\n"
            )

            payloads = {
                ("b", f"pfx/files/md5/{md5_a[:2]}/{md5_a[2:]}"): data_a,
                ("b", f"pfx/files/md5/{md5_b[:2]}/{md5_b[2:]}"): data_b,
            }
            fake_s3 = _FakeS3(payloads)

            with mock.patch.object(fda, "_make_s3_client", return_value=fake_s3):
                result = fda.pull(
                    project,
                    cache_dir=tmp / "cache",
                    log=lambda _: None,
                )

            self.assertEqual(result.fetched, 2)
            self.assertEqual((project / "a.bin").read_bytes(), data_a)
            self.assertEqual((project / "sub" / "b.bin").read_bytes(), data_b)
            self.assertEqual(len(fake_s3.calls), 2)

    def test_pull_raises_aggregated_error(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            project = tmp / "proj"
            (project / ".dvc").mkdir(parents=True)
            (project / ".dvc" / "config").write_text(
                "[core]\n    remote = storage\n"
                "['remote \"storage\"']\n    url = s3://b/pfx\n"
                "    allow_anonymous_login = true\n",
            )
            md5 = "0" * 32  # will mismatch any real content
            (project / "x.bin.dvc").write_text(
                f"outs:\n- md5: {md5}\n  size: 5\n  hash: md5\n  path: x.bin\n"
            )
            fake_s3 = _FakeS3({("b", f"pfx/files/md5/{md5[:2]}/{md5[2:]}"): b"hello"})

            with mock.patch.object(fda, "_make_s3_client", return_value=fake_s3):
                with self.assertRaisesRegex(fda.FetchError, "1 file"):
                    fda.pull(project, cache_dir=None, log=lambda _: None)
            self.assertFalse((project / "x.bin").exists())

    def test_pull_no_dvc_config_raises(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(fda.FetchError, "no .dvc/config"):
                fda.pull(Path(t), log=lambda _: None)


if __name__ == "__main__":
    unittest.main()
