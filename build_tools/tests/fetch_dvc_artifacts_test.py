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
import concurrent.futures
import errno
import json
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

# Add build_tools to path so fetch_dvc_artifacts is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import fetch_dvc_artifacts as fda


def _md5_hex(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


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
        r = fda._Remote(
            name="storage", bucket="b", prefix="rocm-libraries", anonymous=True
        )
        self.assertEqual(
            fda._s3_key(r, "35b9082b72e661dc5e37f14de1d9d4ed"),
            "rocm-libraries/files/md5/35/b9082b72e661dc5e37f14de1d9d4ed",
        )

    def test_without_prefix(self):
        r = fda._Remote(name="storage", bucket="b", prefix="", anonymous=True)
        self.assertEqual(
            fda._s3_key(r, "35b9082b72e661dc5e37f14de1d9d4ed"),
            "files/md5/35/b9082b72e661dc5e37f14de1d9d4ed",
        )

    def test_dir_suffix_preserved(self):
        r = fda._Remote(name="storage", bucket="b", prefix="p", anonymous=True)
        self.assertEqual(
            fda._s3_key(r, "a" * 32 + ".dir"),
            "p/files/md5/aa/" + "a" * 30 + ".dir",
        )

    def test_invalid_length_rejected(self):
        r = fda._Remote(name="storage", bucket="b", prefix="p", anonymous=True)
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
        return fda._Remote(name="storage", bucket="bkt", prefix="p", anonymous=True)

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


class TestStoreInCacheConcurrency(unittest.TestCase):
    """Regression test for #5943.

    Multiple .dvc pointer files can reference files with the same content hash
    (e.g. identical small tensors in different subdirectories). When fetched
    concurrently, all workers call _store_in_cache with the same cache_file
    path. Windows publication must leave a competing winner in place and
    verify its contents before accepting a collision as success.
    """

    def test_concurrent_calls_with_same_cache_file_do_not_raise(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            data = b"shared content"
            md5 = _md5_hex(data)
            cache_file = fda._cache_path(tmp / "cache", md5)

            # Both workers arrive at publication concurrently. Caller 1
            # writes cache_file; caller 2 then gets PermissionError, as on
            # Windows when publication collides with a competing publisher.
            # We model this with a two-phase barrier:
            # both workers reach the gate, then caller 1 does the real replace
            # and only after that does caller 2 raise PermissionError, so
            # cache_file already exists when the exception handler checks it.
            original_link = os.link
            call_count = [0]
            lock = threading.Lock()
            gate = threading.Barrier(2)
            done = threading.Event()

            def patched_publish(src, dst):
                with lock:
                    call_count[0] += 1
                    n = call_count[0]
                gate.wait(timeout=5)
                if n == 1:
                    original_link(src, dst)
                    done.set()
                else:
                    done.wait(timeout=5)
                    raise PermissionError("[WinError 5] Access is denied")

            src1 = tmp / "src1.bin"
            src2 = tmp / "src2.bin"
            src1.write_bytes(data)
            src2.write_bytes(data)

            errors = []

            def worker(src):
                try:
                    fda._store_in_cache(src, cache_file)
                except Exception as e:
                    errors.append(e)

            # Exercise Windows publication and its PermissionError handling.
            with (
                mock.patch.object(fda, "sys", platform="win32"),
                mock.patch.object(fda.os, "link", side_effect=patched_publish),
            ):
                threads = [
                    threading.Thread(target=worker, args=(src1,)),
                    threading.Thread(target=worker, args=(src2,)),
                ]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()

            self.assertEqual(errors, [], f"unexpected errors: {errors}")
            self.assertTrue(cache_file.exists())
            self.assertEqual(cache_file.read_bytes(), data)
            leftover = list(cache_file.parent.glob("*.tmp"))
            self.assertEqual(leftover, [], f"leftover temp files: {leftover}")

    def test_windows_publication_preserves_winning_cache_entry(self) -> None:
        data = b"shared content"
        for winner_data in (data, b"broken content", b"truncated"):
            with self.subTest(
                winner_data=winner_data
            ), tempfile.TemporaryDirectory() as t:
                tmp = Path(t)
                src = tmp / "source"
                src.write_bytes(data)
                winner = tmp / "winner"
                winner.write_bytes(winner_data)
                cache_file = fda._cache_path(tmp / "cache", _md5_hex(data))
                original_link = os.link
                original_rename = os.rename

                def link_and_publish_winner(source: Path, target: Path) -> None:
                    # Another worker publishes after our cache-miss check,
                    # immediately before our attempted publication.
                    if not cache_file.exists():
                        original_link(winner, cache_file)
                    original_link(source, target)

                def windows_rename(source: Path, target: Path) -> None:
                    if sys.platform == "win32":
                        original_rename(source, target)
                    else:
                        # Model Windows' no-overwrite rename with real files.
                        original_link(source, target)
                        source.unlink()

                with (
                    mock.patch.object(fda, "sys", platform="win32"),
                    mock.patch.object(
                        fda.os, "link", side_effect=link_and_publish_winner
                    ),
                    mock.patch.object(fda.os, "rename", side_effect=windows_rename),
                ):
                    if winner_data == data:
                        fda._store_in_cache(src, cache_file)
                    else:
                        with self.assertRaises(FileExistsError):
                            fda._store_in_cache(src, cache_file)

                # Replacing the winner creates a Windows delete/rename window
                # during which readers can get PermissionError opening it.
                self.assertTrue(cache_file.samefile(winner))
                self.assertEqual(cache_file.read_bytes(), winner_data)
                self.assertEqual(list(cache_file.parent.glob("*.tmp")), [])


class TestCacheSynchronization(unittest.TestCase):
    def test_duplicate_outputs_and_manifests_fetch_once(self):
        data = b"tiny tensor"
        md5 = _md5_hex(data)
        manifest = json.dumps([{"md5": md5, "relpath": "child.bin"}]).encode()
        manifest_md5 = _md5_hex(manifest) + ".dir"
        for directory in (False, True):
            with self.subTest(directory=directory), tempfile.TemporaryDirectory() as t:
                tmp = Path(t)
                project = tmp / "project"
                (project / ".dvc").mkdir(parents=True)
                (project / ".dvc/config").write_text(
                    "[core]\n remote = storage\n['remote \"storage\"']\n"
                    " url = s3://b/pfx\n allow_anonymous_login = true\n"
                )
                out_md5 = manifest_md5 if directory else md5
                size = len(data)
                for i in range(12):
                    (project / f"{i}.dvc").write_text(
                        f"outs:\n- md5: {out_md5}\n  size: {size}\n  path: out-{i}\n"
                    )
                payloads = {("b", f"pfx/files/md5/{md5[:2]}/{md5[2:]}"): data}
                if directory:
                    payloads[
                        ("b", f"pfx/files/md5/{manifest_md5[:2]}/{manifest_md5[2:]}")
                    ] = manifest
                fake = _FakeS3(payloads)
                messages = []
                with mock.patch.object(fda, "_make_s3_client", return_value=fake):
                    try:
                        result = fda.pull(
                            project,
                            jobs=12,
                            cache_dir=tmp / "cache",
                            log=messages.append,
                        )
                    except fda.FetchError:
                        self.fail("\n".join(messages))
                self.assertEqual(result.fetched, 1)
                self.assertEqual(result.cached, 11)
                self.assertEqual(len(fake.calls), 2 if directory else 1)
                for i in range(12):
                    dest = project / f"out-{i}"
                    if directory:
                        dest /= "child.bin"
                    self.assertEqual(dest.read_bytes(), data)
                self.assertEqual(list(tmp.rglob("*.tmp")), [])

    def test_independent_hashes_remain_parallel(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            remote = fda._Remote("storage", "b", "pfx", True)
            payloads = [b"alpha", b"beta"]
            barrier = threading.Barrier(2)

            class ConcurrentS3:
                def download_fileobj(self, bucket, key, file):
                    barrier.wait(timeout=5)
                    file.write(
                        next(
                            data
                            for data in payloads
                            if fda._s3_key(remote, _md5_hex(data)) == key
                        )
                    )

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                futures = [
                    ex.submit(
                        fda._materialize_file,
                        ConcurrentS3(),
                        remote,
                        _md5_hex(data),
                        len(data),
                        tmp / str(i),
                        tmp / "cache",
                        lambda _: None,
                    )
                    for i, data in enumerate(payloads)
                ]
                for future in futures:
                    self.assertEqual(future.result(timeout=6).fetched, 1)

    def test_cache_locks_share_canonical_path_and_are_released(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            lock = fda._cache_lock(tmp / "cache", "a" * 32)
            self.assertIs(lock, fda._cache_lock(tmp / "sub/../cache", "a" * 32))
            self.assertIsNot(lock, fda._cache_lock(tmp / "cache", "b" * 32))
            with lock:
                self.assertFalse(lock.acquire(blocking=False))
            self.assertTrue(lock.acquire(blocking=False))
            lock.release()
            del lock
            self.assertFalse(fda._cache_locks)

    def test_cache_lock_is_stable_across_entry_creation_and_eviction(self):
        with tempfile.TemporaryDirectory() as t:
            cache_dir = Path(t) / "cache"
            md5 = _md5_hex(b"tiny")
            cache_file = fda._cache_path(cache_dir, md5)
            real_resolve = Path.resolve

            def resolve_directory_only(path, *args, **kwargs):
                if path == cache_file:
                    raise AssertionError(
                        "cache entry resolution is unstable on Windows"
                    )
                return real_resolve(path, *args, **kwargs)

            with mock.patch.object(Path, "resolve", resolve_directory_only):
                lock = fda._cache_lock(cache_dir, md5)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(b"corrupt")
                with lock:
                    self.assertIs(lock, fda._cache_lock(cache_dir, md5))
                    cache_file.unlink()
                    self.assertIs(lock, fda._cache_lock(cache_dir, md5))
                    source = Path(t) / "source"
                    source.write_bytes(b"tiny")
                    os.link(source, cache_file)
                    self.assertIs(lock, fda._cache_lock(cache_dir, md5))
                self.assertEqual(cache_file.read_bytes(), b"tiny")

    def test_downloads_do_not_share_staging_file(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            data = b"small payload"
            remote = fda._Remote("storage", "b", "pfx", True)
            first_open = threading.Event()
            second_done = threading.Event()

            class OrderedS3:
                def download_fileobj(self, bucket, key, file):
                    file.write(data)
                    file.flush()
                    if threading.current_thread().name == "first":
                        first_open.set()
                        if not second_done.wait(5):
                            raise RuntimeError("second download timed out")

            errors = []

            def download(name):
                try:
                    fda._download_blob(
                        OrderedS3(), remote, _md5_hex(data), tmp / "same", len(data)
                    )
                except Exception as err:
                    errors.append(err)
                finally:
                    if name == "second":
                        second_done.set()

            first = threading.Thread(target=download, args=("first",), name="first")
            first.start()
            self.assertTrue(first_open.wait(5))
            second = threading.Thread(target=download, args=("second",), name="second")
            second.start()
            first.join(6)
            second.join(6)
            self.assertEqual(errors, [])
            self.assertEqual((tmp / "same").read_bytes(), data)
            self.assertEqual(list(tmp.glob("*.tmp")), [])


class TestDirectCachePublication(unittest.TestCase):
    def test_publication_does_not_rename_cache_path(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            source, cache = tmp / "source", tmp / "cache/key"
            source.write_bytes(b"tiny")
            with mock.patch.object(
                fda.os, "rename", side_effect=AssertionError("rename window")
            ), mock.patch.object(
                fda.os, "replace", side_effect=AssertionError("replace window")
            ):
                fda._store_in_cache(source, cache)
            self.assertTrue(source.samefile(cache))

    def test_concurrent_publication_keeps_cache_readable(self):
        # #8505: real Windows rename/replace publication can expose a cache
        # pathname that ordinary Python readers temporarily cannot open.
        # Force simultaneous misses, but do not inject filesystem errors.
        # The small payload and bounded rounds keep this in routine CI.
        workers = 12
        data = b"same verified tensor contents" * 100
        real_exists = Path.exists
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            sources = [root / f"source-{i}" for i in range(workers)]
            for source in sources:
                source.write_bytes(data)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                for iteration in range(100):
                    with self.subTest(iteration=iteration):
                        cache_file = root / f"cache-{iteration}" / "entry"
                        gate = threading.Barrier(workers)
                        state = threading.local()
                        done = threading.Event()
                        observed = threading.Event()
                        reader_errors = []

                        def synchronized_miss(path):
                            if path == cache_file and not getattr(
                                state, "checked", False
                            ):
                                present = real_exists(path)
                                state.checked = True
                                gate.wait(timeout=5)
                                return present
                            return real_exists(path)

                        def read_cache():
                            try:
                                while not done.is_set():
                                    if real_exists(cache_file):
                                        contents = cache_file.read_bytes()
                                        if contents != data:
                                            raise AssertionError(
                                                "incorrect published contents"
                                            )
                                        observed.set()
                                    else:
                                        time.sleep(0)
                            except Exception as error:
                                reader_errors.append(error)
                                observed.set()

                        reader = threading.Thread(target=read_cache, daemon=True)
                        reader.start()
                        try:
                            with mock.patch.object(Path, "exists", synchronized_miss):
                                futures = [
                                    executor.submit(
                                        fda._store_in_cache, source, cache_file
                                    )
                                    for source in sources
                                ]
                                # Wait for all publishers before reporting an error, so
                                # no worker outlives the patched miss gate or temp files.
                                concurrent.futures.wait(futures)
                                for future in futures:
                                    future.result()
                            self.assertTrue(
                                observed.wait(5), "reader did not open the cache"
                            )
                        finally:
                            done.set()
                            reader.join(timeout=5)
                        self.assertFalse(reader.is_alive(), "cache reader did not stop")
                        self.assertEqual(
                            reader_errors, [], "publication denied a cache reader"
                        )
                        self.assertEqual(cache_file.read_bytes(), data)
                        self.assertEqual(list(cache_file.parent.glob("*.tmp")), [])

    def test_cross_volume_source_uses_cache_local_link(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            source, cache = tmp / "source", tmp / "cache/key"
            source.write_bytes(b"tiny")
            original_link = os.link

            def link(src, dest):
                if src == source:
                    raise OSError(errno.EXDEV, "different volume")
                return original_link(src, dest)

            with mock.patch.object(fda.os, "link", side_effect=link), mock.patch.object(
                fda.os, "rename", side_effect=AssertionError("rename window")
            ), mock.patch.object(
                fda.os, "replace", side_effect=AssertionError("replace window")
            ):
                fda._store_in_cache(source, cache)
            self.assertEqual(cache.read_bytes(), b"tiny")
            self.assertFalse(source.samefile(cache))
            self.assertEqual(list(cache.parent.glob("*.tmp")), [])

    def test_no_hardlink_filesystem_retains_copy_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            source, cache = tmp / "source", tmp / "cache/key"
            source.write_bytes(b"tiny")
            with mock.patch.object(
                fda.os, "link", side_effect=OSError(errno.ENOTSUP, "no hardlinks")
            ):
                fda._store_in_cache(source, cache)
            self.assertEqual(cache.read_bytes(), b"tiny")
            self.assertEqual(list(cache.parent.glob("*.tmp")), [])


class TestMaterializeFile(unittest.TestCase):
    """Tests for the cache fast paths and download fallback."""

    def _remote(self) -> fda._Remote:
        return fda._Remote(name="storage", bucket="bkt", prefix="p", anonymous=True)

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

    def test_pull_preserves_cache_failure_tracebacks(self):
        # CI's single cache-path error previously lost the operation that
        # failed. Exercise a real cache hit through the worker and aggregate
        # two failures, including the cause of an exception from a helper.
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            project = tmp / "proj"
            (project / ".dvc").mkdir(parents=True)
            (project / ".dvc" / "config").write_text(
                "[core]\n    remote = storage\n"
                "['remote \"storage\"']\n    url = s3://b/pfx\n"
                "    allow_anonymous_login = true\n"
            )
            data = b"cached content"
            md5 = _md5_hex(data)
            for name in ("a", "b"):
                (project / f"{name}.dvc").write_text(
                    f"outs:\n- md5: {md5}\n  size: {len(data)}\n"
                    f"  hash: md5\n  path: {name}.bin\n"
                )
            cache_dir = tmp / "cache"
            cache_file = fda._cache_path(cache_dir, md5)
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(data)
            real_open = Path.open

            def denied_cache_open(path, *args, **kwargs):
                if path == cache_file:
                    try:
                        raise OSError("underlying filesystem failure")
                    except OSError as cause:
                        raise PermissionError(
                            13, "Permission denied", str(path)
                        ) from cause
                return real_open(path, *args, **kwargs)

            messages = []
            with (
                mock.patch.object(fda, "_make_s3_client", return_value=_FakeS3({})),
                mock.patch.object(Path, "open", denied_cache_open),
            ):
                with self.assertRaisesRegex(fda.FetchError, "2 file\\(s\\)"):
                    fda.pull(project, jobs=2, cache_dir=cache_dir, log=messages.append)

            for name in ("a", "b"):
                self.assertTrue(
                    any(
                        f"ERROR     {project.resolve() / f'{name}.dvc'}:" in m
                        for m in messages
                    )
                )
            stacks = [m for m in messages if "Traceback (most recent call last)" in m]
            self.assertEqual(len(stacks), 2)
            for stack in stacks:
                self.assertIn("in _process_pointer", stack)
                self.assertIn("in _materialize_file", stack)
                self.assertIn("in _md5_of", stack)
                self.assertIn("in denied_cache_open", stack)
                self.assertIn("underlying filesystem failure", stack)
                self.assertIn("direct cause", stack)
                self.assertIn("PermissionError: [Errno 13] Permission denied", stack)
                self.assertIn(repr(str(cache_file)), stack)


class TestParsePerOutputRemote(unittest.TestCase):
    """Pointer entries may carry a per-output `remote:` key (dvc routing)."""

    def _write(self, tmp: Path, content: str) -> Path:
        p = tmp / "x.dvc"
        p.write_text(content)
        return p

    def test_remote_key_parsed(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- md5: " + "a" * 32 + "\n  size: 1\n"
                "  hash: md5\n  remote: golden-data\n  path: f.bin\n",
            )
            outs = fda._parse_dvc_pointer(p)
            self.assertEqual(outs[0].remote, "golden-data")

    def test_remote_key_absent_is_none(self):
        with tempfile.TemporaryDirectory() as t:
            p = self._write(
                Path(t),
                "outs:\n- md5: " + "a" * 32 + "\n  size: 1\n  hash: md5\n  path: f\n",
            )
            outs = fda._parse_dvc_pointer(p)
            self.assertIsNone(outs[0].remote)


class TestParseDvcRemotes(unittest.TestCase):
    """Tests for multi-remote parsing and per-output resolution."""

    _CONFIG = (
        "[core]\n    remote = storage\n"
        "['remote \"storage\"']\n    url = s3://therock-dvc/rocm-libraries\n"
        "    allow_anonymous_login = true\n"
        "['remote \"golden-data\"']\n"
        "    url = s3://therock-dvc/rocm-libraries/hipdnn/golden-data\n"
        "    allow_anonymous_login = false\n"
    )

    def _write(self, tmp: Path, content: str) -> Path:
        p = tmp / "config"
        p.write_text(content)
        return p

    def test_parses_all_remotes(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = fda._parse_dvc_remotes(self._write(Path(t), self._CONFIG))
            self.assertEqual(cfg.default, "storage")
            self.assertEqual(set(cfg.remotes), {"storage", "golden-data"})
            self.assertEqual(
                cfg.remotes["golden-data"].prefix, "rocm-libraries/hipdnn/golden-data"
            )
            # Per-remote anonymous flags are independent.
            self.assertTrue(cfg.remotes["storage"].anonymous)
            self.assertFalse(cfg.remotes["golden-data"].anonymous)

    def test_resolve_default_when_none(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = fda._parse_dvc_remotes(self._write(Path(t), self._CONFIG))
            self.assertEqual(cfg.resolve(None, Path("x.dvc")).name, "storage")

    def test_resolve_named_remote(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = fda._parse_dvc_remotes(self._write(Path(t), self._CONFIG))
            self.assertEqual(
                cfg.resolve("golden-data", Path("x.dvc")).name, "golden-data"
            )

    def test_resolve_unknown_remote_raises(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = fda._parse_dvc_remotes(self._write(Path(t), self._CONFIG))
            with self.assertRaisesRegex(fda.FetchError, "not defined"):
                cfg.resolve("ghost", Path("x.dvc"))

    def test_default_remote_backward_compat(self):
        # _parse_dvc_config still returns the [core] default remote unchanged.
        with tempfile.TemporaryDirectory() as t:
            r = fda._parse_dvc_config(self._write(Path(t), self._CONFIG))
            self.assertEqual(r.name, "storage")
            self.assertEqual(r.prefix, "rocm-libraries")


class TestPullPerOutputRemoteRouting(unittest.TestCase):
    """End-to-end: a pull routes each blob to the remote named in its pointer."""

    def test_blobs_fetched_from_their_named_remotes(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            project = tmp / "proj"
            (project / ".dvc").mkdir(parents=True)
            (project / ".dvc" / "config").write_text(
                "[core]\n    remote = storage\n"
                "['remote \"storage\"']\n    url = s3://bkt/rocm-libraries\n"
                "    allow_anonymous_login = true\n"
                "['remote \"golden-data\"']\n"
                "    url = s3://bkt/rocm-libraries/hipdnn/golden-data\n"
                "    allow_anonymous_login = true\n"
            )
            data_def = b"default-remote blob"
            data_gld = b"golden-data blob"
            md5_def = _md5_hex(data_def)
            md5_gld = _md5_hex(data_gld)
            # Pointer 1: no remote key -> default (storage).
            (project / "a.bin.dvc").write_text(
                f"outs:\n- md5: {md5_def}\n  size: {len(data_def)}\n"
                f"  hash: md5\n  path: a.bin\n"
            )
            # Pointer 2: remote: golden-data.
            (project / "b.bin.dvc").write_text(
                f"outs:\n- md5: {md5_gld}\n  size: {len(data_gld)}\n"
                f"  hash: md5\n  remote: golden-data\n  path: b.bin\n"
            )

            # A single fake S3 keyed by (bucket, key). Because the two remotes
            # share a bucket but differ in prefix, the keys differ - so a blob
            # served only under the golden-data prefix proves it was routed
            # there and NOT looked up under the storage prefix (the #9183 bug).
            payloads = {
                (
                    "bkt",
                    f"rocm-libraries/files/md5/{md5_def[:2]}/{md5_def[2:]}",
                ): data_def,
                (
                    "bkt",
                    "rocm-libraries/hipdnn/golden-data/files/md5/"
                    f"{md5_gld[:2]}/{md5_gld[2:]}",
                ): data_gld,
            }
            fake = _FakeS3(payloads)

            with mock.patch.object(fda, "_make_s3_client", return_value=fake):
                result = fda.pull(project, cache_dir=tmp / "cache", log=lambda _: None)

            self.assertEqual(result.fetched, 2)
            self.assertEqual((project / "a.bin").read_bytes(), data_def)
            self.assertEqual((project / "b.bin").read_bytes(), data_gld)
            fetched_keys = {key for _, key in fake.calls}
            self.assertIn(
                "rocm-libraries/hipdnn/golden-data/files/md5/"
                f"{md5_gld[:2]}/{md5_gld[2:]}",
                fetched_keys,
            )

    def test_unknown_remote_in_pointer_is_reported(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            project = tmp / "proj"
            (project / ".dvc").mkdir(parents=True)
            (project / ".dvc" / "config").write_text(
                "[core]\n    remote = storage\n"
                "['remote \"storage\"']\n    url = s3://b/pfx\n"
                "    allow_anonymous_login = true\n"
            )
            md5 = _md5_hex(b"z")
            (project / "z.bin.dvc").write_text(
                f"outs:\n- md5: {md5}\n  size: 1\n  hash: md5\n"
                f"  remote: nonexistent\n  path: z.bin\n"
            )
            fake = _FakeS3({})
            logs: list[str] = []
            with mock.patch.object(fda, "_make_s3_client", return_value=fake):
                # An unknown per-output remote surfaces via the normal aggregated
                # per-pointer failure path; the specific cause is logged.
                with self.assertRaisesRegex(fda.FetchError, "1 file"):
                    fda.pull(project, cache_dir=None, log=logs.append)
            self.assertTrue(
                any("not defined" in line for line in logs),
                f"cause not logged; logs={logs}",
            )
            self.assertFalse((project / "z.bin").exists())


if __name__ == "__main__":
    unittest.main()
