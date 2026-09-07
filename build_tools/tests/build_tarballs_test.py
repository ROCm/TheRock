#!/usr/bin/env python
"""Unit tests for build_tarballs.py."""

import json
import os
import sys
import tarfile
import tempfile
import unittest
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from types import TracebackType
from typing import NamedTuple
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from build_tarballs import (
    _tar_is_gnu,
    compress_tarball,
    determine_compress_workers,
    is_kpack_split,
    main,
    reproducible_tar_flags,
)


class MainMocks(NamedTuple):
    fetch: mock.Mock
    compress: mock.Mock
    kpack: mock.Mock


class InlineProcessPoolExecutor:
    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> "InlineProcessPoolExecutor":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    def submit(
        self,
        fn: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> Future[object]:
        future: Future[object] = Future()
        future.set_result(fn(*args, **kwargs))
        return future


class TestIsKpackSplit(unittest.TestCase):
    def _write_manifest(self, tmpdir: Path, flags: dict):
        manifest_dir = tmpdir / "share" / "therock"
        manifest_dir.mkdir(parents=True)
        manifest = {"flags": flags}
        (manifest_dir / "therock_manifest.json").write_text(json.dumps(manifest))

    def test_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._write_manifest(tmpdir, {"KPACK_SPLIT_ARTIFACTS": True})
            self.assertTrue(is_kpack_split(tmpdir))

    def test_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._write_manifest(tmpdir, {"KPACK_SPLIT_ARTIFACTS": False})
            self.assertFalse(is_kpack_split(tmpdir))

    def test_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(is_kpack_split(Path(tmpdir)))


class TestCompressTarball(unittest.TestCase):
    def test_creates_tarball(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            src = tmpdir / "src"
            src.mkdir()
            (src / "bin").mkdir()
            (src / "bin" / "hello").write_text("hello world")
            (src / "lib").mkdir()
            (src / "lib" / "libfoo.so").write_bytes(b"\x00" * 1024)

            tarball_path = tmpdir / "output" / "test.tar.gz"
            compress_tarball(source_dir=src, tarball_path=tarball_path)

            self.assertTrue(tarball_path.exists())
            self.assertGreater(tarball_path.stat().st_size, 0)

            with tarfile.open(tarball_path, "r:gz") as tf:
                names = tf.getnames()
                self.assertIn("./bin/hello", names)
                self.assertIn("./lib/libfoo.so", names)
                hello_file = tf.extractfile("./bin/hello")
                self.assertIsNotNone(hello_file)
                self.assertEqual(hello_file.read(), b"hello world")

    def test_creates_tarball_with_system_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            src = tmpdir / "src"
            src.mkdir()
            (src / "hello").write_text("hello world")

            tarball_path = tmpdir / "test.tar.gz"
            compress_tarball(
                source_dir=src,
                tarball_path=tarball_path,
                compression_backend="system-gzip",
            )

            with tarfile.open(tarball_path, "r:gz") as tf:
                self.assertIn("./hello", tf.getnames())


class TestReproducibleTarballs(unittest.TestCase):
    """Release tarballs must depend only on their content.

    `tar` has no filter hook, so unlike the Python archive writers this relies
    on command-line flags, and those are GNU extensions.
    """

    FIXED_EPOCH = 1700000000

    def _write_tree(self, root: Path) -> None:
        # Created out of sorted order so a passing order assertion cannot be
        # explained by creation order.
        (root / "sub").mkdir(parents=True)
        (root / "z.txt").write_text("z")
        (root / "a.txt").write_text("a")
        (root / "sub" / "m.txt").write_text("m")

    def test_flags_are_omitted_without_a_timestamp(self):
        self.assertEqual(reproducible_tar_flags(None), [])

    @unittest.skipUnless(_tar_is_gnu(), "Reproducibility flags are GNU tar extensions")
    def test_flags_pin_order_metadata_and_time(self):
        flags = reproducible_tar_flags(self.FIXED_EPOCH)
        self.assertIn("--sort=name", flags)
        self.assertIn(f"--mtime=@{self.FIXED_EPOCH}", flags)
        self.assertIn("--owner=0", flags)
        self.assertIn("--group=0", flags)

    @unittest.skipUnless(_tar_is_gnu(), "Reproducibility flags are GNU tar extensions")
    def test_same_content_different_mtimes_gives_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            src = tmpdir / "src"
            self._write_tree(src)

            first = tmpdir / "first.tar.gz"
            compress_tarball(
                source_dir=src,
                tarball_path=first,
                source_date_epoch=self.FIXED_EPOCH,
            )

            # Simulate the same content produced by a later build.
            for path in src.rglob("*"):
                os.utime(path, (1600000000, 1600000000))

            second = tmpdir / "second.tar.gz"
            compress_tarball(
                source_dir=src,
                tarball_path=second,
                source_date_epoch=self.FIXED_EPOCH,
            )

            self.assertEqual(first.read_bytes(), second.read_bytes())

    @unittest.skipUnless(_tar_is_gnu(), "Reproducibility flags are GNU tar extensions")
    def test_members_carry_the_pinned_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            src = tmpdir / "src"
            self._write_tree(src)
            tarball = tmpdir / "out.tar.gz"
            compress_tarball(
                source_dir=src,
                tarball_path=tarball,
                source_date_epoch=self.FIXED_EPOCH,
            )

            with tarfile.open(tarball, "r:gz") as tf:
                members = tf.getmembers()
            self.assertTrue(members)
            for member in members:
                self.assertEqual(member.mtime, self.FIXED_EPOCH, member.name)
                self.assertEqual(member.uid, 0, member.name)
                self.assertEqual(member.gid, 0, member.name)
            names = [m.name for m in members]
            self.assertEqual(names, sorted(names))


class TestDetermineCompressWorkers(unittest.TestCase):
    @mock.patch("build_tarballs.available_cpu_count", return_value=32)
    def test_zlib_ng_reserves_cpus_per_archive(self, _: mock.Mock) -> None:
        self.assertEqual(
            determine_compress_workers(
                task_count=10,
                requested_workers=None,
                compression_backend="zlib-ng",
                compression_threads=8,
            ),
            3,
        )

    @mock.patch("build_tarballs.available_cpu_count", return_value=32)
    def test_requested_workers_are_capped_by_task_count(self, _: mock.Mock) -> None:
        self.assertEqual(
            determine_compress_workers(
                task_count=4,
                requested_workers=8,
                compression_backend="zlib-ng",
                compression_threads=8,
            ),
            4,
        )


class TestMain(unittest.TestCase):
    def _run_main_with_mocks(
        self,
        argv: list[str],
        *,
        kpack_split: bool = False,
    ) -> MainMocks:
        patches = [
            mock.patch("build_tarballs.fetch_and_flatten"),
            mock.patch("build_tarballs.compress_tarball"),
            mock.patch("build_tarballs.is_kpack_split", return_value=kpack_split),
            mock.patch("build_tarballs.ProcessPoolExecutor", InlineProcessPoolExecutor),
        ]
        with patches[0] as fetch_mock:
            with patches[1] as compress_mock:
                with patches[2] as kpack_mock:
                    with patches[3]:
                        main(argv)
        return MainMocks(fetch_mock, compress_mock, kpack_mock)

    def test_default_builds_tarballs_without_tests_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "tarballs"
            fetch_mock, compress_mock, _ = self._run_main_with_mocks(
                [
                    "--run-id=123",
                    "--dist-amdgpu-families=gfx94X-dcgpu",
                    "--platform=linux",
                    "--package-version=7.13.0",
                    f"--output-dir={output_dir}",
                ]
            )

        self.assertEqual(fetch_mock.call_count, 1)
        self.assertEqual(fetch_mock.call_args.kwargs["exclude_components"], ["test"])
        self.assertEqual(fetch_mock.call_args.kwargs["exclude_artifacts"], ["fftw3"])

        compressed_names = [
            call.kwargs["tarball_path"].name for call in compress_mock.call_args_list
        ]
        self.assertEqual(
            compressed_names,
            ["therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz"],
        )
        self.assertEqual(
            compress_mock.call_args.kwargs["compression_backend"], "zlib-ng"
        )
        self.assertEqual(compress_mock.call_args.kwargs["compression_level"], 9)
        self.assertEqual(compress_mock.call_args.kwargs["compression_threads"], 8)

    def test_kpack_builds_common_tarball_with_one_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "tarballs"
            fetch_mock, compress_mock, _ = self._run_main_with_mocks(
                [
                    "--run-id=123",
                    "--dist-amdgpu-families=gfx94X-dcgpu",
                    "--platform=linux",
                    "--package-version=7.13.0",
                    f"--output-dir={output_dir}",
                ],
                kpack_split=True,
            )

        self.assertEqual(fetch_mock.call_count, 2)

        compressed_names = [
            call.kwargs["tarball_path"].name for call in compress_mock.call_args_list
        ]
        self.assertEqual(
            compressed_names,
            [
                "therock-dist-linux-multiarch-7.13.0.tar.gz",
                "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz",
            ],
        )

    def test_include_test_tarballs_builds_both_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "tarballs"
            fetch_mock, compress_mock, _ = self._run_main_with_mocks(
                [
                    "--run-id=123",
                    "--dist-amdgpu-families=gfx94X-dcgpu",
                    "--platform=linux",
                    "--package-version=7.13.0",
                    f"--output-dir={output_dir}",
                    "--include-test-tarballs",
                ]
            )

        self.assertEqual(fetch_mock.call_count, 2)
        self.assertEqual(
            fetch_mock.call_args_list[0].kwargs["exclude_components"], ["test"]
        )
        self.assertEqual(
            fetch_mock.call_args_list[0].kwargs["exclude_artifacts"], ["fftw3"]
        )
        self.assertNotIn("exclude_components", fetch_mock.call_args_list[1].kwargs)
        self.assertNotIn("exclude_artifacts", fetch_mock.call_args_list[1].kwargs)

        compressed_names = [
            call.kwargs["tarball_path"].name for call in compress_mock.call_args_list
        ]
        self.assertEqual(
            compressed_names,
            [
                "therock-dist-linux-gfx94X-dcgpu-tests-7.13.0.tar.gz",
                "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz",
            ],
        )

    def test_include_test_tarballs_builds_kpack_multiarch_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "tarballs"
            fetch_mock, compress_mock, _ = self._run_main_with_mocks(
                [
                    "--run-id=123",
                    "--dist-amdgpu-families=gfx94X-dcgpu;gfx110X-all",
                    "--platform=linux",
                    "--package-version=7.13.0",
                    f"--output-dir={output_dir}",
                    "--include-test-tarballs",
                ],
                kpack_split=True,
            )

        self.assertEqual(fetch_mock.call_count, 6)
        self.assertEqual(
            fetch_mock.call_args_list[-2].kwargs["exclude_components"], ["test"]
        )
        self.assertEqual(
            fetch_mock.call_args_list[-2].kwargs["exclude_artifacts"], ["fftw3"]
        )
        self.assertNotIn("exclude_components", fetch_mock.call_args_list[-1].kwargs)
        self.assertNotIn("exclude_artifacts", fetch_mock.call_args_list[-1].kwargs)

        compressed_names = [
            call.kwargs["tarball_path"].name for call in compress_mock.call_args_list
        ]
        self.assertEqual(
            compressed_names,
            [
                "therock-dist-linux-multiarch-tests-7.13.0.tar.gz",
                "therock-dist-linux-multiarch-7.13.0.tar.gz",
                "therock-dist-linux-gfx94X-dcgpu-tests-7.13.0.tar.gz",
                "therock-dist-linux-gfx110X-all-tests-7.13.0.tar.gz",
                "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz",
                "therock-dist-linux-gfx110X-all-7.13.0.tar.gz",
            ],
        )


if __name__ == "__main__":
    unittest.main()
