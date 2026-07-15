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

from build_tarballs import compress_tarball, is_kpack_split, main


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
            sorted(compressed_names),
            [
                "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz",
                "therock-dist-linux-multiarch-7.13.0.tar.gz",
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
            sorted(compressed_names),
            [
                "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz",
                "therock-dist-linux-gfx94X-dcgpu-tests-7.13.0.tar.gz",
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
            sorted(compressed_names),
            [
                "therock-dist-linux-gfx110X-all-7.13.0.tar.gz",
                "therock-dist-linux-gfx110X-all-tests-7.13.0.tar.gz",
                "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz",
                "therock-dist-linux-gfx94X-dcgpu-tests-7.13.0.tar.gz",
                "therock-dist-linux-multiarch-7.13.0.tar.gz",
                "therock-dist-linux-multiarch-tests-7.13.0.tar.gz",
            ],
        )


if __name__ == "__main__":
    unittest.main()
