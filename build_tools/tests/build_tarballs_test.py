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

        # Two fetches: the default tarball and the opt-in HPC expansion tarball.
        self.assertEqual(fetch_mock.call_count, 2)
        # First call = default tarball fetch (excludes tests, fftw3, and HPC libs).
        default_call = fetch_mock.call_args_list[0]
        self.assertEqual(default_call.kwargs["exclude_components"], ["test"])
        self.assertEqual(
            default_call.kwargs["exclude_artifacts"],
            ["fftw3", "hiptensor", "rocalution"],
        )
        # Second call = HPC tarball fetch (includes only the HPC libs).
        hpc_call = fetch_mock.call_args_list[1]
        self.assertEqual(
            hpc_call.kwargs["include_artifacts"], ["hiptensor", "rocalution"]
        )

        # fetch_and_flatten is mocked and creates no files, so the HPC output
        # dir is empty and only the default tarball is compressed.
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

        # Four fetches: default per-family, HPC per-family, default multiarch,
        # HPC multiarch.
        self.assertEqual(fetch_mock.call_count, 4)

        # fetch_and_flatten is mocked (creates no files), so HPC output dirs are
        # empty and only the default tarballs are compressed.
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

        # Four fetches per family: default, tests, HPC, HPC-tests.
        self.assertEqual(fetch_mock.call_count, 4)
        # Call 0 = default tarball (excludes tests, fftw3, HPC libs).
        self.assertEqual(
            fetch_mock.call_args_list[0].kwargs["exclude_components"], ["test"]
        )
        self.assertEqual(
            fetch_mock.call_args_list[0].kwargs["exclude_artifacts"],
            ["fftw3", "hiptensor", "rocalution"],
        )
        # Call 1 = default tests tarball (no exclusions).
        self.assertNotIn("exclude_components", fetch_mock.call_args_list[1].kwargs)
        self.assertNotIn("exclude_artifacts", fetch_mock.call_args_list[1].kwargs)
        # Call 2 = HPC tarball (includes only HPC libs, excludes test component).
        self.assertEqual(
            fetch_mock.call_args_list[2].kwargs["include_artifacts"],
            ["hiptensor", "rocalution"],
        )
        self.assertEqual(
            fetch_mock.call_args_list[2].kwargs["exclude_components"], ["test"]
        )
        # Call 3 = HPC tests tarball (includes only HPC libs, keeps test component).
        self.assertEqual(
            fetch_mock.call_args_list[3].kwargs["include_artifacts"],
            ["hiptensor", "rocalution"],
        )
        self.assertNotIn("exclude_components", fetch_mock.call_args_list[3].kwargs)

        # fetch_and_flatten is mocked (creates no files), so HPC output dirs are
        # empty and only the default tarballs are compressed.
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

        # 2 families x 4 (default, tests, HPC, HPC-tests) = 8, plus multiarch x 4
        # (default, tests, HPC, HPC-tests) = 12 total.
        self.assertEqual(fetch_mock.call_count, 12)
        # The multiarch block runs last: default, tests, HPC, HPC-tests. So the
        # multiarch default tarball fetch is [-4] and multiarch tests is [-3].
        self.assertEqual(
            fetch_mock.call_args_list[-4].kwargs["exclude_components"], ["test"]
        )
        self.assertEqual(
            fetch_mock.call_args_list[-4].kwargs["exclude_artifacts"],
            ["fftw3", "hiptensor", "rocalution"],
        )
        self.assertNotIn("exclude_components", fetch_mock.call_args_list[-3].kwargs)
        self.assertNotIn("exclude_artifacts", fetch_mock.call_args_list[-3].kwargs)
        # [-2] = multiarch HPC, [-1] = multiarch HPC-tests.
        self.assertEqual(
            fetch_mock.call_args_list[-2].kwargs["include_artifacts"],
            ["hiptensor", "rocalution"],
        )
        self.assertEqual(
            fetch_mock.call_args_list[-1].kwargs["include_artifacts"],
            ["hiptensor", "rocalution"],
        )

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
