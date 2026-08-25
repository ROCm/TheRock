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
        hpc_present: bool = False,
    ) -> MainMocks:
        # fetch_and_flatten is mocked and creates no files, so the real
        # has_contents() probe would always be False. Patch it to simulate
        # whether the build contains HPC artifacts (controls superset emission).
        patches = [
            mock.patch("build_tarballs.fetch_and_flatten"),
            mock.patch("build_tarballs.compress_tarball"),
            mock.patch("build_tarballs.is_kpack_split", return_value=kpack_split),
            mock.patch("build_tarballs.ProcessPoolExecutor", InlineProcessPoolExecutor),
            mock.patch("build_tarballs.has_contents", return_value=hpc_present),
        ]
        with patches[0] as fetch_mock:
            with patches[1] as compress_mock:
                with patches[2] as kpack_mock:
                    with patches[3]:
                        with patches[4]:
                            main(argv)
        return MainMocks(fetch_mock, compress_mock, kpack_mock)

    def test_default_builds_core_and_superset_when_hpc_present(self) -> None:
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
                hpc_present=True,
            )

        # Three fetches: Core tarball, HPC presence probe, core+hpc superset.
        self.assertEqual(fetch_mock.call_count, 3)
        # Call 0 = Core tarball fetch (excludes tests, fftw3, and HPC libs).
        core_call = fetch_mock.call_args_list[0]
        self.assertEqual(core_call.kwargs["exclude_components"], ["test"])
        self.assertEqual(
            core_call.kwargs["exclude_artifacts"],
            ["fftw3", "hiptensor", "rocalution"],
        )
        # Call 1 = HPC presence probe (includes only the HPC libs).
        probe_call = fetch_mock.call_args_list[1]
        self.assertEqual(
            probe_call.kwargs["include_artifacts"], ["hiptensor", "rocalution"]
        )
        # Call 2 = superset fetch: full Core set with HPC libs kept (only fftw3
        # dropped), no include filter.
        superset_call = fetch_mock.call_args_list[2]
        self.assertEqual(superset_call.kwargs["exclude_components"], ["test"])
        self.assertEqual(superset_call.kwargs["exclude_artifacts"], ["fftw3"])
        self.assertNotIn("include_artifacts", superset_call.kwargs)

        compressed_names = [
            call.kwargs["tarball_path"].name for call in compress_mock.call_args_list
        ]
        self.assertEqual(
            sorted(compressed_names),
            [
                "therock-dist-core+hpc-linux-gfx94X-dcgpu-7.13.0.tar.gz",
                "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz",
            ],
        )

    def test_default_builds_only_core_when_no_hpc(self) -> None:
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
                hpc_present=False,
            )

        # Two fetches: the Core tarball and the HPC presence probe (which finds
        # nothing, so no superset fetch and no superset tarball).
        self.assertEqual(fetch_mock.call_count, 2)
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
                hpc_present=False,
            )

        # No HPC present: per-family Core + probe, multiarch Core + probe = 4.
        self.assertEqual(fetch_mock.call_count, 4)

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

    def test_kpack_builds_core_and_superset_when_hpc_present(self) -> None:
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
                hpc_present=True,
            )

        # Per-family: Core + probe + superset = 3. Multiarch: Core + probe +
        # superset = 3. Total 6.
        self.assertEqual(fetch_mock.call_count, 6)

        compressed_names = [
            call.kwargs["tarball_path"].name for call in compress_mock.call_args_list
        ]
        self.assertEqual(
            sorted(compressed_names),
            [
                "therock-dist-core+hpc-linux-gfx94X-dcgpu-7.13.0.tar.gz",
                "therock-dist-core+hpc-linux-multiarch-7.13.0.tar.gz",
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
                ],
                hpc_present=True,
            )

        # Five fetches: Core, Core-tests, HPC probe, superset, superset-tests.
        self.assertEqual(fetch_mock.call_count, 5)
        # Call 0 = Core tarball (excludes tests, fftw3, HPC libs).
        self.assertEqual(
            fetch_mock.call_args_list[0].kwargs["exclude_components"], ["test"]
        )
        self.assertEqual(
            fetch_mock.call_args_list[0].kwargs["exclude_artifacts"],
            ["fftw3", "hiptensor", "rocalution"],
        )
        # Call 1 = Core tests tarball (no exclusions).
        self.assertNotIn("exclude_components", fetch_mock.call_args_list[1].kwargs)
        self.assertNotIn("exclude_artifacts", fetch_mock.call_args_list[1].kwargs)
        # Call 2 = HPC presence probe (includes only HPC libs, excludes test).
        self.assertEqual(
            fetch_mock.call_args_list[2].kwargs["include_artifacts"],
            ["hiptensor", "rocalution"],
        )
        self.assertEqual(
            fetch_mock.call_args_list[2].kwargs["exclude_components"], ["test"]
        )
        # Call 3 = superset (full Core + HPC libs, only fftw3 dropped, no tests).
        self.assertEqual(
            fetch_mock.call_args_list[3].kwargs["exclude_components"], ["test"]
        )
        self.assertEqual(
            fetch_mock.call_args_list[3].kwargs["exclude_artifacts"], ["fftw3"]
        )
        self.assertNotIn("include_artifacts", fetch_mock.call_args_list[3].kwargs)
        # Call 4 = superset-tests (keeps test component, only fftw3 dropped).
        self.assertNotIn("exclude_components", fetch_mock.call_args_list[4].kwargs)
        self.assertEqual(
            fetch_mock.call_args_list[4].kwargs["exclude_artifacts"], ["fftw3"]
        )
        self.assertNotIn("include_artifacts", fetch_mock.call_args_list[4].kwargs)

        compressed_names = [
            call.kwargs["tarball_path"].name for call in compress_mock.call_args_list
        ]
        self.assertEqual(
            sorted(compressed_names),
            [
                "therock-dist-core+hpc-linux-gfx94X-dcgpu-7.13.0.tar.gz",
                "therock-dist-core+hpc-linux-gfx94X-dcgpu-tests-7.13.0.tar.gz",
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
                hpc_present=True,
            )

        # 2 families x 5 (Core, Core-tests, probe, superset, superset-tests) = 10,
        # plus multiarch x 5 = 15 total.
        self.assertEqual(fetch_mock.call_count, 15)
        # The multiarch block runs last in order: Core, Core-tests, probe,
        # superset, superset-tests -> [-5], [-4], [-3], [-2], [-1].
        self.assertEqual(
            fetch_mock.call_args_list[-5].kwargs["exclude_components"], ["test"]
        )
        self.assertEqual(
            fetch_mock.call_args_list[-5].kwargs["exclude_artifacts"],
            ["fftw3", "hiptensor", "rocalution"],
        )
        # [-4] = multiarch Core-tests (no exclusions).
        self.assertNotIn("exclude_components", fetch_mock.call_args_list[-4].kwargs)
        self.assertNotIn("exclude_artifacts", fetch_mock.call_args_list[-4].kwargs)
        # [-3] = multiarch HPC presence probe.
        self.assertEqual(
            fetch_mock.call_args_list[-3].kwargs["include_artifacts"],
            ["hiptensor", "rocalution"],
        )
        # [-2] = multiarch superset, [-1] = multiarch superset-tests.
        self.assertEqual(
            fetch_mock.call_args_list[-2].kwargs["exclude_artifacts"], ["fftw3"]
        )
        self.assertNotIn("include_artifacts", fetch_mock.call_args_list[-2].kwargs)
        self.assertEqual(
            fetch_mock.call_args_list[-1].kwargs["exclude_artifacts"], ["fftw3"]
        )
        self.assertNotIn("include_artifacts", fetch_mock.call_args_list[-1].kwargs)

        compressed_names = [
            call.kwargs["tarball_path"].name for call in compress_mock.call_args_list
        ]
        self.assertEqual(
            sorted(compressed_names),
            [
                "therock-dist-core+hpc-linux-gfx110X-all-7.13.0.tar.gz",
                "therock-dist-core+hpc-linux-gfx110X-all-tests-7.13.0.tar.gz",
                "therock-dist-core+hpc-linux-gfx94X-dcgpu-7.13.0.tar.gz",
                "therock-dist-core+hpc-linux-gfx94X-dcgpu-tests-7.13.0.tar.gz",
                "therock-dist-core+hpc-linux-multiarch-7.13.0.tar.gz",
                "therock-dist-core+hpc-linux-multiarch-tests-7.13.0.tar.gz",
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
