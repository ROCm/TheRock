#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for verify_host_tsan_linkage.py."""

import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import verify_host_tsan_linkage


@contextmanager
def _test_directory():
    root = Path(__file__).parent / f"_verify_host_tsan_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root)


class IsElfTest(unittest.TestCase):
    def test_recognizes_elf_magic(self):
        with _test_directory() as root:
            path = root / "test"
            path.write_bytes(b"\x7fELFpayload")

            self.assertTrue(verify_host_tsan_linkage.is_elf(path))

    def test_rejects_non_elf_and_unreadable_paths(self):
        with _test_directory() as root:
            path = root / "test"
            path.write_bytes(b"text")

            self.assertFalse(verify_host_tsan_linkage.is_elf(path))
            self.assertFalse(verify_host_tsan_linkage.is_elf(root / "missing"))


class DiscoverElfExecutablesTest(unittest.TestCase):
    def test_profiler_inventories_are_explicit_and_fail_closed(self):
        expected_counts = {
            "aqlprofile": 15,
            "rocprofiler-compute": 2,
            "rocprofiler-sdk": 3,
            "rocprofiler-systems": 1,
        }
        for component, expected_count in expected_counts.items():
            with self.subTest(component=component):
                inventory = verify_host_tsan_linkage.COMPONENT_INVENTORIES[component]
                self.assertEqual(len(inventory["executables"]), expected_count)
                self.assertNotIn("ctest_dir", inventory)

    def test_unknown_component_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError, "no host-TSAN linkage inventory for unknown"
        ):
            verify_host_tsan_linkage.discover_elf_executables(Path("/rocm"), "unknown")

    def test_explicit_executable_inventory_requires_every_valid_elf(self):
        with _test_directory() as rocm_root, mock.patch.dict(
            verify_host_tsan_linkage.COMPONENT_INVENTORIES,
            {"strict": {"executables": ("bin/required",)}},
        ):
            executable = rocm_root / "bin" / "required"

            with self.assertRaisesRegex(RuntimeError, "executable is missing"):
                verify_host_tsan_linkage.discover_elf_executables(
                    rocm_root, "strict"
                )

            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"\x7fELFpayload")
            with mock.patch.object(
                verify_host_tsan_linkage.os, "access", return_value=False
            ), self.assertRaisesRegex(RuntimeError, "is not executable"):
                verify_host_tsan_linkage.discover_elf_executables(
                    rocm_root, "strict"
                )

            executable.write_bytes(b"not-elf")
            with mock.patch.object(
                verify_host_tsan_linkage.os, "access", return_value=True
            ), self.assertRaisesRegex(RuntimeError, "is not ELF"):
                verify_host_tsan_linkage.discover_elf_executables(
                    rocm_root, "strict"
                )

            executable.write_bytes(b"\x7fELFpayload")
            with mock.patch.object(
                verify_host_tsan_linkage.os, "access", return_value=True
            ):
                self.assertEqual(
                    verify_host_tsan_linkage.discover_elf_executables(
                        rocm_root, "strict"
                    ),
                    [executable.resolve()],
                )

    def test_hipfile_uses_exact_direct_executable_without_ctest_discovery(self):
        with _test_directory() as rocm_root:
            executable = rocm_root / "share" / "hipfile" / "test" / "internal_tests"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"\x7fELFpayload")
            with mock.patch.object(
                verify_host_tsan_linkage.os, "access", return_value=True
            ), mock.patch.object(
                verify_host_tsan_linkage.subprocess, "run"
            ) as run:
                result = verify_host_tsan_linkage.discover_elf_executables(
                    rocm_root, "hipfile"
                )

            self.assertEqual(result, [executable.resolve()])
            run.assert_not_called()

    def test_discovers_all_rocgdb_elf_variants_not_shell_launcher(self):
        with _test_directory() as rocm_root:
            bin_dir = rocm_root / "bin"
            bin_dir.mkdir(parents=True)
            launcher = bin_dir / "rocgdb"
            pynone = bin_dir / "rocgdb-pynone"
            py312 = bin_dir / "rocgdb-py3.12"
            for path in (launcher, pynone, py312):
                path.touch()

            with mock.patch.object(
                verify_host_tsan_linkage.os, "access", return_value=True
            ), mock.patch.object(
                verify_host_tsan_linkage,
                "is_elf",
                side_effect=lambda path: path != launcher,
            ):
                result = verify_host_tsan_linkage.discover_elf_executables(
                    rocm_root, "rocgdb-cpu"
                )

            self.assertEqual(result, sorted((pynone.resolve(), py312.resolve())))

    def test_discovers_media_library_without_execute_bit(self):
        with _test_directory() as rocm_root:
            library = rocm_root / "lib" / "librocdecode.so"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"\x7fELFpayload")

            result = verify_host_tsan_linkage.discover_elf_executables(
                rocm_root, "rocdecode"
            )

            self.assertEqual(result, [library.resolve()])


class VerifyLinkageTest(unittest.TestCase):
    def test_requires_artifact_readelf(self):
        with mock.patch.object(
            verify_host_tsan_linkage.os, "access", return_value=False
        ):
            with self.assertRaisesRegex(
                RuntimeError, "artifact llvm-readelf not found"
            ):
                verify_host_tsan_linkage.verify_linkage(Path("/rocm"), "hipfile")

    def test_requires_at_least_one_elf_linkage_target(self):
        with mock.patch.object(
            verify_host_tsan_linkage.os, "access", return_value=True
        ), mock.patch.object(
            verify_host_tsan_linkage,
            "discover_elf_executables",
            return_value=[],
        ):
            with self.assertRaisesRegex(
                RuntimeError, "no ELF host-TSAN linkage targets found for hipfile"
            ):
                verify_host_tsan_linkage.verify_linkage(Path("/rocm"), "hipfile")

    def test_reports_every_executable_without_direct_tsan_dependency(self):
        executables = [Path("/tests/instrumented"), Path("/tests/plain")]
        results = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="NEEDED Shared library: [libclang_rt.tsan-x86_64.so]",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[], returncode=0, stdout="NEEDED libc.so.6", stderr=""
            ),
        ]
        with mock.patch.object(
            verify_host_tsan_linkage.os, "access", return_value=True
        ), mock.patch.object(
            verify_host_tsan_linkage,
            "discover_elf_executables",
            return_value=executables,
        ), mock.patch.object(
            verify_host_tsan_linkage.subprocess,
            "run",
            side_effect=results,
        ):
            with self.assertRaises(RuntimeError) as raised:
                verify_host_tsan_linkage.verify_linkage(Path("/rocm"), "hipfile")

        self.assertIn(str(executables[1]), str(raised.exception))

    def test_returns_all_instrumented_executables(self):
        executable = Path("/tests/instrumented")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="NEEDED Shared library: [libclang_rt.tsan-x86_64.so]",
            stderr="",
        )
        with mock.patch.object(
            verify_host_tsan_linkage.os, "access", return_value=True
        ), mock.patch.object(
            verify_host_tsan_linkage,
            "discover_elf_executables",
            return_value=[executable],
        ), mock.patch.object(
            verify_host_tsan_linkage.subprocess,
            "run",
            return_value=completed,
        ) as run:
            result = verify_host_tsan_linkage.verify_linkage(Path("/rocm"), "hipfile")

        self.assertEqual(result, [executable])
        run.assert_called_once_with(
            [
                os.fspath(Path("/rocm/llvm/bin/llvm-readelf")),
                "--dynamic",
                os.fspath(executable),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


class MainTest(unittest.TestCase):
    def test_prints_verified_executables(self):
        executable = Path("/tests/instrumented")
        output = StringIO()
        with mock.patch.object(
            verify_host_tsan_linkage,
            "verify_linkage",
            return_value=[executable],
        ) as verify, redirect_stdout(output):
            result = verify_host_tsan_linkage.main(
                ["--rocm-root", "/rocm", "--component", "hipfile"]
            )

        self.assertEqual(result, 0)
        verify.assert_called_once_with(Path("/rocm").resolve(), "hipfile")
        self.assertIn(
            "Verified direct host-TSAN linkage for 1 hipfile", output.getvalue()
        )
        self.assertIn(str(executable), output.getvalue())


if __name__ == "__main__":
    unittest.main()
