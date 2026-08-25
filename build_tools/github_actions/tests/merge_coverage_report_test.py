# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import merge_coverage_report


def is_windows() -> bool:
    return platform.system() == "Windows"


class TempDirTestBase(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)

    def touch(self, relative_path: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path


class FindLlvmToolTest(TempDirTestBase):
    def test_prefers_the_rocm_distribution_copy(self):
        bin_dir = self.root / "lib" / "llvm" / "bin"
        bin_dir.mkdir(parents=True)
        expected = bin_dir / f"llvm-cov{merge_coverage_report.EXECUTABLE_SUFFIX}"
        expected.touch()

        with mock.patch("shutil.which", return_value="/usr/bin/llvm-cov"):
            self.assertEqual(
                merge_coverage_report.find_llvm_tool(bin_dir, "llvm-cov"), expected
            )

    def test_falls_back_to_path(self):
        bin_dir = self.root / "empty"
        bin_dir.mkdir()

        with mock.patch("shutil.which", return_value="/usr/bin/llvm-cov"):
            self.assertEqual(
                merge_coverage_report.find_llvm_tool(bin_dir, "llvm-cov"),
                Path("/usr/bin/llvm-cov"),
            )

    def test_raises_when_the_tool_is_missing_everywhere(self):
        bin_dir = self.root / "empty"
        bin_dir.mkdir()

        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError):
                merge_coverage_report.find_llvm_tool(bin_dir, "llvm-cov")


class FindProfrawFilesTest(TempDirTestBase):
    def test_searches_recursively_and_ignores_other_files(self):
        self.touch("shard0/a-1234.profraw")
        self.touch("shard1/nested/b-5678.profraw")
        self.touch("shard1/test-output.xml")

        found = merge_coverage_report.find_profraw_files(self.root)

        self.assertEqual([f.name for f in found], ["a-1234.profraw", "b-5678.profraw"])

    def test_empty_directory_yields_nothing(self):
        self.assertEqual(merge_coverage_report.find_profraw_files(self.root), [])


class ResolveObjectsTest(TempDirTestBase):
    @unittest.skipIf(is_windows(), "symlinks require elevated privileges on Windows")
    def test_versioned_symlinks_collapse_to_one_object(self):
        real = self.touch("lib/libhiprand.so.1.0")
        (self.root / "lib" / "libhiprand.so").symlink_to(real)
        (self.root / "lib" / "libhiprand.so.1").symlink_to(real)

        objects = merge_coverage_report.resolve_objects(
            self.root, ["lib/libhiprand.so*"]
        )

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].resolve(), real.resolve())

    def test_multiple_globs_are_combined(self):
        self.touch("lib/libhiprand.so")
        self.touch("bin/hiprand_tool")

        objects = merge_coverage_report.resolve_objects(
            self.root, ["lib/libhiprand.so*", "bin/hiprand_*"]
        )

        self.assertEqual(
            sorted(o.name for o in objects), ["hiprand_tool", "libhiprand.so"]
        )

    def test_directories_are_not_treated_as_objects(self):
        (self.root / "lib" / "libhiprand.so.d").mkdir(parents=True)

        self.assertEqual(
            merge_coverage_report.resolve_objects(self.root, ["lib/libhiprand.so*"]), []
        )

    def test_unmatched_glob_yields_nothing(self):
        self.assertEqual(
            merge_coverage_report.resolve_objects(self.root, ["lib/nothing*"]), []
        )


class CommandConstructionTest(TempDirTestBase):
    def test_merge_passes_every_profraw_file(self):
        profraw_files = [self.touch("a.profraw"), self.touch("b.profraw")]
        output = self.root / "out" / "coverage.profdata"

        with mock.patch("subprocess.run") as run:
            merge_coverage_report.merge_profraw(
                Path("/llvm/llvm-profdata"), profraw_files, output
            )

        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["/llvm/llvm-profdata", "merge", "-sparse"])
        self.assertEqual(command[-2:], [str(profraw_files[0]), str(profraw_files[1])])
        self.assertTrue(output.parent.is_dir())

    def test_export_passes_the_first_object_positionally(self):
        objects = [self.touch("lib/a.so"), self.touch("lib/b.so")]
        output = self.root / "out" / "coverage.info"

        with mock.patch("subprocess.run") as run:
            merge_coverage_report.export_lcov(
                Path("/llvm/llvm-cov"), Path("/tmp/coverage.profdata"), objects, output
            )

        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["/llvm/llvm-cov", "export", str(objects[0])])
        self.assertIn("-object", command)
        self.assertEqual(
            command[-2:], ["-instr-profile=/tmp/coverage.profdata", "--format=lcov"]
        )


class MainTest(TempDirTestBase):
    def _argv(self, *extra: str) -> list[str]:
        return [
            "--profraw-dir",
            os.fspath(self.root / "profraw"),
            "--rocm-dir",
            os.fspath(self.root / "rocm"),
            "--object-globs",
            "lib/libhiprand.so*",
            *extra,
        ]

    def test_missing_profiles_fail_by_default(self):
        (self.root / "profraw").mkdir()

        self.assertEqual(merge_coverage_report.main(self._argv()), 1)

    def test_missing_profiles_are_tolerated_with_allow_empty(self):
        (self.root / "profraw").mkdir()

        self.assertEqual(merge_coverage_report.main(self._argv("--allow-empty")), 0)

    def test_unmatched_object_globs_fail(self):
        self.touch("profraw/a.profraw")
        llvm_bin_dir = self.root / "rocm" / "lib" / "llvm" / "bin"
        llvm_bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            (llvm_bin_dir / f"{tool}{merge_coverage_report.EXECUTABLE_SUFFIX}").touch()

        self.assertEqual(merge_coverage_report.main(self._argv()), 1)

    def test_happy_path_invokes_both_llvm_tools(self):
        self.touch("profraw/shard0/a.profraw")
        self.touch("rocm/lib/libhiprand.so")
        llvm_bin_dir = self.root / "rocm" / "lib" / "llvm" / "bin"
        llvm_bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            (llvm_bin_dir / f"{tool}{merge_coverage_report.EXECUTABLE_SUFFIX}").touch()

        with mock.patch("subprocess.run") as run:
            exit_code = merge_coverage_report.main(
                self._argv(
                    "--profdata-output",
                    os.fspath(self.root / "out" / "coverage.profdata"),
                    "--lcov-output",
                    os.fspath(self.root / "out" / "coverage.info"),
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_count, 2)
        invoked = [Path(call.args[0][0]).name for call in run.call_args_list]
        self.assertEqual(
            invoked,
            [
                f"llvm-profdata{merge_coverage_report.EXECUTABLE_SUFFIX}",
                f"llvm-cov{merge_coverage_report.EXECUTABLE_SUFFIX}",
            ],
        )


if __name__ == "__main__":
    unittest.main()
