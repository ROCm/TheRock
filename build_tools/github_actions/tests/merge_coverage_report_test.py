# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import platform
import subprocess
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

    def test_bang_glob_takes_matches_back_out(self):
        self.touch("bin/test_basic")
        self.touch("bin/test_hipcub_basic")

        objects = merge_coverage_report.resolve_objects(
            self.root, ["bin/test_*", "!bin/test_hipcub_*"]
        )

        self.assertEqual([o.name for o in objects], ["test_basic"])


class CountProfiledFunctionsTest(unittest.TestCase):
    def _count(self, stdout: str):
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=stdout)) as run:
            count = merge_coverage_report.count_profiled_functions(
                Path("/llvm/llvm-profdata"), Path("/tmp/c.profdata")
            )
        self.assertEqual(run.call_args.args[0][1], "show")
        return count

    def test_reads_the_summary_line(self):
        self.assertEqual(
            self._count("Instrumentation level: Front-end\nTotal functions: 1305\n"),
            1305,
        )

    def test_profile_without_counters_reads_as_zero(self):
        self.assertEqual(self._count("Total functions: 0\n"), 0)

    def test_unrecognized_output_skips_the_check(self):
        self.assertIsNone(self._count("something else entirely\n"))


class CountLcovLinesTest(TempDirTestBase):
    def test_sums_over_every_file(self):
        lcov = self.root / "coverage.info"
        lcov.write_text(
            "SF:a.cpp\nDA:1,1\nLF:10\nLH:4\nend_of_record\n"
            "SF:b.cpp\nDA:1,0\nLF:5\nLH:0\nend_of_record\n"
        )

        self.assertEqual(merge_coverage_report.count_lcov_lines(lcov), (15, 4))


class CommandConstructionTest(TempDirTestBase):
    def test_merge_passes_every_profraw_file(self):
        profraw_files = [self.touch("a.profraw"), self.touch("b.profraw")]
        output = self.root / "out" / "coverage.profdata"
        llvm_profdata = Path("/llvm/llvm-profdata")

        with mock.patch("subprocess.run") as run:
            merge_coverage_report.merge_profraw(llvm_profdata, profraw_files, output)

        command = run.call_args.args[0]
        self.assertEqual(command[:3], [str(llvm_profdata), "merge", "-sparse"])
        self.assertEqual(command[-2:], [str(profraw_files[0]), str(profraw_files[1])])
        self.assertTrue(output.parent.is_dir())

    def test_export_passes_the_first_object_positionally(self):
        objects = [self.touch("lib/a.so"), self.touch("lib/b.so")]
        output = self.root / "out" / "coverage.info"
        llvm_cov = Path("/llvm/llvm-cov")
        profdata = Path("/tmp/coverage.profdata")

        with mock.patch("subprocess.run") as run:
            merge_coverage_report.export_lcov(llvm_cov, profdata, objects, output)

        command = run.call_args.args[0]
        self.assertEqual(command[:3], [str(llvm_cov), "export", str(objects[0])])
        self.assertIn("-object", command)
        self.assertEqual(command[-2:], [f"-instr-profile={profdata}", "--format=lcov"])

    def test_summary_reports_and_captures_the_table(self):
        objects = [self.touch("lib/a.so")]
        output = self.root / "out" / "coverage_summary.txt"
        table = "Filename  Cover\nTOTAL     77.78%\n"

        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=table)) as run:
            merge_coverage_report.write_summary(
                Path("/llvm/llvm-cov"), Path("/tmp/c.profdata"), objects, output
            )

        self.assertEqual(run.call_args.args[0][1], "report")
        self.assertEqual(output.read_text(), table)

    def test_html_renders_into_a_directory(self):
        objects = [self.touch("lib/a.so")]
        output_dir = self.root / "out" / "html"

        with mock.patch("subprocess.run") as run:
            merge_coverage_report.write_html(
                Path("/llvm/llvm-cov"),
                Path("/tmp/c.profdata"),
                objects,
                output_dir,
                project_title="hiprand",
                demangler=Path("/llvm/llvm-cxxfilt"),
            )

        command = run.call_args.args[0]
        self.assertEqual(command[1], "show")
        self.assertIn("--format=html", command)
        self.assertIn(f"-output-dir={output_dir}", command)
        self.assertIn("--project-title=hiprand", command)
        self.assertIn(f"-Xdemangler={Path('/llvm/llvm-cxxfilt')}", command)

    def test_path_equivalence_reaches_every_rendering(self):
        objects = [self.touch("lib/a.so")]
        remap = "/__w/TheRock/TheRock,/home/runner/work"

        for build in (
            lambda: merge_coverage_report.export_lcov(
                Path("/llvm/llvm-cov"),
                Path("/tmp/c.profdata"),
                objects,
                self.root / "out" / "coverage.info",
                remap,
            ),
            lambda: merge_coverage_report.write_html(
                Path("/llvm/llvm-cov"),
                Path("/tmp/c.profdata"),
                objects,
                self.root / "out" / "html",
                remap,
            ),
        ):
            with mock.patch("subprocess.run") as run:
                build()
            self.assertIn(f"-path-equivalence={remap}", run.call_args.args[0])


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

        with mock.patch(
            "subprocess.run", return_value=mock.Mock(stdout="Total functions: 3\n")
        ) as run:
            exit_code = merge_coverage_report.main(
                self._argv(
                    "--profdata-output",
                    os.fspath(self.root / "out" / "coverage.profdata"),
                    "--lcov-output",
                    os.fspath(self.root / "out" / "coverage.info"),
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_count, 3)
        invoked = [Path(call.args[0][0]).name for call in run.call_args_list]
        self.assertEqual(
            invoked,
            [
                f"llvm-profdata{merge_coverage_report.EXECUTABLE_SUFFIX}",
                f"llvm-profdata{merge_coverage_report.EXECUTABLE_SUFFIX}",
                f"llvm-cov{merge_coverage_report.EXECUTABLE_SUFFIX}",
            ],
        )

    def test_summary_and_html_are_opt_in(self):
        self.touch("profraw/shard0/a.profraw")
        self.touch("rocm/lib/libhiprand.so")
        llvm_bin_dir = self.root / "rocm" / "lib" / "llvm" / "bin"
        llvm_bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            (llvm_bin_dir / f"{tool}{merge_coverage_report.EXECUTABLE_SUFFIX}").touch()

        with mock.patch("subprocess.run", return_value=mock.Mock(stdout="")) as run:
            exit_code = merge_coverage_report.main(
                self._argv(
                    "--profdata-output",
                    os.fspath(self.root / "out" / "coverage.profdata"),
                    "--lcov-output",
                    os.fspath(self.root / "out" / "coverage.info"),
                    "--summary-output",
                    os.fspath(self.root / "out" / "coverage_summary.txt"),
                    "--html-output",
                    os.fspath(self.root / "out" / "html"),
                )
            )

        self.assertEqual(exit_code, 0)
        # merge and check the index, then export, report and show off it.
        subcommands = [call.args[0][1] for call in run.call_args_list]
        self.assertEqual(subcommands, ["merge", "show", "export", "report", "show"])


class MainRejectsEmptyReportsTest(TempDirTestBase):
    """Every way the pipeline breaks upstream still leaves profraw files behind."""

    def setUp(self):
        super().setUp()
        self.touch("profraw/shard0/a.profraw")
        self.touch("rocm/lib/libhiprand.so")
        llvm_bin_dir = self.root / "rocm" / "lib" / "llvm" / "bin"
        llvm_bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            (llvm_bin_dir / f"{tool}{merge_coverage_report.EXECUTABLE_SUFFIX}").touch()

    def _main(
        self, *extra: str, functions: int = 3, lcov: str = "", export_fails=False
    ):
        def fake_run(command, **kwargs):
            tool, subcommand = Path(command[0]).stem, command[1]
            if (tool, subcommand) == ("llvm-profdata", "show"):
                return mock.Mock(stdout=f"Total functions: {functions}\n")
            if subcommand == "export":
                if export_fails:
                    raise subprocess.CalledProcessError(1, command)
                kwargs["stdout"].write(lcov)
            return mock.Mock(stdout="")

        with mock.patch("subprocess.run", side_effect=fake_run) as run:
            exit_code = merge_coverage_report.main(
                [
                    "--profraw-dir",
                    os.fspath(self.root / "profraw"),
                    "--rocm-dir",
                    os.fspath(self.root / "rocm"),
                    "--object-globs",
                    "lib/libhiprand.so*",
                    "--profdata-output",
                    os.fspath(self.root / "out" / "coverage.profdata"),
                    "--lcov-output",
                    os.fspath(self.root / "out" / "coverage.info"),
                    *extra,
                ]
            )
        return exit_code, [call.args[0][1] for call in run.call_args_list]

    def test_profiles_without_counters_fail_before_export(self):
        # What an uninstrumented library writes: a header and nothing else.
        exit_code, subcommands = self._main(functions=0)

        self.assertEqual(exit_code, 1)
        self.assertEqual(subcommands, ["merge", "show"])

    def test_profiles_without_counters_are_tolerated_with_allow_empty(self):
        exit_code, _ = self._main("--allow-empty", functions=0)

        self.assertEqual(exit_code, 0)

    def test_objects_without_a_coverage_mapping_fail(self):
        exit_code, _ = self._main(export_fails=True)

        self.assertEqual(exit_code, 1)

    def test_report_that_never_reaches_the_objects_fails(self):
        exit_code, subcommands = self._main(
            "--summary-output",
            os.fspath(self.root / "out" / "coverage_summary.txt"),
            lcov="SF:a.cpp\nLF:10\nLH:0\nend_of_record\n",
        )

        self.assertEqual(exit_code, 1)
        # The summary is still written, so the job page shows the zero.
        self.assertEqual(subcommands, ["merge", "show", "export", "report"])

    def test_report_with_any_line_hit_passes(self):
        exit_code, _ = self._main(lcov="SF:a.cpp\nLF:10\nLH:1\nend_of_record\n")

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
