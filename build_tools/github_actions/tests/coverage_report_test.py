#!/usr/bin/env python
"""Unit tests for coverage_report.py.

The LLVM tools are replaced with scripts that record their own argv, so these
tests assert on the commands the report builds rather than on coverage numbers.
That is where the behaviour that matters lives: which profraw files get merged,
which binaries end up as -object arguments, and which situations must fail the
job instead of publishing a misleading report.
"""

import contextlib
import io
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

# Add build_tools to path so _therock_utils is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
# Add github_actions to path so coverage_report is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import coverage_report

# Emulates just enough of each tool: `merge` creates the file named by -o,
# `report` prints a TOTAL line, and `export` prints an lcov record.
STUB_TOOL = """#!/bin/bash
echo "$(basename $0) $@" >> "{log}"
if [[ "$1" == "merge" ]]; then
  while [[ $# -gt 0 ]]; do [[ "$1" == "-o" ]] && touch "$2"; shift; done
  exit 0
fi
if [[ "$1" == "report" ]]; then
  echo "Filename Regions Missed Cover"
  echo "TOTAL        120     30 75.00%"
  exit 0
fi
echo "SF:/src/component.cpp"
echo "end_of_record"
"""


@unittest.skipIf(
    sys.platform == "win32",
    "The tool stubs are POSIX shell scripts. Coverage is Linux-only: "
    "coverage_nightly.yml builds portable Linux, so coverage_report.py never "
    "runs on Windows, and teaching find_llvm_tool to look for .bat stubs would "
    "mean changing production code for a platform it is not used on.",
)
class CoverageReportTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        self.rocm_dir = self.root / "build"
        self.profraw_dir = self.root / "profraw"
        self.output_dir = self.root / "coverage-report"
        self.call_log = self.root / "calls.log"

        bin_dir = self.rocm_dir / "lib" / "llvm" / "bin"
        bin_dir.mkdir(parents=True)
        for tool in ("llvm-profdata", "llvm-cov"):
            path = bin_dir / tool
            path.write_text(STUB_TOOL.format(log=self.call_log))
            path.chmod(path.stat().st_mode | stat.S_IEXEC)

        self.profraw_dir.mkdir()

    def write_profraw(self, name: str) -> Path:
        path = self.profraw_dir / name
        path.write_bytes(b"profraw")
        return path

    def write_executable(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/true\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def run_report(self, component: str, extra_args: list[str] | None = None) -> int:
        return coverage_report.main(
            [
                "--component",
                component,
                "--rocm-dir",
                os.fspath(self.rocm_dir),
                "--profraw-dir",
                os.fspath(self.profraw_dir),
                "--output-dir",
                os.fspath(self.output_dir),
            ]
            + (extra_args or [])
        )

    def calls(self) -> list[str]:
        return self.call_log.read_text().splitlines()


class TestSharedLibraryComponent(CoverageReportTestCase):
    """A component that ships a shared library is reported against it."""

    def setUp(self):
        super().setUp()
        (self.rocm_dir / "lib").mkdir(parents=True, exist_ok=True)
        self.library = self.rocm_dir / "lib" / "libhiprand.so"
        self.library.write_text("")

    def test_merges_profraw_from_every_shard(self):
        # A per-shard report would only reflect that shard's slice of the
        # suite, so every shard's files have to reach the merge.
        self.write_profraw("hiprand-shard1-1-a.profraw")
        self.write_profraw("hiprand-shard2-2-b.profraw")

        self.assertEqual(self.run_report("hiprand"), 0)

        merge = next(c for c in self.calls() if c.startswith("llvm-profdata merge"))
        self.assertIn("hiprand-shard1-1-a.profraw", merge)
        self.assertIn("hiprand-shard2-2-b.profraw", merge)
        self.assertIn("-sparse", merge)

    def test_reports_against_shared_library(self):
        self.write_profraw("hiprand-shard1-1-a.profraw")

        self.assertEqual(self.run_report("hiprand"), 0)

        export = next(c for c in self.calls() if c.startswith("llvm-cov export"))
        self.assertIn(f"-object {self.library}", export)
        self.assertIn("--format=lcov", export)

    def test_writes_lcov_and_text_reports(self):
        self.write_profraw("hiprand-shard1-1-a.profraw")

        self.assertEqual(self.run_report("hiprand"), 0)

        self.assertIn(
            "SF:/src/component.cpp", (self.output_dir / "coverage.info").read_text()
        )
        self.assertIn("TOTAL", (self.output_dir / "coverage.txt").read_text())

    def test_excludes_test_sources_by_default(self):
        # Otherwise a component could raise its coverage by adding test code.
        self.write_profraw("hiprand-shard1-1-a.profraw")

        self.assertEqual(self.run_report("hiprand"), 0)

        export = next(c for c in self.calls() if c.startswith("llvm-cov export"))
        self.assertIn("--ignore-filename-regex=", export)
        self.assertIn("/test/", export)

    def test_explicit_object_overrides_discovery(self):
        self.write_profraw("hiprand-shard1-1-a.profraw")
        other = self.write_executable(self.rocm_dir / "bin" / "custom_binary")

        self.assertEqual(self.run_report("hiprand", ["--object", os.fspath(other)]), 0)

        export = next(c for c in self.calls() if c.startswith("llvm-cov export"))
        self.assertIn(f"-object {other}", export)
        self.assertNotIn(os.fspath(self.library), export)


class TestHtmlReport(CoverageReportTestCase):
    """The annotated HTML report is opt-in, since it needs the sources."""

    def setUp(self):
        super().setUp()
        (self.rocm_dir / "lib").mkdir(parents=True, exist_ok=True)
        (self.rocm_dir / "lib" / "libhiprand.so").write_text("")
        self.write_profraw("hiprand-shard1-1-a.profraw")

    def test_html_is_not_written_unless_requested(self):
        self.assertEqual(self.run_report("hiprand"), 0)

        self.assertFalse((self.output_dir / "coverage.html").exists())
        self.assertNotIn("llvm-cov show", "\n".join(self.calls()))

    def test_html_flag_writes_a_single_annotated_file(self):
        # A directory tree would have to be unzipped and browsed; one file
        # opens straight from the artifact download.
        self.assertEqual(self.run_report("hiprand", ["--html"]), 0)

        show = next(c for c in self.calls() if c.startswith("llvm-cov show"))
        self.assertIn("--format=html", show)
        self.assertNotIn("--output-dir", show)
        self.assertTrue((self.output_dir / "coverage.html").is_file())


class TestSourceAvailability(unittest.TestCase):
    """Whether `show` will be able to annotate the sources it was given."""

    LCOV = "SF:/absent/one.cpp\nend_of_record\nSF:/absent/two.cpp\nend_of_record\n"

    def test_parses_source_paths_from_lcov(self):
        self.assertEqual(
            coverage_report.sources_from_lcov(self.LCOV),
            [Path("/absent/one.cpp"), Path("/absent/two.cpp")],
        )

    @staticmethod
    def capture(*args) -> str:
        """Returns what warn_on_missing_sources logged, which goes to stdout."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            coverage_report.warn_on_missing_sources(*args)
        return buffer.getvalue()

    def test_warns_when_sources_are_not_at_the_recorded_paths(self):
        # The paths come from the build machine, so a report job that did not
        # check the sources out produces an HTML report with no source in it.
        output = self.capture(self.LCOV)

        self.assertIn("[WARN]", output)
        self.assertIn("2 of 2 source file(s)", output)

    def test_silent_when_every_source_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            present = Path(tmp) / "present.cpp"
            present.write_text("int main() { return 0; }\n")

            self.assertEqual(self.capture(f"SF:{present}\n"), "")


class TestHeaderOnlyComponent(CoverageReportTestCase):
    """Header-only components have no library, so test binaries stand in."""

    def test_expands_each_test_binary_into_its_own_object_flag(self):
        # llvm-cov's -object takes a single path and does not expand globs.
        self.write_profraw("rocprim-shard1-1-a.profraw")
        test_a = self.write_executable(self.rocm_dir / "bin" / "rocprim" / "test_a")
        test_b = self.write_executable(self.rocm_dir / "bin" / "rocprim" / "test_b")

        self.assertEqual(self.run_report("rocprim"), 0)

        export = next(c for c in self.calls() if c.startswith("llvm-cov export"))
        self.assertIn(f"-object {test_a}", export)
        self.assertIn(f"-object {test_b}", export)

    def test_ignores_non_test_files_in_the_test_directory(self):
        self.write_profraw("rocprim-shard1-1-a.profraw")
        test_a = self.write_executable(self.rocm_dir / "bin" / "rocprim" / "test_a")
        data = self.rocm_dir / "bin" / "rocprim" / "CTestTestfile.cmake"
        data.write_text("")

        self.assertEqual(self.run_report("rocprim"), 0)

        export = next(c for c in self.calls() if c.startswith("llvm-cov export"))
        self.assertIn(f"-object {test_a}", export)
        self.assertNotIn("CTestTestfile", export)

    def test_uses_installed_directory_name_when_it_differs_from_job_name(self):
        # The hiprand job installs its tests under bin/hipRAND.
        self.write_profraw("hiprand-shard1-1-a.profraw")
        test_a = self.write_executable(self.rocm_dir / "bin" / "hipRAND" / "test_a")

        self.assertEqual(self.run_report("hiprand"), 0)

        export = next(c for c in self.calls() if c.startswith("llvm-cov export"))
        self.assertIn(f"-object {test_a}", export)


class TestFailureModes(CoverageReportTestCase):
    """Coverage problems must fail the job, not publish a misleading number."""

    def test_missing_profraw_fails(self):
        # No profraw at all means the binaries were not instrumented or
        # LLVM_PROFILE_FILE was wrong. Reporting 0% would hide that.
        (self.rocm_dir / "lib" / "libhiprand.so").write_text("")

        with self.assertRaises(FileNotFoundError):
            self.run_report("hiprand")

    def test_empty_profraw_files_do_not_count_as_output(self):
        (self.rocm_dir / "lib" / "libhiprand.so").write_text("")
        (self.profraw_dir / "hiprand-shard1-1-a.profraw").write_bytes(b"")

        with self.assertRaises(FileNotFoundError):
            self.run_report("hiprand")

    def test_no_library_and_no_tests_fails(self):
        self.write_profraw("rocblas-shard1-1-a.profraw")

        with self.assertRaises(FileNotFoundError):
            self.run_report("rocblas")


class TestLlvmToolDiscovery(unittest.TestCase):
    """Where the LLVM tools are taken from.

    This needs no tool stubs, only paths, so unlike the cases above it is
    meaningful on every platform.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.rocm_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.bin_dir = self.rocm_dir / "lib" / "llvm" / "bin"
        self.bin_dir.mkdir(parents=True)

    def test_prefers_the_toolchain_that_built_the_artifacts(self):
        # Coverage data is only readable by a tool at least as new as the
        # compiler that produced it, so a system tool must not win.
        bundled = self.bin_dir / "llvm-profdata"
        bundled.write_text("")

        with mock.patch.object(shutil, "which", return_value="/usr/bin/llvm-profdata"):
            self.assertEqual(
                coverage_report.find_llvm_tool(self.rocm_dir, "llvm-profdata"), bundled
            )

    def test_falls_back_to_path_for_local_runs(self):
        with mock.patch.object(shutil, "which", return_value="/usr/bin/llvm-profdata"):
            self.assertEqual(
                coverage_report.find_llvm_tool(self.rocm_dir, "llvm-profdata"),
                Path("/usr/bin/llvm-profdata"),
            )

    def test_missing_llvm_tools_fails(self):
        # shutil.which is stubbed rather than left to the machine: hosted
        # Windows runners ship LLVM on PATH, so a real lookup would find a tool
        # and this would assert on whether the runner has LLVM installed.
        with mock.patch.object(shutil, "which", return_value=None):
            with self.assertRaises(FileNotFoundError):
                coverage_report.find_llvm_tool(self.rocm_dir, "llvm-profdata")


class TestTotalLineExtraction(unittest.TestCase):
    def test_extracts_and_normalizes_total_line(self):
        report = "Filename Regions\nfoo.cpp 10\nTOTAL      120     30    75.00%\n"
        self.assertEqual(
            coverage_report.extract_total_line(report), "TOTAL 120 30 75.00%"
        )

    def test_returns_empty_when_absent(self):
        self.assertEqual(coverage_report.extract_total_line("no totals here"), "")


if __name__ == "__main__":
    unittest.main()
