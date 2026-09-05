#!/usr/bin/env python
"""Unit tests for coverage_report.py.

The LLVM tools are replaced with scripts that record their own argv, so these
tests assert on the commands the report builds rather than on coverage numbers.
That is where the behaviour that matters lives: which profraw files get merged,
which binaries end up as -object arguments, and which situations must fail the
job instead of publishing a misleading report.
"""

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

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

    def test_missing_llvm_tools_fails(self):
        for tool in ("llvm-profdata", "llvm-cov"):
            (self.rocm_dir / "lib" / "llvm" / "bin" / tool).unlink()

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
