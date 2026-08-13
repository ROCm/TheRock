"""
Unit tests for test_tools/list_pytest_failed_tests.py
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TEST_TOOLS_DIR = THIS_DIR.parents[2] / "test_tools"

sys.path.insert(0, os.fspath(TEST_TOOLS_DIR))

import list_pytest_failed_tests as lister


def write_report(directory: Path, name: str, tests: list[dict]) -> Path:
    path = directory / name
    path.write_text(json.dumps({"tests": tests}))
    return path


class FailedTestsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_reports_failed_and_error_outcomes(self):
        report = write_report(
            self.dir,
            "report.json",
            [
                {"nodeid": "tests/a_test.py::test_pass", "outcome": "passed"},
                {"nodeid": "tests/a_test.py::test_fail", "outcome": "failed"},
                {"nodeid": "tests/a_test.py::test_error", "outcome": "error"},
                {"nodeid": "tests/a_test.py::test_skip", "outcome": "skipped"},
            ],
        )

        self.assertEqual(
            lister.failed_tests(report),
            ["tests/a_test.py::test_fail", "tests/a_test.py::test_error"],
        )

    def test_keeps_parametrized_nodeids_intact(self):
        # Parametrized ids contain spaces, and a retry passes each as one
        # argument, so they must survive unsplit.
        nodeid = "tests/lax_test.py::LaxTest::test_conv[shape=(2, 3)]"
        report = write_report(
            self.dir, "report.json", [{"nodeid": nodeid, "outcome": "failed"}]
        )

        self.assertEqual(lister.failed_tests(report), [nodeid])

    def test_report_without_tests_key(self):
        path = self.dir / "empty.json"
        path.write_text(json.dumps({"exitcode": 1}))

        self.assertEqual(lister.failed_tests(path), [])


class MainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def run_main(self, argv: list[str]) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = lister.main(argv)
        return code, out.getvalue()

    def test_combines_reports_and_skips_missing_ones(self):
        first = write_report(
            self.dir,
            "single.json",
            [{"nodeid": "tests/a_test.py::test_one", "outcome": "failed"}],
        )
        second = write_report(
            self.dir,
            "multi.json",
            [{"nodeid": "tests/b_test.py::test_two", "outcome": "failed"}],
        )
        missing = self.dir / "absent.json"

        code, out = self.run_main(
            [os.fspath(first), os.fspath(missing), os.fspath(second)]
        )

        self.assertEqual(code, 0)
        self.assertEqual(
            out.split(), ["tests/a_test.py::test_one", "tests/b_test.py::test_two"]
        )

    def test_refuses_to_retry_more_than_max_tests(self):
        report = write_report(
            self.dir,
            "single.json",
            [
                {"nodeid": f"tests/a_test.py::test_{i}", "outcome": "failed"}
                for i in range(5)
            ],
        )

        code, out = self.run_main([os.fspath(report), "--max-tests", "4"])

        self.assertEqual(code, 1)
        self.assertEqual(out.strip(), "")

    def test_max_tests_zero_disables_the_cap(self):
        report = write_report(
            self.dir,
            "single.json",
            [
                {"nodeid": f"tests/a_test.py::test_{i}", "outcome": "failed"}
                for i in range(5)
            ],
        )

        code, out = self.run_main([os.fspath(report), "--max-tests", "0"])

        self.assertEqual(code, 0)
        self.assertEqual(len(out.split()), 5)


if __name__ == "__main__":
    unittest.main()
