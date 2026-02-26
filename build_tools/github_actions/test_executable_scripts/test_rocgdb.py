import argparse
import glob
import logging
import os
import platform
import re
import resource
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Status symbols used throughout.
STATUS_PASS = "[✓]"
STATUS_FAIL = "[X]"
STATUS_WARN = "[!]"
STATUS_FLAKY = "[~]"

# A list of tests we know FAIL but we are OK with it.
#
# The "Generic" key tracks failures we should ignore for any compiler
# label.
#
# Each compiler label key contains a list of tests for which failures should
# be ignored.
XFAILED_TESTS = {
    "Generic": [
        "gdb.rocm/corefile.exp",
        "gdb.rocm/device-interrupt.exp",
        "gdb.rocm/load-core-remote-system.exp",
    ],
    "GCC": [],
    "LLVM": [
        "gdb.dwarf2/ada-valprint-error.exp",
        "gdb.dwarf2/dw2-case-insensitive.exp",
        "gdb.dwarf2/dw2-cp-infcall-ref-static.exp",
        "gdb.dwarf2/dw2-empty-inline-ranges.exp",
        "gdb.dwarf2/dw2-entry-pc.exp",
        # Testcase expects ASLR OFF and amd-llvm generates
        # PIE executables by default.  Fix being pursued upstream.
        # REMOVE when the fix makes its way to our branch.
        "gdb.dwarf2/dw2-entry-value.exp",
        "gdb.dwarf2/dw2-inline-param.exp",
        "gdb.dwarf2/dw2-param-error.exp",
        "gdb.dwarf2/dw2-skip-prologue.exp",
        "gdb.dwarf2/dw2-undefined-ret-addr.exp",
        "gdb.dwarf2/dw2-unresolved.exp",
        "gdb.dwarf2/dw2-unexpected-entry-pc.exp",
        "gdb.dwarf2/fission-base.exp",
        "gdb.dwarf2/fission-dw-form-strx.exp",
        "gdb.dwarf2/pr13961.exp",
    ],
}

# Test result categories.
TEST_CATEGORIES = [
    "PASS",
    "FAIL",
    "ERROR",
    "UNRESOLVED",
    "TIMEOUT",
    "UNTESTED",
    "UNSUPPORTED",
    "XFAIL",
    "KFAIL",
    "FLAKY",  # Tests that failed initially but passed on retry.
]

# Category display configuration: (prefix, suffix).
CATEGORY_DISPLAY = {
    "PASS": (STATUS_PASS, ""),
    "FAIL": (STATUS_FAIL, ""),
    "ERROR": (STATUS_FAIL, ""),
    "UNRESOLVED": (STATUS_FAIL, ""),
    "TIMEOUT": (STATUS_WARN, ""),
    "UNTESTED": (STATUS_WARN, ""),
    "UNSUPPORTED": (STATUS_WARN, ""),
    "XFAIL": (STATUS_WARN, ""),
    "KFAIL": (STATUS_WARN, ""),
    "FLAKY": (STATUS_FLAKY, ""),
}


def _log_error_and_exit(message: str, exit_code: int = 1) -> None:
    """
    Log an error message and exit the program.

    Args:
        message: Error message to log.
        exit_code: Exit code (default 1).
    """
    logger.info(f"{STATUS_FAIL} Error: {message}")
    sys.exit(exit_code)


def _run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = False,
    check: bool = True,
    error_msg: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """
    Run a command with consistent error handling.

    Args:
        cmd: Command and arguments to execute.
        cwd: Working directory for command execution.
        env: Environment variables for command.
        capture_output: Whether to capture stdout/stderr.
        check: Whether to raise on non-zero exit.
        error_msg: Custom error message on failure.

    Returns:
        CompletedProcess result.

    Exits:
        Code 1 if check=True and command fails.
    """
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=check,
            capture_output=capture_output,
            env=env,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        if error_msg:
            _log_error_and_exit(f"{error_msg}: {e}")
        raise


def _read_file_lines(file_path: Path, error_context: str = "") -> List[str]:
    """
    Read file lines with error handling.

    Args:
        file_path: Path to file to read.
        error_context: Context for error message.

    Returns:
        List of stripped lines from file.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        _log_error_and_exit(
            f"{file_path} not found{'. ' + error_context if error_context else ''}"
        )


def _extract_test_file_from_name(test_name: str) -> str:
    """
    Extract test file path from full test name.

    Args:
        test_name: Full test name (e.g., "gdb.rocm/foo.exp: test description").

    Returns:
        Test file path without description (e.g., "gdb.rocm/foo.exp").
    """
    return test_name.split(": ", 1)[0] if ": " in test_name else test_name


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for the ROCgdb test suite.

    Returns:
        Parsed arguments with validated paths, timeout, and retry settings.

    Exits:
        Code 1 if arguments are invalid or paths don't exist.
    """
    parser = argparse.ArgumentParser(
        description="Run ROCgdb test suite with different compilers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python %(prog)s --testsuite-dir /path/to/testsuite --rocgdb-bin /path/to/rocgdb
  python %(prog)s --tests gdb.base/break.exp gdb.base/call-ar-st.exp
  python %(prog)s --parallel
  python %(prog)s --group-results
  python %(prog)s --timeout 600
  python %(prog)s --max-failed-retries 2
  python %(prog)s --optimization="-O0"
  python %(prog)s --runtestflags="--target_board=hip -debug"
  python %(prog)s --no-xfail
  python %(prog)s --quiet
  python %(prog)s --dump-failed-test-log

        """,
    )

    parser.add_argument(
        "--dump-failed-test-log",
        action="store_false",
        help="For failed tests, dump gdb.log to the console at the end of the run. Default is on.",
    )
    parser.add_argument(
        "--group-results",
        action="store_true",
        help="Group test results in summary output. Default is off.",
    )
    parser.add_argument(
        "--max-failed-retries",
        type=int,
        default=3,
        help="Maximum number of times to retry failed tests. Default is 3.",
    )
    parser.add_argument(
        "--no-xfail",
        action="store_true",
        help="Do not use ignore lists for failed tests. Default is to use the ignore lists.",
    )
    parser.add_argument(
        "--optimization",
        type=str,
        default="",
        help="Optimization level to pass to compiler (e.g., -O0, -Os, -Og).",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run tests in parallel. Default is off."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not output configure/testsuite commands. Default is off.",
    )
    parser.add_argument(
        "--rocgdb-bin",
        type=Path,
        help="Path to ROCgdb executable. Default is to look for TheRock env variables.",
    )
    parser.add_argument(
        "--runtestflags",
        type=str,
        default="",
        help="Additional flags for RUNTESTFLAGS (e.g., '--target_board=hip -debug').",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["gdb.rocm", "gdb.dwarf2"],
        help="List of tests to run. Default is gdb.rocm/*.exp and gdb.dwarf2/*.exp",
    )
    parser.add_argument(
        "--testsuite-dir",
        type=Path,
        help="Path to GDB testsuite directory. Default is to look for TheRock env variables.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=100,
        help="Timeout value in seconds for individual tests (max: 600). Default is 100.",
    )

    args = parser.parse_args()

    # Enforce both testsuite-dir and rocgdb-bin being provided together.
    if (args.testsuite_dir is None) != (args.rocgdb_bin is None):
        _log_error_and_exit(
            "Both --testsuite-dir and --rocgdb-bin must be provided together, or neither."
        )

    # Validate paths if provided.
    if args.testsuite_dir is not None:
        validate_path(args.testsuite_dir, is_dir=True)
        validate_path(args.rocgdb_bin, is_file=True)

    # Validate timeout value.
    if not (0 < args.timeout <= 600):
        _log_error_and_exit(
            f"Timeout must be between 1 and 600 seconds. Got {args.timeout}."
        )

    # Validate max_failed_retries value.
    if args.max_failed_retries < 0:
        _log_error_and_exit(
            f"Max failed retries must be non-negative. Got {args.max_failed_retries}."
        )

    return args


class TestResults:
    """Class to store and manage test results across multiple compiler runs."""

    def __init__(self) -> None:
        """Initialize test results storage with empty data structures."""
        # Flag to control whether to group results in output.
        self.group_results: bool = False

        # Our main data structure for storing results. The mapping goes like this:
        #
        # - 1 compiler label maps to N labels (PASS, FAIL etc).
        # - 1 label maps to N test files (gdb.rocm/simple.exp for example).
        # - 1 test file maps to N test descriptions (the complete test
        #   line output by dejagnu).
        self.test_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Track currently failed tests by compiler_label.
        self.failed_tests: Dict[str, Set[str]] = defaultdict(set)

        # Track flaky tests (failed initially but passed on retry) by compiler_label.
        self.flaky_tests: Dict[str, Set[str]] = defaultdict(set)

    def cleanup_old_entries(self, compiler_label: str, tests: str) -> None:
        """
        Remove stale test results before retry runs to prevent duplicate entries.

        Args:
            compiler_label: Compiler identifier (e.g., "GCC", "LLVM").
            tests: Space-separated test file names to remove.

        Returns:
            None
        """
        # Case: first run for this compiler — nothing to clean.
        if not self.test_data[compiler_label]:
            return

        logger.info(f"Retry run. Cleaning dictionary entries for {tests}.")

        # Clear the failed tests set for this compiler label.
        self.failed_tests[compiler_label].clear()

        # Clear tests entries from the main test results database for this
        # compiler label.
        for test in tests.split():
            for category in TEST_CATEGORIES:
                category_results = self.test_data[compiler_label].get(category)
                if category_results and test in category_results:
                    del category_results[test]

    def extract_errors(self, results_file: str) -> Dict[str, List[str]]:
        """
        Extract ERROR messages from test results, grouped by test file.

        Args:
            results_file: Path to the GDB test results file.

        Returns:
            Dictionary mapping test file names to lists of error messages.
            Test paths are shortened to last two components (e.g., 'gdb.base/break.exp').
        """
        errors_dict = defaultdict(list)
        current_test_file = None

        for line in _read_file_lines(Path(results_file), "during error extraction"):
            # Detect the start of a test case.
            match = re.search(r"Running\s+(\S+\.exp)", line)
            if match:
                full_path = match.group(1).rstrip(":.")
                segments = os.path.normpath(full_path).split(os.path.sep)
                current_test_file = "/".join(segments[-2:])

            # Detect ERROR lines.
            if line.startswith("ERROR:"):
                msg = line[len("ERROR:") :].strip()
                test_file = current_test_file or "UNKNOWN"
                errors_dict[test_file].append(f"{test_file}: {msg}")

        return dict(errors_dict)

    def update_results(
        self, compiler_label: str, tests: str, results_file: str
    ) -> None:
        """
        Parse and update test results from a DejaGnu results file.

        Args:
            compiler_label: Compiler name (e.g., "GCC", "LLVM").
            tests: Space-separated test file names to clear before updating.
            results_file: Path to results file with lines like "STATUS: test_description".

        Returns:
            None
        """
        self.cleanup_old_entries(compiler_label, tests)

        result_regex = re.compile(
            r"^(PASS|FAIL|XFAIL|UNTESTED|UNSUPPORTED|KFAIL|UNRESOLVED): (.+)"
        )
        timeout_regex = re.compile(r"\(timeout\)")

        for line in _read_file_lines(Path(results_file), "during results update"):
            if not line:
                continue

            match = result_regex.match(line)
            if match:
                status, test_name = match.groups()
                test_file = _extract_test_file_from_name(test_name)
                is_timeout = timeout_regex.search(test_name)

                # Add the entry to our test results database.
                self.test_data[compiler_label][status][test_file].append(test_name)

                # For timeouts, we add a duplicate since we want to
                # explicitly list tests that ran into timeouts.
                if is_timeout:
                    self.test_data[compiler_label]["TIMEOUT"][test_file].append(
                        test_name
                    )

                # Track the current list of failed tests.
                if status in ["FAIL", "UNRESOLVED"] or is_timeout:
                    self.failed_tests[compiler_label].add(test_file)

        # Handle ERROR entries separately since we need to do extra work to
        # find out the testcase information.
        self.test_data[compiler_label]["ERROR"] = self.extract_errors(results_file)

        # Update failed tests with ERROR entries.
        for test_file in self.test_data[compiler_label]["ERROR"].keys():
            self.failed_tests[compiler_label].add(test_file)

    def get_failed_tests(self, compiler_label: str) -> List[str]:
        """
        Get currently failed tests for a compiler.

        Args:
            compiler_label: Compiler identifier.

        Returns:
            List of failed test file paths.
        """
        return list(self.failed_tests[compiler_label])

    def mark_flaky_tests(self, compiler_label: str, test_files: Set[str]) -> None:
        """
        Mark tests as flaky (failed initially but passed on retry).

        Args:
            compiler_label: Compiler identifier (e.g., "GCC", "LLVM").
            test_files: Set of test file paths that were flaky.

        Returns:
            None
        """
        if not test_files:
            return

        # Add to flaky tests set.
        self.flaky_tests[compiler_label].update(test_files)

        # Add to test_data under FLAKY category with marker.
        for test_file in test_files:
            flaky_description = f"{test_file}: initially failed, passed on retry"
            self.test_data[compiler_label]["FLAKY"][test_file].append(flaky_description)

        logger.info(
            f"{STATUS_FLAKY} {len(test_files)} test(s) became stable after retry for {compiler_label}"
        )

    def print_all_summaries(self) -> None:
        """
        Print test summaries for all compilers.

        Returns:
            None
        """
        if not self.test_data:
            logger.info(f"\n{STATUS_FAIL} No test results to display.")
            return

        print_section("COMBINED GDB TESTSUITE ANALYSIS")

        for compiler_label in sorted(self.test_data.keys()):
            self._print_summary(compiler_label, self.test_data[compiler_label])

        self.print_comparison()

    def _print_summary(self, compiler_label: str, details: Dict[str, Set[str]]) -> None:
        """
        Print test summary for a single compiler.

        Args:
            compiler_label: Compiler identifier.
            details: Mapping from status category to test files and descriptions.

        Returns:
            None
        """
        print_section(f"{compiler_label}", border_char="-", inline=True)
        for cat, (prefix_indicator, suffix_indicator) in CATEGORY_DISPLAY.items():
            count = sum(len(tests) for tests in details[cat].values())

            if prefix_indicator and count > 0:
                header = f"  {prefix_indicator} {cat}: {count} {suffix_indicator}"
                logger.info(header)

            # Print details for non-PASS tests.
            if count > 0 and cat != "PASS":
                if self.group_results:
                    self._print_grouped_tests(details[cat])
                else:
                    for test_file in details[cat]:
                        for test_name in details[cat][test_file]:
                            logger.info(f"          {test_name}")

    def _print_grouped_tests(self, test_list: Dict[str, List[str]]) -> None:
        """
        Print tests grouped by directory, file, and description.

        Args:
            test_list: Mapping of test file paths to descriptions.

        Returns:
            None
        """
        grouped = defaultdict(lambda: defaultdict(list))

        for file in test_list.keys():
            for test_description in test_list[file]:
                test_dir, test_file = os.path.split(file)
                grouped[test_dir][test_file].append(test_description[len(file) + 2 :])

        for directory in sorted(grouped.keys()):
            logger.info(f"     * {directory}")
            for filename in sorted(grouped[directory].keys()):
                descriptions = grouped[directory][filename]
                if len(descriptions) == 1 and descriptions[0]:
                    logger.info(f"          * {filename}: {descriptions[0]}")
                elif len(descriptions) == 1:
                    logger.info(f"          * {filename}")
                else:
                    logger.info(f"          * {filename}")
                    for desc in descriptions:
                        if desc:
                            logger.info(f"              * {desc}")

    def print_comparison(self) -> None:
        """
        Print failing tests exclusive to each compiler.

        Returns:
            None
        """
        if len(self.failed_tests) <= 1:
            return

        print_section("EXCLUSIVE FAILING TESTS COMPARISON")

        for compiler_label in sorted(self.failed_tests.keys()):
            other_failed = set.union(
                *(
                    self.failed_tests[other]
                    for other in self.failed_tests
                    if other != compiler_label
                )
            )
            exclusive_tests = self.failed_tests[compiler_label] - other_failed

            print_section(f"{compiler_label}", border_char="-", inline=True)
            if exclusive_tests:
                for test in sorted(exclusive_tests):
                    logger.info(f"  {STATUS_FAIL} {test}")
            else:
                logger.info("  No exclusive failing tests.")

    def check_all_pass_or_xfailed(
        self, xfailed_tests: Dict[str, List[str]]
    ) -> Tuple[bool, Dict[str, Dict]]:
        """
        Check if all tests passed or are in the expected failure list.

        Args:
            xfailed_tests: Compiler labels mapped to expected failing test paths.
                Must include "Generic" key for cross-compiler failures.

        Returns:
            Tuple of (overall_pass, details) where:
                - overall_pass: True if no unexpected failures.
                - details: Per-compiler dict with passed status, total_failed count,
                  unexpected_failures, unused_xfails, expected_generic,
                  expected_compiler_specific lists, and flaky_tests count.
        """
        overall_pass = True
        details = {}
        generic_xfailed = set(xfailed_tests.get("Generic", []))

        # Include all compilers that have either failed or flaky tests.
        all_compilers = set(self.failed_tests.keys()) | set(self.flaky_tests.keys())

        for compiler_label in all_compilers:
            failed_test_files = {
                _extract_test_file_from_name(test)
                for test in self.failed_tests.get(compiler_label, set())
            }
            compiler_xfailed = set(xfailed_tests.get(compiler_label, []))
            all_xfailed = generic_xfailed | compiler_xfailed

            unexpected_failures = failed_test_files - all_xfailed
            unused_xfails = all_xfailed - failed_test_files
            expected_generic = failed_test_files & generic_xfailed
            expected_compiler_specific = failed_test_files & compiler_xfailed

            compiler_pass = len(unexpected_failures) == 0
            overall_pass = overall_pass and compiler_pass

            flaky_count = len(self.flaky_tests.get(compiler_label, set()))

            details[compiler_label] = {
                "passed": compiler_pass,
                "total_failed": len(failed_test_files),
                "unexpected_failures": sorted(unexpected_failures),
                "unused_xfails": sorted(unused_xfails),
                "expected_generic": sorted(expected_generic),
                "expected_compiler_specific": sorted(expected_compiler_specific),
                "flaky_tests": sorted(self.flaky_tests.get(compiler_label, set())),
                "flaky_count": flaky_count,
            }

        return overall_pass, details

    def print_final_status(
        self, xfailed_tests: Dict[str, List[str]], no_xfail: bool
    ) -> bool:
        """
        Print final PASS/FAIL status for all compilers.

        Args:
            xfailed_tests: Compiler labels mapped to expected failures.
            no_xfail: If True, ignore the expected failure lists.

        Returns:
            True if all tests passed or are expected failures, False otherwise.
        """
        if no_xfail:
            xfailed_tests.clear()

        overall_pass, details = self.check_all_pass_or_xfailed(xfailed_tests)

        print_section("FINAL TEST STATUS")

        for compiler_label in sorted(details.keys()):
            detail = details[compiler_label]
            status_symbol = STATUS_PASS if detail["passed"] else STATUS_FAIL
            status_text = "PASS" if detail["passed"] else "FAIL"

            logger.info(f"{status_symbol} {compiler_label}: {status_text}")

            if detail["total_failed"]:
                logger.info(
                    f"{STATUS_FAIL} Total Failed Tests: {detail['total_failed']}"
                )

            if detail["flaky_count"] > 0:
                logger.info(
                    f"{STATUS_FLAKY} Flaky Tests (passed after retry) ({detail['flaky_count']}):"
                )
                for test in detail["flaky_tests"]:
                    logger.info(f"       {test}")

            if detail["expected_generic"]:
                logger.info(
                    f"{STATUS_WARN} Ignored Failures (Generic) ({len(detail['expected_generic'])}):"
                )
                for test in detail["expected_generic"]:
                    logger.info(f"       {test}")

            if detail["expected_compiler_specific"]:
                logger.info(
                    f"{STATUS_WARN} Ignored Failures ({compiler_label}) ({len(detail['expected_compiler_specific'])}):"
                )
                for test in detail["expected_compiler_specific"]:
                    logger.info(f"       {test}")

            if detail["unexpected_failures"]:
                logger.info(
                    f"{STATUS_FAIL} Unexpected Failures ({len(detail['unexpected_failures'])}):"
                )
                for test in detail["unexpected_failures"]:
                    logger.info(f"       {test}")

            if detail["unused_xfails"]:
                logger.info(
                    f"{STATUS_WARN} Unused Ignored Failures ({len(detail['unused_xfails'])}):"
                )
                for test in detail["unused_xfails"]:
                    logger.info(f"       {test}")
            logger.info("")

        overall_status = "PASS" if overall_pass else "FAIL"
        overall_symbol = STATUS_PASS if overall_pass else STATUS_FAIL
        print_section(f"{overall_symbol} OVERALL STATUS: {overall_status}")
        logger.info("")

        return overall_pass


def print_section(
    title: str,
    border_char: str = "=",
    width: int = 80,
    center: bool = True,
    inline: bool = False,
    color: Optional[str] = None,
) -> None:
    """
    Print a formatted section header to console.

    Args:
        title: Text to display in the header.
        border_char: Character for border line (default "=").
        width: Total header width (default 80).
        center: Center title in multi-line mode (default True).
        inline: Single-line format if True, multi-line if False (default False).
        color: ANSI color code (e.g., "\\033[92m" for green).

    Returns:
        None
    """
    reset = "\033[0m"
    apply_color = (
        (lambda text: f"{color}{text}{reset}") if color else (lambda text: text)
    )

    # Always add a newline to the beginning of the section.
    logger.info("")

    if inline:
        # Prepare inline title.
        title_str = f" {title} "
        remaining = width - len(title_str)
        if remaining < 0:
            remaining = 0
        left = border_char * (remaining // 2)
        right = border_char * (remaining - len(left))
        logger.info(apply_color(f"{left}{title_str}{right}"))
    else:
        # Multi-line section style.
        border = border_char * width
        title_line = f"{title:^{width}}" if center else title
        logger.info(apply_color(border))
        logger.info(apply_color(title_line))
        logger.info(apply_color(border))


def validate_required_files(required_files: Dict[str, Path]) -> None:
    """
    Validate presence of required files and print status.

    Args:
        required_files: File descriptions mapped to filesystem paths.

    Returns:
        None

    Exits:
        Code 1 if any required file is missing.
    """
    print_section("Required files")
    all_valid = True

    for name, path in required_files.items():
        status = STATUS_PASS if path.is_file() else STATUS_FAIL
        logger.info(f"{status} {name}: {path}")
        if not path.is_file():
            all_valid = False

    if not all_valid:
        _log_error_and_exit("One or more required files are missing.")


def setup_environment(artifacts_dir: Path) -> Dict[str, str]:
    """
    Configure environment variables for test suite execution.

    Args:
        artifacts_dir: Path to artifacts directory with binaries and libraries.

    Returns:
        Updated environment variables dictionary.
    """
    print_section("Setting up environment variables")

    # Copy current environment.
    env_vars = os.environ.copy()

    # Add ROCgdb and LLVM binaries to PATH.
    env_vars["PATH"] = f"{artifacts_dir}/bin:{artifacts_dir}/llvm/bin:" + env_vars.get(
        "PATH", ""
    )
    logger.info(f"PATH: {artifacts_dir}/bin:{artifacts_dir}/llvm/bin")

    # Add ROCgdb and LLVM libraries to LD_LIBRARY_PATH.
    env_vars["LD_LIBRARY_PATH"] = (
        f"{artifacts_dir}/lib:{artifacts_dir}/llvm/lib:"
        + env_vars.get("LD_LIBRARY_PATH", "")
    )
    logger.info(f"LD_LIBRARY_PATH: {artifacts_dir}/lib:{artifacts_dir}/llvm/lib")

    # Configure GPU core dump pattern.
    env_vars["HSA_COREDUMP_PATTERN"] = "gpucore.%p"
    logger.info(f"HSA_COREDUMP_PATTERN: {env_vars['HSA_COREDUMP_PATTERN']}")

    # Check if we are running within a github actions context, where a
    # non-system version of Python is being used. If so, we have
    # pythonLocation set to the base location of the Python interpreter.
    if "pythonLocation" in env_vars:
        # Set PYTHONHOME so rocgdb can initialize Python properly.
        env_vars["PYTHONHOME"] = env_vars["pythonLocation"]
        logger.info(
            f"Found 'pythonLocation'. Setting 'PYTHONHOME' to {env_vars['pythonLocation']}"
        )
    else:
        logger.info("'pythonLocation' is not set. Using system defaults.")

    # Apply settings to the current process.
    os.environ.update(env_vars)

    return env_vars


def check_executables(executables: List[str]) -> None:
    """
    Verify required executables are in system PATH.

    Args:
        executables: List of executable names to check.

    Returns:
        None

    Exits:
        Code 1 if any executable is missing.
    """
    print_section("Required executables")
    missing = []

    for exe in executables:
        path = shutil.which(exe)
        if path:
            logger.info(f"{STATUS_PASS} {exe:15} found at: {path}")
        else:
            missing.append(exe)
            logger.info(f"{STATUS_FAIL} {exe:15} NOT found on PATH")

    if missing:
        _log_error_and_exit(f"Missing {len(missing)} executables required for testing.")


def set_core_file_limit() -> bool:
    """
    Set system core file size limit to unlimited.

    Returns:
        True if successfully set to unlimited, False otherwise.
    """
    print_section("Core file size")

    try:
        resource.setrlimit(
            resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)

        if soft == resource.RLIM_INFINITY:
            logger.info(f"{STATUS_PASS} Core file size limit set to unlimited")
            return True
        else:
            logger.info(f"   Warning: Core file size limit is {soft}, not unlimited")
            logger.info("   Core file tests may not execute properly")
            return False

    except (ValueError, Exception) as e:
        logger.info(f"{STATUS_FAIL} Error: Unable to set core file size limit: {e}")
        logger.info("   Warning: Core file tests will not be executed")
        return False


def cleanup_test_suite(
    test_suite_dir: Path, env_vars: Dict[str, str], quiet: bool = False
) -> None:
    """
    Clean build artifacts and recreate site.exp configuration.

    Args:
        test_suite_dir: Path to test suite directory.
        env_vars: Environment variables for cleanup commands.
        quiet: Suppress command output if True (default False).

    Returns:
        None

    Exits:
        Code 1 if cleanup fails.
    """
    logger.info("Cleaning test suite directory...")
    cmd = ["make", "clean"]
    logger.info(f"Executing cleanup: {shlex.join(cmd)}")
    _run_command(
        cmd,
        cwd=test_suite_dir,
        env=env_vars,
        capture_output=quiet,
        error_msg="Failed to clean test directory",
    )

    logger.info("Removing old site.exp and site.bak...")
    for filename in ["site.exp", "site.bak"]:
        (test_suite_dir / filename).unlink(missing_ok=True)

    logger.info("Creating site.exp...")
    cmd = ["make", "site.exp"]
    logger.info(f"Executing cleanup: {shlex.join(cmd)}")
    _run_command(
        cmd,
        cwd=test_suite_dir,
        env=env_vars,
        capture_output=quiet,
        error_msg="Failed to create site.exp",
    )


def configure_test_suite(
    test_suite_dir: Path, env_vars: Dict[str, str], quiet: bool = False
) -> None:
    """
    Run test suite configuration script.

    Args:
        test_suite_dir: Path to test suite directory with configure script.
        env_vars: Environment variables for configuration.
        quiet: Suppress command output if True (default False).

    Returns:
        None

    Exits:
        Code 1 if configuration fails.
    """
    configure_script = test_suite_dir / "configure"
    cmd = ["sh", str(configure_script)]
    logger.info(f"Executing: {shlex.join(cmd)}")
    _run_command(
        cmd,
        cwd=test_suite_dir,
        env=env_vars,
        capture_output=quiet,
        error_msg="Test suite configuration failed",
    )
    cleanup_test_suite(test_suite_dir, env_vars, quiet)


def set_test_timeout(test_suite_dir: Path, timeout_value: int) -> None:
    """
    Set gdb_test_timeout in site.exp configuration file.

    Args:
        test_suite_dir: Path to test suite directory with site.exp file.
        timeout_value: Timeout in seconds for test runs.

    Returns:
        None

    Exits:
        Code 1 if site.exp is missing or cannot be written.
    """
    site_exp_file = test_suite_dir / "site.exp"
    try:
        with open(site_exp_file, "a") as f:
            f.write(f"\nset gdb_test_timeout {timeout_value}\n")
        logger.info(
            f"{STATUS_PASS} Successfully set gdb_test_timeout to {timeout_value} in {site_exp_file}"
        )
    except (FileNotFoundError, IOError) as e:
        _log_error_and_exit(f"Failed to write timeout to {site_exp_file}: {e}")


def _extract_test_files(test_names: List[str]) -> List[str]:
    """
    Extract unique test file paths from full test names.

    Args:
        test_names: Full test names (e.g., "gdb.rocm/foo.exp: description").

    Returns:
        Sorted list of unique test file paths without descriptions.
    """
    return sorted(set(_extract_test_file_from_name(name) for name in test_names))


def expand_test_paths(test_list: List[str], testsuite_dir: Path) -> str:
    """
    Expand test paths to space-separated .exp files, validating existence.

    Args:
        test_list: Test paths (individual .exp files or directories).
        testsuite_dir: Base directory to resolve relative paths.

    Returns:
        Space-separated list of .exp test file paths.

    Exits:
        Code 1 if any test file is missing.
    """
    expanded_tests = []

    for test_path in test_list:
        if not test_path.endswith(".exp"):
            pattern = f"{test_path}/*.exp"
            matches = glob.glob(pattern, root_dir=str(testsuite_dir))
            if matches:
                expanded_tests.extend(matches)
                logger.info(
                    f"Expanded directory '{test_path}' to {len(matches)} test files"
                )
            else:
                logging.warning(f"No test files found matching pattern: {pattern}")
        else:
            expanded_tests.append(test_path)
            logger.info(f"Added test file: {test_path}")

    # Verify that all expanded files exist.
    for file_path in expanded_tests:
        abs_path = testsuite_dir / file_path
        if not abs_path.is_file():
            _log_error_and_exit(f"Missing or invalid test file: {file_path}")

    return " ".join(expanded_tests)


def print_env_variables() -> None:
    """
    Print all environment variables for debugging.

    Returns:
        None
    """
    print_section("Environment Variables")
    for key, value in os.environ.items():
        logger.info(f"{key}: {value}")


def validate_path(path: Path, is_dir: bool = False, is_file: bool = False) -> None:
    """
    Validate path exists and matches expected type.

    Args:
        path: Path to validate.
        is_dir: Require directory if True (default False).
        is_file: Require file if True (default False).

    Returns:
        None

    Exits:
        Code 1 if path doesn't exist or is wrong type.
    """
    resolved_path = path.resolve()

    if is_dir and not resolved_path.is_dir():
        _log_error_and_exit(f"Directory does not exist: {resolved_path}")
    if is_file and not resolved_path.is_file():
        _log_error_and_exit(f"File does not exist: {resolved_path}")


def validate_rocgdb(rocgdb_bin: Path) -> None:
    """
    Validate ROCgdb executable can run successfully.

    Args:
        rocgdb_bin: Path to ROCgdb executable.

    Returns:
        None

    Raises:
        RuntimeError: If ROCgdb fails to execute.
    """
    print_section("ROCgdb launcher data")
    env = os.environ.copy()
    env["ROCGDB_WRAPPER_DEBUG"] = "1"

    try:
        # First invoke the rocgdb launcher in debug mode.
        print_section("ROCgdb launcher start", border_char="-", inline=True)
        result = _run_command(
            [str(rocgdb_bin), "--version"], env=env, capture_output=True
        )

        for line in result.stderr.splitlines():
            logger.info(line)
        logger.info("ROCgdb launcher ran successfully.")

        # Now validate that we can launch rocgdb at all.
        print_section("ROCgdb executable start", border_char="-", inline=True)
        result = _run_command([str(rocgdb_bin), "--version"], capture_output=True)

        for line in result.stdout.splitlines():
            logger.info(line)
        logger.info("ROCgdb executable ran successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run ROCgdb: {e.stderr}")
        raise RuntimeError("ROCgdb did not run successfully.")

    # Validate internal Python environment.
    print_section("ROCgdb internal python data", border_char="-", inline=True)

    py_cmd = (
        "import sys, os, sysconfig, gdb; "
        "v = sys.version_info; "
        "version = f'{v.major}.{v.minor}.{v.micro}'; "
        "supports = bool(sysconfig.get_config_var('Py_ENABLE_SHARED')); "
        "lib_dir = sysconfig.get_config_var('LIBDIR'); "
        "ld_lib = sysconfig.get_config_var('LDLIBRARY'); "
        "lib_path = os.path.join(lib_dir, ld_lib) if (supports and lib_dir and ld_lib) else 'N/A'; "
        "print(f'Executable: {sys.executable}'); "
        "print(f'Version: {version}'); "
        "print(f'Supports libpython: {supports}'); "
        "print(f'libpython Path: {lib_path}'); "
        'print(f\'libpython Version: {sysconfig.get_config_var("VERSION") or "N/A"}\'); '
        "print(f'GDB python modules path: {gdb.PYTHONDIR}'); "
    )

    try:
        result = _run_command(
            [str(rocgdb_bin), "-batch", "-ex", f"python {py_cmd}"],
            capture_output=True,
            check=False,
        )

        if result.returncode == 0:
            for line in result.stdout.splitlines():
                logger.info(line)
            logger.info("ROCgdb internal python validation successful.")
        elif "Python scripting is not supported in this copy of GDB" in result.stderr:
            logger.warning(
                "Python scripting is not supported in this copy of GDB. Testing will proceed without Python support."
            )
        else:
            logger.error(f"Failed ROCgdb Python validation: {result.stderr}")
            raise RuntimeError("ROCgdb did not run successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed ROCgdb Python validation: {e.stderr}")
        raise RuntimeError("ROCgdb did not run successfully.")


def run_tests(
    test_suite_dir: Path,
    rocgdb_bin: Path,
    env_vars: Dict[str, str],
    tests: str,
    cc: str,
    cxx: str,
    fc: str,
    compiler_label: str,
    test_results: "TestResults",
    args: argparse.Namespace,
) -> None:
    """
    Run ROCgdb test suite with retry logic for failed tests.

    Args:
        test_suite_dir: Path to test suite directory.
        rocgdb_bin: Path to ROCgdb binary.
        env_vars: Environment variables for test execution.
        tests: Space-separated test file names.
        cc: C compiler command.
        cxx: C++ compiler command.
        fc: Fortran compiler command.
        compiler_label: Compiler identifier (e.g., "GCC").
        test_results: TestResults object for storing results.
        args: Parsed command-line arguments.

    Returns:
        None
    """
    max_iterations = 1 + args.max_failed_retries
    current_tests = tests

    for iteration in range(1, max_iterations + 1):
        print_section(f"ROCgdb tests - {compiler_label} - Iteration {iteration}")
        logger.info(f"Number of tests to run: {len(current_tests.split())}")

        # Track which tests we're running in this iteration (for flaky detection).
        tests_in_iteration = set(current_tests.split())

        configure_test_suite(test_suite_dir, env_vars, args.quiet)

        # If re-running due to timeout failures, apply timeout.
        if iteration != 1 and test_results.test_data[compiler_label].get("TIMEOUT"):
            set_test_timeout(test_suite_dir, args.timeout)

        start_time = time.perf_counter()

        runtestflags_str = _build_runtestflags(
            rocgdb_bin, cc, cxx, fc, args.optimization, args.runtestflags
        )

        cmd = [
            "make",
            "check",
            f"RUNTESTFLAGS={runtestflags_str}",
            f"TESTS={current_tests}",
        ]
        if args.parallel:
            cmd += ["FORCE_PARALLEL=1", "-j"]

        logger.info(
            f"Executing tests with {compiler_label} - Iteration {iteration}: {shlex.join(cmd)}"
        )
        _run_command(
            cmd,
            cwd=test_suite_dir,
            env=env_vars,
            capture_output=args.quiet,
            check=False,
        )

        duration = time.perf_counter() - start_time

        print_section(
            f"Tests with {compiler_label} - Iteration {iteration} completed in {duration:.4f} seconds."
        )

        # Parse test results from gdb.sum.
        results_file = f"{test_suite_dir}/gdb.sum"
        test_results.update_results(compiler_label, current_tests, results_file)

        failed_tests = test_results.get_failed_tests(compiler_label)

        # Detect flaky tests: tests that were failing in previous iteration but now pass.
        if iteration > 1:
            failed_test_files = set(_extract_test_files(failed_tests))
            newly_passed = tests_in_iteration - failed_test_files
            if newly_passed:
                test_results.mark_flaky_tests(compiler_label, newly_passed)

        if not failed_tests:
            logger.info(
                f"{STATUS_PASS} No failing tests for {compiler_label}. Stopping iterations."
            )
            break
        elif iteration < max_iterations:
            # Only rerun failed tests next time.
            current_tests = " ".join(_extract_test_files(failed_tests))
            logger.info(
                f"{STATUS_FAIL}  {len(failed_tests)} failing test(s) found for {compiler_label}. "
                f"Proceeding to iteration {iteration + 1} with only failed tests."
            )
        else:
            logger.info(
                f"{STATUS_FAIL}  {len(failed_tests)} failing test(s) remain for {compiler_label} "
                f"after {max_iterations} iterations."
            )

            # Print the contents of gdb.log in a visually uncluttered way.
            if args.dump_failed_test_log:
                gdb_log_file = test_suite_dir / "gdb.log"
                print_section("Contents of gdb.log")
                for line in _read_file_lines(gdb_log_file, "during gdb.log dump"):
                    logger.info(line)


def main() -> None:
    """
    Main entry point for ROCgdb test suite runner.

    Orchestrates argument parsing, environment setup, test execution,
    and result reporting for GCC and LLVM compilers.

    Returns:
        None

    Exits:
        Code 0 if all tests pass or are expected failures, Code 1 otherwise.
    """
    args = parse_arguments()

    start_time = time.perf_counter()

    # Determine paths either from arguments or environment variables.
    if args.testsuite_dir is None:
        the_rock_bin_dir = validate_env_var("THEROCK_BIN_DIR")
        artifacts_dir = validate_env_var("OUTPUT_ARTIFACTS_DIR")
        rocgdb_bin = the_rock_bin_dir / "rocgdb"
        rocgdb_testsuite_dir = artifacts_dir / "tests" / "rocgdb" / "gdb" / "testsuite"
    else:
        rocgdb_testsuite_dir = args.testsuite_dir
        rocgdb_bin = args.rocgdb_bin
        artifacts_dir = rocgdb_testsuite_dir.parent.parent.parent

    rocgdb_configure_script = rocgdb_testsuite_dir / "configure"

    # Print env variables.
    print_env_variables()

    # Print Python information.
    print_python_info()

    # Show configuration summary.
    print_configuration(rocgdb_bin, rocgdb_testsuite_dir, rocgdb_configure_script, args)

    # Validate critical files exist.
    validate_required_files(
        {
            "ROCgdb launcher": rocgdb_bin,
            "ROCgdb configure script": rocgdb_configure_script,
        }
    )

    if platform.system() == "Linux":
        # Set core dump file limit.
        set_core_file_limit()

    # Prepare environment for tests.
    env_vars = setup_environment(artifacts_dir)

    # Validate that we can run rocgdb.
    validate_rocgdb(rocgdb_bin)

    # Verify executables presence.
    check_executables(
        ["hipcc", "gcc", "g++", "gfortran", "clang", "clang++", "flang", "runtest"]
    )

    print_section("Expanding test paths")
    tests = expand_test_paths(args.tests, rocgdb_testsuite_dir)

    if not tests:
        _log_error_and_exit("No test files found")

    # Compiler configurations.
    compilers = [
        ("gcc", "g++", "gfortran", "GCC"),
        ("clang", "clang++", "flang", "LLVM"),
    ]

    # Initialize test result tracking.
    test_results = TestResults()
    test_results.group_results = args.group_results

    # Run tests for all configured compilers.
    for cc, cxx, fc, compiler_label in compilers:
        run_tests(
            rocgdb_testsuite_dir,
            rocgdb_bin,
            env_vars,
            tests,
            cc,
            cxx,
            fc,
            compiler_label,
            test_results,
            args,
        )

    # Final summaries.
    test_results.print_all_summaries()
    overall_pass = test_results.print_final_status(XFAILED_TESTS, args.no_xfail)

    duration = time.perf_counter() - start_time
    logger.info(f"Total test run duration: {duration:.4f} seconds.")

    sys.exit(0 if overall_pass else 1)


def validate_env_var(var_name: str) -> Path:
    """
    Validate and resolve environment variable to Path.

    Args:
        var_name: Environment variable name.

    Returns:
        Resolved Path from environment variable.

    Exits:
        Code 1 if variable is not set.
    """
    value = os.getenv(var_name)
    if value is None:
        _log_error_and_exit(f"{var_name} environment variable is not set")
    return Path(value).resolve()


def _build_runtestflags(
    rocgdb_bin: Path,
    cc: str,
    cxx: str,
    fc: str,
    optimization: str,
    runtestflags: str,
) -> str:
    """
    Build RUNTESTFLAGS string for test execution.

    Args:
        rocgdb_bin: Path to ROCgdb binary.
        cc: C compiler command.
        cxx: C++ compiler command.
        fc: Fortran compiler command.
        optimization: Optimization flags.
        runtestflags: Additional flags.

    Returns:
        Complete RUNTESTFLAGS string.
    """
    parts = [
        f"GDB={rocgdb_bin}",
        f"CC_FOR_TARGET={cc}",
        f"CXX_FOR_TARGET={cxx}",
        f"F77_FOR_TARGET={fc}",
        f"F90_FOR_TARGET={fc}",
    ]
    if optimization:
        parts.append(f"CFLAGS_FOR_TARGET={optimization}")
    if runtestflags:
        parts.append(runtestflags)

    return " ".join(parts)


def print_python_info() -> None:
    """
    Print Python interpreter information including version and libpython details.

    Returns:
        None
    """
    print_section("Python information")

    version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    supports_libpython = bool(sysconfig.get_config_var("Py_ENABLE_SHARED"))

    lib_path = "N/A"
    lib_version = "N/A"

    if supports_libpython:
        lib_dir = sysconfig.get_config_var("LIBDIR")
        ld_library = sysconfig.get_config_var("LDLIBRARY")
        if lib_dir and ld_library:
            lib_path = os.path.join(lib_dir, ld_library)
        lib_version = str(sysconfig.get_config_var("VERSION") or "N/A")

    logger.info(f"Executable: {sys.executable}")
    logger.info(f"Version: {version}")

    # Determine support status.
    if supports_libpython:
        if lib_path != "N/A" and os.path.exists(lib_path):
            logger.info("Supports libpython: Yes")
        else:
            logger.info("Supports libpython: Supported but libpython is missing")
    else:
        logger.info("Supports libpython: No")

    # Print library path with missing indicator if needed.
    if lib_path != "N/A" and not os.path.exists(lib_path):
        logger.info(f"libpython Path: {lib_path} (missing)")
    else:
        logger.info(f"libpython Path: {lib_path}")

    logger.info(f"libpython Version: {lib_version}")


def print_configuration(
    rocgdb_bin: Path,
    testsuite_dir: Path,
    configure_script: Path,
    args: argparse.Namespace,
) -> None:
    """
    Display the ROCgdb test configuration in a formatted table.

    Args:
        rocgdb_bin (Path):
            Path to the ROCgdb binary.
        testsuite_dir (Path):
            Path to the test suite directory.
        configure_script (Path):
            Path to the configure script for the test suite.
        args (argparse.Namespace):
            Parsed command-line arguments containing additional configuration values.

    Returns:
        None

    Notes:
        - Prints an ASCII table summarizing binary location, directories, test selection,
          and execution settings.
        - Fields come directly from parsed CLI arguments.
    """

    print_section("ROCgdb Test Suite Configuration")
    logger.info(f"  OS:                   {platform.system()}")
    logger.info(f"  ROCgdb Binary:        {rocgdb_bin}")
    logger.info(f"  Testsuite Directory:  {testsuite_dir}")
    logger.info(f"  Configure Script:     {configure_script}")
    logger.info(f"  Tests:                {' '.join(args.tests)}")
    logger.info(f"  Parallel Execution:   {'Enabled' if args.parallel else 'Disabled'}")
    logger.info(f"  Use FAIL ignore list: {'Not using' if args.no_xfail else 'Using'}")
    logger.info(
        f"  Group Results:        {'Enabled' if args.group_results else 'Disabled'}"
    )
    logger.info(f"  Timeout Value:        {args.timeout} seconds")
    logger.info(f"  Max Failed Retries:   {args.max_failed_retries}")
    logger.info(
        f"  Optimization:         {args.optimization if args.optimization else 'None'}"
    )
    logger.info(
        f"  Additional Runtest Flags: {args.runtestflags if args.runtestflags else 'None'}"
    )


if __name__ == "__main__":
    main()
