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
        "gdb.dwarf2/dw2-cp-infcall-ref-static.exp",
        # Testcase expects ASLR OFF and amd-llvm generates
        # PIE executables by default.  Fix being pursued upstream.
        # REMOVE when the fix makes its way to our branch.
        "gdb.dwarf2/dw2-entry-value.exp",
        "gdb.dwarf2/dw2-inline-param.exp",
        "gdb.dwarf2/dw2-param-error.exp",
        "gdb.dwarf2/dw2-skip-prologue.exp",
        "gdb.dwarf2/dw2-undefined-ret-addr.exp",
        "gdb.dwarf2/dw2-unresolved.exp",
        "gdb.dwarf2/fission-base.exp",
        "gdb.dwarf2/fission-dw-form-strx.exp",
        "gdb.dwarf2/pr13961.exp",
    ],
}


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for running the ROCgdb test suite.

    Returns:
        argparse.Namespace: Parsed and validated arguments.

    Exits:
        - System exit with code 1 if required arguments are missing or invalid.

    Notes:
        - Ensures `--testsuite-dir` and `--rocgdb-bin` are provided together or not at all.
        - Validates directories/files when provided.
        - Restricts timeout value to a maximum of 600 seconds.
        - Enforces non-negative `max_failed_retries`.
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
        "--install-packages",
        action="store_true",
        help="Install required packages before running tests. Default is off.",
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
        logger.info(
            "[X] Error: Both --testsuite-dir and --rocgdb-bin must be provided together, or neither."
        )
        sys.exit(1)

    # Validate paths if provided.
    if args.testsuite_dir is not None:
        validate_path(args.testsuite_dir, is_dir=True)
        validate_path(args.rocgdb_bin, is_file=True)

    # Validate timeout value.
    if not (0 < args.timeout <= 600):
        logger.info(
            f"[X] Error: Timeout must be between 1 and 600 seconds. Got {args.timeout}."
        )
        sys.exit(1)

    # Validate max_failed_retries value.
    if args.max_failed_retries < 0:
        logger.info(
            f"[X] Error: Max failed retries must be non-negative. Got {args.max_failed_retries}."
        )
        sys.exit(1)

    return args


class TestResults:
    """Class to store and manage test results across multiple compiler runs."""

    def __init__(self) -> None:
        """Initialize test results storage."""
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

        # Categories recognized by the test framework.
        self.categories = [
            "PASS",
            "FAIL",
            "ERROR",
            "UNRESOLVED",
            "TIMEOUT",
            "UNTESTED",
            "UNSUPPORTED",
            "XFAIL",
            "KFAIL",
        ]

    def cleanup_old_entries(self, compiler_label: str, tests: str) -> None:
        """
        Remove stale test results and failure tracking entries for given tests
        under the specified compiler label.

        This is mainly used during retry runs, ensuring that any previous
        results for the same tests do not persist and cause duplicate or
        outdated output.

        Args:
            compiler_label (str):
                The identifier for the compiler/toolchain used in this test run.
                This should match a key in `self.test_data` and `self.failed_tests`.
                Example: "GCC", "LLVM".

            tests (str):
                A space-separated string containing test file names
                (e.g. "gdb.rocm/simple.exp gdb.rocm/device-interrupt.exp").
                Each will be removed from stored results for the given compiler,
                across all categories.

        Behavior:
            - If no previous results exist for the given compiler label, logs a
              message and does nothing (first run case).
            - If results are found (retry run case):
                * Clears the set of failed tests for this compiler label.
                * Iterates through each provided test name and removes its entries
                  from every test status category in `self.test_data` for the given
                  compiler label.

        Notes:
            - This prevents duplication by ensuring that fresh test results replace
              old ones.
            - The `tests` argument is split by whitespace; individual test names
              should not contain spaces.
        """
        # Case: first run for this compiler — nothing to clean.
        if not self.test_data[compiler_label]:
            return

        logger.info(f"Retry run. Cleaning dictionary entries for {tests}.")

        # Clear the failed tests set for this compiler label.
        del self.failed_tests[compiler_label]

        # Clear tests entries from the main test results database for this
        # compiler label.
        for test in tests.split():
            for category in self.categories:
                # Only attempt removal if category exists for this compiler.
                category_results = self.test_data[compiler_label].get(category)
                if category_results and test in category_results:
                    del category_results[test]

    def extract_errors(self, results_file: str) -> Dict[str, List[str]]:
        """
        Parse a GDB test results file and extract ERROR messages grouped by test file.

        This function scans the file for lines indicating the start of a test case
        (lines starting with 'Running' and containing a `.exp` file) and for lines
        starting with `"ERROR:"`.

        For each ERROR line found, it associates it with the most recently detected
        test case. Test file paths are shortened to only the last two path components.

        Example:
            '/path/to/tests/gdb.base/break.exp' -> 'gdb.base/break.exp'

        Args:
            results_file (str):
                Path to the GDB test results file.

        Returns:
            Dict[str, List[str]]:
                A mapping of test file names (shortened to two levels) to a list
                of extracted error messages prefixed with the test file name.
        """
        errors_dict: Dict[str, List[str]] = defaultdict(list)
        current_test_file: str | None = None

        try:
            with open(results_file, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()

                    # Detect the start of a test case.
                    match = re.search(r"Running\s+(\S+\.exp)", stripped)
                    if match:
                        # Remove trailing ':' or '.'.
                        full_path = match.group(1).rstrip(":.")
                        segments = os.path.normpath(full_path).split(os.path.sep)
                        # Keep last two directory levels.
                        current_test_file = "/".join(segments[-2:])

                    # Detect ERROR lines.
                    if stripped.startswith("ERROR:"):
                        msg = stripped[len("ERROR:") :].strip()
                        test_file = current_test_file or "UNKNOWN"
                        errors_dict[test_file].append(f"{test_file}: {msg}")

        except FileNotFoundError:
            logger.info(f"[X] Error: {results_file} not found.")

        return dict(errors_dict)

    def update_results(
        self, compiler_label: str, tests: str, results_file: str
    ) -> None:
        """
        Refresh test results for a given compiler from a results file.

        Args:
            compiler_label (str):
                Name of the compiler/toolchain (e.g. "GCC", "LLVM").
            tests (str):
                Space-separated list of test file names to clear before updating.
            results_file (str):
                Path to a DejaGnu-style results file, with lines like:
                STATUS: test_description

        Returns:
            None

        Workflow:
            1. Remove old entries for `tests` under `compiler_label`.
            2. Parse `results_file` for valid status lines.
            3. Store results in `self.test_data` by status and test file.
            4. Record TIMEOUT tests separately if "(timeout)" appears.
            5. Update `self.failed_tests` for FAIL/ERROR/UNRESOLVED or TIMEOUT cases.

        Notes:
            - Old entries are removed via `cleanup_old_entries`.
            - Ignores empty lines.
            - Prints an error if the results file is missing.
        """
        self.cleanup_old_entries(compiler_label, tests)

        # Regex to match test result lines.
        result_regex = re.compile(
            r"^(PASS|FAIL|XFAIL|UNTESTED|UNSUPPORTED|KFAIL|UNRESOLVED): (.+)"
        )

        # Regex to detect timeout in test description.
        timeout_regex = re.compile(r"\(timeout\)")

        try:
            with open(results_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines.
                        continue
                    match = result_regex.match(line)
                    if match:
                        status, test_name = match.groups()
                        test_file = self.extract_test_file(test_name)
                        is_timeout = timeout_regex.search(test_name)

                        # Add the entry to our test results database.
                        self.test_data[compiler_label][status][test_file].append(
                            test_name
                        )

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
        except FileNotFoundError:
            logger.info(f"[X] Error: {results_file} not found.")

    def extract_test_file(self, test_name: str) -> str:
        """
        Extract the test file path from a full test name.

        Args:
            test_name (str):
                Full test name, e.g. "gdb.rocm/foo.exp: test description".

        Returns:
            str: Test file path without the description, e.g. "gdb.rocm/foo.exp".
        """
        if ": " in test_name:
            return test_name.split(": ", 1)[0]
        return test_name

    def get_failed_tests(self, compiler_label: str) -> List[str]:
        """
        Retrieve the list of currently failed tests for a given compiler.

        Args:
            compiler_label (str):
                Name or label identifying the compiler/toolchain.

        Returns:
            List[str]: Test file paths for tests that are currently marked as failed.
        """
        return list(self.failed_tests[compiler_label])

    def print_all_summaries(self) -> None:
        """
        Print all stored test summaries, grouped by compiler label.

        Args:
            None

        Returns:
            None

        Notes:
            - If no test data is available, prints a message and exits.
            - Summaries are produced by calling `_print_summary` for each compiler.
            - Output is displayed in a combined analysis format.
        """
        if not self.test_data:
            logger.info("\n[X] No test results to display.")
            return

        print_section("COMBINED GDB TESTSUITE ANALYSIS")

        for compiler_label in sorted(self.test_data.keys()):
            self._print_summary(compiler_label, self.test_data[compiler_label])

        self.print_comparison()

    def _print_summary(self, compiler_label: str, details: Dict[str, Set[str]]) -> None:
        """
        Print a single compiler's test summary, grouped by status category.

        Args:
            compiler_label (str):
                Name or label identifying the compiler/toolchain.
            details (Dict[str, Set[str]]):
                Mapping from status category (e.g. "PASS", "FAIL") to a dictionary
                of test files → sets/lists of test descriptions.

        Returns:
            None

        Notes:
            - Displays the number of tests per category and relevant indicators.
            - For non-PASS categories, optionally groups results if
              `self.group_results` is True, otherwise prints each test.
            - Uses `categories_display` to define emoji/prefix indicators in output.
        """
        # Define display properties for each test status category.
        categories_display = {
            "PASS": (None, "[✓]"),
            "FAIL": ("", "[X]"),
            "ERROR": ("", "[X]"),
            "UNRESOLVED": ("", "[X]"),
            "TIMEOUT": ("", " "),
            "UNTESTED": ("", " "),
            "UNSUPPORTED": ("", " "),
            "XFAIL": ("", " "),
            "KFAIL": ("", " "),
        }

        print_section(f"{compiler_label}", border_char="-", inline=True)
        for cat, (prefix_indicator, suffix_indicator) in categories_display.items():
            # Count tests in this category for the given compiler.
            count = sum(len(tests) for tests in details[cat].values())

            header = f"  {cat}: {count}"
            if prefix_indicator:
                header = f"  {prefix_indicator} {cat}: {count}"
            if suffix_indicator and count > 0:
                header = f"{header} {suffix_indicator}"
            logger.info(header)

            # Print details for non-PASS tests.
            if count > 0 and cat != "PASS":
                if self.group_results:
                    self._print_grouped_tests(details[cat])
                else:
                    for test_file in details[cat]:
                        for test_name in details[cat][test_file]:
                            logger.info(f"      [!] {test_name}")

    def _print_grouped_tests(self, test_list: Dict[str, List[str]]) -> None:
        """
        Print tests grouped as: directory → file → descriptions.

        Args:
            test_list (Dict[str, List[str]]): Test file paths → descriptions.

        Returns:
            None
        """
        grouped: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for file in test_list.keys():
            for test_description in test_list[file]:
                test_dir, test_file = os.path.split(file)
                grouped[test_dir][test_file].append(test_description[len(file) + 2 :])

        for directory in sorted(grouped.keys()):
            logger.info(f"    [!] {directory}")
            for filename in sorted(grouped[directory].keys()):
                descriptions = grouped[directory][filename]
                if len(descriptions) == 1 and descriptions[0]:
                    logger.info(f"        [!] {filename}: {descriptions[0]}")
                elif len(descriptions) == 1:
                    logger.info(f"        [!] {filename}")
                else:
                    logger.info(f"        [!] {filename}")
                    for desc in descriptions:
                        if desc:
                            logger.info(f"            [!] {desc}")

    def print_comparison(self) -> None:
        """
        Print a summary of failing tests that are exclusive to each compiler.

        Args:
            None

        Returns:
            None

        Notes:
            - Only runs if there are results for more than one compiler.
            - An "exclusive" failing test is one that fails for a single compiler
              and passes (or is absent) for all others.
        """
        if len(self.failed_tests) <= 1:
            return

        print_section("EXCLUSIVE FAILING TESTS COMPARISON")

        # Compare each compiler's failed tests set against others.
        for compiler_label in sorted(self.failed_tests.keys()):
            exclusive_tests = self.failed_tests[compiler_label] - set.union(
                *(
                    self.failed_tests[other]
                    for other in self.failed_tests
                    if other != compiler_label
                )
            )

            print_section(f"{compiler_label}", border_char="-", inline=True)
            if exclusive_tests:
                for test in sorted(exclusive_tests):
                    logger.info(f"  [X] {test}")
            else:
                logger.info("  No exclusive failing tests.")

    def check_all_pass_or_xfailed(
        self, xfailed_tests: Dict[str, List[str]]
    ) -> Tuple[bool, Dict[str, Dict]]:
        """
        Check whether all tests either passed or failed in the expected (XFAIL) list.

        Args:
            xfailed_tests (Dict[str, List[str]]):
                Mapping of compiler labels to lists of expected failing test file paths.
                Must include a "Generic" key for failures expected across all compilers.

        Returns:
            Tuple[bool, Dict[str, Dict]]:
                - bool: True if all compilers have no unexpected failures.
                - dict: Per-compiler breakdown with:
                    * "passed": bool indicating if there were no unexpected failures.
                    * "total_failed": int count of failed tests.
                    * "unexpected_failures": list of tests that failed but weren't expected.
                    * "unused_xfails": list of expected failures that actually passed.
                    * "expected_generic": list of passed tests in the generic XFAIL set.
                    * "expected_compiler_specific": list of passed tests in compiler-specific XFAIL set.
        """
        overall_pass = True
        details = {}

        # Generic XFAIL list applies to all compilers.
        generic_xfailed = set(xfailed_tests.get("Generic", []))

        for compiler_label in self.failed_tests.keys():
            # Failed test files without descriptions.
            failed_test_files = {
                self.extract_test_file(test)
                for test in self.failed_tests[compiler_label]
            }

            # Compiler-specific XFAIL list.
            compiler_xfailed = set(xfailed_tests.get(compiler_label, []))

            # Full set of expected failures for this compiler.
            all_xfailed = generic_xfailed | compiler_xfailed

            # Tests that failed but weren't expected.
            unexpected_failures = failed_test_files - all_xfailed

            # Expected failures that actually passed.
            unused_xfails = all_xfailed - failed_test_files

            # Intersection for reporting.
            expected_generic = failed_test_files & generic_xfailed
            expected_compiler_specific = failed_test_files & compiler_xfailed

            # Compiler passes if there are no unexpected failures.
            compiler_pass = len(unexpected_failures) == 0
            overall_pass = overall_pass and compiler_pass

            details[compiler_label] = {
                "passed": compiler_pass,
                "total_failed": len(failed_test_files),
                "unexpected_failures": sorted(unexpected_failures),
                "unused_xfails": sorted(unused_xfails),
                "expected_generic": sorted(expected_generic),
                "expected_compiler_specific": sorted(expected_compiler_specific),
            }

        return overall_pass, details

    def print_final_status(
        self, xfailed_tests: Dict[str, List[str]], no_xfail: bool
    ) -> bool:
        """
        Print the final PASS/FAIL status based on test results.

        Args:
            xfailed_tests: Dictionary mapping compiler labels to expected failures.
                          Must include a "Generic" key for common expected failures.

            no_xfail: If true, do not use the ignore lists. Otherwise apply the
                      ignore lists.

        Returns:
            bool: True if overall status is PASS, False otherwise.
        """

        # If we're skipping the ignore lists, just clear the lists here.
        if no_xfail:
            xfailed_tests.clear()

        overall_pass, details = self.check_all_pass_or_xfailed(xfailed_tests)

        print_section("FINAL TEST STATUS")

        for compiler_label in sorted(details.keys()):
            detail = details[compiler_label]
            status_symbol = "[✓]" if detail["passed"] else "[X]"
            status_text = "PASS" if detail["passed"] else "FAIL"

            logger.info(f"{status_symbol} {compiler_label}: {status_text}")
            logger.info(f"   Total Failed Tests: {detail['total_failed']}")

            if detail["expected_generic"]:
                logger.info(
                    f"[X]  Ignored Failures (Generic) ({len(detail['expected_generic'])}):"
                )
                for test in detail["expected_generic"]:
                    logger.info(f"      [X] {test}")

            if detail["expected_compiler_specific"]:
                logger.info(
                    f"[X]  Ignored Failures ({compiler_label}) ({len(detail['expected_compiler_specific'])}):"
                )
                for test in detail["expected_compiler_specific"]:
                    logger.info(f"      [X] {test}")

            if detail["unexpected_failures"]:
                logger.info(
                    f"   Unexpected Failures ({len(detail['unexpected_failures'])}):"
                )
                for test in detail["unexpected_failures"]:
                    logger.info(f"      [X] {test}")
            else:
                logger.info(f"   Unexpected Failures: 0")

            if detail["unused_xfails"]:
                logger.info(
                    f"[!] Unused Ignored Failures ({len(detail['unused_xfails'])}):"
                )
                for test in detail["unused_xfails"]:
                    logger.info(f"      [!] {test}")
            logger.info("")

        overall_status = "PASS" if overall_pass else "FAIL"
        overall_symbol = "[✓]" if overall_pass else "[X]"
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
    Print a visually distinct section header for console output.

    Supports two modes:
    1. Full multi-line section:
        ==============================
              Section Title
        ==============================
    2. Inline single-line section:
        -------- Section Title --------

    Args:
        title (str):
            The text to display inside the section header.
        border_char (str, optional):
            Character used for the border line (default: "=").
        width (int, optional):
            Total width of the header including borders (default: 80).
        center (bool, optional):
            Whether to center the title text for multi-line sections (default: True).
        inline (bool, optional):
            If True, print a single-line header with title inline (default: False).
        color (str, optional):
            ANSI color escape code applied to both title and borders (default: None, no color).

            Examples:
                "\033[92m" → Green
                "\033[93m" → Yellow
                "\033[94m" → Blue
                "\033[91m" → Red
                "\033[0m" resets color

    Example:
        print_section("EXCLUSIVE FAILING TESTS COMPARISON")
        print_section("LLVM", border_char="-", inline=True)
        print_section("WARNING", border_char="!", width=50, color="\033[93m")

    Notes:
        - Works in standard terminals and logs.
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
    Validate the presence of required files and print their status.

    Args:
        required_files (Dict[str, Path]):
            Mapping of file descriptions to their filesystem paths.

    Returns:
        None

    Exits:
        System exit with code 1 if any required file is missing.

    Notes:
        - Prints a check mark [✓] for files found and a cross [X] for missing files.
        - Continues checking all files before deciding to exit.
    """
    print_section("Required files")
    all_valid = True

    for name, path in required_files.items():
        status = "[✓]" if path.is_file() else "[X]"
        logger.info(f"{status} {name}: {path}")
        if not path.is_file():
            all_valid = False

    if not all_valid:
        logger.info("[X] Error: One or more required files are missing.")
        sys.exit(1)


def install_required_packages() -> None:
    """
    Install required packages for testing using `apt`.

    Args:
        None

    Returns:
        None

    Exits:
        System exit with code 1 if package installation fails.

    Notes:
        - Installs: dejagnu, gcc, g++.
        - Uses `sudo apt install ... -y` for non-interactive installation.
        - Prints the command before executing.
    """
    logger.info("Installing dejagnu, gcc, and g++...")
    cmd = ["sudo", "apt", "install", "dejagnu", "gcc", "g++", "-y"]

    # Display the exact command to be run.
    logger.info(f"Executing: {shlex.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.info(f"[X] Error: Failed to install required packages: {e}")
        sys.exit(1)


def setup_environment(artifacts_dir: Path) -> Dict[str, str]:
    """
    Configure environment variables for running the test suite.

    Args:
        artifacts_dir (Path):
            Path to the artifacts directory containing binaries and libraries.

    Returns:
        Dict[str, str]: Updated environment variables.

    Notes:
        - Adds ROCgdb and LLVM `bin` directories to PATH.
        - Adds ROCgdb and LLVM `lib` directories to LD_LIBRARY_PATH.
        - Sets GPU core dump pattern via HSA_COREDUMP_PATTERN.
        - Updates both the returned environment mapping and `os.environ`.
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
    Verify that required executables are available in the system PATH.

    Args:
        executables (List[str]):
            Names of executables to check (e.g., ["gdb", "gcc", "dejagnu"]).

    Returns:
        None

    Exits:
        System exit with code 1 if any required executable is missing.

    Notes:
        - Prints a [✓] when an executable is found, [X] when missing.
        - Checks are performed using `shutil.which`.
    """

    print_section("Required executables")
    paths = {exe: shutil.which(exe) for exe in executables}
    missing_executables = 0

    for exe, path in paths.items():
        if path:
            logger.info(f"[✓] {exe:15} found at: {path}")
        else:
            missing_executables += 1
            logger.info(f"[X] {exe:15} NOT found on PATH")

    if missing_executables > 0:
        logger.info(
            f"[X] Error: Missing {missing_executables} executables required for testing."
        )
        sys.exit(1)


def set_core_file_limit() -> bool:
    """
    Set the system core file size limit to unlimited.

    Returns:
        bool: True if limit was successfully set to unlimited, False otherwise.

    Notes:
        - Uses the `resource` module to set RLIMIT_CORE soft and hard limits.
        - Prints warnings if unable to set unlimited or if an exception occurs.
        - A non-unlimited setting may cause core file–related tests to fail.
    """
    print_section("Core file size")

    try:
        # Set soft and hard limits to unlimited.
        resource.setrlimit(
            resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )

        # Verify if limits are applied correctly.
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
        if soft == resource.RLIM_INFINITY:
            logger.info("[✓] Core file size limit set to unlimited")
            return True
        else:
            logger.info(f"   Warning: Core file size limit is {soft}, not unlimited")
            logger.info("   Core file tests may not execute properly")
            return False

    except ValueError as ve:
        logger.info(f"[X] Error: Unable to set core file size limit: {ve}")
        logger.info("   Warning: Core file tests will not be executed")
        return False

    except Exception as e:
        logger.info(f"[X] Error: Unexpected error setting core file limit: {e}")
        logger.info("   Warning: Core file tests will not be executed")
        return False


def cleanup_test_suite(
    test_suite_dir: Path, env_vars: Dict[str, str], quiet: bool = False
) -> None:
    """
    Remove build artifacts from the test suite directory before running tests.
    Recreate the site.exp configuration file.

    Args:
        test_suite_dir (Path):
            Path to the test suite directory.
        env_vars (Dict[str, str]):
            Environment variables to use when running the cleanup command.

    Returns:
        None

    Exits:
        System exit with code 1 if cleanup fails.
    """
    logger.info("Cleaning test suite directory...")
    cmd = ["make", "clean"]
    logger.info(f"Executing cleanup: {shlex.join(cmd)}")

    try:
        subprocess.run(
            cmd, cwd=str(test_suite_dir), check=True, capture_output=quiet, env=env_vars
        )
    except subprocess.CalledProcessError as e:
        logger.info(f"[X] Error: Failed to clean test directory: {e}")
        sys.exit(1)

    logger.info("Removing old site.exp and site.bak...")

    # Files make clean does not remove.
    files_to_remove = ["site.exp", "site.bak"]

    for filename in files_to_remove:
        (test_suite_dir / filename).unlink(missing_ok=True)

    logger.info("Creating site.exp...")
    cmd = ["make", "site.exp"]
    logger.info(f"Executing cleanup: {shlex.join(cmd)}")

    try:
        subprocess.run(
            cmd, cwd=str(test_suite_dir), check=True, capture_output=quiet, env=env_vars
        )
    except subprocess.CalledProcessError as e:
        logger.info(f"[X] Error: Failed to create site.exp: {e}")
        sys.exit(1)


def configure_test_suite(
    test_suite_dir: Path, env_vars: Dict[str, str], quiet: bool = False
) -> None:
    """
    Run the configuration script for the test suite.

    Args:
        test_suite_dir (Path):
            Path to the test suite directory containing the `configure` script.
        env_vars (Dict[str, str]):
            Environment variables to use when running the configuration command.

    Returns:
        None

    Exits:
        System exit with code 1 if configuration fails.

    Notes:
        - Executes the `configure` script via `sh` in the test suite directory.
        - Uses the provided `env_vars` so configuration is environment-aware.
    """
    configure_script = test_suite_dir / "configure"
    cmd = ["sh", str(configure_script)]
    logger.info(f"Executing: {shlex.join(cmd)}")
    try:
        subprocess.run(
            cmd, cwd=str(test_suite_dir), check=True, capture_output=quiet, env=env_vars
        )
    except subprocess.CalledProcessError as e:
        logger.info(f"[X] Error: Test suite configuration failed: {e}")
        sys.exit(1)

    cleanup_test_suite(test_suite_dir, env_vars, quiet)


def set_test_timeout(test_suite_dir: Path, timeout_value: int) -> None:
    """
    Append a gdb_test_timeout setting to the `site.exp` file for the test suite.

    Args:
        test_suite_dir (Path):
            Path to the test suite directory containing the `site.exp` file.
        timeout_value (int):
            Timeout value (in seconds) to set for test runs.

    Returns:
        None

    Exits:
        System exit with code 1 if `site.exp` is missing or cannot be written.

    Notes:
        - The gdb_test_timeout is appended to the file, allowing it to override existing values.
        - File is written in append mode so prior configuration is preserved.
    """
    site_exp_file = test_suite_dir / "site.exp"
    try:
        with open(site_exp_file, "a") as f:
            f.write(f"\nset gdb_test_timeout {timeout_value}\n")
        logger.info(
            f"[✓] Successfully set gdb_test_timeout to {timeout_value} in {site_exp_file}"
        )
    except FileNotFoundError:
        logger.info(f"[X] Error: {site_exp_file} not found.")
        sys.exit(1)
    except IOError as e:
        logger.info(f"[X] Error: Failed to write to {site_exp_file}: {e}")
        sys.exit(1)


def extract_test_files(test_names: List[str]) -> List[str]:
    """
    Extract unique test file paths from a list of full test names.

    Args:
        test_names (List[str]):
            Full test names, e.g., "gdb.rocm/foo.exp: test description".

    Returns:
        List[str]: Sorted list of unique test file paths without descriptions.

    Notes:
        - Splits each test name at the first ": " to isolate the file path.
        - Removes duplicates by using a set, then returns a sorted list.
    """
    test_files = set()
    for test_name in test_names:
        if ": " in test_name:
            test_file = test_name.split(": ", 1)[0]
        else:
            test_file = test_name
        test_files.add(test_file)

    return sorted(test_files)


def expand_test_paths(test_list: List[str], testsuite_dir: Path) -> str:
    """
    Expand given test paths into a space-separated list of `.exp` test files.
    Verifies that each file exists and outputs an error message for any missing one.

    Args:
        test_list (List[str]):
            Paths to tests — may be individual `.exp` files or directories.
        testsuite_dir (Path):
            Base directory for the test suite to resolve relative paths.

    Returns:
        str: Space-separated list of `.exp` test file paths.

    Notes:
        - Directories are expanded to all `*.exp` files within them.
        - Files are kept as-is.
    """
    expanded_tests: List[str] = []

    for test_path in test_list:
        # If it's not an .exp file, treat it as a directory.
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
    final_paths: List[str] = []
    for file_path in expanded_tests:
        abs_path = testsuite_dir / file_path  # resolve relative to testsuite_dir.
        if abs_path.is_file():
            final_paths.append(file_path)
        else:
            logging.error(f"Missing or invalid test file: {file_path}")
            sys.exit(1)

    return " ".join(expanded_tests)


def print_env_variables() -> None:
    # Print all of the environment variables for debugging purposes.
    print_section("Environment Variables")
    for key, value in os.environ.items():
        logger.info(f"{key}: {value}")


def validate_path(path: Path, is_dir: bool = False, is_file: bool = False) -> None:
    """
    Validate that the given path exists and matches the expected type.

    Args:
        path (Path):
            Path to validate.
        is_dir (bool, optional):
            If True, ensure the path is an existing directory.
        is_file (bool, optional):
            If True, ensure the path is an existing file.

    Returns:
        None

    Exits:
        System exit with code 1 if the path does not exist or is of the wrong type.

    Notes:
        - Path is resolved to its absolute form before validation.
        - Both `is_dir` and `is_file` can be False, in which case only existence is checked.
    """
    try:
        resolved_path = path.resolve()

        if is_dir and not resolved_path.is_dir():
            raise ValueError(f"Directory does not exist: {resolved_path}")
        if is_file and not resolved_path.is_file():
            raise ValueError(f"File does not exist: {resolved_path}")

    except Exception as e:
        logger.info(f"[X] Error: {e}")
        sys.exit(1)


def validate_rocgdb(rocgdb_bin: Path) -> None:
    """
    Validate that ROCgdb can be executed correctly.

    Args:
        rocgdb_bin (Path): Path to the ROCgdb executable.

    Returns:
        None

    Raises:
        RuntimeError: If ROCgdb fails to run successfully.
    """

    print_section("ROCgdb launcher data")
    env = os.environ.copy()
    env["ROCGDB_WRAPPER_DEBUG"] = "1"
    try:
        # First invoke the rocgdb launcher in debug mode so we have an
        # overview of what python versions we have and what rocgdb executable
        # we are picking up.
        print_section("ROCgdb launcher start", border_char="-", inline=True)
        result = subprocess.run(
            [rocgdb_bin, "--version"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        # Log each line of stdout separately.
        for line in result.stderr.splitlines():
            logger.info(line)
        logger.info(f"ROCgdb launcher ran successfully.")

        # Now validate that we can launch rocgdb at all.
        print_section("ROCgdb executable start", border_char="-", inline=True)
        result = subprocess.run(
            [rocgdb_bin, "--version"], capture_output=True, text=True, check=True
        )

        # Log each line of stdout separately.
        for line in result.stdout.splitlines():
            logger.info(line)
        logger.info(f"ROCgdb executable ran successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run ROCgdb: {e.stderr}")
        raise RuntimeError("ROCgdb did not run successfully.")

    try:
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

        # Capture output to check for specific error strings.
        result = subprocess.run(
            [str(rocgdb_bin), "-batch", "-ex", f"python {py_cmd}"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Log each line of stdout separately.
            for line in result.stdout.splitlines():
                logger.info(line)

            logger.info("ROCgdb internal python validation successful.")
        else:
            # Check for the specific GDB error regarding Python support.
            if "Python scripting is not supported in this copy of GDB" in result.stderr:
                logger.warning(
                    "Python scripting is not supported in this copy of GDB. Testing will proceed without Python support."
                )
            else:
                # If it failed for another reason, treat it as a standard failure.
                result.check_returncode()

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
    Run the ROCgdb test suite for a specific compiler configuration.

    Supports retrying failed tests up to `max_failed_retries` times.
    Each retry runs only the previously failing tests.

    Args:
        test_suite_dir (Path):
            Path to the test suite directory.
        rocgdb_bin (Path):
            Path to the ROCgdb binary.
        env_vars (Dict[str, str]):
            Environment variables for the test run.
        tests (str):
            Space-separated list of test files or directories.
        cc (str):
            C compiler command.
        cxx (str):
            C++ compiler command.
        fc  (str):
            Fortran compiler command.
        compiler_label (str):
            Label for the compiler (e.g., "GCC").
        test_results (TestResults):
            Object for storing and updating test results.
        args (argparse.Namespace):
            Various user options to control behavior of the testing procedure.
    Returns:
        None

    Notes:
        - Iterations: First run includes all tests, subsequent runs only failing tests.
        - Timeout adjustments are only applied after a timeout-related failure.
        - Test results are parsed from `gdb.sum`.
        - Does not raise on test failures — continues or retries based on input parameters.
    """
    max_iterations = 1 + args.max_failed_retries
    current_tests = tests

    for iteration in range(1, max_iterations + 1):
        print_section(f"ROCgdb tests - {compiler_label} - Iteration {iteration}")
        logger.info(f"Number of tests to run: {len(current_tests.split())}")

        configure_test_suite(test_suite_dir, env_vars, args.quiet)

        # If re-running due to timeout failures, apply timeout.
        if iteration != 1 and test_results.test_data[compiler_label].get("TIMEOUT"):
            set_test_timeout(test_suite_dir, args.timeout)

        start_time = time.perf_counter()

        # Build RUNTESTFLAGS.
        runtest_parts = [
            f"GDB={rocgdb_bin}",
            f"CC_FOR_TARGET={cc}",
            f"CXX_FOR_TARGET={cxx}",
            f"F77_FOR_TARGET={fc}",
            f"F90_FOR_TARGET={fc}",
        ]
        if args.optimization:
            runtest_parts += [f"CFLAGS_FOR_TARGET={args.optimization}"]
        if args.runtestflags:
            runtest_parts.append(args.runtestflags)

        runtestflags_str = " ".join(runtest_parts)

        # Construct make command.
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
        subprocess.run(
            cmd,
            cwd=str(test_suite_dir),
            check=False,
            capture_output=args.quiet,
            env=env_vars,
        )

        duration = time.perf_counter() - start_time

        print_section(
            f"Tests with {compiler_label} - Iteration {iteration} completed in {duration:.4f} seconds."
        )

        # Parse test results from gdb.sum.
        results_file = f"{test_suite_dir}/gdb.sum"
        test_results.update_results(compiler_label, current_tests, results_file)

        failed_tests = test_results.get_failed_tests(compiler_label)
        if not failed_tests:
            print_section(
                f"[✓] No failing tests for {compiler_label}. Stopping iterations.",
                border_char="-",
                inline=True,
            )
            break
        elif iteration < max_iterations:
            # Only rerun failed tests next time.
            current_tests = " ".join(extract_test_files(failed_tests))
            print_section(
                f"[X]  {len(failed_tests)} failing test(s) found for {compiler_label}. "
                f"Proceeding to iteration {iteration + 1} with only failed tests.",
                border_char="-",
                inline=True,
            )
        else:
            print_section(
                f"[X]  {len(failed_tests)} failing test(s) remain for {compiler_label} "
                f"after {max_iterations} iterations.",
                border_char="-",
                inline=True,
            )

            # Print the contents of gdb.log in a visually uncluttered way.
            if args.dump_failed_test_log:
                gdb_log_file = test_suite_dir / "gdb.log"
                try:
                    with open(gdb_log_file, "r") as log_file:
                        print_section("Contents of gdb.log")
                        for line in log_file:
                            logger.info(line.strip())
                except FileNotFoundError:
                    logger.info(f"[X] Error: {gdb_log_file} not found.")
                except IOError as e:
                    logger.info(f"[X] Error: Failed to read {gdb_log_file}: {e}")


def main():
    """
    Main entry point for the ROCgdb test suite runner.

    Orchestrates the entire test execution process:

    1. Parse and validate command-line arguments.
    2. Determine paths from arguments or environment variables.
    3. Configure logging and display the run configuration.
    4. Validate required files and environment setup.
    5. Optionally install required packages.
    6. Prepare environment variables.
    7. Check availability of essential executables.
    8. Set core file size limits.
    9. Expand and validate test paths.
    10. Run the test suite with each configured compiler.
    11. Display test summaries and final pass/fail status.
    12. Exit with appropriate status code.

    Returns:
        No return — exits the interpreter upon completion.

    Exits:
        - Code 0 if all tests pass (or failures are in the expected list).
        - Code 1 otherwise.

    Notes:
        - The set of expected failures is defined in `XFAILED_TESTS`.
        - Paths may come from CLI args or environment variables THEROCK_BIN_DIR
          and OUTPUT_ARTIFACTS_DIR.
    """
    args = parse_arguments()

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

    start_time = time.perf_counter()

    rocgdb_configure_script = rocgdb_testsuite_dir / "configure"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

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

        # Install packages if requested via CLI.
        if args.install_packages:
            install_required_packages()

    # Prepare environment for tests.
    env_vars = setup_environment(artifacts_dir)

    # Validate that we can run rocgdb.
    validate_rocgdb(rocgdb_bin)

    # Verify executables presence.
    check_executables(
        ["hipcc", "gcc", "g++", "gfortran", "clang", "clang++", "flang", "runtest"]
    )

    print_section("Expanding test paths")
    # Resolve and expand test paths.
    try:
        tests = expand_test_paths(args.tests, rocgdb_testsuite_dir)
    except Exception as e:
        logger.info(f"[X] Error: Failed to expand test paths: {e}")
        sys.exit(1)

    if not tests:
        logger.info("[X] Error: No test files found")
        sys.exit(1)

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
    Validate and resolve an environment variable to a Path.

    Args:
        var_name: Name of the environment variable to validate.

    Returns:
        Resolved Path object from the environment variable.
    """
    try:
        return Path(os.getenv(var_name)).resolve()
    except (TypeError, AttributeError):
        logger.info(f"[X] Error: {var_name} environment variable is not set")
        sys.exit(1)


def print_python_info() -> None:
    """
    Prints runtime information about the current Python interpreter,
    including executable path, full version, and libpython details.
    """
    print_section("Python information")

    # Find the current python executable path.
    executable: str = sys.executable

    # Construct the full version number (e.g., 3.12.1).
    version: str = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # Check for shared libpython support.
    is_shared = sysconfig.get_config_var("Py_ENABLE_SHARED")
    supports_libpython: bool = bool(is_shared)

    lib_path: str = "N/A"
    lib_version: str = "N/A"

    if supports_libpython:
        lib_dir: str | None = sysconfig.get_config_var("LIBDIR")
        ld_library: str | None = sysconfig.get_config_var("LDLIBRARY")
        if lib_dir and ld_library:
            lib_path = os.path.join(lib_dir, ld_library)

        lib_version = str(sysconfig.get_config_var("VERSION") or "N/A")

    logger.info(f"Executable: {executable}")
    logger.info(f"Version: {version}")

    # Check if lib_path exists before printing support status.
    if supports_libpython:
        if lib_path != "N/A" and os.path.exists(lib_path):
            logger.info(f"Supports libpython: Yes")
        else:
            logger.info(f"Supports libpython: Supported but libpython is missing")
    else:
        logger.info(f"Supports libpython: No")

    # Point out if the library is missing when printing the path.
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
