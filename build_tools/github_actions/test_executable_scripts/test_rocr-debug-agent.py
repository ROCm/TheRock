# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import json
import logging
import os
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

STATUS_PASS = "[✓]"
STATUS_FAIL = "[X]"
STATUS_WARN = "[!]"
STATUS_SKIP = "[~]"


def print_section(title: str, border_char: str = "=", width: int = 80) -> None:
    border = border_char * width
    logger.info("")
    logger.info(border)
    logger.info(f"{title:^{width}}")
    logger.info(border)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ROCm Debug Agent tests with configurable paths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (used when CLI args are not provided):
  THEROCK_BIN_DIR          Directory containing rocm-debug-agent-test
  OUTPUT_ARTIFACTS_DIR     Directory containing run-test.py script
        """,
    )
    parser.add_argument(
        "--test-bin",
        type=Path,
        help="Path to rocm-debug-agent-test binary.",
    )
    parser.add_argument(
        "--test-script",
        type=Path,
        help="Path to run-test.py script.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of test retry attempts (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Base delay in seconds between retries (default: 5).",
    )

    args = parser.parse_args()

    args_provided = [args.test_bin, args.test_script]
    args_count = sum(arg is not None for arg in args_provided)
    if args_count not in (0, 2):
        parser.error(
            "Error: Either provide both arguments (--test-bin, --test-script) or none."
        )

    return args


def set_core_dump_limit() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        logger.info(f"  {STATUS_PASS} Core dump limit set to 0.")
    except (ValueError, OSError) as e:
        logger.warning(f"  {STATUS_WARN} Failed to set core dump limit: {e}")


def validate_path(path: Path, path_type: str, must_exist: bool = True) -> Path:
    try:
        resolved = path.resolve(strict=must_exist)
        if must_exist and not resolved.exists():
            logger.error(f"{STATUS_FAIL} {path_type} does not exist: {resolved}")
            sys.exit(1)
        return resolved
    except (OSError, RuntimeError) as e:
        logger.error(f"{STATUS_FAIL} Could not resolve {path_type} '{path}': {e}")
        sys.exit(1)


def get_default_paths() -> Dict[str, Path]:
    therock_bin_dir_str = os.getenv("THEROCK_BIN_DIR")
    artifacts_dir_str = os.getenv("OUTPUT_ARTIFACTS_DIR")

    if therock_bin_dir_str is None:
        logger.error(f"{STATUS_FAIL} THEROCK_BIN_DIR environment variable is not defined.")
        sys.exit(1)

    if artifacts_dir_str is None:
        logger.error(f"{STATUS_FAIL} OUTPUT_ARTIFACTS_DIR environment variable is not defined.")
        sys.exit(1)

    therock_bin_dir = validate_path(Path(therock_bin_dir_str), "THEROCK_BIN_DIR")
    artifacts_dir = validate_path(Path(artifacts_dir_str), "OUTPUT_ARTIFACTS_DIR")

    return {
        "test_bin": therock_bin_dir / "rocm-debug-agent-test",
        "test_script": artifacts_dir / "src" / "rocm-debug-agent-test" / "run-test.py",
    }


def get_python_executable() -> str:
    if not sys.executable or not os.path.exists(sys.executable):
        logger.error(f"{STATUS_FAIL} Could not identify a valid Python executable path.")
        sys.exit(1)
    return sys.executable


def extract_json_summary(output: str) -> Optional[Dict]:
    """Extract the JSON summary block emitted by run-test.py --json on stdout."""
    brace_start = output.find("{")
    if brace_start == -1:
        return None

    try:
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(output, brace_start)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def print_ci_summary(summary: Dict) -> None:
    """Print a concise CI summary — failures only, passes just counted."""
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)

    print_section("ROCm Debug Agent CI Summary")
    logger.info(f"  Total:              {total}")
    logger.info(f"  Passed:             {passed}")
    logger.info(f"  Failed:             {failed}")
    logger.info(f"  Skipped/Unsupported: {skipped}")

    tests = summary.get("tests", [])
    failed_tests = [t for t in tests if t.get("status") == "FAIL"]
    skipped_tests = [t for t in tests if t.get("status") in ("SKIP", "UNSUPPORTED")]

    if failed_tests:
        logger.info("")
        logger.info("  Failed tests:")
        for t in failed_tests:
            name = t.get("name", "<unknown>")
            duration = t.get("duration_s", 0)
            logger.info(f"    {STATUS_FAIL} {name} ({duration:.2f}s)")
            if t.get("message"):
                logger.info(f"           Reason: {t['message']}")
            if t.get("unmatched_patterns"):
                logger.info(f"           Missing patterns:")
                for pat in t["unmatched_patterns"]:
                    logger.info(f"             - {pat}")

    if skipped_tests:
        logger.info("")
        logger.info("  Skipped tests:")
        for t in skipped_tests:
            name = t.get("name", "<unknown>")
            logger.info(f"    {STATUS_SKIP} {name}")
            if t.get("message"):
                logger.info(f"           Reason: {t['message']}")

    all_pass = summary.get("all_pass", False)
    overall = STATUS_PASS if all_pass else STATUS_FAIL
    status_text = "PASS" if all_pass else "FAIL"
    print_section(f"{overall} OVERALL: {status_text}")


def run_tests(
    python_executable: str,
    test_script: Path,
    working_dir: Path,
    test_bin_dir: Path,
    env_vars: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> None:
    if env_vars is None:
        env_vars = os.environ.copy()

    cmd = [python_executable, str(test_script), str(test_bin_dir), "--json"]

    for attempt in range(1, max_retries + 1):
        print_section(f"Attempt {attempt}/{max_retries}")
        logger.info(f"  Exec [{working_dir}]$ {shlex.join(cmd)}")

        start_time = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(working_dir),
            check=False,
            env=env_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = proc.stdout or ""
        duration = time.perf_counter() - start_time
        summary = extract_json_summary(output)

        if proc.returncode == 0:
            if summary:
                print_ci_summary(summary)
            else:
                print_section(
                    f"{STATUS_PASS} Tests succeeded (attempt {attempt}, {duration:.2f}s)"
                )
            return

        for line in output.splitlines():
            logger.info(line)

        logger.error(
            f"  {STATUS_FAIL} Attempt {attempt}/{max_retries} failed "
            f"(exit code {proc.returncode}, {duration:.2f}s)"
        )

        if summary:
            print_ci_summary(summary)

        if attempt < max_retries:
            wait_time = attempt * retry_delay
            logger.info(f"  Retrying in {wait_time}s...")
            time.sleep(wait_time)
        else:
            print_section(f"{STATUS_FAIL} All {max_retries} attempts failed.")
            sys.exit(1)


def main() -> None:
    args = parse_arguments()

    print_section("ROCm Debug Agent CI Test Runner")

    if args.test_bin is not None:
        logger.info("  Using paths from command-line arguments.")
        rocr_debug_agent_test_bin = validate_path(args.test_bin, "--test-bin")
        rocr_debug_agent_test_script = validate_path(args.test_script, "--test-script")
    else:
        logger.info("  Using default paths from environment variables.")
        defaults = get_default_paths()
        rocr_debug_agent_test_bin = defaults["test_bin"]
        rocr_debug_agent_test_script = defaults["test_script"]

    test_bin_dir = rocr_debug_agent_test_bin.parent

    logger.info(f"  Test Binary:  {rocr_debug_agent_test_bin}")
    logger.info(f"  Test Script:  {rocr_debug_agent_test_script}")
    logger.info(f"  Test Bin Dir: {test_bin_dir}")

    python_executable = get_python_executable()
    logger.info(f"  Python:       {python_executable}")

    logger.info("")
    set_core_dump_limit()

    run_tests(
        python_executable=python_executable,
        test_script=rocr_debug_agent_test_script,
        working_dir=test_bin_dir,
        test_bin_dir=test_bin_dir,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )


if __name__ == "__main__":
    main()
