import argparse
import logging
import os
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def set_core_dump_limit() -> None:
    """
    Set core dump size limit to 0 (equivalent to ulimit -c 0).
    """
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        logger.info("✅ Core dump limit set to 0.")
    except (ValueError, OSError) as e:
        logger.warning(f"⚠️  Failed to set core dump limit: {e}")
        logger.warning("Core files may be generated and consume disk space.")


def validate_path(path: Path, path_type: str, must_exist: bool = True) -> Path:
    """
    Validate and resolve a path.

    Args:
        path: Path to validate.
        path_type: Description of the path (for error messages).
        must_exist: Whether the path must exist.

    Returns:
        Resolved path.

    Raises:
        SystemExit: If path validation fails.
    """
    try:
        resolved = path.resolve(strict=must_exist)
        if must_exist and not resolved.exists():
            sys.exit(f"❌ Error: {path_type} does not exist: {resolved}")
        return resolved
    except (OSError, RuntimeError) as e:
        sys.exit(f"❌ Error: Could not resolve {path_type} '{path}': {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments with all-or-nothing logic.
    """
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

    # Check if any arguments are provided.
    args_provided = [args.test_bin, args.test_script]
    args_count = sum(arg is not None for arg in args_provided)

    # Either all arguments or none.
    if args_count not in (0, 2):
        parser.error(
            "❌ Error: Either provide both arguments (--test-bin, --test-script) or none."
        )

    return args


def get_default_paths() -> Dict[str, Path]:
    """
    Get default paths from environment variables.

    Returns:
        Dictionary containing 'test_bin' and 'test_script' paths.

    Raises:
        SystemExit: If environment variables are not defined or paths cannot be resolved.
    """
    therock_bin_dir_str = os.getenv("THEROCK_BIN_DIR")
    artifacts_dir_str = os.getenv("OUTPUT_ARTIFACTS_DIR")

    # Check if environment variables are defined.
    if therock_bin_dir_str is None:
        sys.exit("❌ Error: THEROCK_BIN_DIR environment variable is not defined.")

    if artifacts_dir_str is None:
        sys.exit("❌ Error: OUTPUT_ARTIFACTS_DIR environment variable is not defined.")

    # Resolve and validate paths.
    therock_bin_dir = validate_path(Path(therock_bin_dir_str), "THEROCK_BIN_DIR")
    artifacts_dir = validate_path(Path(artifacts_dir_str), "OUTPUT_ARTIFACTS_DIR")

    return {
        "test_bin": therock_bin_dir / "rocm-debug-agent-test",
        "test_script": artifacts_dir / "src/rocm-debug-agent-test/run-test.py",
    }


def get_python_executable() -> str:
    """
    Validates and returns the Python executable path.

    Returns:
        Path to Python executable.

    Raises:
        SystemExit: If valid Python executable cannot be found.
    """
    if not sys.executable or not os.path.exists(sys.executable):
        sys.exit("❌ Error: Could not identify a valid Python executable path.")
    return sys.executable


def run_tests(
    python_executable: str,
    test_script: Path,
    working_dir: Path,
    test_bin_dir: Path,
    env_vars: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> None:
    """
    Runs the testsuite with a retry mechanism.

    Args:
        python_executable: Path to Python interpreter.
        test_script: Path to test script.
        working_dir: Working directory for test execution.
        test_bin_dir: Directory containing test binaries.
        env_vars: Environment variables for test execution.
        max_retries: Maximum number of retry attempts.
        retry_delay: Base delay in seconds between retries.

    Raises:
        SystemExit: If all retry attempts fail.
    """
    if env_vars is None:
        env_vars = os.environ.copy()

    cmd = [python_executable, str(test_script), str(test_bin_dir)]

    for attempt in range(1, max_retries + 1):
        logger.info(
            f"++ Exec (Attempt {attempt}/{max_retries}) [{working_dir}]$ {shlex.join(cmd)}"
        )

        start_time = time.perf_counter()
        try:
            subprocess.run(cmd, cwd=str(working_dir), check=True, env=env_vars)

            duration = time.perf_counter() - start_time
            logger.info(
                f"✅ Tests succeeded on attempt {attempt}. Duration: {duration:.2f}s"
            )
            return

        except subprocess.CalledProcessError as e:
            duration = time.perf_counter() - start_time
            logger.error(
                f"❌ Attempt {attempt}/{max_retries} failed with exit code {e.returncode} "
                f"after {duration:.2f}s"
            )

            if attempt < max_retries:
                # Exponential-ish backoff: multiply base delay by attempt number
                wait_time = attempt * retry_delay
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # We failed all the attempts.
                logger.error(f"❌ All {max_retries} attempts failed.")
                sys.exit(1)


def main() -> None:
    """
    Main entry point for the script.
    """
    # Parse command-line arguments.
    args = parse_arguments()

    # Determine paths to use.
    if args.test_bin is not None:
        logger.info("Using paths from command-line arguments.")
        rocr_debug_agent_test_bin = validate_path(args.test_bin, "--test-bin")
        rocr_debug_agent_test_script = validate_path(args.test_script, "--test-script")
    else:
        # Use default logic.
        logger.info("Using default paths from environment variables.")
        defaults = get_default_paths()
        rocr_debug_agent_test_bin = defaults["test_bin"]
        rocr_debug_agent_test_script = defaults["test_script"]

    # Derive the test binary directory.
    test_bin_dir = rocr_debug_agent_test_bin.parent

    logger.info(f"  Test Binary: {rocr_debug_agent_test_bin}")
    logger.info(f"  Test Script: {rocr_debug_agent_test_script}")
    logger.info(f"  Test Bin Dir: {test_bin_dir}")

    # Setup Python executable.
    python_executable = get_python_executable()

    # Set core dump limit to 0 (ulimit -c 0).
    set_core_dump_limit()

    # Run tests.
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
