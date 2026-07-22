import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

THEROCK_BIN_DIR_STR = os.getenv("THEROCK_BIN_DIR")
if THEROCK_BIN_DIR_STR is None:
    logging.error(
        "++ Error: env(THEROCK_BIN_DIR) is not set. Please set it before executing tests."
    )
    sys.exit(1)

THEROCK_BIN_DIR = Path(THEROCK_BIN_DIR_STR).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
ROCM_PATH = THEROCK_BIN_DIR.parent

SHARD_INDEX = int(os.getenv("SHARD_INDEX", "1"))  # 1-based
TOTAL_SHARDS = int(os.getenv("TOTAL_SHARDS", "1"))

CTS_BIN_DIR = ROCM_PATH / "share" / "opencl" / "opencl-cts" / "Release"
AMDOCL_PATH = ROCM_PATH / "lib" / "opencl" / "libamdocl64.so"

logging.info(f"THEROCK_BIN_DIR: {THEROCK_BIN_DIR}")
logging.info(f"ROCM_PATH: {ROCM_PATH}")
logging.info(f"CTS_BIN_DIR: {CTS_BIN_DIR}")


def verify_opencl_runtime():
    """Verify OpenCL runtime is available using clinfo"""
    logging.info("++ Verifying OpenCL runtime availability")

    clinfo_path = ROCM_PATH / "bin" / "clinfo"
    if not clinfo_path.exists():
        logging.warning(f"clinfo not found at {clinfo_path}, skipping verification")
        return

    try:
        cmd = [str(clinfo_path)]
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=THEROCK_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logging.info("OpenCL runtime verification successful")
            lines = result.stdout.split("\n")[:10]
            for line in lines:
                if line.strip():
                    logging.info(f"  {line}")
        else:
            logging.warning(f"clinfo returned non-zero exit code: {result.returncode}")
            logging.warning(f"stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        logging.warning("clinfo verification timed out")
    except Exception as e:
        logging.warning(f"Error running clinfo: {e}")


def find_test_executables():
    """Find all test_* executables in the CTS bin directory"""
    if not CTS_BIN_DIR.exists():
        logging.error(
            f"OpenCL-CTS bin directory not found at {CTS_BIN_DIR}. "
            "Please ensure opencl-cts was built and artifacts were created."
        )
        sys.exit(1)

    # Entirely disabled test binaries.
    # test_bruteforce, test_conversions: prohibitively slow, exceed 4.5h CI timeout.
    # test_basic: crashes with GPU hang in 'constant' sub-test (gfx942).
    # test_compiler: crashes with SIGSEGV in 'get_program_info_kernel_names' (gfx942).
    # test_half: crashes with GPU hang in 'vstore_half_rtp' sub-test (gfx942).
    # test_spir: cl_khr_spir not supported on gfx942, exits 156.
    # test_vectors: crashes with GPU hang in 'vec_align_packed_struct' sub-test (gfx942).
    # test_workgroups: crashes with GPU hang in 'work_group_reduce_add' sub-test (gfx942).
    DISABLED_TESTS = {
        "test_bruteforce",
        "test_conversions",
        "test_basic",
        "test_compiler",
        "test_half",
        "test_spir",
        "test_vectors",
        "test_workgroups",
    }

    # Disabled sub-tests within otherwise-passing binaries.
    # Sub-tests are passed as positional arguments; only the allowed subset is run.
    # test_api: 'min_max_constant_buffer_size' fails on gfx942.
    # test_printf: 'vector' and 'length_specifier' fail on gfx942.
    # test_svm: 'svm_migrate' fails on gfx942.
    DISABLED_SUBTESTS: dict[str, set[str]] = {
        "test_api": {"min_max_constant_buffer_size"},
        "test_printf": {"vector", "length_specifier"},
        "test_svm": {"svm_migrate"},
    }

    test_executables = []
    for test_exe in CTS_BIN_DIR.rglob("test_*"):
        if test_exe.is_file() and os.access(test_exe, os.X_OK):
            if test_exe.name in DISABLED_TESTS:
                logging.info(f"Skipping disabled test: {test_exe.name}")
                continue
            subtests = None
            if test_exe.name in DISABLED_SUBTESTS:
                disabled = DISABLED_SUBTESTS[test_exe.name]
                result = subprocess.run(
                    [str(test_exe), "--list"],
                    cwd=str(test_exe.parent),
                    capture_output=True,
                    text=True,
                )
                all_subtests = [
                    line.strip() for line in result.stdout.splitlines() if line.strip()
                ]
                subtests = [s for s in all_subtests if s not in disabled]
                logging.info(
                    f"Skipping sub-tests in {test_exe.name}: {sorted(disabled)}"
                )
            test_executables.append((test_exe, subtests))

    if not test_executables:
        logging.error(f"No test executables found in {CTS_BIN_DIR}")
        sys.exit(1)

    test_executables.sort(key=lambda x: x[0])
    test_executables = test_executables[SHARD_INDEX - 1 :: TOTAL_SHARDS]
    return test_executables


def run_test(test_exe, subtests=None):
    """Run a single test executable and return True if it passes"""
    test_name = test_exe.name
    logging.info(f"++ Running test: {test_name}")

    cmd = [str(test_exe)]
    if subtests:
        cmd.extend(subtests)
    logging.info(f"++ Exec [{test_exe.parent}]$ {shlex.join(cmd)}")

    try:
        start = time.monotonic()
        with subprocess.Popen(
            cmd,
            cwd=str(test_exe.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
            returncode = proc.wait()
        elapsed = (time.monotonic() - start) / 60

        if returncode == 0:
            logging.info(f"✓ PASSED: {test_name} ({elapsed:.1f} min)")
            return True
        else:
            logging.error(
                f"✗ FAILED: {test_name} ({elapsed:.1f} min, exit code: {returncode})"
            )
            return False

    except Exception as e:
        logging.error(f"✗ ERROR: {test_name} - {e}")
        return False


def register_icd():
    """Register the AMD OpenCL ICD so all tools (clinfo, CTS) can find it."""
    icd_dir = Path("/etc/OpenCL/vendors")
    icd_dir.mkdir(parents=True, exist_ok=True)
    icd_file = icd_dir / "amdocl64.icd"
    icd_file.write_text(str(AMDOCL_PATH) + "\n")
    logging.info(f"Registered AMD OpenCL ICD: {icd_file} -> {AMDOCL_PATH}")


def run_tests():
    """Run all OpenCL CTS test executables"""
    logging.info(f"++ Running OpenCL-CTS tests (shard {SHARD_INDEX}/{TOTAL_SHARDS})")

    test_executables = find_test_executables()
    logging.info(f"Found {len(test_executables)} test executables")

    passed = 0
    failed_tests = []
    for test_exe, subtests in test_executables:
        if run_test(test_exe, subtests):
            passed += 1
        else:
            failed_tests.append(test_exe.name)

    total = passed + len(failed_tests)
    logging.info("=" * 70)
    logging.info("OpenCL-CTS Test Summary:")
    logging.info(f"  Total:  {total}")
    logging.info(f"  Passed: {passed}")
    logging.info(f"  Failed: {len(failed_tests)}")
    if failed_tests:
        logging.error("Failed tests:")
        for name in failed_tests:
            logging.error(f"  - {name}")
    logging.info("=" * 70)

    if failed_tests:
        logging.error(f"{len(failed_tests)} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        register_icd()
        verify_opencl_runtime()
        run_tests()
        logging.info("++ OpenCL-CTS tests completed successfully")

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {e.cmd}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
