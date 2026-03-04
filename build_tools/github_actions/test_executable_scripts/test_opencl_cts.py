import logging
import os
import shlex
import subprocess
import sys
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

CTS_BIN_DIR = THEROCK_BIN_DIR / "Release"
OPENCL_ICD_FILENAMES = ROCM_PATH / "lib" / "opencl" / "libamdocl64.so"

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

    test_executables = []
    for test_exe in CTS_BIN_DIR.rglob("test_*"):
        if test_exe.is_file() and os.access(test_exe, os.X_OK):
            test_executables.append(test_exe)

    if not test_executables:
        logging.error(f"No test executables found in {CTS_BIN_DIR}")
        sys.exit(1)

    test_executables.sort()
    return test_executables


def run_test(test_exe, env):
    """Run a single test executable and return True if it passes"""
    test_name = test_exe.name
    logging.info(f"++ Running test: {test_name}")

    cmd = [str(test_exe)]
    logging.info(f"++ Exec [{test_exe.parent}]$ {shlex.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(test_exe.parent),
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode == 0:
            logging.info(f"✓ PASSED: {test_name}")
            return True
        else:
            logging.error(f"✗ FAILED: {test_name} (exit code: {result.returncode})")
            if result.stdout:
                logging.error(f"  stdout: {result.stdout[:500]}")
            if result.stderr:
                logging.error(f"  stderr: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        logging.error(f"✗ TIMEOUT: {test_name} (exceeded 300 seconds)")
        return False
    except Exception as e:
        logging.error(f"✗ ERROR: {test_name} - {e}")
        return False


def run_tests():
    """Run all OpenCL CTS test executables"""
    logging.info("++ Running OpenCL-CTS tests")

    env = os.environ.copy()
    env["OCL_ICD_FILENAMES"] = str(OPENCL_ICD_FILENAMES)

    lib_dir = ROCM_PATH / "lib"
    if lib_dir.exists():
        ld_library_path = str(lib_dir)
        if "LD_LIBRARY_PATH" in env:
            ld_library_path = f"{ld_library_path}:{env['LD_LIBRARY_PATH']}"
        env["LD_LIBRARY_PATH"] = ld_library_path
        logging.info(f"Set LD_LIBRARY_PATH to include: {lib_dir}")

    test_executables = find_test_executables()
    logging.info(f"Found {len(test_executables)} test executables")

    passed = 0
    failed = 0
    for test_exe in test_executables:
        if run_test(test_exe, env):
            passed += 1
        else:
            failed += 1

    total = passed + failed
    logging.info("=" * 70)
    logging.info("OpenCL-CTS Test Summary:")
    logging.info(f"  Total:  {total}")
    logging.info(f"  Passed: {passed}")
    logging.info(f"  Failed: {failed}")
    logging.info("=" * 70)

    if failed > 0:
        logging.error(f"{failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        verify_opencl_runtime()
        run_tests()
        logging.info("++ OpenCL-CTS tests completed successfully")

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {e.cmd}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
