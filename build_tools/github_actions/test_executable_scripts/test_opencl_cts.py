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

CTS_BUILD_DIR = THEROCK_BIN_DIR / "opencl-cts"
OPENCL_ICD_FILENAMES = ROCM_PATH / "lib" / "opencl" / "libamdocl64.so"

logging.info(f"THEROCK_BIN_DIR: {THEROCK_BIN_DIR}")
logging.info(f"ROCM_PATH: {ROCM_PATH}")
logging.info(f"CTS_BUILD_DIR: {CTS_BUILD_DIR}")


def verify_opencl_runtime():
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


def run_tests():
    logging.info("++ Running OpenCL-CTS tests")

    if not CTS_BUILD_DIR.exists():
        logging.error(
            f"OpenCL-CTS build directory not found at {CTS_BUILD_DIR}. "
            "Please ensure opencl-cts was built during the build stage."
        )
        sys.exit(1)

    test_conformance_dir = CTS_BUILD_DIR / "test_conformance"
    run_conformance_script = test_conformance_dir / "run_conformance.py"
    test_csv = test_conformance_dir / "opencl_conformance_tests_full.csv"

    if not run_conformance_script.exists():
        logging.error(f"run_conformance.py not found at {run_conformance_script}")
        sys.exit(1)

    if not test_csv.exists():
        logging.error(f"Test CSV not found at {test_csv}")
        sys.exit(1)

    cmd = [
        "python",
        str(run_conformance_script),
        str(test_csv),
    ]
    logging.info(f"++ Exec [{test_conformance_dir}]$ {shlex.join(cmd)}")
    env = os.environ.copy()
    env["OCL_ICD_FILENAMES"] = str(OPENCL_ICD_FILENAMES)
    subprocess.run(cmd, cwd=str(test_conformance_dir), check=True, env=env)


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
