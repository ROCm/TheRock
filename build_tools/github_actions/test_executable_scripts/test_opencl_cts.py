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

CTS_SOURCE_DIR = THEROCK_DIR / "build" / "opencl-cts-source"
CTS_BUILD_DIR = THEROCK_DIR / "build" / "opencl-cts-build"

OPENCL_CTS_REPO = "https://github.com/ROCm/OpenCL-CTS.git"
OPENCL_CTS_BRANCH = os.getenv("OPENCL_CTS_BRANCH", "main")

TEST_TYPE = os.getenv("TEST_TYPE", "full")

logging.info(f"THEROCK_BIN_DIR: {THEROCK_BIN_DIR}")
logging.info(f"ROCM_PATH: {ROCM_PATH}")
logging.info(f"CTS_SOURCE_DIR: {CTS_SOURCE_DIR}")
logging.info(f"CTS_BUILD_DIR: {CTS_BUILD_DIR}")
logging.info(f"OpenCL-CTS Branch: {OPENCL_CTS_BRANCH}")
logging.info(f"Test Type: {TEST_TYPE}")


def setup_environment():
    logging.info("++ Setting up environment variables")

    env = os.environ.copy()

    env["ROCM_PATH"] = str(ROCM_PATH)

    ocl_icd_path = ROCM_PATH / "lib" / "opencl" / "libamdocl64.so"
    if not ocl_icd_path.exists():
        ocl_icd_path = ROCM_PATH / "lib" / "libamdocl64.so"

    if ocl_icd_path.exists():
        env["OCL_ICD_FILENAMES"] = str(ocl_icd_path)
        logging.info(f"OCL_ICD_FILENAMES: {ocl_icd_path}")
    else:
        logging.warning("OpenCL library not found at expected locations")

    rocm_lib = str(ROCM_PATH / "lib")
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = f"{rocm_lib}:{env['LD_LIBRARY_PATH']}"
    else:
        env["LD_LIBRARY_PATH"] = rocm_lib
    logging.info(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")

    rocm_bin = str(ROCM_PATH / "bin")
    if "PATH" in env:
        env["PATH"] = f"{rocm_bin}:{env['PATH']}"
    else:
        env["PATH"] = rocm_bin

    return env


def verify_opencl_runtime(env):
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
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logging.info("OpenCL runtime verification successful")
            # Log first few lines of clinfo output for debugging
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


def clone_opencl_cts():
    logging.info(f"++ Cloning OpenCL-CTS from {OPENCL_CTS_REPO}")

    CTS_SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        OPENCL_CTS_BRANCH,
        OPENCL_CTS_REPO,
        str(CTS_SOURCE_DIR),
    ]

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)

    logging.info("++ Initializing git submodules")
    cmd = ["git", "submodule", "update", "--init", "--recursive"]
    logging.info(f"++ Exec [{CTS_SOURCE_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=CTS_SOURCE_DIR, check=True)


def configure_build(env):
    logging.info("++ Configuring OpenCL-CTS build")

    CTS_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    opencl_lib_dir = ROCM_PATH / "lib"
    opencl_include_dir = ROCM_PATH / "include"

    cmd = [
        "cmake",
        "-S",
        str(CTS_SOURCE_DIR),
        "-B",
        str(CTS_BUILD_DIR),
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DOPENCL_LIBRARIES={opencl_lib_dir}",
        f"-DOPENCL_INCLUDE_DIR={opencl_include_dir}",
        f"-DCL_INCLUDE_DIR={opencl_include_dir}",
        f"-DCL_LIB_DIR={opencl_lib_dir}",
    ]

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=env)


def build_tests(env):
    logging.info("++ Building OpenCL-CTS tests")

    cpu_count = os.cpu_count() or 4
    cmd = [
        "cmake",
        "--build",
        str(CTS_BUILD_DIR),
        "--config",
        "Release",
        "--parallel",
        str(cpu_count),
    ]

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=env)


def run_tests(env):
    logging.info(f"++ Running OpenCL-CTS tests (type: {TEST_TYPE})")

    cmd = [
        "ctest",
        "--test-dir",
        str(CTS_BUILD_DIR),
        "--output-on-failure",
        "--timeout",
        "600",
    ]
    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=env)


if __name__ == "__main__":
    try:
        env = setup_environment()
        verify_opencl_runtime(env)
        clone_opencl_cts()
        configure_build(env)
        build_tests(env)
        run_tests(env)
        logging.info("++ OpenCL-CTS tests completed successfully")

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {e.cmd}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
