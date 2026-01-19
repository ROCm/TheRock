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
HEADERS_SOURCE_DIR = THEROCK_DIR / "build" / "opencl-headers-source"
SPIRV_HEADERS_DIR = THEROCK_DIR / "build" / "spirv-headers-source"
CTS_BUILD_DIR = THEROCK_DIR / "build" / "opencl-cts-build"

OPENCL_CTS_REPO = "https://github.com/ROCm/OpenCL-CTS.git"
OPENCL_CTS_BRANCH = os.getenv("OPENCL_CTS_BRANCH", "main")
OPENCL_HEADERS_REPO = "https://github.com/KhronosGroup/OpenCL-Headers.git"
OPENCL_HEADERS_BRANCH = os.getenv("OPENCL_HEADERS_BRANCH", "main")
SPIRV_HEADERS_REPO = "https://github.com/KhronosGroup/SPIRV-Headers.git"
SPIRV_HEADERS_BRANCH = os.getenv("SPIRV_HEADERS_BRANCH", "main")


logging.info(f"THEROCK_BIN_DIR: {THEROCK_BIN_DIR}")
logging.info(f"ROCM_PATH: {ROCM_PATH}")
logging.info(f"CTS_SOURCE_DIR: {CTS_SOURCE_DIR}")
logging.info(f"CTS_BUILD_DIR: {CTS_BUILD_DIR}")
logging.info(f"OpenCL-CTS Branch: {OPENCL_CTS_BRANCH}")
OPENCL_ICD_FILENAMES = ROCM_PATH / "lib" / "opencl" / "libamdocl64.so"


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


def clone_opencl():
    logging.info(f"++ Cloning OpenCL dependencies from {OPENCL_CTS_REPO}")

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

    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        OPENCL_HEADERS_BRANCH,
        OPENCL_HEADERS_REPO,
        str(HEADERS_SOURCE_DIR),
    ]

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)

    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        SPIRV_HEADERS_BRANCH,
        SPIRV_HEADERS_REPO,
        str(SPIRV_HEADERS_DIR),
    ]

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


def configure_build():
    logging.info("++ Configuring OpenCL-CTS build")

    CTS_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    opencl_lib_dir = ROCM_PATH / "lib"

    cmd = [
        "cmake",
        "-S",
        str(CTS_SOURCE_DIR),
        "-B",
        str(CTS_BUILD_DIR),
        f"-DCL_LIB_DIR={opencl_lib_dir}",
        f"-DCL_INCLUDE_DIR={HEADERS_SOURCE_DIR}",
        f"-DSPIRV_INCLUDE_DIR={SPIRV_HEADERS_DIR}",
        "-DOPENCL_LIBRARIES=OpenCL",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    ]

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


def build_tests():
    logging.info("++ Building OpenCL-CTS tests")

    cmd = [
        "cmake",
        "--build",
        str(CTS_BUILD_DIR),
        "--config",
        "Release",
    ]

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


def run_tests():
    logging.info("++ Running OpenCL-CTS tests ")

    cmd = [
        "python",
        str(CTS_BUILD_DIR / "test_conformance" / "run_conformance.py"),
        str(CTS_BUILD_DIR / "test_conformance" / "opencl_conformance_tests_full.csv"),
    ]
    logging.info(f"++ Exec [{CTS_BUILD_DIR}]$ {shlex.join(cmd)}")
    env = {"OCL_ICD_FILENAMES": str(OPENCL_ICD_FILENAMES)}
    subprocess.run(
        cmd, cwd=str(CTS_BUILD_DIR / "test_conformance"), check=True, env=env
    )


if __name__ == "__main__":
    try:
        verify_opencl_runtime()
        clone_opencl()
        configure_build()
        build_tests()
        run_tests()
        logging.info("++ OpenCL-CTS tests completed successfully")

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {e.cmd}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
