import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

environ_vars = os.environ.copy()

# Resolve absolute paths
OUTPUT_ARTIFACTS_PATH = Path(OUTPUT_ARTIFACTS_DIR).resolve()
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()

# Set up ROCm/HIP environment
environ_vars["ROCM_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["HIP_DEVICE_LIB_PATH"] = str(OUTPUT_ARTIFACTS_PATH / "lib/llvm/amdgcn/bitcode/")
environ_vars["HIP_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["CMAKE_PREFIX_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["HIP_PLATFORM"] = "amd"

# Add ROCm binaries to PATH
rocm_bin = str(THEROCK_BIN_PATH)
if "PATH" in environ_vars:
    environ_vars["PATH"] = f"{rocm_bin}:{environ_vars['PATH']}"
else:
    environ_vars["PATH"] = rocm_bin

# Set library paths
rocm_lib = str(OUTPUT_ARTIFACTS_PATH / "lib")
if "LD_LIBRARY_PATH" in environ_vars:
    environ_vars["LD_LIBRARY_PATH"] = f"{rocm_lib}:{environ_vars['LD_LIBRARY_PATH']}"
else:
    environ_vars["LD_LIBRARY_PATH"] = rocm_lib

logging.info(f"ROCM_PATH: {environ_vars['ROCM_PATH']}")
logging.info(f"HIP_PATH: {environ_vars['HIP_PATH']}")
logging.info(f"PATH: {environ_vars['PATH']}")

LIBHIPCXX_BUILD_DIR = OUTPUT_ARTIFACTS_PATH / "libhipcxx"

try:
    os.chdir(LIBHIPCXX_BUILD_DIR)
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    os.chdir(build_dir)
    logging.info(f"Changed working directory to: {os.getcwd()}")
except FileNotFoundError as e:
    logging.error(f"Error: Directory '{LIBHIPCXX_BUILD_DIR}' does not exist.")
    raise


# Configure with CMake
cmd = [
    "cmake",
    "..",
    f"-DCMAKE_PREFIX_PATH={OUTPUT_ARTIFACTS_PATH}",
    #f"-DCMAKE_HIP_COMPILER={THEROCK_BIN_PATH}/hipcc",
    f"-DCMAKE_CXX_COMPILER={THEROCK_BIN_PATH}/hipcc",
    #f"-DCMAKE_CXX_FLAGS=--rocm-path={OUTPUT_ARTIFACTS_PATH}",
    f"-DHIP_HIPCC_EXECUTABLE={THEROCK_BIN_PATH}/hipcc",
    "-GNinja",
]

logging.info(f"++ Exec [{os.getcwd()}]$ {shlex.join(cmd)}")
subprocess.run(cmd, check=True, env=environ_vars)


# Run the tests using lit
# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    # For smoke tests, we could filter specific tests if needed
    # For now, run all tests but could be customized later
    pass

cmd = [
    "ninja",
    "check-hipcxx"
]
logging.info(f"++ Exec [{os.getcwd()}]$ {shlex.join(cmd)}")

subprocess.run(cmd, check=True, env=environ_vars)
