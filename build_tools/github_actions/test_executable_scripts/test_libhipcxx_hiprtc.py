# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import logging
import os
import shlex
import sys
import subprocess
from pathlib import Path
import platform

from libhipcxx_utils import (
    get_gpu_architecture_portable,
    prepend_env_path,
)

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

gpu_arch = get_gpu_architecture_portable(OUTPUT_ARTIFACTS_DIR)
logging.info(f"++ Detected GPU architecture: {gpu_arch}")


# Load ROCm version from version.json
def load_rocm_version() -> str:
    """Loads the rocm-version from the repository's version.json file."""
    version_file = THEROCK_DIR / "version.json"
    logging.info(f"Loading ROCm version from: {version_file}")
    with open(version_file, "rt") as f:
        loaded_file = json.load(f)
        return loaded_file["rocm-version"]


ROCM_VERSION = load_rocm_version()
logging.info(f"ROCm version: {ROCM_VERSION}")

environ_vars = os.environ.copy()

# Resolve absolute paths
OUTPUT_ARTIFACTS_PATH = Path(OUTPUT_ARTIFACTS_DIR).resolve()
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()

# Set up ROCm/HIP environment
environ_vars["ROCM_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["HIP_DEVICE_LIB_PATH"] = str(
    OUTPUT_ARTIFACTS_PATH / "lib/llvm/amdgcn/bitcode/"
)
environ_vars["HIP_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["CMAKE_PREFIX_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["HIP_PLATFORM"] = "amd"
environ_vars["ROCM_VERSION"] = str(ROCM_VERSION)
environ_vars["CMAKE_GENERATOR"] = "Ninja"

# Add ROCm binaries to PATH
rocm_bin = str(THEROCK_BIN_PATH)
prepend_env_path(environ_vars, "PATH", rocm_bin)

# Set library paths
rocm_lib = str(OUTPUT_ARTIFACTS_PATH / "lib")
prepend_env_path(environ_vars, "LD_LIBRARY_PATH", rocm_lib)

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

is_windows = platform.system() == "Windows"
is_linux = platform.system() == "Linux"

compiler_ext = ".exe" if is_windows else ""

HIPCC_BINARY_NAME = f"hipcc{compiler_ext}"

HIP_COMPILER_ROCM_ROOT = OUTPUT_ARTIFACTS_PATH
if is_windows:
    environ_vars["HIPCXX"] = str(
        OUTPUT_ARTIFACTS_PATH / "lib" / "llvm" / "bin" / "amdclang++.exe"
    )
    print("HIPCXX:", environ_vars["HIPCXX"], str(OUTPUT_ARTIFACTS_PATH))


cmd = [
    "cmake",
    f"-DCMAKE_PREFIX_PATH={OUTPUT_ARTIFACTS_PATH}",
    f"-DHIP_HIPCC_EXECUTABLE={(THEROCK_BIN_PATH / HIPCC_BINARY_NAME).as_posix()}",
    f"-DCMAKE_CXX_COMPILER={THEROCK_BIN_PATH / HIPCC_BINARY_NAME}",
    f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={HIP_COMPILER_ROCM_ROOT.as_posix()}",
    f"-DCMAKE_HIP_ARCHITECTURES={gpu_arch}",
    "-DLIBHIPCXX_TEST_WITH_HIPRTC=ON",
]

# Add rc compiler for windows
if is_windows:
    cmd.append("-DCMAKE_RC_COMPILER=rc.exe")

cmd.extend(["-GNinja", ".."])

logging.info(f"++ Exec [{os.getcwd()}]$ {shlex.join(cmd)}")
subprocess.run(cmd, check=True, env=environ_vars)

# Run the tests using lit
if is_windows:
    cmd = [
        "ninja",
        "check-hipcxx",
    ]
else:
    cmd = [
        "bash",
        "../ci/hiprtc_libhipcxx.sh",
        "-cmake-options",
        f"-DHIP_HIPCC_EXECUTABLE={THEROCK_BIN_PATH / HIPCC_BINARY_NAME} -DCMAKE_HIP_COMPILER_ROCM_ROOT={HIP_COMPILER_ROCM_ROOT} -DCMAKE_HIP_ARCHITECTURES={gpu_arch}",
    ]

logging.info(f"++ Exec [{os.getcwd()}]$ {shlex.join(cmd)}")

subprocess.run(cmd, check=True, env=environ_vars)
