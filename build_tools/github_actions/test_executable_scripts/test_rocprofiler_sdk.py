import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# Env setup
os.environ["HIP_PLATFORM"] = "amd"

# CMake Init
cmake_init_cmd = [
    "cmake",
    "-B",
    "/tmp/rocprofiler-sdk-build-tests",
    f"-DCMAKE_PREFIX_PATH={OUTPUT_ARTIFACTS_DIR}",
    f"{OUTPUT_ARTIFACTS_DIR}/share/rocprofiler-sdk/tests",
]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmake_init_cmd)}")
subprocess.run(cmake_init_cmd, cwd=THEROCK_DIR, check=True)

# Cmake Build
cmake_cmd = [
    "cmake",
    "--build",
    "/tmp/rocprofiler-sdk-build-tests",
    "--target",
    "all",
    "--parallel",
    "8",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmake_cmd)}")
subprocess.run(cmake_cmd, cwd=THEROCK_DIR, check=True)

# Run tests
cmd = [
    "ctest",
    "--test-dir",
    f"{OUTPUT_ARTIFACTS_DIR}/share/rocprofiler-sdk/tests",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "900",
    "--repeat",
    "until-pass:3",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
