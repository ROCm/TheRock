import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
environ_vars = os.environ.copy()
environ_vars["ROCM_PATH"] = str(ROCM_PATH)

# Set up PYTHONPATH
old_pythonpath = os.getenv("PYTHONPATH", "")
test_dir = f"{OUTPUT_ARTIFACTS_DIR}/libexec/rocprofiler-compute/tests"
if old_pythonpath:
    os.environ["PYTHONPATH"] = f"{test_dir}:{old_pythonpath}"
else:
    os.environ["PYTHONPATH"] = test_dir

# Set up PATH
old_path = os.getenv("PATH", "")
if old_path:
    os.environ["PATH"] = f"{THEROCK_BIN_DIR}:{old_path}"
else:
    os.environ["PATH"] = THEROCK_BIN_DIR

# Set up LD_LIBRARY_PATH
old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "")
if old_ld_lib_path:
    os.environ["LD_LIBRARY_PATH"] = (
        f"{OUTPUT_ARTIFACTS_DIR}/lib:{OUTPUT_ARTIFACTS_DIR}/lib/rocm_sysdeps/lib:{old_ld_lib_path}"
    )
else:
    os.environ["LD_LIBRARY_PATH"] = f"{OUTPUT_ARTIFACTS_DIR}/lib"

# Run tests
cmd = [
    "ctest",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "1800",
]
logging.info(
    f"++ Exec [{OUTPUT_ARTIFACTS_DIR}/libexec/rocprofiler-compute]$ {shlex.join(cmd)}"
)
subprocess.run(
    cmd,
    cwd=f"{OUTPUT_ARTIFACTS_DIR}/libexec/rocprofiler-compute",
    check=True,
    env=environ_vars,
)
