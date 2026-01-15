import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Set up ROCm path
ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
environ_vars = os.environ.copy()
environ_vars["ROCM_PATH"] = str(ROCM_PATH)

# Resolve absolute paths
OUTPUT_ARTIFACTS_PATH = Path(OUTPUT_ARTIFACTS_DIR).resolve()
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()

# Set up PYTHONPATH
old_pythonpath = os.getenv("PYTHONPATH", "")
module_dir = f"{THEROCK_BIN_PATH}/libexec/rocprofiler-compute/tests"
if old_pythonpath:
    os.environ["PYTHONPATH"] = f"{module_dir}:{old_pythonpath}"
else:
    os.environ["PYTHONPATH"] = module_dir

# Set up PATH
old_path = os.getenv("PATH", "")
rocm_bin = str(THEROCK_BIN_PATH)
if old_path:
    os.environ["PATH"] = f"{rocm_bin}:{old_path}"
else:
    os.environ["PATH"] = rocm_bin

# Set up LD_LIBRARY_PATH
old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "")
rocm_lib = str(OUTPUT_ARTIFACTS_PATH / "lib")
if old_ld_lib_path:
    os.environ["LD_LIBRARY_PATH"] = (
        f"{rocm_lib}:{rocm_lib}/rocm_sysdeps/lib:{old_ld_lib_path}"
    )
else:
    os.environ["LD_LIBRARY_PATH"] = rocm_lib

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
