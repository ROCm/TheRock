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

old_pythonpath = os.getenv("PYTHONPATH", "")
old_path = os.getenv("PATH", "")
test_dir = f"{OUTPUT_ARTIFACTS_DIR}/libexec/rocprofiler-compute/tests"

if old_pythonpath:
    os.environ["PYTHONPATH"] = f"{test_dir}:{old_pythonpath}"
else:
    os.environ["PYTHONPATH"] = test_dir

if old_path:
    os.environ["PATH"] = f"{THEROCK_BIN_DIR}:{old_path}"
else:
    os.environ["PATH"] = THEROCK_BIN_DIR

# Run tests
cmd = [
    "ctest",
    "--test-dir",
    f"{OUTPUT_ARTIFACTS_DIR}/libexec/rocprofiler-compute",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "1800",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
