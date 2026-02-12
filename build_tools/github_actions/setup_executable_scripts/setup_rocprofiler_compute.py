import logging
import shlex
import subprocess
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_OUTPUT_DIR = str(THEROCK_DIR / "build")

VENV_DIR = os.getenv("VENV_DIR")
PYTHON_EXECUTABLE = VENV_DIR + "/bin/python"

# Set up pip
setup_cmd = [
    f"{PYTHON_EXECUTABLE}",
    "-m",
    "ensurepip",
    "--upgrade",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(setup_cmd)}")
subprocess.run(
    setup_cmd,
    cwd=THEROCK_DIR,
    check=True,
)

# Install requirements
requirements_dir = f"{THEROCK_OUTPUT_DIR}/libexec/rocprofiler-compute"
cmd = [
    f"{PYTHON_EXECUTABLE}",
    "-m",
    "pip",
    "install",
    "-r",
    f"{requirements_dir}/requirements.txt",
    "-r",
    f"{requirements_dir}/requirements-test.txt",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
