import logging
import os
import shlex
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_OUTPUT_DIR = str(THEROCK_DIR / "build")

environ_vars = os.environ.copy()

requirements_dir = f"{THEROCK_OUTPUT_DIR}/share/rocprofiler-sdk/tests"
cmd = [
    "CC=clang"
    "CXX=clang++"
    "pip",
    "install",
    "-r",
    f"{requirements_dir}/requirements.txt",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
