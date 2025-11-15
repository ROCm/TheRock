import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

cmd = [
    f"ctest",
    f"--test-dir",
    f"{THEROCK_BIN_DIR}/../share/hip/catch_tests"
]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True)
