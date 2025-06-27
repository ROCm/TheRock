import os
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

subprocess.run(
    [
        "ctest",
        "--test-dir",
        f"{THEROCK_BIN_DIR}/rocthrust",
        "--output-on-failure",
        "--parallel",
        "8",
        "--exclude-regex",
        "^copy.hip$|scan.hip",
        "--timeout",
        "60",
        "--repeat",
        "until-pass:3",
    ],
    cwd=THEROCK_DIR,
    check=True,
)
