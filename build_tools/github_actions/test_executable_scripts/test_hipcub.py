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
        f"{THEROCK_BIN_DIR}/hipcub",
        "--output-on-failure",
        "--parallel",
        "8",
        "--timeout",
        "300",
        "--repeat",
        "until-pass:3",
    ],
    cwd=THEROCK_DIR,
    check=True,
)
