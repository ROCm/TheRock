import os
import subprocess

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

subprocess.run([
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
    "until-pass:3"
    ""
    ], cwd=THEROCK_BIN_DIR, check=True)
