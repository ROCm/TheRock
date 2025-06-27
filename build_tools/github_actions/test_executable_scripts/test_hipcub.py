import os
import subprocess

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

subprocess.run([
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipcub",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "300",
    "--repeat",
    "until-pass:3"
    ""
    ], cwd=THEROCK_BIN_DIR, check=True)
