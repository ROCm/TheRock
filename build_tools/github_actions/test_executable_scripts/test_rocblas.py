import os
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

subprocess.run(
    [
        f"{THEROCK_BIN_DIR}/rocblas-test",
        "--yaml",
        f"{THEROCK_BIN_DIR}/rocblas_smoke.yaml",
    ],
    cwd=THEROCK_DIR,
    check=True,
)
