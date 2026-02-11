#!/usr/bin/env python3
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo + dirs
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(os.getenv("OUTPUT_ARTIFACTS_DIR")).resolve()
env = os.environ.copy()

env["LD_LIBRARY_PATH"] = THEROCK_DIR / "lib"
cmd = ["echo", "hello"]

logging.info(f"++ Exec {' '.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=env,
)
