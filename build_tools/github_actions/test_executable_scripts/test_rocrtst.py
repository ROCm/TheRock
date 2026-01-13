#!/usr/bin/env python3
import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo + dirs
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(os.getenv("OUTPUT_ARTIFACTS_DIR")).resolve()
env = os.environ.copy()
platform = os.getenv("RUNNER_OS", "linux").lower()

cwd_dir = THEROCK_DIR / "bin" / "gfx942"
cmd = "./rocrtst64"

logging.info(f"++ Exec {cmd}")

subprocess.run(
    cmd,
    cwd=cwd_dir,
    check=True,
    env=env,
)
