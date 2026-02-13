#!/usr/bin/env python3
import logging
import os
import resource
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo + dirs
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(os.getenv("OUTPUT_ARTIFACTS_DIR")).resolve()
env = os.environ.copy()

env["LD_LIBRARY_PATH"] = THEROCK_DIR / "lib"
cmd = THEROCK_DIR / "share" / "hsa-amd-aqlprofile" / "run_tests.sh"

logging.info(f"++ Exec {cmd}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=env,
)
peak_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
logging.info(f"Peak RSS (children): {peak_rss} KB = {peak_rss / 1024 / 1024:.1f} Gi")
