#!/usr/bin/env python3
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

def is_asan():
    ARTIFACT_GROUP = os.getenv("ARTIFACT_GROUP")
    return "asan" in ARTIFACT_GROUP

# repo + dirs
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(os.getenv("OUTPUT_ARTIFACTS_DIR")).resolve()
env = os.environ.copy()

# Importing get_asan_lib_path from github_actions_utils.py
sys.path.append(str(SCRIPT_DIR.parent))
from github_actions_utils import get_asan_lib_path

if is_asan():
    asan_lib_path = get_asan_lib_path(THEROCK_DIR / "bin")
    env["LD_PRELOAD"] = asan_lib_path

env["LD_LIBRARY_PATH"] = THEROCK_DIR / "lib"
cmd = THEROCK_DIR / "share" / "hsa-amd-aqlprofile" / "run_tests.sh"

logging.info(f"++ Exec {cmd}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=env,
)
