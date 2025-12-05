#!/usr/bin/env python3
import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo + dirs
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR", "")
env = os.environ.copy()
platform = os.getenv("RUNNER_OS", "linux").lower()

# Sharding
env = os.environ.copy()
env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
env["GTEST_TOTAL_SHARDS"] = str(int(os.getenv("TOTAL_SHARDS", "1")))

env["LD_LIBRARY_PATH"] = f"../../lib/"
cmd = "run_tests.sh"
cmd = f"{OUTPUT_ARTIFACTS_DIR}/share/hsa-amd-aqlprofile/run_tests.sh"

logging.info(f"++ Exec {cmd}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=env,
)
