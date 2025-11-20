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
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR", "")
env = os.environ.copy()
platform = os.getenv("RUNNER_OS", "linux").lower()

# Sharding
env = os.environ.copy()
env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
env["GTEST_TOTAL_SHARDS"] = str(int(os.getenv("TOTAL_SHARDS", "1")))

env["LD_LIBRARY_PATH"] = f"../../lib/"
cmd = "run_tests.sh"
TEST_DIR = f"{THEROCK_BIN_DIR}/../share/hsa-amd-aqlprofile"
cmd = f"{THEROCK_BIN_DIR}/../share/hsa-amd-aqlprofile/run_tests.sh"

logging.info(f"++ Exec [{TEST_DIR}]$ {cmd}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=env,
)
