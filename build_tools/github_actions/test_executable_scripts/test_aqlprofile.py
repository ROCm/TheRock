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
platform = os.getenv("RUNNER_OS", "linux").lower()

# Sharding
env = os.environ.copy()
env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
env["GTEST_TOTAL_SHARDS"] = str(int(os.getenv("TOTAL_SHARDS", "1")))

cmd = "run_test.sh"
TEST_DIR = f"{THEROCK_DIR}/../share/share/hsa-amd-aqlprofile"

logging.info(f"++ Exec [{TEST_DIR}]$ {cmd}")

subprocess.run(
    cmd,
    cwd=TEST_DIR,
    check=True,
    env=environ_vars,
)
