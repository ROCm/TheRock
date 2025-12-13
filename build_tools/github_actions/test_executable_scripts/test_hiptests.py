import logging
import os
import shlex
import subprocess
from pathlib import Path
import glob
import shutil
import json
import sys


THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
CATCH_TESTS_PATH = f"{THEROCK_BIN_DIR}/../share/hip/catch_tests"

env = os.environ.copy()

def get_test_count():
    cmd = ["ctest", "--show-only=json-v1"]
    result = subprocess.run(
        cmd,
        cwd=CATCH_TESTS_PATH,
        check=True,
        capture_output=True,
    )
    jdata = json.loads(result.stdout)
    tests = jdata["tests"]
    return len(tests)


def get_test_range_per_shard(total_test_count: int, total_shards, shard_index):
    tests_per_shard = int(total_test_count / total_shards)
    current_index = (tests_per_shard * (shard_index - 1)) + 1
    end_index = current_index + tests_per_shard - 1
    if shard_index == total_shards:
        # Adjust last few tests
        end_index = total_test_count
    logging.info(
        """
        Shard index: {shard_index},
        total shards: {total_shards},
        tests per shard: {tests_per_shard},
        current index: {current_index},
        to: {end_index}
        """
    )
    return [current_index, end_index]


if sys.platform == "win32":
    # hip and comgr dlls need to be copied to the same folder as exectuable
    dlls_pattern = ["amdhip64*.dll", "amd_comgr*.dll", "hiprtc*.dll"]
    dlls_to_copy = []
    for pattern in dlls_pattern:
        dlls_to_copy.extend(Path(THEROCK_BIN_DIR).glob(pattern))
    for dll in dlls_to_copy:
        try:
            shutil.copy(dll, CATCH_TESTS_PATH)
            logging.info(f"++ Copied: {dll} to {CATCH_TESTS_PATH}")
        except Exception as e:
            logging.info(f"Error copying {dll}: {e}")

# catch/ctest framework 
# Linux
#   does not honor LD_LIBRARY_PATH on Linux
#   tests are hardcoded to look at THEROCK_BIN_DIR or /opt/rocm/lib path
# Windows
#   tests load the dlls present in the local exe folder
# Set ROCM Path, to find rocm_agent_enum etc
ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
env["ROCM_PATH"] = str(ROCM_PATH)

total_tests = get_test_count()
test_range = get_test_range_per_shard(total_tests, int(TOTAL_SHARDS), int(SHARD_INDEX))
index_start = test_range[0]
index_end = test_range[1]
cmd = [
    "ctest",
    "-I",
    f"{index_start},{index_end}",
    "--test-dir",
    CATCH_TESTS_PATH,
    "--output-on-failure",
    "--repeat",
    "until-pass:3",
    "--timeout",
    "600"
]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=env)
