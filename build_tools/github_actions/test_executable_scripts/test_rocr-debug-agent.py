import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# The root of our artifact instal directory.
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Important directories we need to know about.
THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()
ARTIFACTS_DIR = Path(os.getenv("OUTPUT_ARTIFACTS_DIR")).resolve()

# The test executable.
ROCR_DEBUG_AGENT_TESTS_BIN = THEROCK_BIN_DIR / "rocm-debug-agent-test"
# The test script.
ROCR_DEBUG_AGENT_TEST_SCRIPT = ARTIFACTS_DIR / "src/rocm-debug-agent-test/run-test.py"

# Check if we have a python executable and extract the path to it.
if not sys.executable or not os.path.exists(sys.executable):
    sys.exit("Error: Could not identify a valid Python executable path.")
PYTHON_EXECUTABLE = sys.executable

environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

logging.basicConfig(level=logging.INFO)

# Validate that we have the files we need for the tests.
required_files = {
    "Debug Agent Binary": ROCR_DEBUG_AGENT_TESTS_BIN,
    "Test Runner Script": ROCR_DEBUG_AGENT_TEST_SCRIPT,
}

for name, path in required_files.items():
    status = "✅" if path.is_file() else "❌"
    print(f"{status} {name}: {path}")

###########################################
# Testing logic for rocr-debug-agent
###########################################

# For rocr-debug-agent, we just need to install the artifacts and
# invoke the python test script, which in turn invokes all the tests.
#
# Optionally it would be nice to set "ulimit -c 0" to prevent a known
# non-fatal error, but it should not cause tests to fail.
cmd = [PYTHON_EXECUTABLE, str(ROCR_DEBUG_AGENT_TEST_SCRIPT), str(THEROCK_BIN_DIR)]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

# Time our run for statistical purposes.
start_time = time.perf_counter()

# Run tests.
subprocess.run(cmd, cwd=str(THEROCK_DIR), check=True, env=environ_vars)

end_time = time.perf_counter()
duration = end_time - start_time

print(f"Tests took {duration:.4f} seconds.")
