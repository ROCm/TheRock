import logging
import os
import shlex
import subprocess
from pathlib import Path
import glob
import shutil
import json
import sys
import platform

logging.basicConfig(level=logging.INFO)
THEROCK_BIN_DIR_STR = os.getenv("THEROCK_BIN_DIR")
if THEROCK_BIN_DIR_STR is None:
    logging.info(
        "++ Error: env(THEROCK_BIN_DIR) is not set. Please set it before executing tests."
    )
    sys.exit(1)
THEROCK_BIN_DIR = Path(THEROCK_BIN_DIR_STR)
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
CATCH_TESTS_PATH = str(Path(THEROCK_BIN_DIR).parent / "share" / "hip" / "catch_tests")
if not os.path.isdir(CATCH_TESTS_PATH):
    logging.info(f"++ Error: catch tests not found in {CATCH_TESTS_PATH}")
    sys.exit(1)
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
        # Retrieve remaining tests
        end_index = total_test_count
    logging.info(
        f"""++ hip-tests ctest: shard {shard_index} / {total_shards}. Running:{tests_per_shard} tests"""
    )
    return [current_index, end_index]


def copy_dlls_exe_path():
    if platform.system() == "Windows":
        # hip and comgr dlls need to be copied to the same folder as exectuable
        dlls_pattern = ["amdhip64*.dll", "amd_comgr*.dll", "hiprtc*.dll"]
        dlls_to_copy = []
        for pattern in dlls_pattern:
            dlls_to_copy.extend(THEROCK_BIN_DIR.glob(pattern))
        for dll in dlls_to_copy:
            try:
                shutil.copy(dll, CATCH_TESTS_PATH)
                logging.info(f"++ Copied: {dll} to {CATCH_TESTS_PATH}")
            except Exception as e:
                logging.info(f"++ Error copying {dll}: {e}")


def setup_env(env):
    # catch/ctest framework
    # Linux
    #   LD_LIBRARY_PATH needs to be used
    #   tests are hardcoded to look at THEROCK_BIN_DIR or /opt/rocm/lib path
    # Windows
    #   tests load the dlls present in the local exe folder
    # Set ROCM Path, to find rocm_agent_enum etc
    ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
    env["ROCM_PATH"] = str(ROCM_PATH)
    if platform.system() == "Linux":
        HIP_LIB_PATH = Path(THEROCK_BIN_DIR).parent / "lib"
        logging.info(f"++ Setting LD_LIBRARY_PATH={HIP_LIB_PATH}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{HIP_LIB_PATH}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = HIP_LIB_PATH
    else:
        copy_dlls_exe_path()


def execute_tests(env):
    total_tests = get_test_count()
    index_start, index_end = get_test_range_per_shard(
        total_tests, int(TOTAL_SHARDS), int(SHARD_INDEX)
    )
    cmd = [
        "ctest",
        "-I",
        f"{index_start},{index_end}",
        "--test-dir",
        CATCH_TESTS_PATH,
        "--output-on-failure",
        "--timeout",
        "600",
    ]
    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=env)


if __name__ == "__main__":
    setup_env(env)
    execute_tests(env)
