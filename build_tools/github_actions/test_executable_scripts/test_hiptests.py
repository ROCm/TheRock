import logging
import os
import shlex
import subprocess
from pathlib import Path
import platform
import shutil
import json
import sys
import platform

logging.basicConfig(level=logging.INFO)
THEROCK_BIN_DIR_STR = os.getenv("THEROCK_BIN_DIR")
THEROCK_BIN_DIR = Path(THEROCK_BIN_DIR_STR)
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
SHARD_INDEX = int(os.getenv("SHARD_INDEX", 1)) - 1
TOTAL_SHARDS = int(os.getenv("TOTAL_SHARDS", 1))
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
os_type = platform.system().lower()
CATCH_TESTS_PATH = str(Path(THEROCK_BIN_DIR).parent / "share" / "hip" / "catch_tests")

# Importing is_asan from github_actions_utils.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import is_asan

env = os.environ.copy()

if THEROCK_BIN_DIR_STR is None:
    logging.info(
        "++ Error: env(THEROCK_BIN_DIR) is not set. Please set it before executing tests."
    )
    sys.exit(1)

if not os.path.isdir(CATCH_TESTS_PATH):
    logging.info(f"++ Error: catch tests not found in {CATCH_TESTS_PATH}")
    sys.exit(1)

# TODO(#3204): Re-enable tests once issues are resolved
TEST_TO_IGNORE = {
    "gfx94X-dcgpu": {
        "linux": [
            "Unit_hipGetProcAddress_spt_Stream",
        ]
    },
    "gfx950-dcgpu": {
        "linux": [
            "Unit_hipManagedKeyword_SingleGpu",
            "Unit_hipGetProcAddress_spt_Stream",
        ]
    },
}


def get_asan_lib_path():
    arch = platform.machine()
    CLANG_PATH = str(Path(THEROCK_BIN_DIR).parent / "lib" / "llvm" / "bin" / "clang++")
    cmd = [f"{CLANG_PATH}", f"--print-file-name=libclang_rt.asan-{arch}.so"]
    logging.info(f"++ Exec [{CLANG_PATH}]$ {shlex.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


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
        # For ASAN mode, we preload it for test count query and test running
        if is_asan():
            env["LD_PRELOAD"] = get_asan_lib_path()
            # TODO: enable this when we have symbolizer patch in
            # env["ASAN_SYMBOLIZER_PATH"] = str(Path(THEROCK_BIN_DIR).parent / "lib" / "llvm" / "bin" / "llvm-symbolizer")
    else:
        copy_dlls_exe_path()


def execute_tests(env):
    # Allow for more time in ASAN mode to run the tests.
    timeout = 1500 if is_asan() else 600
    cmd = [
        "ctest",
        "--tests-information",
        f"{SHARD_INDEX},,{TOTAL_SHARDS}",
        "--test-dir",
        CATCH_TESTS_PATH,
        "--output-on-failure",
        "--timeout",
        f"{timeout}",
    ]

    if AMDGPU_FAMILIES in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[AMDGPU_FAMILIES]:
        ignored_tests = TEST_TO_IGNORE[AMDGPU_FAMILIES][os_type]
        cmd.extend(["--exclude-regex", "|".join(ignored_tests)])

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=env)


if __name__ == "__main__":
    setup_env(env)
    execute_tests(env)
