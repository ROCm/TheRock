import logging
import os
import shlex
import subprocess
from pathlib import Path
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
env = os.environ.copy()


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
        OCL_LIB_PATH = Path(THEROCK_BIN_DIR).parent / "lib"
        logging.info(f"++ Setting LD_LIBRARY_PATH={OCL_LIB_PATH}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{OCL_LIB_PATH}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = OCL_LIB_PATH


def execute_tests(env):
    if platform.system() == "Linux":
        OCLTST_PATH = str(Path(THEROCK_BIN_DIR).parent / "share" / "opencl" / "ocltst")
        cmd = [
            "./ocltst", "-J", "-m", "oclruntime.so", "-A", "oclruntime.exclude",
        ]
    elif platform.system() == "Windows":
        OCLTST_PATH = str(Path(THEROCK_BIN_DIR).parent / "tests" / "ocltst")
        cmd = [
            "ocltst.exe", "-J", "-m", "oclruntime.dll", "-A", "oclruntime.exclude",
        ]
    else:
        logging.info(f"++ Error: unsupported system: {platform.system()}")
        sys.exit(1)

    logging.info(f"++ Exec [{OCLTST_PATH}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=OCLTST_PATH, check=True, env=env, shell=True)


if __name__ == "__main__":
    setup_env(env)
    execute_tests(env)
