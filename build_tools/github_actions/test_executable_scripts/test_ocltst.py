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
THEROCK_DIR = Path(THEROCK_BIN_DIR).resolve().parent
env = os.environ.copy()
is_windows = platform.system() == "Windows"

def setup_env(env):
    ROCM_PATH = Path(THEROCK_DIR)
    env["ROCM_PATH"] = str(ROCM_PATH)
    if not is_windows:
        ROCK_LIB_PATH = Path(THEROCK_DIR) / "lib"
        OCL_LIB = Path(ROCK_LIB_PATH) / "opencl"
        LLVM_LIB = Path(ROCK_LIB_PATH) / "llvm" / "lib"
        ROCM_SYSDEPS_LIB = Path(ROCK_LIB_PATH) / "rocm_sysdeps" / "lib"
        OCL_ICD_VENDORS = Path(THEROCK_DIR) / "etc" / "OpenCL" / "vendors"
        LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH")
        if LD_LIBRARY_PATH is not None:
            LD_LIBRARY_PATH = Path(LD_LIBRARY_PATH)
        env["LD_LIBRARY_PATH"] = (
            f"{ROCK_LIB_PATH}:{OCL_LIB}:{LLVM_LIB}:{ROCM_SYSDEPS_LIB}:{LD_LIBRARY_PATH}"
        )
        env["OCL_ICD_VENDORS"] = f"{OCL_ICD_VENDORS}/"


def execute_tests(env):
    if not is_windows:
        OCLTST_PATH = Path(THEROCK_DIR) / "share" / "opencl" / "ocltst"
        cmd = [
            "./ocltst",
            "-J",
            "-m",
            "liboclruntime.so",
            "-A",
            "oclruntime.exclude",
        ]

        env["LD_LIBRARY_PATH"] = f"{OCLTST_PATH}:{env['LD_LIBRARY_PATH']}"
        logging.info(f"++ Setting LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
        logging.info(f"++ Setting OCL_ICD_VENDORS={env['OCL_ICD_VENDORS']}")
        shell_var = False
    else:
        OCLTST_PATH = Path(THEROCK_DIR) / "tests" / "ocltst"
        cmd = [
            "ocltst.exe",
            "-J",
            "-m",
            "oclruntime.dll",
            "-A",
            "oclruntime.exclude",
        ]
        shell_var = True

    logging.info(f"++ Exec [{OCLTST_PATH}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=OCLTST_PATH, check=True, env=env, shell=shell_var)


if __name__ == "__main__":
    setup_env(env)
    execute_tests(env)
