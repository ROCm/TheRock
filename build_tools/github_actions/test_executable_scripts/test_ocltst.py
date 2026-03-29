import logging
import os
import shlex
import subprocess
from pathlib import Path
import sys
import platform
import shutil

logging.basicConfig(level=logging.INFO)
THEROCK_BIN_DIR_STR = os.getenv("THEROCK_BIN_DIR")
if THEROCK_BIN_DIR_STR is None:
    logging.info(
        "++ Error: env(THEROCK_BIN_DIR) is not set. Please set it before executing tests."
    )
    sys.exit(1)
THEROCK_BIN_DIR = Path(THEROCK_BIN_DIR_STR).resolve()
THEROCK_DIR = Path(THEROCK_BIN_DIR).parent
env = os.environ.copy()
is_windows = platform.system() == "Windows"


# copies the dlls to local ocltst path.
# to overwrite the registry entries
def copy_dlls_exe_path(ocltst_path):
    if platform.system() == "Windows":
        # hip and comgr dlls need to be copied to the same folder as exectuable
        dlls_pattern = ["amdocl*.dll", "amd_comgr*.dll", "OpenCL.dll"]
        dlls_to_copy = []
        for pattern in dlls_pattern:
            dlls_to_copy.extend(THEROCK_BIN_DIR.glob(pattern))
        for dll in dlls_to_copy:
            try:
                shutil.copy(dll, ocltst_path)
                logging.info(f"++ Copied: {dll} to {ocltst_path}")
            except Exception as e:
                logging.info(f"++ Error copying {dll}: {e}")


# returns ocltst path
def setup_env(env):
    ROCM_PATH = Path(THEROCK_DIR)
    env["ROCM_PATH"] = str(ROCM_PATH)
    if not is_windows:
        ROCK_LIB_PATH = Path(THEROCK_DIR) / "lib"
        OCL_LIB = Path(ROCK_LIB_PATH) / "opencl"
        LLVM_LIB = Path(ROCK_LIB_PATH) / "llvm" / "lib"
        ROCM_SYSDEPS_LIB = Path(ROCK_LIB_PATH) / "rocm_sysdeps" / "lib"
        OCL_ICD_VENDORS = Path(THEROCK_DIR) / "etc" / "OpenCL" / "vendors"
        OCLTST_PATH = Path(THEROCK_DIR) / "share" / "opencl" / "ocltst"
        LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH")
        if LD_LIBRARY_PATH is not None:
            LD_LIBRARY_PATH = Path(LD_LIBRARY_PATH)
        env["LD_LIBRARY_PATH"] = (
            f"{ROCK_LIB_PATH}:{OCL_LIB}:{LLVM_LIB}:{ROCM_SYSDEPS_LIB}:{LD_LIBRARY_PATH}:{OCLTST_PATH}"
        )
        env["OCL_ICD_VENDORS"] = f"{OCL_ICD_VENDORS}/"
        logging.info(f"++ Setting LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
        logging.info(f"++ Setting OCL_ICD_VENDORS={env['OCL_ICD_VENDORS']}")
    else:

        OCLTST_PATH = Path(THEROCK_DIR) / "tests" / "ocltst"
        copy_dlls_exe_path(OCLTST_PATH)
        OCL_DLL_FILE = Path(OCLTST_PATH) / "amdocl64.dll"
        OCL_ICD_DLL = Path(THEROCK_BIN_DIR) / "OpenCL.dll"
        env["OCL_ICD_FILENAMES"] = str(OCL_DLL_FILE)
        logging.info(f"++ Setting OCL_ICD_FILENAMES={env['OCL_ICD_FILENAMES']}")
    return OCLTST_PATH


def execute_tests(env):
    OCLTST_PATH = setup_env(env)
    OCLTST = Path(OCLTST_PATH) / "ocltst"
    module = "liboclruntime.so" if not is_windows else "oclruntime.dll"
    # command to execute ocltst tests
    cmd = [
        f"{OCLTST}",
        "-m",  # module to test
        f"{module}",
        "-s",  # threads to spawn/use
        f"16",
        "-A",  # exclude tests in the file
        "oclruntime.exclude",  # perf related tests skipped
    ]
    logging.info(f"++ Exec [{OCLTST_PATH}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=OCLTST_PATH, check=True, env=env, shell=False)


if __name__ == "__main__":
    execute_tests(env)
