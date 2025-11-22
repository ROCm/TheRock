import logging
import os
import shlex
import subprocess
from pathlib import Path
import glob
import shutil

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
env = os.environ.copy()
if os.name == 'nt':
    # hip and comgr dlls need to be copied to the same folder as exectuable
    dlls_pattern = ["amdhip64*.dll", "amd_comgr*.dll", "hiprtc*.dll"]
    dlls_to_copy = []
    catch_tests_path = f"{THEROCK_BIN_DIR}/../catch_tests"
    for pattern in dlls_pattern:
        dlls_to_copy.append(glob.glob(os.path.join(f"{THEROCK_BIN_DIR}", pattern)))
    # convert list of lists to list
    dlls_to_copy = [item for sublist in dlls_to_copy for item in sublist]
    for dll in dlls_to_copy:
        try:
            shutil.copy(dll, catch_tests_path)
            print(f"Copied: {dll} to {catch_tests_path}")
        except Exception as e:
            print(f"Error copying {file_path}: {e}")

    cmd = [
        f"ctest",
        f"--test-dir",
        catch_tests_path,
        "--output-on-failure",
        "--timeout",
        "600"
    ]
else:
    hip_library_path = f"{THEROCK_BIN_DIR}/../lib"
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = f"{hip_library_path}:{env['LD_LIBRARY_PATH']}"
    else:
        env["LD_LIBRARY_PATH"] = hip_library_path
    # Set ROCM Path, to find rocm_agent_enum etc
    ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
    env["ROCM_PATH"] = str(ROCM_PATH)

    cmd = [
        "sudo",
        "-E",
        f"ctest",
        "--test-dir",
        f"{THEROCK_BIN_DIR}/../share/hip/catch_tests",
        "--output-on-failure",
        "--repeat",
        "until-pass:3",
        "--timeout",
        "600"
    ]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=env)
