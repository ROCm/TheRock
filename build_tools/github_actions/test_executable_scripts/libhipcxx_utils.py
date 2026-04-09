import os
import platform
import subprocess
import sys
import logging


def prepend_env_path(env: dict, var_name: str, new_path: str):
    """
    Prepend a new path to an environment variable that contains a list of paths (e.g., PATH, LD_LIBRARY_PATH).
    """
    existing = env.get(var_name)
    if existing:
        env[var_name] = f"{new_path}{os.pathsep}{existing}"
    else:
        env[var_name] = new_path


def get_gpu_architecture_portable(therock_build_dir):
    """
    Executes rocm_agent_enumerator for Linux and offload-arch for Windows and returns last line of the output.

    Returns:
        str: The gfx architecture of the running system, or None if not available.
    """
    therock_build_dir = str(therock_build_dir)
    file_ending = ".exe" if platform.system() == "Windows" else ""
    try:
        executable = therock_build_dir + f"/lib/llvm/bin/offload-arch{file_ending}"
        result = subprocess.run(
            [executable], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        logging.info(f"DEBUG:{lines}")
        return lines[-1]

    except subprocess.CalledProcessError as e:
        print(f"Error executing offload-arch: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: offload-arch command not found", file=sys.stderr)
        return None
