import os
import platform
import re
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


def _try_offload_arch(therock_build_dir: str, file_ending: str):
    """Try offload-arch; return gfxNNNN string or None."""
    executable = therock_build_dir + f"/lib/llvm/bin/offload-arch{file_ending}"
    try:
        result = subprocess.run([executable], capture_output=True, text=True)
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        logging.info(f"DEBUG offload-arch:{lines}")
        if lines:
            last = lines[-1]
            if last.startswith("gfx"):
                if result.returncode != 0:
                    logging.warning(
                        f"offload-arch exited {result.returncode} but returned '{last}'; using it"
                    )
                return last
        if result.returncode != 0:
            logging.warning(f"offload-arch failed (exit {result.returncode}): {result.stderr}")
    except FileNotFoundError:
        logging.warning("offload-arch not found")
    return None


def _try_rocminfo(therock_build_dir: str, file_ending: str):
    """Try rocminfo (uses HSA runtime, works on MxGPU); return gfxNNNN or None."""
    executable = therock_build_dir + f"/bin/rocminfo{file_ending}"
    try:
        result = subprocess.run([executable], capture_output=True, text=True)
        # rocminfo output contains lines like:  Name:                    gfx1101
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Name:"):
                match = re.search(r"gfx\d+", line)
                if match:
                    arch = match.group(0)
                    logging.info(f"DEBUG rocminfo detected: {arch}")
                    return arch
    except FileNotFoundError:
        logging.warning("rocminfo not found")
    return None


def get_gpu_architecture_portable(therock_build_dir):
    """
    Detect the GPU gfxNNNN architecture string.

    Tries offload-arch first; falls back to rocminfo (which uses the HSA
    runtime and works correctly on MxGPU virtual GPUs where offload-arch
    may return an incorrect default or fail entirely).

    Returns:
        str: The gfx architecture of the running system, or None if not available.
    """
    therock_build_dir = str(therock_build_dir)
    file_ending = ".exe" if platform.system() == "Windows" else ""

    arch = _try_offload_arch(therock_build_dir, file_ending)
    if arch:
        return arch

    arch = _try_rocminfo(therock_build_dir, file_ending)
    if arch:
        return arch

    return None
