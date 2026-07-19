import os
import platform
import re
import subprocess
import sys
import logging
import tempfile
import textwrap


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


def _try_hip_runtime_detection(therock_build_dir: str, file_ending: str):
    """
    Compile and run a tiny host-only HIP program that calls hipGetDeviceProperties
    to read gcnArchName from the HIP runtime. This works on MxGPU virtual GPUs
    where offload-arch and rocminfo may fail or return wrong results.
    """
    # amdclang++ lives in lib/llvm/bin/ on both Linux and Windows in a TheRock
    # build; on Linux bin/amdclang++ is a symlink to the same binary.
    amdclangpp = None
    for candidate in [
        therock_build_dir + f"/lib/llvm/bin/amdclang++{file_ending}",
        therock_build_dir + f"/bin/amdclang++{file_ending}",
    ]:
        if os.path.exists(candidate):
            amdclangpp = candidate
            break

    if not amdclangpp:
        logging.warning("amdclang++ not found for HIP runtime GPU detection")
        return None

    src = textwrap.dedent("""\
        #include <hip/hip_runtime.h>
        #include <cstdio>
        int main() {
            hipDeviceProp_t p;
            if (hipGetDeviceProperties(&p, 0) == hipSuccess)
                printf("%s\\n", p.gcnArchName);
            return 0;
        }
    """)

    with tempfile.NamedTemporaryFile(
        suffix=".cpp", mode="w", delete=False, prefix="detect_gpu_"
    ) as f:
        f.write(src)
        src_path = f.name

    exe_path = src_path.replace(".cpp", file_ending or "")

    try:
        hip_include = os.path.join(therock_build_dir, "include")
        hip_lib = os.path.join(therock_build_dir, "lib")

        compile_result = subprocess.run(
            [
                amdclangpp,
                "--cuda-host-only",
                "-x", "hip",
                f"-I{hip_include}",
                src_path,
                f"-L{hip_lib}",
                "-lamdhip64",
                "-o", exe_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if compile_result.returncode != 0:
            logging.warning(
                f"GPU detect compile failed: {compile_result.stderr[:300]}"
            )
            return None

        run_result = subprocess.run(
            [exe_path], capture_output=True, text=True, timeout=10
        )
        arch = run_result.stdout.strip()
        if arch.startswith("gfx"):
            logging.info(f"HIP runtime detected GPU arch: {arch}")
            return arch
    except Exception as e:
        logging.warning(f"HIP runtime detection failed: {e}")
    finally:
        for p in [src_path, exe_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

    return None


def get_gpu_architecture_portable(therock_build_dir):
    """
    Detect the GPU gfxNNNN architecture string.

    Tries multiple detection methods in order:
    1. offload-arch (fast, but fails on some MxGPU virtual GPUs)
    2. rocminfo (uses HSA runtime)
    3. Compile+run a tiny host HIP program (uses HIP runtime, most reliable)

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

    arch = _try_hip_runtime_detection(therock_build_dir, file_ending)
    if arch:
        return arch

    return None
