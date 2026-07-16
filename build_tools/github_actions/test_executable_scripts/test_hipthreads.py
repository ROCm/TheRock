# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runs the hipThreads lit test suite against pre-built TheRock artifacts.

Unlike libhipcxx (which re-runs CMake + a ci/ shell script), hipThreads ships a
self-contained lit suite driven entirely by environment variables (see the
hipThreads test/lit.cfg). This script:

  1. Points lit at the test sources packaged in the `test` artifact
     (OUTPUT_ARTIFACTS_DIR/hipthreads, produced by HIPTHREADS_COPY_TO_BUILD).
  2. Makes the pre-built static library (libhipthreads.a / hipthreads.lib)
     discoverable at the
     `<HIPTHREADS_BUILD_DIR>/lib` path that lit.cfg's link flags expect.
  3. Invokes lit directly.
"""

import json
import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path

from libhipcxx_utils import get_gpu_architecture_portable, prepend_env_path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

IS_WINDOWS = platform.system() == "Windows"
# The static library is libhipthreads.a on Linux but hipthreads.lib on Windows.
STATIC_LIB_NAME = "hipthreads.lib" if IS_WINDOWS else "libhipthreads.a"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_rocm_version() -> str:
    """Loads the rocm-version from the repository's version.json file."""
    version_file = THEROCK_DIR / "version.json"
    logging.info(f"Loading ROCm version from: {version_file}")
    with open(version_file, "rt") as f:
        loaded_file = json.load(f)
        return loaded_file["rocm-version"]


ROCM_VERSION = load_rocm_version()
logging.info(f"ROCm version: {ROCM_VERSION}")

environ_vars = os.environ.copy()

# Resolve absolute paths
OUTPUT_ARTIFACTS_PATH = Path(OUTPUT_ARTIFACTS_DIR).resolve()
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()

# Set up ROCm/HIP environment
environ_vars["ROCM_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["HIP_DEVICE_LIB_PATH"] = str(
    OUTPUT_ARTIFACTS_PATH / "lib/llvm/amdgcn/bitcode/"
)
environ_vars["HIP_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
environ_vars["HIP_PLATFORM"] = "amd"
environ_vars["ROCM_VERSION"] = str(ROCM_VERSION)
# Kill any individual test that hangs (e.g. a hip::thread scheduler deadlock)
# after this many seconds, so one stuck test can't burn the whole job timeout.
# Honored by the vendored test executor (test/utils/run.py).
environ_vars["HIPTHREADS_TEST_TIMEOUT"] = "20"
# hipThreads' persistent scheduler kernel spawns numVcores = CU_count *
# HIPTHREADS_VCORES_PER_WGP workgroups; the default (16) occupies the whole GPU
# and can deadlock the suite under shared/over-subscribed CI runners. This is a
# RUNTIME setting (read via getenv in thread.cxx), so we dial it to 1 here for the
# test run only — the shipped library keeps its default. lit.cfg forwards this var
# to each spawned test via run.py's --env.
environ_vars["HIPTHREADS_VCORES_PER_WGP"] = "1"

# Add ROCm binaries to PATH (also the hipcc location).
prepend_env_path(environ_vars, "PATH", str(THEROCK_BIN_PATH))

# Set library / loader paths. On Windows the ROCm DLLs live in bin/ and PATH is
# the DLL search path; on Linux the shared libs are under lib/ on LD_LIBRARY_PATH.
if IS_WINDOWS:
    prepend_env_path(environ_vars, "PATH", str(OUTPUT_ARTIFACTS_PATH / "bin"))
else:
    prepend_env_path(
        environ_vars, "LD_LIBRARY_PATH", str(OUTPUT_ARTIFACTS_PATH / "lib")
    )

# Windows ONLY: hipcc has no GPU auto-detection there, so without --offload-arch
# it defaults to gfx906 and links against the wrong arch. lit.cfg turns the
# HIP_ARCHITECTURES env var into --offload-arch, so resolve a CONCRETE arch (e.g.
# gfx1100) here and pass it through. We probe offload-arch (like libhipcxx) rather
# than parsing AMDGPU_FAMILIES, whose gfx110X testers report the wildcard family
# "gfx110X-all" that is not a compilable target. offload-arch.exe loads ROCm DLLs
# from PATH, so prime os.environ's PATH first (the helper runs it with os.environ);
# see TheRock #2019 / test_sanity.py.
#
# On Linux we deliberately do NOT set this: hipcc auto-detects the GPU and the lit
# suite has always run green without --offload-arch, so leave that path untouched.
if IS_WINDOWS:
    os.environ["PATH"] = (
        str(OUTPUT_ARTIFACTS_PATH / "bin") + os.pathsep + os.environ.get("PATH", "")
    )
    gpu_arch = get_gpu_architecture_portable(OUTPUT_ARTIFACTS_DIR)
    logging.info(f"++ Detected GPU architecture: {gpu_arch}")
    if gpu_arch:
        environ_vars["HIP_ARCHITECTURES"] = gpu_arch
    else:
        # A miss means hipcc's gfx906 default kicks in and the link fails loudly
        # on its own; surface why here.
        logging.warning(
            "Could not detect GPU architecture; lit will fall back to hipcc's "
            "gfx906 default and likely fail to link."
        )

# The hipThreads lit suite is self-contained and resolves all of its paths from
# these three environment variables (see hipThreads test/lit.cfg):
#   HIPTHREADS_SOURCE_DIR  -> the source tree (test/, inc/) packaged in the artifact
#   HIPTHREADS_BUILD_DIR   -> where the pre-built libhipthreads.a is found (links -L <dir>/lib)
#   ROCM_PATH              -> hipcc + ROCm/libhipcxx headers
HIPTHREADS_SOURCE_DIR = OUTPUT_ARTIFACTS_PATH / "hipthreads"
environ_vars["HIPTHREADS_SOURCE_DIR"] = str(HIPTHREADS_SOURCE_DIR)

# lit.cfg links against `-L <HIPTHREADS_BUILD_DIR>/lib -lhipthreads`. The dev artifact
# stages the static library at <artifacts>/lib/hipthreads/<lib> (libhipthreads.a on
# Linux, hipthreads.lib on Windows), so we point HIPTHREADS_BUILD_DIR at a location
# whose `lib/` contains that archive.
HIPTHREADS_BUILD_DIR = OUTPUT_ARTIFACTS_PATH / "hipthreads"
hipthreads_lib_dir = HIPTHREADS_BUILD_DIR / "lib"
hipthreads_lib_dir.mkdir(parents=True, exist_ok=True)

staged_lib_dir = OUTPUT_ARTIFACTS_PATH / "lib" / "hipthreads"
staged_lib = staged_lib_dir / STATIC_LIB_NAME
if not staged_lib.exists():
    # Fall back to globbing in case the staged name/path differs from expectation,
    # so a naming surprise surfaces a clear log instead of a silent link failure.
    candidates = sorted(staged_lib_dir.glob("*hipthreads*"))
    logging.error(
        f"Pre-built library not found at {staged_lib}. "
        f"Candidates in {staged_lib_dir}: {[p.name for p in candidates]}"
    )
    if not candidates:
        raise FileNotFoundError(staged_lib)
    staged_lib = candidates[0]
    logging.warning(f"Falling back to staged library: {staged_lib}")

linked_lib = hipthreads_lib_dir / staged_lib.name
if not linked_lib.exists():
    # Hard link (fall back to copy) so lit's -L <build>/lib finds the archive.
    try:
        os.link(staged_lib, linked_lib)
    except OSError:
        import shutil

        shutil.copy2(staged_lib, linked_lib)
environ_vars["HIPTHREADS_BUILD_DIR"] = str(HIPTHREADS_BUILD_DIR)

logging.info(f"ROCM_PATH: {environ_vars['ROCM_PATH']}")
logging.info(f"HIPTHREADS_SOURCE_DIR: {environ_vars['HIPTHREADS_SOURCE_DIR']}")
logging.info(f"HIPTHREADS_BUILD_DIR: {environ_vars['HIPTHREADS_BUILD_DIR']}")
logging.info(f"PATH: {environ_vars['PATH']}")

# Report the size of the (now target-neutral) static archive so each CI run
# records how large the all-architecture library is.
_lib_size_bytes = staged_lib.stat().st_size
logging.info(
    f"{staged_lib.name} size: {_lib_size_bytes} bytes "
    f"({_lib_size_bytes / (1024 * 1024):.2f} MiB) at {staged_lib}"
)

if not HIPTHREADS_SOURCE_DIR.exists():
    logging.error(f"Test sources not found at: {HIPTHREADS_SOURCE_DIR}")
    raise FileNotFoundError(HIPTHREADS_SOURCE_DIR)

# Run the lit suite.
cmd = [
    "lit",
    "-v",
    "-j",
    "1",
    str(HIPTHREADS_SOURCE_DIR / "test"),
]
logging.info(f"++ Exec [{os.getcwd()}]$ {shlex.join(cmd)}")
subprocess.run(cmd, check=True, env=environ_vars)
