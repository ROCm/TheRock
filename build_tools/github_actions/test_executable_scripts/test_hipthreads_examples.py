# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds and runs hipThreads example apps against pre-built TheRock artifacts.

Unlike test_hipthreads.py (which runs the lit unit-test suite), this script
exercises the public consumer path: it configures each example with
find_package(hipthreads) / find_package(rocthrust) against the packaged ROCm
artifact, builds it, runs it, and checks for a clean exit plus an expected
output marker. This catches breakage in the installed CMake package and headers
that the unit tests do not.

The example sources are packaged in the `test` artifact (HIPTHREADS_COPY_TO_BUILD
copies the source tree, including examples/, into the build dir which the test
artifact globs as hipthreads/**/*).
"""

import json
import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path

from libhipcxx_utils import (
    get_gpu_architecture_portable,
    prepend_env_path,
)

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Each example: where it lives under <artifacts>/hipthreads/examples, the binary
# name produced, the argv to run it with, and a substring expected on stdout.
# stdout is captured to a file (the raytracer emits a multi-million-line PPM) and
# scanned for the marker.
EXAMPLES = [
    {
        "name": "saxpy",
        "subdir": "saxpy/step3-simdize",
        "binary": "saxpy",
        "args": [],
        "marker": "Time to run saxpy(",
    },
    {
        "name": "inOneWeekend",
        "subdir": "InOneWeekendRaytracer/step4-simdize",
        "binary": "inOneWeekend",
        "args": [],
        "marker": "P3",
    },
    {
        # Pass an explicit matrix path so spmm runs ONE tiny matrix instead of its
        # default pair (the other default matrices are LFS files that CI does not
        # `git lfs pull`). test_general.mtx is committed as plain text (exempted
        # from the *.mtx LFS rule in .gitattributes); the path is relative to the
        # example dir, which is run_example's cwd.
        "name": "spmm",
        "subdir": "sparse-mat-mul/step3-hipthread-port",
        "binary": "spmm",
        "args": ["../data/test_general.mtx"],
        "marker": "Time(",
    },
]

# Per-example wall-clock cap. The examples run at full benchmark sizes and the CI
# run sets HIPTHREADS_VCORES_PER_WGP=1 (see build_environment), so they can be
# slow; keep this comfortably under the job timeout in fetch_test_configurations.py.
RUN_TIMEOUT_SECONDS = 1800

IS_WINDOWS = platform.system() == "Windows"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""


def load_rocm_version() -> str:
    """Loads the rocm-version from the repository's version.json file."""
    version_file = THEROCK_DIR / "version.json"
    logging.info(f"Loading ROCm version from: {version_file}")
    with open(version_file, "rt") as f:
        loaded_file = json.load(f)
        return loaded_file["rocm-version"]


def build_environment() -> dict:
    """Returns an environment dict configured for hipcc + ROCm from the artifact."""
    environ_vars = os.environ.copy()

    environ_vars["ROCM_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
    environ_vars["HIP_DEVICE_LIB_PATH"] = str(
        OUTPUT_ARTIFACTS_PATH / "lib/llvm/amdgcn/bitcode/"
    )
    environ_vars["HIP_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
    environ_vars["CMAKE_PREFIX_PATH"] = str(OUTPUT_ARTIFACTS_PATH)
    environ_vars["HIP_PLATFORM"] = "amd"
    environ_vars["ROCM_VERSION"] = str(ROCM_VERSION)
    environ_vars["CMAKE_GENERATOR"] = "Ninja"
    # RUNTIME setting (read via getenv in thread.cxx): dial the scheduler's
    # per-WGP vcore count down to 1 so the example binaries don't over-subscribe a
    # shared CI GPU and deadlock. The shipped library keeps its default (16).
    environ_vars["HIPTHREADS_VCORES_PER_WGP"] = "1"

    prepend_env_path(environ_vars, "PATH", str(THEROCK_BIN_PATH))
    if IS_WINDOWS:
        # ROCm DLLs live in bin/ on Windows; PATH is also the DLL search path.
        prepend_env_path(environ_vars, "PATH", str(OUTPUT_ARTIFACTS_PATH / "bin"))
    else:
        prepend_env_path(
            environ_vars, "LD_LIBRARY_PATH", str(OUTPUT_ARTIFACTS_PATH / "lib")
        )
    return environ_vars


def configure_and_build(example: dict, gpu_arch: str, environ_vars: dict) -> Path:
    """Configures + builds one example; returns the built binary path."""
    source_dir = EXAMPLES_ROOT / example["subdir"]
    if not source_dir.exists():
        raise FileNotFoundError(f"Example source dir not found: {source_dir}")

    build_dir = source_dir / "build"
    llvm_bin = OUTPUT_ARTIFACTS_PATH / "lib" / "llvm" / "bin"
    clang = llvm_bin / f"clang{EXE_SUFFIX}"
    clangxx = llvm_bin / f"clang++{EXE_SUFFIX}"

    configure_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(source_dir),
        "-GNinja",
        f"-DCMAKE_PREFIX_PATH={OUTPUT_ARTIFACTS_PATH}",
        # The examples are project(... LANGUAGES CXX HIP). Pin BOTH the CXX and
        # HIP compilers to the ROCm clang from the artifact. Otherwise CMake
        # picks the system C++ compiler for CXX (e.g. clang-18 at /usr/bin/c++)
        # while HIP uses ROCm clang, and the final link fails looking for the
        # system toolchain's compiler-rt (libclang_rt.builtins) which isn't
        # installed in the CI container.
        f"-DCMAKE_C_COMPILER={clang.as_posix()}",
        f"-DCMAKE_CXX_COMPILER={clangxx.as_posix()}",
        f"-DCMAKE_HIP_COMPILER={clangxx.as_posix()}",
        f"-DCMAKE_HIP_ARCHITECTURES={gpu_arch}",
        # The packaged hipthreads.lib is built Release (dynamic CRT, msvcrt,
        # _ITERATOR_DEBUG_LEVEL=0). Pin the examples to Release too: without an
        # explicit build type the Windows/Ninja default links the debug CRT
        # (msvcrtd, _ITERATOR_DEBUG_LEVEL=2) and lld-link /failifmismatch aborts.
        # Matches the documented per-example Windows build command.
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if IS_WINDOWS:
        # clang needs the MSVC resource compiler for the CXX language on Windows.
        configure_cmd.append("-DCMAKE_RC_COMPILER=rc.exe")
    logging.info(f"++ Configure [{example['name']}]$ {shlex.join(configure_cmd)}")
    subprocess.run(configure_cmd, check=True, env=environ_vars)

    build_cmd = ["cmake", "--build", str(build_dir)]
    logging.info(f"++ Build [{example['name']}]$ {shlex.join(build_cmd)}")
    subprocess.run(build_cmd, check=True, env=environ_vars)

    binary = build_dir / "bin" / f"{example['binary']}{EXE_SUFFIX}"
    if not binary.exists():
        raise FileNotFoundError(f"Built binary not found: {binary}")
    return binary


def run_example(example: dict, binary: Path, environ_vars: dict) -> None:
    """Runs one example; requires exit 0 and the expected output marker."""
    build_dir = binary.parent.parent
    stdout_path = build_dir / f"{example['name']}.stdout.log"

    cmd = [str(binary), *example["args"]]
    logging.info(f"++ Run [{example['name']}]$ {shlex.join(cmd)}")
    with open(stdout_path, "wt") as stdout_file:
        # cwd = source dir so any relative paths in the example resolve as it
        # expects when run from its own directory.
        result = subprocess.run(
            cmd,
            cwd=EXAMPLES_ROOT / example["subdir"],
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
            env=environ_vars,
            timeout=RUN_TIMEOUT_SECONDS,
        )

    if result.returncode != 0:
        logging.error(f"{example['name']} exited with code {result.returncode}")
        _dump_tail(stdout_path)
        raise subprocess.CalledProcessError(result.returncode, cmd)

    marker = example["marker"]
    if not _file_contains(stdout_path, marker):
        logging.error(f"{example['name']} output missing marker {marker!r}")
        _dump_tail(stdout_path)
        raise RuntimeError(f"{example['name']}: expected marker {marker!r} not found")

    logging.info(f"PASS: {example['name']} (exit 0, found marker {marker!r})")


def _file_contains(path: Path, needle: str) -> bool:
    with open(path, "rt", errors="replace") as f:
        for line in f:
            if needle in line:
                return True
    return False


def _dump_tail(path: Path, lines: int = 40) -> None:
    try:
        with open(path, "rt", errors="replace") as f:
            tail = f.readlines()[-lines:]
        logging.error("---- last output ----\n%s", "".join(tail))
    except OSError:
        pass


if OUTPUT_ARTIFACTS_DIR is None or THEROCK_BIN_DIR is None:
    raise EnvironmentError(
        "OUTPUT_ARTIFACTS_DIR and THEROCK_BIN_DIR must both be set."
    )

OUTPUT_ARTIFACTS_PATH = Path(OUTPUT_ARTIFACTS_DIR).resolve()
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
EXAMPLES_ROOT = OUTPUT_ARTIFACTS_PATH / "hipthreads" / "examples"

ROCM_VERSION = load_rocm_version()
logging.info(f"ROCm version: {ROCM_VERSION}")

# offload-arch must resolve a CONCRETE arch (e.g. gfx1100) for
# -DCMAKE_HIP_ARCHITECTURES. The AMDGPU_FAMILIES env var is unsuitable here: for
# the gfx110X testers it holds the wildcard family "gfx110X-all", which is not a
# compilable target. So we probe offload-arch like libhipcxx does. On Windows
# offload-arch.exe loads ROCm DLLs from PATH, which is also the DLL search path,
# so prepend the artifact bin/ dir before probing (see TheRock #2019, mirrored in
# test_sanity.py) or detection returns None.
if IS_WINDOWS:
    os.environ["PATH"] = (
        str(OUTPUT_ARTIFACTS_PATH / "bin") + os.pathsep + os.environ.get("PATH", "")
    )

gpu_arch = get_gpu_architecture_portable(OUTPUT_ARTIFACTS_DIR)
logging.info(f"++ Detected GPU architecture: {gpu_arch}")
if not gpu_arch:
    raise RuntimeError("Could not detect GPU architecture (offload-arch failed).")

if not EXAMPLES_ROOT.exists():
    raise FileNotFoundError(
        f"Examples not found at {EXAMPLES_ROOT}. The hipthreads test artifact "
        "must be fetched (HIPTHREADS_COPY_TO_BUILD packages examples/)."
    )

environ_vars = build_environment()
logging.info(f"ROCM_PATH: {environ_vars['ROCM_PATH']}")
logging.info(f"EXAMPLES_ROOT: {EXAMPLES_ROOT}")

for example in EXAMPLES:
    binary = configure_and_build(example, gpu_arch, environ_vars)
    run_example(example, binary, environ_vars)

logging.info("All hipThreads example apps built and ran successfully.")
