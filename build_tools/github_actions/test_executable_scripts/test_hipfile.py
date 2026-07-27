# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Runs hipFile's unit tests from the installed/packaged artifact tree.
#
# hipFile installs a relocatable ctest tree at share/hipfile/test (a top-level
# CTestTestfile.cmake plus script/hipfile_discover.cmake). Discovery and the
# runtime library path are handled inside that tree relative to its own
# location, so this script only needs to point ctest at it -- no LD_LIBRARY_PATH
# setup is required here.
#
# The one exception is ASAN builds: the test binaries are built with
# -shared-libsan and dynamically depend on libclang_rt.asan-<arch>.so, which
# lives in the clang resource dir outside the relocatable tree. Preload it so
# the loader can satisfy that dependency.
#
# Only the "unit" label is run: the "system" tests need a real GPU and the
# "stress" tests are gdb-wrapped concurrency testers, both excluded from the
# packaged unit suite.

import logging
import platform
import shlex
import subprocess
import sys
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Importing is_asan from amdgpu_family_matrix.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from amdgpu_family_matrix import is_asan

if THEROCK_BIN_DIR is None:
    logging.error("env(THEROCK_BIN_DIR) is not set. Set it before running tests.")
    raise SystemExit(1)

# THEROCK_BIN_DIR is <install>/bin; the relocatable test tree is alongside it.
HIPFILE_TEST_DIR = Path(THEROCK_BIN_DIR).resolve().parent / "share" / "hipfile" / "test"

if not HIPFILE_TEST_DIR.is_dir():
    logging.error(f"hipFile test directory not found: {HIPFILE_TEST_DIR}")
    raise SystemExit(1)


def get_asan_lib_path():
    arch = platform.machine()
    clang_path = str(Path(THEROCK_BIN_DIR).parent / "lib" / "llvm" / "bin" / "clang++")
    asan_lib = f"libclang_rt.asan-{arch}.so"
    cmd = [clang_path, f"-print-file-name={asan_lib}"]
    logging.info(f"++ Exec [{clang_path}]$ {shlex.join(cmd)}")
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    resolved = result.stdout.strip()
    if not resolved or resolved == asan_lib or not Path(resolved).is_file():
        raise FileNotFoundError(
            f"Could not locate ASan runtime '{asan_lib}' via {clang_path} "
            f"(got: '{resolved}')"
        )
    return str(Path(resolved).resolve())


env = os.environ.copy()
if is_asan():
    asan_lib = get_asan_lib_path()
    existing_preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = (
        f"{existing_preload}:{asan_lib}" if existing_preload else asan_lib
    )

cmd = [
    "ctest",
    "--test-dir",
    str(HIPFILE_TEST_DIR),
    "-L",
    "unit",
    "--output-on-failure",
    "--no-tests=error",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=env,
)
