import logging
import os
import re
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
THEROCK_LIB_DIR = THEROCK_BIN_DIR.resolve().parent / "lib"
THEROCK_CLANG_PATH = THEROCK_LIB_DIR / "llvm" / "bin" / "amdclang"
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_TEST_DIR = Path(THEROCK_DIR) / "build"

# Determine host triple
host_triple = ""
if THEROCK_CLANG_PATH.exists():
    try:
        host_triple = subprocess.run(
            [str(THEROCK_CLANG_PATH), "--print-target-triple"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(
            f"'{THEROCK_CLANG_PATH} --print-target-triple' failed; "
            "this suggests a broken toolchain."
        ) from exc
if host_triple:
  THEROCK_LLVM_LIB_HOST_TRIPLE_PATH = THEROCK_LIB_DIR / "llvm" / "lib" / host_triple

RPP_TEST_PATH = str(Path(THEROCK_BIN_DIR).resolve().parent / "share" / "rpp" / "test")
if not os.path.isdir(RPP_TEST_PATH):
    logging.info(f"++ Error: rpp tests not found in {RPP_TEST_PATH}")
    sys.exit(1)
else:
    logging.info(f"++ INFO: rpp tests found in {RPP_TEST_PATH}")
env = os.environ.copy()

# GitHub Actions passes an empty string when a workflow input is left blank.
TEST_TYPE = (os.getenv("TEST_TYPE") or "standard").lower()


def test_filter_args():
    """CTest filter for the requested category, per docs/development/test_filtering.md.

    The two `test_type_1` entries are RPP's performance suites and take ~58% of
    the total runtime, so they are reserved for native comprehensive/full. The
    FFM (Full Function Model emulator) tiers mirror their native counterparts but
    exclude the perf suites at every level: perf numbers off a software emulator
    are meaningless and the runs are prohibitively slow.
    """
    if TEST_TYPE in ("quick", "ffm-quick"):
        return ["-R", "rpp_sanity_test"]
    if TEST_TYPE in ("standard", "ffm-standard"):
        return ["-E", "test_type_1"]
    if TEST_TYPE in ("ffm-comprehensive", "ffm-full"):
        return ["-E", "test_type_1"]
    # Native comprehensive/full: run everything, including the perf suites.
    return []


# set env variables required for tests
def setup_env(env):
    ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
    env["ROCM_PATH"] = str(ROCM_PATH)
    logging.info(f"++ rpp setting ROCM_PATH={ROCM_PATH}")
    if platform.system() == "Linux":
        HIP_LIB_PATH = Path(THEROCK_BIN_DIR).resolve().parent / "lib"
        logging.info(f"++ rpp setting LD_LIBRARY_PATH={HIP_LIB_PATH}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{HIP_LIB_PATH}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = str(HIP_LIB_PATH)
        if host_triple and THEROCK_LLVM_LIB_HOST_TRIPLE_PATH.exists():
            logging.info(f"++ rpp prepending LD_LIBRARY_PATH with {THEROCK_LLVM_LIB_HOST_TRIPLE_PATH}")
            env["LD_LIBRARY_PATH"] = f"{THEROCK_LLVM_LIB_HOST_TRIPLE_PATH}:{env['LD_LIBRARY_PATH']}"
    else:
        logging.info("++ rpp tests only supported on Linux")
        sys.exit(0)


def execute_tests(env):
    RPP_TEST_DIR = Path(THEROCK_TEST_DIR) / "rpp-test"
    RPP_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # rpp ships its tests as CMake source, built here against the installed
    # headers/libs because some test deps are only available from the system.
    # TODO(ROCm/rocm-libraries#10187): drop once prebuilt test binaries ship.
    cmd = [
        "cmake",
        "-GNinja",
        RPP_TEST_PATH,
    ]
    logging.info(f"++ Exec [{RPP_TEST_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=RPP_TEST_DIR, check=True, env=env)

    filter_args = test_filter_args()
    logging.info(f"++ rpp test category TEST_TYPE={TEST_TYPE}")

    cmd = [
        "ctest",
        "-N",
    ] + filter_args
    logging.info(f"++ Exec [{RPP_TEST_DIR}]$ {shlex.join(cmd)}")
    ctest_list = subprocess.run(
        cmd,
        cwd=RPP_TEST_DIR,
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    logging.info(ctest_list.stdout)
    match = re.search(r"Total Tests:\s*(\d+)", ctest_list.stdout)
    if match is None:
        raise RuntimeError(
            "Failed to determine CTest test count from `ctest -N` output"
        )
    if int(match.group(1)) == 0:
        raise RuntimeError(f"CTest discovered zero rpp tests for TEST_TYPE={TEST_TYPE}")

    cmd = [
        "ctest",
        "--extra-verbose",
        "--output-on-failure",
    ] + filter_args
    logging.info(f"++ Exec [{RPP_TEST_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=RPP_TEST_DIR, check=True, env=env)


if __name__ == "__main__":
    setup_env(env)
    execute_tests(env)
