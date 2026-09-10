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

MIVISIONX_TEST_PATH = str(
    Path(THEROCK_BIN_DIR).resolve().parent / "share" / "mivisionx" / "test"
)
if not os.path.isdir(MIVISIONX_TEST_PATH):
    logging.info(f"++ Error: mivisionx tests not found in {MIVISIONX_TEST_PATH}")
    sys.exit(1)
else:
    logging.info(f"++ INFO: mivisionx tests found in {MIVISIONX_TEST_PATH}")
env = os.environ.copy()

TEST_TYPE = (os.getenv("TEST_TYPE") or "standard").lower()


def test_filter_args():
    """CTest filter for the requested category, per docs/development/test_filtering.md.

    MIVisionX test names (from tests/CMakeLists.txt):
      Core/CPU (fast, always run):
        vx_core_test, runvx_test
        openvx_canny, openvx_channel_extract, openvx_color_convert, openvx_accumulate
        openvx_graph, openvx_graph_api, openvx_tensor_api, openvx_data_objects_api
        openvx_user_kernel_api, openvx_tensor_advanced_api, openvx_threshold_query_api
        openvx_graph_import_api, openvx_vxu_api, openvx_coverage_boost
        openvx_vision_coverage, openvx_pipelining_api
        openvx_canny_CPU, openvx_channel_extract_CPU, openvx_color_convert_CPU
      Python-driven CPU suites (slower — standard+):
        openvx_tests_runVisionPython_CPU, openvx_tests_runVisionPython_CPU_5X3
        openvx_tests_runVisionPython_CPU_10X10, openvx_gdf_tests_CPU
      GPU tests (require device — standard+, excluded on FFM):
        openvx_canny_GPU, openvx_channel_extract_GPU, openvx_color_convert_GPU
        openvx_tests_runVisionPython_GPU, openvx_gdf_tests_GPU
      Comprehensive-only (longest GPU test):
        openvx_hip_cu_mask_remap_4K
      Optional (present only when RPP found):
        vx_rpp_test

    Filter strategy:
      quick:              vx_core_test only — validates the OpenVX runtime loads correctly
      ffm-quick:          same as quick (FFM has no GPU, so GPU tests auto-excluded anyway)
      standard:           all CPU tests; exclude GPU-only and comprehensive-only tests
      ffm-standard:       same as standard (GPU tests excluded)
      comprehensive/full: run everything
      ffm-comprehensive/ffm-full: exclude GPU tests (FFM has no real GPU)
    """
    # Regex matching all GPU-accelerated tests
    _GPU_TESTS = (
        "openvx_canny_GPU|openvx_channel_extract_GPU|openvx_color_convert_GPU"
        "|openvx_tests_runVisionPython_GPU|openvx_gdf_tests_GPU"
        "|openvx_hip_cu_mask_remap_4K"
    )

    if TEST_TYPE in ("quick", "ffm-quick"):
        # Only the core OpenVX runtime sanity check — fast, no GPU required.
        return ["-R", "vx_core_test"]

    if TEST_TYPE in ("standard", "ffm-standard"):
        # All tests except GPU-only and the long comprehensive HIP test.
        return ["-E", _GPU_TESTS]

    if TEST_TYPE in ("ffm-comprehensive", "ffm-full"):
        # FFM has no real GPU — exclude all GPU-targeted tests.
        return ["-E", _GPU_TESTS]

    # Native comprehensive and full: run everything including GPU tests.
    return []


def setup_env(env):
    ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
    env["ROCM_PATH"] = str(ROCM_PATH)
    logging.info(f"++ mivisionx setting ROCM_PATH={ROCM_PATH}")
    if platform.system() == "Linux":
        HIP_LIB_PATH = Path(THEROCK_BIN_DIR).resolve().parent / "lib"
        logging.info(f"++ mivisionx setting LD_LIBRARY_PATH={HIP_LIB_PATH}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{HIP_LIB_PATH}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = str(HIP_LIB_PATH)
        if host_triple and THEROCK_LLVM_LIB_HOST_TRIPLE_PATH.exists():
            logging.info(
                f"++ mivisionx prepending LD_LIBRARY_PATH with {THEROCK_LLVM_LIB_HOST_TRIPLE_PATH}"
            )
            env["LD_LIBRARY_PATH"] = (
                f"{THEROCK_LLVM_LIB_HOST_TRIPLE_PATH}:{env['LD_LIBRARY_PATH']}"
            )
    else:
        logging.info("++ mivisionx tests only supported on Linux")
        sys.exit(0)


def execute_tests(env):
    MIVISIONX_TEST_DIR = Path(THEROCK_TEST_DIR) / "mivisionx-test"
    MIVISIONX_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # mivisionx ships its tests as CMake source, built here against the installed tree.
    cmd = [
        "cmake",
        "-GNinja",
        MIVISIONX_TEST_PATH,
    ]
    logging.info(f"++ Exec [{MIVISIONX_TEST_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=MIVISIONX_TEST_DIR, check=True, env=env)

    filter_args = test_filter_args()
    logging.info(f"++ mivisionx test category TEST_TYPE={TEST_TYPE}")

    cmd = ["ctest", "-N"] + filter_args
    logging.info(f"++ Exec [{MIVISIONX_TEST_DIR}]$ {shlex.join(cmd)}")
    ctest_list = subprocess.run(
        cmd,
        cwd=MIVISIONX_TEST_DIR,
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
        raise RuntimeError(
            f"CTest discovered zero mivisionx tests for TEST_TYPE={TEST_TYPE}"
        )

    cmd = ["ctest", "--extra-verbose", "--output-on-failure"] + filter_args
    logging.info(f"++ Exec [{MIVISIONX_TEST_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=MIVISIONX_TEST_DIR, check=True, env=env)


if __name__ == "__main__":
    setup_env(env)
    execute_tests(env)
