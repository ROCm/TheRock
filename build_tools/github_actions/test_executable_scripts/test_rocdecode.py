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
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_TEST_DIR = Path(THEROCK_DIR) / "build"

ROCDECODE_TEST_PATH = str(
    Path(THEROCK_BIN_DIR).resolve().parent / "share" / "rocdecode" / "test"
)
if not os.path.isdir(ROCDECODE_TEST_PATH):
    logging.info(f"++ Error: rocdecode tests not found in {ROCDECODE_TEST_PATH}")
    sys.exit(1)
else:
    logging.info(f"++ INFO: rocdecode tests found in {ROCDECODE_TEST_PATH}")
env = os.environ.copy()

# GitHub Actions passes an empty string when a workflow input is left blank.
TEST_TYPE = (os.getenv("TEST_TYPE") or "standard").lower()


def test_filter_args():
    """CTest filter for the requested category, per docs/development/test_filtering.md.

    rocdecode ships raw-decode tests (`video_decodeRaw-*`, no FFmpeg needed), a
    set of FFmpeg-based extended decode tests, and one performance suite
    (`video_decodePerf-*`). quick runs raw decode only; standard adds the rest of
    the correctness suites but excludes perf. The FFM (Full Function Model
    emulator) tiers mirror their native counterparts but exclude perf at every
    level: perf numbers off a software emulator are meaningless and the runs are
    prohibitively slow.
    """
    if TEST_TYPE in ("quick", "ffm-quick"):
        return ["-R", "video_decodeRaw"]
    if TEST_TYPE in ("standard", "ffm-standard"):
        return ["-E", "video_decodePerf"]
    if TEST_TYPE in ("ffm-comprehensive", "ffm-full"):
        return ["-E", "video_decodePerf"]
    # Native comprehensive/full: run everything, including the perf suite.
    return []


# set env variables required for tests
def setup_env(env):
    ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
    env["ROCM_PATH"] = str(ROCM_PATH)
    logging.info(f"++ rocdecode setting ROCM_PATH={ROCM_PATH}")
    if platform.system() == "Linux":
        HIP_LIB_PATH = Path(THEROCK_BIN_DIR).resolve().parent / "lib"
        logging.info(f"++ rocdecode setting LD_LIBRARY_PATH={HIP_LIB_PATH}")
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{HIP_LIB_PATH}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = str(HIP_LIB_PATH)
        ROCM_SYSDEPS_LIB_PATH = ROCM_PATH / "lib" / "rocm_sysdeps" / "lib"
        LD_PRELOAD_LIBS = [
            str(ROCM_SYSDEPS_LIB_PATH / "librocm_sysdeps_va.so.2"),
            str(ROCM_SYSDEPS_LIB_PATH / "librocm_sysdeps_va-drm.so.2"),
        ]
        LD_PRELOAD_VALUE = ":".join(LD_PRELOAD_LIBS)
        logging.info(f"++ rocdecode setting LD_PRELOAD={LD_PRELOAD_VALUE}")
        env["LD_PRELOAD"] = LD_PRELOAD_VALUE
        logging.info(f"++ rocdecode setting LIBVA_DRIVERS_PATH={ROCM_SYSDEPS_LIB_PATH}")
        env["LIBVA_DRIVERS_PATH"] = str(ROCM_SYSDEPS_LIB_PATH)
    elif platform.system() == "Windows":
        # Windows uses the vaon12 (VA-API-on-D3D12) backend. The patched libva
        # locates vaon12_drv_video.dll relative to ROCM_PATH
        # (<ROCM_PATH>/lib/rocm_sysdeps/bin), so no LIBVA_DRIVERS_PATH is needed.
        # DLLs are resolved via PATH: rocdecode.dll lives in bin/, and the Mesa
        # VA-API DLLs (rocm_sysdeps_va*.dll, vaon12_drv_video.dll) in
        # lib/rocm_sysdeps/bin.
        BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
        SYSDEPS_BIN_PATH = ROCM_PATH / "lib" / "rocm_sysdeps" / "bin"
        new_path = os.pathsep.join(
            [str(BIN_PATH), str(SYSDEPS_BIN_PATH), env.get("PATH", "")]
        )
        logging.info(f"++ rocdecode prepending to PATH: {BIN_PATH};{SYSDEPS_BIN_PATH}")
        env["PATH"] = new_path
    else:
        logging.info(f"++ rocdecode tests only supported on Linux and Windows")
        sys.exit(0)


def execute_tests(env):
    ROCDECODE_TEST_DIR = Path(THEROCK_TEST_DIR) / "rocdecode-test"

    ROCDECODE_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # rocdecode tests are shipped as CMake source and must be built on the target
    # machine. This serves two purposes:
    # 1. Verifies that the installed rocdecode headers and libraries are functional.
    # 2. Some test dependencies (e.g. video codec libraries) are not bundled in the
    #    TheRock artifacts and must be linked from the system at build time.
    # Extended tests require FFmpeg dev libraries, provided on Linux via the
    # specialized media container image. Windows CI has no such image, so it runs
    # only the always-on tests (raw decode, caps, negative API) which need no
    # FFmpeg.
    enable_extended = "OFF" if platform.system() == "Windows" else "ON"
    cmd = [
        "cmake",
        "-GNinja",
        f"-DENABLE_EXTENDED_TESTS={enable_extended}",
        ROCDECODE_TEST_PATH,
    ]
    logging.info(f"++ Exec [{ROCDECODE_TEST_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=ROCDECODE_TEST_DIR, check=True, env=env)

    filter_args = test_filter_args()
    # On Windows, rocdec_Decode-HEVC decodes correctly but the sample returns a
    # non-zero exit code during process teardown under CTest's launch context (the
    # decoded output and frame counts are correct; only the shutdown path is
    # affected). This is adjacent to the vaTerminate teardown crash fixed upstream
    # in ROCm/rocm-systems#12309 and is tracked there. Exclude it on Windows until
    # the teardown exit-code issue is resolved.
    # TODO(ROCm/rocm-systems#12309): re-enable rocdec_Decode-HEVC on Windows.
    if platform.system() == "Windows":
        excluded = "rocdec_Decode-HEVC"
        if "-E" in filter_args:
            i = filter_args.index("-E")
            filter_args[i + 1] = f"({filter_args[i + 1]})|({excluded})"
        else:
            filter_args += ["-E", excluded]
    logging.info(f"++ rocdecode test category TEST_TYPE={TEST_TYPE}")

    cmd = [
        "ctest",
        "-N",
    ] + filter_args
    logging.info(f"++ Exec [{ROCDECODE_TEST_DIR}]$ {shlex.join(cmd)}")
    ctest_list = subprocess.run(
        cmd,
        cwd=ROCDECODE_TEST_DIR,
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
            f"CTest discovered zero rocdecode tests for TEST_TYPE={TEST_TYPE}"
        )

    cmd = [
        "ctest",
        "--extra-verbose",
        "--output-on-failure",
    ] + filter_args
    logging.info(f"++ Exec [{ROCDECODE_TEST_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=ROCDECODE_TEST_DIR, check=True, env=env)


if __name__ == "__main__":
    setup_env(env)
    execute_tests(env)
