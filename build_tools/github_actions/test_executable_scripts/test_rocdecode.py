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
ROCDECODE_TEST_DIR = Path(THEROCK_TEST_DIR) / "rocdecode-test"
env = os.environ.copy()


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
    else:
        logging.info(f"++ rocdecode tests only supported on Linux")
        sys.exit(0)

    # When testing an ASAN artifact, propagate sanitizer flags to test compile steps.
    # CFLAGS/CXXFLAGS are inherited by all cmake child processes, including each
    # isolated cmake invocation spawned by `ctest --build-and-test`, which is the
    # only mechanism that reaches those sub-projects (cmake cache vars do not).
    # librocdecode.so is built with -shared-libsan, so the test executables must
    # link against the same shared ASan runtime: `-shared-libasan` selects it,
    # otherwise clang links the static runtime and the loader reports
    # "incompatible ASan runtimes" at startup.
    asan_enabled = "ASAN_OPTIONS" in env
    if asan_enabled:
        asan_flags = "-fsanitize=address -shared-libasan -fno-omit-frame-pointer"
        env["CFLAGS"] = f"{env.get('CFLAGS', '')} {asan_flags}".strip()
        env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} {asan_flags}".strip()
        logging.info(f"++ rocdecode ASAN detected: setting CFLAGS/CXXFLAGS={asan_flags}")

    _setup_gpu_targets(env, asan_enabled)


def _setup_gpu_targets(env, asan_enabled):
    # The per-test cmake calls are spawned by `ctest --build-and-test` from the
    # installed rocdecode CTestTestfile, with no `--build-options`, so we can't
    # pass `-DGPU_TARGETS=...` to them. Under ASAN amdgpu-arch also fails (the
    # tool isn't linked against the asan runtime), so the nested cmake falls
    # back to gfx906/gfx942 defaults and produces a binary that's incompatible
    # with librocdecode.so. CMAKE_TOOLCHAIN_FILE is the env-driven mechanism
    # CMake honors (>=3.21) on every cmake invocation, including the nested
    # ones, so we point it at a tiny cache-priming file.
    raw = env.get("AMDGPU_TARGETS") or env.get("GPU_TARGETS")
    if not raw:
        return
    targets = [t.strip() for t in re.split(r"[,;]", raw) if t.strip()]
    if asan_enabled:
        # Mirror cmake/therock_sanitizers.cmake: device-side ASAN requires
        # xnack+ on gfx942/gfx950 (HSA_XNACK=1 is set by test_component.yml).
        targets = [
            f"{t}:xnack+" if t in ("gfx942", "gfx950") else t for t in targets
        ]
    gpu_targets_value = ";".join(targets)

    ROCDECODE_TEST_DIR.mkdir(parents=True, exist_ok=True)
    toolchain = ROCDECODE_TEST_DIR / "rocdecode-test-toolchain.cmake"
    toolchain.write_text(
        f'set(GPU_TARGETS "{gpu_targets_value}" CACHE STRING "" FORCE)\n'
        f'set(AMDGPU_TARGETS "{gpu_targets_value}" CACHE STRING "" FORCE)\n'
    )
    env["CMAKE_TOOLCHAIN_FILE"] = str(toolchain)
    logging.info(
        f"++ rocdecode setting GPU_TARGETS={gpu_targets_value} via {toolchain}"
    )


def execute_tests(env):
    ROCDECODE_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # rocdecode tests are shipped as CMake source and must be built on the target
    # machine. This serves two purposes:
    # 1. Verifies that the installed rocdecode headers and libraries are functional.
    # 2. Some test dependencies (e.g. video codec libraries) are not bundled in the
    #    TheRock artifacts and must be linked from the system at build time.
    cmd = [
        "cmake",
        "-GNinja",
        "-DENABLE_EXTENDED_TESTS=ON",
        ROCDECODE_TEST_PATH,
    ]
    logging.info(f"++ Exec [{ROCDECODE_TEST_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=ROCDECODE_TEST_DIR, check=True, env=env)

    cmd = [
        "ctest",
        "-N",
    ]
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
        raise RuntimeError("CTest discovered zero rocdecode tests")

    cmd = [
        "ctest",
        "--extra-verbose",
        "--output-on-failure",
    ]
    logging.info(f"++ Exec [{ROCDECODE_TEST_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=ROCDECODE_TEST_DIR, check=True, env=env)


if __name__ == "__main__":
    setup_env(env)
    execute_tests(env)
