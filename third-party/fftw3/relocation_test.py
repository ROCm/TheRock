# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build FFTW and test its installed CMake packages after moving both prefixes.

Requires CMake >= 3.25, a C compiler, and unpacked FFTW 3.3.10 sources; no GPU.
Run: python third-party/fftw3/relocation_test.py --source-dir /path/to/fftw-3.3.10
Use --unpatched to reproduce the upstream failure with the same consumer.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


CONSUMER = r"""
cmake_minimum_required(VERSION 3.25)
project(fftw_relocation_test C)
enable_testing()
find_package(FFTW3 CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${DOUBLE_CONFIG}")
find_package(FFTW3f CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${FLOAT_CONFIG}")

# Match rocFFT: use legacy include variables and extract library filenames
# instead of linking imported targets (which already propagate include paths).
add_executable(legacy main.c)
target_include_directories(legacy PRIVATE ${FFTW3_INCLUDE_DIRS} ${FFTW3f_INCLUDE_DIRS})
if(WIN32)
  set(location IMPORTED_IMPLIB_RELEASE)
else()
  set(location IMPORTED_LOCATION_RELEASE)
endif()
get_target_property(double_lib FFTW3::fftw3 ${location})
get_target_property(float_lib FFTW3::fftw3f ${location})
target_link_libraries(legacy PRIVATE "${double_lib}" "${float_lib}")

# Keep the existing imported-target API working too.
add_executable(imported main.c)
target_link_libraries(imported PRIVATE FFTW3::fftw3 FFTW3::fftw3f)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_library(double_threads fftw3_threads NO_DEFAULT_PATH
    PATHS ${FFTW3_LIBRARY_DIRS})
  find_library(float_threads fftw3f_threads NO_DEFAULT_PATH
    PATHS ${FFTW3f_LIBRARY_DIRS})
  if(double_threads AND float_threads)
    foreach(target legacy imported)
      target_compile_definitions(${target} PRIVATE TEST_THREADS)
      target_link_libraries(${target} PRIVATE "${double_threads}" "${float_threads}")
    endforeach()
  endif()
endif()
add_test(NAME legacy COMMAND legacy)
add_test(NAME imported COMMAND imported)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/directories.txt"
  "${FFTW3_INCLUDE_DIRS}\n${FFTW3f_INCLUDE_DIRS}\n${FFTW3_LIBRARY_DIRS}\n${FFTW3f_LIBRARY_DIRS}\n")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/threads.txt"
    "${double_threads}\n${float_threads}\n")
endif()
"""

MAIN = r"""
#include <fftw3.h>

int main(void)
{
#ifdef TEST_THREADS
    if (!fftw_init_threads() || !fftwf_init_threads())
        return 1;
#endif
    double input[4] = {1, 1, 1, 1};
    float inputf[4] = {1, 1, 1, 1};
    fftw_complex output[3];
    fftwf_complex outputf[3];
    fftw_plan plan = fftw_plan_dft_r2c_1d(4, input, output, FFTW_ESTIMATE);
    fftwf_plan planf = fftwf_plan_dft_r2c_1d(4, inputf, outputf, FFTW_ESTIMATE);
    if (!plan || !planf)
        return 2;
    fftw_execute(plan);
    fftwf_execute(planf);
    int failed = output[0][0] != 4 || outputf[0][0] != 4;
    fftw_destroy_plan(plan);
    fftwf_destroy_plan(planf);
#ifdef TEST_THREADS
    fftw_cleanup_threads();
    fftwf_cleanup_threads();
#endif
    return failed;
}
"""


def run(*command):
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--libdir", default="lib")
    parser.add_argument("--unpatched", action="store_true")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    if not (source / "FFTW3Config.cmake.in").is_file():
        parser.error("--source-dir must contain unpacked FFTW sources")
    hooks = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="fftw-relocation-") as temporary:
        root = Path(temporary)
        original = root / "runner-a"
        relocated = root / "runner-b with spaces"
        for suffix in ("", "f"):
            build = root / f"build-fftw3{suffix}"
            # Exercise the same end-of-configure hook that TheRock schedules.
            init = root / f"init{suffix}.cmake"
            init.write_text(
                ""
                if args.unpatched
                else "cmake_language(DEFER CALL include "
                f'"{hooks.as_posix()}/post_hook_therock-fftw3{suffix}.cmake")\n'
            )
            print(f"Building FFTW3{suffix}", flush=True)
            run(
                args.cmake,
                "-S",
                str(source),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_SHARED_LIBS=ON",
                "-DBUILD_TESTS=OFF",
                "-DENABLE_THREADS=ON",
                f"-DENABLE_FLOAT={'ON' if suffix else 'OFF'}",
                f"-DCMAKE_INSTALL_PREFIX={original / ('fftw3' + suffix)}",
                f"-DCMAKE_INSTALL_LIBDIR={args.libdir}",
                f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={init}",
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            )
            run(args.cmake, "--build", str(build), "--parallel", str(args.jobs))
            run(args.cmake, "--install", str(build))

        shutil.move(str(original), str(relocated))
        assert not original.exists()
        consumer = root / "consumer"
        consumer.mkdir()
        (consumer / "CMakeLists.txt").write_text(CONSUMER)
        (consumer / "main.c").write_text(MAIN)
        build = root / "consumer-build"
        print(
            "Testing relocated packages with the original prefixes absent", flush=True
        )
        run(
            args.cmake,
            "-S",
            str(consumer),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DDOUBLE_CONFIG={relocated / 'fftw3' / args.libdir / 'cmake/fftw3'}",
            f"-DFLOAT_CONFIG={relocated / 'fftw3f' / args.libdir / 'cmake/fftw3f'}",
        )
        run(args.cmake, "--build", str(build), "--parallel", str(args.jobs))
        ctest = Path(shutil.which(args.cmake) or args.cmake).with_name("ctest")
        run(
            str(ctest), "--test-dir", str(build), "--output-on-failure", "-C", "Release"
        )
        for directory in (build / "directories.txt").read_text().splitlines():
            path = Path(directory).resolve()
            assert path.is_dir() and path.is_relative_to(relocated), directory
        threads = build / "threads.txt"
        if threads.exists():
            for library in threads.read_text().splitlines():
                path = Path(library).resolve()
                assert path.is_file() and path.is_relative_to(relocated), library
        print(
            "PASS: legacy variables, imported targets, and both precisions relocate",
            flush=True,
        )


if __name__ == "__main__":
    main()
