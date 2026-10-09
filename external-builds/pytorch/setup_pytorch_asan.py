# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AddressSanitizer setup for the PyTorch wheel build.

build_prod_wheels.py calls these helpers when --asan is set. The GPU list and
the installed ROCm version stay as the caller supplied them.
"""

import os
import platform
import re
import subprocess
from pathlib import Path

_ASAN_COMPANION_BUILDS = (
    ("build_triton", "--build-triton"),
    ("build_pytorch_audio", "--build-pytorch-audio"),
    ("build_pytorch_vision", "--build-pytorch-vision"),
    ("build_apex", "--build-apex"),
)


def apply_asan_companion_policy(parser, args) -> None:
    """Build torch only.

    An explicit companion ``--build-*`` flag is rejected. An unset flag is
    forced off before the checkout-directory default can turn it on.
    ``--no-build-*`` is left as false.
    """
    if not args.asan:
        return
    requested = [
        flag for attr, flag in _ASAN_COMPANION_BUILDS if getattr(args, attr) is True
    ]
    if requested:
        parser.error(
            "--asan builds torch only and cannot be combined with "
            + ", ".join(requested)
        )
    for attr, _flag in _ASAN_COMPANION_BUILDS:
        if getattr(args, attr) is None:
            setattr(args, attr, False)


def import_sanity_env(env: dict[str, str]) -> dict[str, str]:
    """Environment for ``import torch`` so the shared runtime is loaded first."""
    runtime = env.get("ASAN_RUNTIME_PATH", "")
    if not runtime:
        raise RuntimeError(
            "--asan import check requires ASAN_RUNTIME_PATH from apply_asan_build_env"
        )
    return {
        "LD_PRELOAD": runtime,
        "ASAN_OPTIONS": env.get(
            "ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:print_stacktrace=1"
        ),
        "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", ""),
    }


def append_env_text(env: dict[str, str], name: str, addition: str) -> None:
    """Append one compiler flag and keep the trailing space later appends expect.

    add_env_compiler_flags concatenates the next flag directly onto the current
    value, so a stripped value turns ``-fno-omit-frame-pointer`` plus ``-I...``
    into one unknown argument.
    """
    current = env.get(name, "")
    if addition in current.split():
        if current and not current.endswith((" ", "\t")):
            env[name] = current + " "
        return
    if current and not current.endswith((" ", "\t")):
        current += " "
    env[name] = f"{current}{addition} "


def _capture(args: list[str], cwd: Path) -> str:
    print(f"++ Capture [{cwd}]$ {' '.join(args)}")
    try:
        return subprocess.check_output(
            args, cwd=str(cwd), stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as error:
        print(f"Error capturing output: {error}")
        print(f"Output from the failed command:\n{error.output}")
        return ""


def _resolve_shared_asan_runtime(clangxx: Path, rocm_dir: Path) -> Path:
    """Ask ROCm clang for its shared ASan runtime so the link can find it."""
    runtime_attempts: list[str] = []
    for runtime_name in (
        f"libclang_rt.asan-{platform.machine().lower()}.so",
        "libclang_rt.asan.so",
    ):
        runtime_text = _capture(
            [str(clangxx), f"-print-file-name={runtime_name}"], rocm_dir
        )
        runtime_attempts.append(f"{runtime_name} -> {runtime_text!r}")
        candidate = Path(runtime_text)
        if (
            runtime_text
            and runtime_text != runtime_name
            and candidate.is_absolute()
            and candidate.is_file()
        ):
            return candidate
    raise RuntimeError(
        "ROCm clang++ did not resolve its shared ASAN runtime: "
        + "; ".join(runtime_attempts)
    )


# release/2.12 and release/2.13 vendor a Google Benchmark that uses
# __COUNTER__ under -Werror and -pedantic-errors. ROCm Clang 23+ reports
# that as a C2y extension and stops the build. release/2.14 already adds
# this suppression after the Benchmark subdirectory.
_BENCHMARK_SUBDIR_LINE = (
    "    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)\n"
)
_BENCHMARK_C2Y_SUPPRESSION = """\
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)
    # Clang 23+ classifies __COUNTER__ in preprocessor conditions as a C2y
    # extension. benchmark enables -Werror so this becomes fatal. Suppress
    # only that warning on affected compilers without touching the submodule.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "23.0")
      if(TARGET benchmark)
        target_compile_options(benchmark PRIVATE -Wno-c2y-extensions)
      endif()
      if(TARGET benchmark_main)
        target_compile_options(benchmark_main PRIVATE -Wno-c2y-extensions)
      endif()
    endif()
"""


def suppress_benchmark_c2y_warning(pytorch_dir: Path) -> None:
    """Insert the release/2.14 Benchmark warning suppression when it is absent.

    ``CXXFLAGS`` cannot carry ``-Wno-c2y-extensions``: Benchmark appends
    ``-pedantic-errors`` later, and that turns the warning back into an error.
    """
    path = pytorch_dir / "cmake" / "Dependencies.cmake"
    if not path.is_file():
        return
    text = path.read_text()
    if "-Wno-c2y-extensions" in text or _BENCHMARK_SUBDIR_LINE not in text:
        return
    path.write_text(text.replace(_BENCHMARK_SUBDIR_LINE, _BENCHMARK_C2Y_SUPPRESSION, 1))
    print(f"  Patched {path} to suppress -Wc2y-extensions in Google Benchmark")


# An ASAN ROCm SDK keeps rocSHMEM's device implementations in
# librocshmem_device_<arch>.bc. release/2.12 and release/2.13 link only the
# host archive, so the device link reports those symbols as undefined.
# release/2.14 already passes the bitcode to the offload linker.
_ROCSHMEM_LINK_LINE = re.compile(
    r"^(?P<indent>[ \t]*)target_link_libraries\(torch_rocshmem PRIVATE roc::rocshmem\)\n",
    re.MULTILINE,
)
_ROCSHMEM_DEVICE_BITCODE = """\
# The ASAN ROCm SDK ships the __device__ implementations of the rocSHMEM
# API in per-arch bitcode (librocshmem_device_<arch>.bc), separate from
# the host archive. Pass it straight to the device (offload) linker so it
# joins the device LTO link as bitcode and the symbols our kernels
# reference stay live. Routing it through the normal HIP input path
# instead recompiles it in isolation with -fvisibility=hidden and
# -amdgpu-internalize-symbols, which dead-strips the library symbols
# before the consumer's references are ever seen.
if(USE_ASAN)
  foreach(_arch ${_torch_rocshmem_build_arches})
    string(REGEX REPLACE ":.*" "" _arch_base "${_arch}")
    set(_dev_bc "${ROCM_PATH}/lib/librocshmem_device_${_arch_base}.bc")
    if(EXISTS "${_dev_bc}")
      target_link_options(torch_rocshmem PRIVATE "SHELL:-Xoffload-linker ${_dev_bc}")
    else()
      message(WARNING "rocSHMEM device bitcode not found for ${_arch_base}: ${_dev_bc}")
    endif()
  endforeach()
endif()
"""


def link_rocshmem_device_bitcode(pytorch_dir: Path) -> None:
    """Insert the release/2.14 rocSHMEM device-bitcode link when it is absent."""
    path = pytorch_dir / "caffe2" / "CMakeLists.txt"
    if not path.is_file():
        return
    text = path.read_text()
    if "librocshmem_device_" in text:
        return
    match = _ROCSHMEM_LINK_LINE.search(text)
    if match is None:
        return
    indent = match.group("indent")
    block = "".join(
        indent + line if line.strip() else line
        for line in _ROCSHMEM_DEVICE_BITCODE.splitlines(keepends=True)
    )
    path.write_text(text[: match.end()] + block + text[match.end() :])
    print(f"  Patched {path} to link rocSHMEM device bitcode for ASAN")


def apply_asan_build_env(env: dict[str, str], rocm_dir: Path) -> None:
    """Point the torch build at the ROCm Clang shared ASan runtime.

    PYTORCH_ROCM_ARCH is whatever the caller already put in ``env``. This does
    not select a GPU target and does not rewrite the installed ROCm version.
    """
    if platform.system() == "Windows" or platform.machine().lower() not in (
        "x86_64",
        "amd64",
    ):
        raise RuntimeError("--asan is supported only on Linux x86_64")

    llvm_bin = rocm_dir / "lib" / "llvm" / "bin"
    clang = llvm_bin / "clang"
    clangxx = llvm_bin / "clang++"
    for compiler in (clang, clangxx):
        if not compiler.is_file() or not os.access(compiler, os.X_OK):
            raise RuntimeError(
                f"--asan requires the executable ROCm compiler {compiler}"
            )

    runtime_path = _resolve_shared_asan_runtime(clangxx, rocm_dir)
    env["ASAN_RUNTIME_PATH"] = str(runtime_path)
    inherited_ld_library_path = env.get(
        "LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", "")
    )
    ld_library_parts = [str(runtime_path.parent), str(rocm_dir / "lib")]
    if inherited_ld_library_path:
        ld_library_parts.append(inherited_ld_library_path)

    # PyTorch reads USE_ASAN from the environment. With the image's default GCC
    # the build reports USE_ASAN OFF, because that compiler's libasan is not
    # this ROCm runtime.
    env["USE_ASAN"] = "1"
    # gcc-toolset comes first on PATH in the manylinux image. Its libasan and
    # ROCm's libclang_rt.asan.so abort with "incompatible ASan runtimes".
    env["CC"] = str(clang)
    env["CXX"] = str(clangxx)
    # CMake records these separately from CC and CXX. Pin the same ROCm Clang
    # so configure does not keep gcc-toolset.
    env["CMAKE_C_COMPILER"] = str(clang)
    env["CMAKE_CXX_COMPILER"] = str(clangxx)
    # hipcc and the other build tools load the instrumented ROCm libraries.
    # detect_leaks=0 keeps LeakSanitizer from aborting those tools.
    # abort_on_error=1 fails the build, and print_stacktrace=1 logs the stack.
    # A caller-supplied ASAN_OPTIONS is left in place.
    env["ASAN_OPTIONS"] = os.environ.get(
        "ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:print_stacktrace=1"
    )
    # -shared-libasan looks up libclang_rt.asan.so while linking, and that
    # directory is not on the default linker path. The ROCm lib directory is
    # included so the instrumented libraries those tools load can resolve.
    env["LD_LIBRARY_PATH"] = os.path.pathsep.join(ld_library_parts)
    # ASan stack traces use frame pointers.
    append_env_text(env, "CFLAGS", "-fno-omit-frame-pointer")
    append_env_text(env, "CXXFLAGS", "-fno-omit-frame-pointer")
    # ROCm is linked with the shared sanitizer runtime. A static libasan in
    # the torch link aborts with "incompatible ASan runtimes".
    append_env_text(env, "LDFLAGS", "-shared-libasan")
    # CMake turns on C++ module scanning for Clang. This wheel does not build
    # modules, and the scan fails configure with this ROCm Clang.
    append_env_text(env, "CMAKE_ARGS", "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF")
    print(f"  ASAN compiler: {clangxx}")
    print(f"  ASAN runtime: {runtime_path}")
    print(f"  PYTORCH_ROCM_ARCH: {env.get('PYTORCH_ROCM_ARCH', '')}")
