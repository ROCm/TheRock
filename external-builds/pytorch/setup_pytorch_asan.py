# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AddressSanitizer setup for the PyTorch wheel build.

build_prod_wheels.py calls these helpers when --asan is set. The GPU list and
the installed ROCm version stay as the caller supplied them.
"""

import os
import platform
import subprocess
from pathlib import Path

# ROCm Clang rejects the GCC -Wno-error workarounds that the normal Linux
# wheel build puts in CXXFLAGS and CPPFLAGS.
_GCC_WARNING_FLAGS = (
    "-Wno-error=maybe-uninitialized",
    "-Wno-error=uninitialized",
    "-Wno-error=restrict",
)

TORCH_IMPORT_SANITY_SKIP_MESSAGE = (
    "+++ Skipping torch import sanity check for --asan. "
    "The shared runtime is applied with LD_PRELOAD when the wheel is used."
)


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


def remove_incompatible_warning_flags(env: dict[str, str]) -> None:
    """Drop GCC -Wno-error workarounds that ROCm Clang rejects."""
    for name in ("CFLAGS", "CXXFLAGS", "CPPFLAGS"):
        current = env.get(name)
        if not current:
            continue
        kept = [token for token in current.split() if token not in _GCC_WARNING_FLAGS]
        if kept:
            env[name] = " ".join(kept) + " "
        else:
            env.pop(name, None)


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

    remove_incompatible_warning_flags(env)

    llvm_bin = rocm_dir / "lib" / "llvm" / "bin"
    clang = llvm_bin / "clang"
    clangxx = llvm_bin / "clang++"
    for compiler in (clang, clangxx):
        if not compiler.is_file() or not os.access(compiler, os.X_OK):
            raise RuntimeError(
                f"--asan requires the executable ROCm compiler {compiler}"
            )

    runtime_path = _resolve_shared_asan_runtime(clangxx, rocm_dir)
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
    # Prefer the ROCm Clang that matches this runtime over gcc-toolset.
    env["PATH"] = str(llvm_bin) + os.path.pathsep + env.get("PATH", "")
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
