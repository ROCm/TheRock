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


def with_asan_local_version(version_suffix: str) -> str:
    """Add an asan marker to the torch local version.

    The incoming suffix is whatever the installed ROCm package already
    produced, for example ``+rocm10.1.0rc3``. The torch wheel becomes
    ``+rocm10.1.0rc3.asan``. The ROCm package version itself is unchanged.
    """
    if not version_suffix.startswith("+") or version_suffix == "+":
        raise RuntimeError(
            "--asan expected a PEP 440 local version suffix such as "
            f"+rocm10.1.0rc3, got {version_suffix!r}"
        )
    if version_suffix.endswith(".asan"):
        return version_suffix
    return f"{version_suffix}.asan"


def disable_default_companion_builds(args) -> None:
    """Leave triton, torchaudio, torchvision, and apex off unless requested.

    Runs before the "a checkout directory means build it" defaults. An explicit
    --build-* value is left alone.
    """
    if args.build_triton is None:
        args.build_triton = False
    if args.build_pytorch_audio is None:
        args.build_pytorch_audio = False
    if args.build_pytorch_vision is None:
        args.build_pytorch_vision = False
    if args.build_apex is None:
        args.build_apex = False


def keep_flash_attention_without_triton(asan: bool) -> bool:
    """ASAN builds torch without triton and still enables AOTriton by arch."""
    return asan


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

    env["USE_ASAN"] = "1"
    env["CC"] = str(clang)
    env["CXX"] = str(clangxx)
    env["CMAKE_C_COMPILER"] = str(clang)
    env["CMAKE_CXX_COMPILER"] = str(clangxx)
    env["ASAN_OPTIONS"] = os.environ.get(
        "ASAN_OPTIONS", "detect_leaks=0:abort_on_error=1:print_stacktrace=1"
    )
    env["LD_LIBRARY_PATH"] = os.path.pathsep.join(ld_library_parts)
    env["PATH"] = str(llvm_bin) + os.path.pathsep + env.get("PATH", "")
    append_env_text(env, "CFLAGS", "-fno-omit-frame-pointer")
    append_env_text(env, "CXXFLAGS", "-fno-omit-frame-pointer")
    append_env_text(env, "LDFLAGS", "-shared-libasan")
    append_env_text(env, "CMAKE_ARGS", "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF")
    print(f"  ASAN compiler: {clangxx}")
    print(f"  ASAN runtime: {runtime_path}")
    print(f"  PYTORCH_ROCM_ARCH: {env.get('PYTORCH_ROCM_ARCH', '')}")
