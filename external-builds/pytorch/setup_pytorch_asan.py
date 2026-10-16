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

_ASAN_COMPANION_BUILDS = (
    ("build_triton", "--build-triton"),
    ("build_pytorch_audio", "--build-pytorch-audio"),
    ("build_pytorch_vision", "--build-pytorch-vision"),
    ("build_apex", "--build-apex"),
)
# Torchaudio is the first companion brought up on the shared ASAN compiler
# environment. The others stay off until their own builds are ready.
_ASAN_ALLOWED_COMPANIONS = frozenset({"build_pytorch_audio"})


def apply_asan_companion_policy(parser, args) -> None:
    """Build torch, and torchaudio when ``--build-pytorch-audio`` is set.

    An explicit ``--build-triton``, ``--build-pytorch-vision``, or
    ``--build-apex`` is rejected. An unset flag is forced off before the
    checkout-directory default can turn it on. ``--no-build-*`` is left as
    false, including ``--no-build-pytorch-audio``.
    """
    if not args.asan:
        return
    requested = [
        flag
        for attr, flag in _ASAN_COMPANION_BUILDS
        if getattr(args, attr) is True and attr not in _ASAN_ALLOWED_COMPANIONS
    ]
    if requested:
        parser.error("--asan cannot be combined with " + ", ".join(requested))
    for attr, _flag in _ASAN_COMPANION_BUILDS:
        if getattr(args, attr) is None:
            setattr(args, attr, False)


_AUDIO_EXTENSION_MODULES = "    modules = [\n"
_AUDIO_EXTENSION_COMPILE_ARGS = "            extra_compile_args=extra_compile_args,\n"
_AUDIO_ASAN_FLAGS = """\
    if os.environ.get("USE_ASAN") == "1":
        _asan_flags = [
            "-fsanitize=address",
            "-fno-omit-frame-pointer",
            "-shared-libasan",
        ]
        extra_compile_args["cxx"].extend(_asan_flags)
        extra_compile_args.setdefault("nvcc", []).extend(_asan_flags)
        extra_link_args = _asan_flags
    else:
        extra_link_args = []

"""


_AUDIO_SETUP_IMPORT = "import torch\n"
_AUDIO_SETUP_DROP_PRELOAD = """\
import torch
# LD_PRELOAD lets this import load libclang_rt.asan.so first. Drop it before
# setuptools spawns ninja and hipcc.
os.environ.pop("LD_PRELOAD", None)
"""


def enable_torchaudio_asan(pytorch_audio_dir: Path) -> None:
    """Instrument torchaudio's extensions with the shared ASan runtime.

    ``torch.utils.cpp_extension`` honors ``CXX`` but does not forward
    ``CFLAGS``, ``CXXFLAGS``, or ``LDFLAGS`` into its ninja build. Without
    these flags the extension is not instrumented and the link does not
    select ``libclang_rt.asan.so``. ``setup.py`` also imports torch, so the
    build process is started with ``LD_PRELOAD`` and then removes it before
    the compiler runs.
    """
    extension_path = pytorch_audio_dir / "tools" / "setup_helpers" / "extension.py"
    setup_path = pytorch_audio_dir / "setup.py"
    if not extension_path.is_file():
        raise RuntimeError(f"--asan torchaudio build did not find {extension_path}")
    if not setup_path.is_file():
        raise RuntimeError(f"--asan torchaudio build did not find {setup_path}")

    extension_text = extension_path.read_text()
    if "-shared-libasan" not in extension_text:
        if (
            _AUDIO_EXTENSION_MODULES not in extension_text
            or _AUDIO_EXTENSION_COMPILE_ARGS not in extension_text
        ):
            raise RuntimeError(
                f"--asan torchaudio build does not recognize {extension_path}; "
                "the extension helper has no modules list to instrument"
            )
        extension_text = extension_text.replace(
            _AUDIO_EXTENSION_MODULES,
            _AUDIO_ASAN_FLAGS + _AUDIO_EXTENSION_MODULES,
            1,
        )
        extension_text = extension_text.replace(
            _AUDIO_EXTENSION_COMPILE_ARGS,
            _AUDIO_EXTENSION_COMPILE_ARGS
            + "            extra_link_args=extra_link_args,\n",
        )
        extension_path.write_text(extension_text)
        print(f"  Patched {extension_path} to build torchaudio with ASAN")

    setup_text = setup_path.read_text()
    if 'os.environ.pop("LD_PRELOAD"' not in setup_text:
        if _AUDIO_SETUP_IMPORT not in setup_text:
            raise RuntimeError(
                f"--asan torchaudio build does not recognize {setup_path}"
            )
        setup_path.write_text(
            setup_text.replace(_AUDIO_SETUP_IMPORT, _AUDIO_SETUP_DROP_PRELOAD, 1)
        )
        print(f"  Patched {setup_path} to drop LD_PRELOAD after importing torch")


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


def _command_output(args: list[str], cwd: Path) -> str:
    """Run a command and return its output. A non-zero exit fails the build."""
    print(f"++ Capture [{cwd}]$ {' '.join(args)}")
    return subprocess.check_output(
        args, cwd=str(cwd), stderr=subprocess.STDOUT, text=True
    ).strip()


def _resolve_shared_asan_runtime(clangxx: Path, rocm_dir: Path) -> Path:
    """Ask ROCm clang for its shared ASan runtime so the link can find it."""
    runtime_attempts: list[str] = []
    for runtime_name in (
        f"libclang_rt.asan-{platform.machine().lower()}.so",
        "libclang_rt.asan.so",
    ):
        runtime_text = _command_output(
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

    # USE_ASAN: PyTorch reads this from the environment. With the image's
    # default GCC the build reports USE_ASAN OFF, because that compiler's
    # libasan is not this ROCm runtime.
    # CC, CXX: gcc-toolset comes first on PATH in the manylinux image. Its
    # libasan and ROCm's libclang_rt.asan.so abort with "incompatible ASan
    # runtimes".
    # CMAKE_C_COMPILER, CMAKE_CXX_COMPILER: CMake records these separately
    # from CC and CXX. Pin the same ROCm Clang so configure does not keep
    # gcc-toolset.
    # ASAN_OPTIONS: hipcc and the other build tools load the instrumented
    # ROCm libraries. detect_leaks=0 keeps LeakSanitizer from aborting those
    # tools. abort_on_error=1 fails the build, and print_stacktrace=1 logs
    # the stack. A caller-supplied ASAN_OPTIONS is left in place.
    # LD_LIBRARY_PATH: -shared-libasan looks up libclang_rt.asan.so while
    # linking, and that directory is not on the default linker path. The
    # ROCm lib directory is included so the instrumented libraries those
    # tools load can resolve.
    env.update(
        {
            "USE_ASAN": "1",
            "CC": str(clang),
            "CXX": str(clangxx),
            "CMAKE_C_COMPILER": str(clang),
            "CMAKE_CXX_COMPILER": str(clangxx),
            "ASAN_OPTIONS": os.environ.get(
                "ASAN_OPTIONS",
                "detect_leaks=0:abort_on_error=1:print_stacktrace=1",
            ),
            "LD_LIBRARY_PATH": os.path.pathsep.join(ld_library_parts),
        }
    )
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
