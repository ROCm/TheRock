# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sanitizer setup for the PyTorch wheel build.

build_prod_wheels.py calls these helpers when a sanitizer flag is set. The
table below is the sanitizer setup. AddressSanitizer is the only row: TSan
and LSan get a row when they have a real wheel build. The GPU list and the
installed ROCm version stay as the caller supplied them.
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

_COMPANION_BUILDS = (
    ("build_triton", "--build-triton"),
    ("build_pytorch_audio", "--build-pytorch-audio"),
    ("build_pytorch_vision", "--build-pytorch-vision"),
    ("build_apex", "--build-apex"),
)


@dataclass(frozen=True)
class SanitizerSpec:
    """One sanitizer's compiler, runtime, and import-check settings."""

    name: str
    arg_attr: str
    pytorch_switch: str
    runtime_filenames: tuple[str, ...]
    link_flag: str
    options_var: str
    default_options: str
    runtime_env: str
    preload_hsa: bool

    def runtime_candidates(self) -> tuple[str, ...]:
        machine = platform.machine().lower()
        return tuple(name.format(machine=machine) for name in self.runtime_filenames)


# USE_ASAN: PyTorch reads this from the environment. With the image's default
# GCC the build reports USE_ASAN OFF, because that compiler's libasan is not
# the ROCm runtime.
# -shared-libasan: ROCm is linked with the shared sanitizer runtime. A static
# libasan in the torch link aborts with "incompatible ASan runtimes".
# ASAN_OPTIONS: hipcc and the other build tools load the instrumented ROCm
# libraries. detect_leaks=0 keeps LeakSanitizer from aborting those tools.
# preload_hsa: the ASan runtime resolves its HSA hooks once, at startup.
# libhsa-runtime64 has to be preloaded ahead of a later dlopen or that call
# goes through a null pointer during HIP init.
SANITIZERS = {
    "asan": SanitizerSpec(
        name="asan",
        arg_attr="asan",
        pytorch_switch="USE_ASAN",
        runtime_filenames=(
            "libclang_rt.asan-{machine}.so",
            "libclang_rt.asan.so",
        ),
        link_flag="-shared-libasan",
        options_var="ASAN_OPTIONS",
        default_options="detect_leaks=0:abort_on_error=1:print_stacktrace=1",
        runtime_env="ASAN_RUNTIME_PATH",
        preload_hsa=True,
    ),
}


def selected_sanitizer(args) -> SanitizerSpec | None:
    """Return the sanitizer selected by the build arguments."""
    selected = [
        spec for spec in SANITIZERS.values() if getattr(args, spec.arg_attr, False)
    ]
    if not selected:
        return None
    if len(selected) > 1:
        names = ", ".join(f"--{spec.name}" for spec in selected)
        raise RuntimeError(f"only one sanitizer can be selected, got {names}")
    return selected[0]


def apply_sanitizer_companion_policy(parser, args) -> None:
    """Build torch only.

    An explicit companion ``--build-*`` flag is rejected. An unset flag is
    forced off before the checkout-directory default can turn it on.
    ``--no-build-*`` is left as false.

    An argparse mutually exclusive group cannot express this. The group sees
    that an option was present, and ``BooleanOptionalAction`` registers both
    ``--build-triton`` and ``--no-build-triton`` in that group. ``--asan
    --no-build-triton`` would be rejected even though the stored value is
    false. An omitted flag stays ``None`` until the assignment below.
    """
    spec = selected_sanitizer(args)
    if spec is None:
        return
    requested = [
        flag for attr, flag in _COMPANION_BUILDS if getattr(args, attr) is True
    ]
    if requested:
        parser.error(
            f"--{spec.name} builds torch only and cannot be combined with "
            + ", ".join(requested)
        )
    for attr, _flag in _COMPANION_BUILDS:
        if getattr(args, attr) is None:
            setattr(args, attr, False)


def import_sanity_env(env: dict[str, str], spec: SanitizerSpec) -> dict[str, str]:
    """Environment for ``import torch`` so HIP init survives the shared runtime.

    The sanitizer runtime must be first in ``LD_PRELOAD``. For AddressSanitizer,
    ``libhsa-runtime64.so.1`` has to follow it: without that preload,
    ``torch.cuda.is_available()`` dies with a pc 0 SEGV inside
    ``libamdhip64.so`` while HIP is initializing. ``detect_leaks=0`` keeps
    LeakSanitizer from failing the check on shutdown leaks.
    """
    runtime = env.get(spec.runtime_env, "")
    preloads = [runtime]
    hsa_runtime = ""
    if spec.preload_hsa:
        hsa_runtime = env.get("HSA_RUNTIME_PATH", "")
        preloads.append(hsa_runtime)
    if not runtime or (spec.preload_hsa and not hsa_runtime):
        required = spec.runtime_env
        if spec.preload_hsa:
            required += " and HSA_RUNTIME_PATH"
        raise RuntimeError(
            f"--{spec.name} import check requires {required} "
            "from apply_sanitizer_build_env"
        )
    return {
        "LD_PRELOAD": os.pathsep.join(preloads),
        spec.options_var: env.get(spec.options_var, spec.default_options),
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


def _resolve_hsa_runtime(rocm_dir: Path, spec: SanitizerSpec) -> Path:
    """Return the ROCr runtime that the import check must preload."""
    candidate = rocm_dir / "lib" / "libhsa-runtime64.so.1"
    if not candidate.is_file():
        raise RuntimeError(
            f"--{spec.name} import check requires the ROCr runtime "
            f"{candidate}, next to libamdhip64"
        )
    return candidate.resolve()


def _resolve_shared_runtime(clangxx: Path, rocm_dir: Path, spec: SanitizerSpec) -> Path:
    """Ask ROCm clang for this sanitizer's shared runtime so the link can find it."""
    runtime_attempts: list[str] = []
    for runtime_name in spec.runtime_candidates():
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
        f"ROCm clang++ did not resolve its shared {spec.name} runtime: "
        + "; ".join(runtime_attempts)
    )


def apply_sanitizer_build_env(
    env: dict[str, str], rocm_dir: Path, spec: SanitizerSpec
) -> None:
    """Point the torch build at the ROCm Clang shared sanitizer runtime.

    PYTORCH_ROCM_ARCH is whatever the caller already put in ``env``. This does
    not select a GPU target and does not rewrite the installed ROCm version.
    """
    if platform.system() == "Windows" or platform.machine().lower() not in (
        "x86_64",
        "amd64",
    ):
        raise RuntimeError(f"--{spec.name} is supported only on Linux x86_64")

    llvm_bin = rocm_dir / "lib" / "llvm" / "bin"
    clang = llvm_bin / "clang"
    clangxx = llvm_bin / "clang++"
    for compiler in (clang, clangxx):
        if not compiler.is_file() or not os.access(compiler, os.X_OK):
            raise RuntimeError(
                f"--{spec.name} requires the executable ROCm compiler {compiler}"
            )

    runtime_path = _resolve_shared_runtime(clangxx, rocm_dir, spec)
    env[spec.runtime_env] = str(runtime_path)
    if spec.preload_hsa:
        env["HSA_RUNTIME_PATH"] = str(_resolve_hsa_runtime(rocm_dir, spec))
    inherited_ld_library_path = env.get(
        "LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", "")
    )
    ld_library_parts = [str(runtime_path.parent), str(rocm_dir / "lib")]
    if inherited_ld_library_path:
        ld_library_parts.append(inherited_ld_library_path)

    # CC, CXX: gcc-toolset comes first on PATH in the manylinux image. Its
    # sanitizer runtime and ROCm's libclang_rt runtime abort with
    # "incompatible ASan runtimes".
    # CMAKE_C_COMPILER, CMAKE_CXX_COMPILER: CMake records these separately
    # from CC and CXX. Pin the same ROCm Clang so configure does not keep
    # gcc-toolset.
    # LD_LIBRARY_PATH: the shared-runtime link flag looks up libclang_rt while
    # linking, and that directory is not on the default linker path. The ROCm
    # lib directory is included so the instrumented libraries those tools load
    # can resolve. A caller-supplied options variable is left in place.
    env.update(
        {
            spec.pytorch_switch: "1",
            "CC": str(clang),
            "CXX": str(clangxx),
            "CMAKE_C_COMPILER": str(clang),
            "CMAKE_CXX_COMPILER": str(clangxx),
            spec.options_var: os.environ.get(spec.options_var, spec.default_options),
            "LD_LIBRARY_PATH": os.pathsep.join(ld_library_parts),
        }
    )
    # Sanitizer stack traces use frame pointers.
    append_env_text(env, "CFLAGS", "-fno-omit-frame-pointer")
    append_env_text(env, "CXXFLAGS", "-fno-omit-frame-pointer")
    append_env_text(env, "LDFLAGS", spec.link_flag)
    # CMake turns on C++ module scanning for Clang. This wheel does not build
    # modules, and the scan fails configure with this ROCm Clang.
    append_env_text(env, "CMAKE_ARGS", "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF")
    print(f"  {spec.name} compiler: {clangxx}")
    print(f"  {spec.name} runtime: {runtime_path}")
    if spec.preload_hsa:
        print(f"  HSA runtime: {env['HSA_RUNTIME_PATH']}")
    print(f"  PYTORCH_ROCM_ARCH: {env.get('PYTORCH_ROCM_ARCH', '')}")
