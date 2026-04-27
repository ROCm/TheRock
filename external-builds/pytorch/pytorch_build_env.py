# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared environment setup and utilities for PyTorch ROCm builds.

Used by both build_prod_wheels.py (CI/production) and build_dev.py (development).
"""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap

from packaging.version import parse

is_windows = platform.system() == "Windows"

# List of library preloads for Linux to generate into _rocm_init.py
LINUX_LIBRARY_PRELOADS = [
    "amd_comgr",
    "amdhip64",
    "rocprofiler-sdk",  # Linux only: needed by torch since kineto uses rocprofiler-sdk.
    "rocprofiler-sdk-roctx",  # Linux only for the moment.
    # TODO: Remove roctracer64 and roctx64 once fully switched to rocprofiler-sdk.
    "roctracer64",  # Linux only for the moment.
    "roctx64",  # Linux only for the moment.
    "hiprtc",
    "hipblas",
    "hipfft",
    "hiprand",
    "hipsparse",
    "hipsparselt",
    "hipsolver",
    "rccl",  # Linux only for the moment.
    "hipblaslt",
    "miopen",
    "hipdnn",
    "rocm_sysdeps_liblzma",
    "rocm-openblas",
    "rocm_smi64",
]

# List of library preloads for Windows to generate into _rocm_init.py
WINDOWS_LIBRARY_PRELOADS = [
    "amd_comgr",
    "amdhip64",
    "hiprtc",
    "hipblas",
    "hipfft",
    "hiprand",
    "hipsparse",
    "hipsparselt",
    "hipsolver",
    "hipblaslt",
    "miopen",
    "hipdnn",
    "rocm-openblas",
]


def run_command(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None):
    args = [str(arg) for arg in args]
    full_env = dict(os.environ)
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    if env:
        print(f":: Env:")
        for k, v in env.items():
            print(f"  {k}={v}")
        full_env.update(env)
    subprocess.check_call(args, cwd=str(cwd), env=full_env)


def capture(args: list[str | Path], cwd: Path) -> str:
    args = [str(arg) for arg in args]
    print(f"++ Capture [{cwd}]$ {shlex.join(args)}")
    try:
        return subprocess.check_output(
            args, cwd=str(cwd), stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"Error capturing output: {e}")
        print(f"Output from the failed command:\n{e.output}")
        return ""


def _get_rocm_sdk_version() -> str:
    return capture(
        [sys.executable, "-m", "rocm_sdk", "version"], cwd=Path.cwd()
    ).strip()


def _get_rocm_sdk_targets() -> str:
    # Run `rocm-sdk targets` to get the default architecture
    targets = capture([sys.executable, "-m", "rocm_sdk", "targets"], cwd=Path.cwd())
    if not targets:
        print("Warning: rocm-sdk targets returned empty or failed")
        return ""
    # Convert space-separated targets to comma-separated for PYTORCH_ROCM_ARCH
    return targets.replace(" ", ",")


def get_installed_package_version(dist_package_name: str) -> str:
    lines = capture(
        [sys.executable, "-m", "pip", "show", dist_package_name], cwd=Path.cwd()
    ).splitlines()
    if not lines:
        raise ValueError(f"Did not find installed package '{dist_package_name}'")
    prefix = "Version: "
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :]
    joined_lines = "\n".join(lines)
    raise ValueError(
        f"Did not find Version for installed package '{dist_package_name}' in output:\n{joined_lines}"
    )


def get_version_suffix_for_installed_rocm_package() -> str:
    rocm_version = get_installed_package_version("rocm")
    print(f"Computing version suffix for installed rocm package: {rocm_version}")
    # Compute a version suffix to be used as a local version identifier:
    # https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers
    # This logic is copied from build_tools/github_actions/determine_version.py.
    parsed_version = parse(rocm_version)
    base_name = "devrocm" if "dev" in rocm_version else "rocm"
    version_suffix = f"+{base_name}{str(parsed_version).replace('+','-')}"
    print(f"Version suffix is: {version_suffix}")
    return version_suffix


def _get_rocm_path(path_name: str) -> Path:
    return Path(
        capture(
            [sys.executable, "-m", "rocm_sdk", "path", f"--{path_name}"], cwd=Path.cwd()
        ).strip()
    )


def _get_rocm_init_contents():
    """Gets the contents of the _rocm_init.py file to add to the build."""
    sdk_version = _get_rocm_sdk_version()
    library_preloads = (
        WINDOWS_LIBRARY_PRELOADS if is_windows else LINUX_LIBRARY_PRELOADS
    )
    library_preloads_formatted = ", ".join(f"'{s}'" for s in library_preloads)
    return textwrap.dedent(
        f"""
        def initialize():
            import rocm_sdk
            rocm_sdk.initialize_process(
                preload_shortnames=[{library_preloads_formatted}],
                check_version='{sdk_version}')
        """
    )


def remove_dir_if_exists(dir: Path):
    if dir.exists():
        print(f"++ Removing {dir}")
        shutil.rmtree(dir)


def find_built_wheel(dist_dir: Path, dist_package: str) -> Path:
    dist_package = dist_package.replace("-", "_")
    glob = f"{dist_package}-*.whl"
    all_wheels = list(dist_dir.glob(glob))
    if not all_wheels:
        raise RuntimeError(f"No wheels matching '{glob}' found in {dist_dir}")
    if len(all_wheels) != 1:
        raise RuntimeError(f"Found multiple wheels matching '{glob}' in {dist_dir}")
    return all_wheels[0]


def directory_if_exists(dir: Path) -> Path | None:
    if dir.exists():
        return dir
    else:
        return None


def _add_env_compiler_flags(env: dict[str, str], flagname: str, *compiler_flags: str):
    current = env.get(flagname, "")
    append = ""
    for compiler_flag in compiler_flags:
        append += f"{compiler_flag} "
    env[flagname] = f"{current}{append}"
    print(f"-- Appended {flagname}+={append}")


def find_dir_containing(file_name: str, *possible_paths: Path) -> Path:
    for path in possible_paths:
        if (path / file_name).exists():
            return path
    raise ValueError(f"No directory contains {file_name}: {possible_paths}")


def do_install_rocm(args: argparse.Namespace):
    # Because the rocm package caches current GPU selection and such, we
    # always purge it to ensure a clean rebuild.
    #
    # This can fail in environments where the pip cache is disabled or
    # unwritable (e.g. manylinux containers), which is fine — if there's no
    # cache, there's nothing stale to purge.
    cache_dir_args = (
        ["--cache-dir", str(args.pip_cache_dir)] if args.pip_cache_dir else []
    )
    try:
        run_command(
            [sys.executable, "-m", "pip", "cache", "remove", "rocm"] + cache_dir_args,
            cwd=Path.cwd(),
        )
    except subprocess.CalledProcessError:
        print("Warning: pip cache remove failed (cache may be disabled), continuing")

    # Do the main pip install.
    pip_args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
    ]
    if args.pre:
        pip_args.extend(["--pre"])
    if args.index_url:
        pip_args.extend(["--index-url", args.index_url])
    if args.find_links:
        pip_args.extend(["--find-links", args.find_links])
    if args.pip_cache_dir:
        pip_args.extend(["--cache-dir", str(args.pip_cache_dir)])
    rocm_sdk_version = args.rocm_sdk_version if args.rocm_sdk_version else ""
    extras = "libraries,devel"
    if args.rocm_extras:
        extras += f",{args.rocm_extras}"
    pip_args.extend([f"rocm[{extras}]{rocm_sdk_version}"])
    run_command(pip_args, cwd=Path.cwd())
    print(f"Installed version: {_get_rocm_sdk_version()}")


def _setup_common_build_env(
    cmake_prefix: Path,
    rocm_dir: Path,
    pytorch_rocm_arch: str,
    triton_dir: Path | None,
    is_windows: bool,
) -> dict[str, str]:
    """Construct the common environment dict shared by all wheel builds."""
    env: dict[str, str] = {
        "PYTHONUTF8": "1",  # Some build files use utf8 characters, force IO encoding
        "CMAKE_PREFIX_PATH": str(cmake_prefix),
        "ROCM_HOME": str(rocm_dir),
        "ROCM_PATH": str(rocm_dir),
        "PYTORCH_ROCM_ARCH": pytorch_rocm_arch,
        "USE_KINETO": os.environ.get("USE_KINETO", "ON" if not is_windows else "OFF"),
    }

    # GLOO enabled for only Linux
    if not is_windows:
        env["USE_GLOO"] = "ON"

    # At checkout, we compute some additional env vars that influence the way that
    # the wheel is named/versioned.
    if triton_dir:
        triton_env_file = triton_dir / "build_env.json"
        if triton_env_file.exists():
            with open(triton_env_file, "r") as f:
                addl_triton_env = json.load(f)
                print(f"-- Additional triton build env vars: {addl_triton_env}")
            env.update(addl_triton_env)
        # With `CMAKE_PREFIX_PATH` set, `find_package(LLVM)` (called in
        # `MLIRConfig.cmake` shipped as part of the LLVM bundled with
        # trition) may pick up TheRock's LLVM instead of triton's.
        # Here, `CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH` is set
        # and passed via `TRITON_APPEND_CMAKE_ARGS` to avoid this.
        # See also https://github.com/ROCm/TheRock/issues/1999.
        env["TRITON_APPEND_CMAKE_ARGS"] = (
            "-DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE"
        )

    if is_windows:
        llvm_dir = rocm_dir / "lib" / "llvm" / "bin"
        env.update(
            {
                "HIP_CLANG_PATH": str(llvm_dir.resolve().as_posix()),
                "CC": str((llvm_dir / "clang-cl.exe").resolve()),
                "CXX": str((llvm_dir / "clang-cl.exe").resolve()),
            }
        )
    else:
        env.update(
            {
                # Workaround GCC12 compiler flags.
                "CXXFLAGS": " -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict ",
                "CPPFLAGS": " -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict ",
            }
        )

    # Workaround missing devicelib bitcode
    # TODO: When "ROCM_PATH" and/or "ROCM_HOME" is set in the environment, the
    # clang frontend ignores its default heuristics and (depending on version)
    # finds the wrong path to the device library. This is bad/annoying. But
    # the PyTorch build shouldn't even need these to be set. Unfortunately, it
    # has been hardcoded for a long time. So we use a clang env var to force
    # a specific device lib path to workaround the hack to get pytorch to build.
    # This may or may not only affect the Python wheels with their own quirks
    # on directory layout.
    # Obviously, this should be completely burned with fire once the root causes
    # are eliminted.
    hip_device_lib_path = rocm_dir / "lib" / "llvm" / "amdgcn" / "bitcode"
    if not hip_device_lib_path.exists():
        print(
            "WARNING: Default location of device libs not found. Relying on "
            "clang heuristics which are known to be buggy in this configuration"
        )
    else:
        env["HIP_DEVICE_LIB_PATH"] = str(hip_device_lib_path)

    # OpenBLAS path setup
    host_math_path = rocm_dir / "lib" / "host-math"
    if not host_math_path.exists():
        print(
            "WARNING: Default location of host-math not found. "
            "Will not build with OpenBLAS support."
        )
    else:
        env["BLAS"] = "OpenBLAS"
        env["OpenBLAS_HOME"] = str(host_math_path)
        env["OpenBLAS_LIB_NAME"] = "rocm-openblas"

    return env


def get_rocm_env(
    pytorch_rocm_arch: str | None = None,
    triton_dir: Path | None = None,
) -> tuple[Path, dict[str, str]]:
    """Resolve ROCm environment: paths, arch, and build env dict.

    Returns (rocm_dir, env_dict).
    """
    rocm_sdk_version = _get_rocm_sdk_version()
    cmake_prefix = _get_rocm_path("cmake")
    bin_dir = _get_rocm_path("bin")
    rocm_dir = _get_rocm_path("root")

    print(f"rocm version {rocm_sdk_version}:")
    print(f"  PYTHON VERSION: {sys.version}")
    print(f"  CMAKE_PREFIX_PATH = {cmake_prefix}")
    print(f"  BIN = {bin_dir}")
    print(f"  ROCM_HOME = {rocm_dir}")

    system_path = str(bin_dir) + os.path.pathsep + os.environ.get("PATH", "")
    print(f"  PATH = {system_path}")

    if pytorch_rocm_arch is None:
        pytorch_rocm_arch = _get_rocm_sdk_targets()
        print(
            f"  Using default PYTORCH_ROCM_ARCH from rocm-sdk targets: {pytorch_rocm_arch}"
        )
    else:
        print(f"  Using provided PYTORCH_ROCM_ARCH: {pytorch_rocm_arch}")

    if not pytorch_rocm_arch:
        raise ValueError(
            "No --pytorch-rocm-arch provided and rocm-sdk targets returned empty. "
            "Please specify --pytorch-rocm-arch (e.g., gfx942)."
        )

    env = _setup_common_build_env(
        cmake_prefix,
        rocm_dir,
        pytorch_rocm_arch,
        triton_dir,
        is_windows,
    )
    return rocm_dir, env


def prepare_pytorch_build(
    env: dict[str, str],
    pytorch_dir: Path,
    *,
    version_suffix: str,
    triton_requirement: str | None = None,
    enable_flash_attention_linux: bool | None = None,
    enable_flash_attention_windows: bool | None = None,
    enable_fbgemm_genai_linux: bool | None = None,
    pip_cache_dir: Path | None = None,
):
    """Common pre-build setup for pytorch.

    Mutates ``env`` with ROCm build flags and feature flags, writes
    ``torch/_rocm_init.py``, configures sysdeps (Linux), uninstalls any
    existing torch, and installs build requirements. Also sets
    ``os.environ["PKG_CONFIG_PATH"]`` and ``os.environ["LD_LIBRARY_PATH"]``
    for sysdeps on Linux.

    Returns the computed pytorch build version string (e.g.
    ``"2.12.0a0+rocm7.13.0"``).  Callers that build wheels should set
    ``env["PYTORCH_BUILD_VERSION"]`` and ``env["PYTORCH_BUILD_NUMBER"]``
    from this.
    """
    # Compute version.
    pytorch_build_version = (pytorch_dir / "version.txt").read_text().strip()
    pytorch_build_version += version_suffix
    pytorch_build_version_parsed = parse(pytorch_build_version)
    print(f"  Using PYTORCH_BUILD_VERSION: {pytorch_build_version}")

    is_pytorch_2_9 = pytorch_build_version_parsed.release[:2] == (2, 9)
    is_pytorch_2_11_or_later = pytorch_build_version_parsed.release[:2] >= (2, 11)

    # aotriton is not supported on certain architectures yet.
    # gfx900/gfx906/gfx908/gfx101X/gfx103X: https://github.com/ROCm/TheRock/issues/1925
    AOTRITON_UNSUPPORTED_ARCHS = ["gfx900", "gfx906", "gfx908", "gfx101", "gfx103"]
    # gfx1152/53: supported in aotriton 0.11.2b+ (https://github.com/ROCm/aotriton/pull/142),
    #   which is pinned by pytorch >= 2.11. Older versions don't include it.
    if not is_pytorch_2_11_or_later:
        AOTRITON_UNSUPPORTED_ARCHS += ["gfx1152", "gfx1153"]

    ## Enable FBGEMM_GENAI on Linux for PyTorch, as it is available only for 2.9 on rocm/pytorch
    ## and causes build failures for other PyTorch versions
    ## Warn user when enabling it manually.
    ## https://github.com/ROCm/TheRock/issues/2056
    if not is_windows:
        # Enabling/Disabling FBGEMM_GENAI based on Pytorch version in Linux
        if is_pytorch_2_9:
            # Default ON for 2.9.x, unless explicitly disabled
            # args.enable_pytorch_fbgemm_genai_linux can be set to false
            # by passing --no-enable-pytorch-fbgemm-genai-linux as input
            if enable_fbgemm_genai_linux is False:
                use_fbgemm_genai = "OFF"
                print(f"  [WARN] User-requested override to set FBGEMM_GENAI = OFF.")
            else:
                use_fbgemm_genai = "ON"
        else:
            # Default OFF for all other versions, unless explicitly enabled
            if enable_fbgemm_genai_linux is True:
                use_fbgemm_genai = "ON"
            else:
                use_fbgemm_genai = "OFF"

            if use_fbgemm_genai == "ON":
                print(f"  [WARN] User-requested override to set FBGEMM_GENAI = ON.")
                print(
                    f"""  [WARN] Please note that FBGEMM_GENAI is not available for PyTorch 2.7, and enabling it may cause build failures
                    for PyTorch >= 2.8 (Except 2.9). See status of issue https://github.com/ROCm/TheRock/issues/2056
                      """
                )

        env["USE_FBGEMM_GENAI"] = use_fbgemm_genai
        print(f"FBGEMM_GENAI enabled: {env['USE_FBGEMM_GENAI'] == 'ON'}")

        if enable_flash_attention_linux is None:
            # Default behavior — determined by if triton is build
            use_flash_attention = "ON" if triton_requirement else "OFF"

            if any(
                arch in env["PYTORCH_ROCM_ARCH"] for arch in AOTRITON_UNSUPPORTED_ARCHS
            ):
                use_flash_attention = "OFF"
            print(
                f"Flash Attention default behavior (based on triton and gpu): {use_flash_attention}"
            )
        else:
            # Explicit override: user has set the flag to true/false
            if enable_flash_attention_linux:
                assert (
                    triton_requirement
                ), "Must build with triton if wanting to use flash attention"
                use_flash_attention = "ON"
            else:
                use_flash_attention = "OFF"

            print(f"Flash Attention override set by flag: {use_flash_attention}")

        env.update(
            {
                "USE_FLASH_ATTENTION": use_flash_attention,
                "USE_MEM_EFF_ATTENTION": use_flash_attention,
            }
        )
        print(
            f"Flash Attention and Memory efficiency enabled: {env['USE_FLASH_ATTENTION'] == 'ON'}"
        )

    env["USE_ROCM"] = "ON"
    env["USE_CUDA"] = "OFF"
    env["USE_MPI"] = "OFF"
    env["USE_NUMA"] = "OFF"

    # Determine which install requirements to add.
    install_requirements = [
        f"rocm[libraries]=={_get_rocm_sdk_version()}",
    ]
    if triton_requirement:
        install_requirements.append(triton_requirement)
    env["PYTORCH_EXTRA_INSTALL_REQUIREMENTS"] = "|".join(install_requirements)
    print(
        f"--- PYTORCH_EXTRA_INSTALL_REQUIREMENTS = {env['PYTORCH_EXTRA_INSTALL_REQUIREMENTS']}"
    )

    # Add the _rocm_init.py file.
    (pytorch_dir / "torch" / "_rocm_init.py").write_text(_get_rocm_init_contents())

    # Windows-specific settings.
    if is_windows:
        _copy_msvc_libomp_to_torch_lib(pytorch_dir)

        use_flash_attention = "0"
        if enable_flash_attention_windows and not any(
            arch in env["PYTORCH_ROCM_ARCH"] for arch in AOTRITON_UNSUPPORTED_ARCHS
        ):
            use_flash_attention = "1"

        env.update(
            {
                "USE_FLASH_ATTENTION": use_flash_attention,
                "USE_MEM_EFF_ATTENTION": use_flash_attention,
                "DISTUTILS_USE_SDK": "1",
                # Workaround compile errors in 'aten/src/ATen/test/hip/hip_vectorized_test.hip'
                # on Torch 2.7.0: https://gist.github.com/ScottTodd/befdaf6c02a8af561f5ac1a2bc9c7a76.
                #   error: no member named 'modern' in namespace 'at::native'
                #     using namespace at::native::modern::detail;
                #   error: no template named 'has_same_arg_types'
                #     static_assert(has_same_arg_types<func1_t>::value, "func1_t has the same argument types");
                # We may want to fix that and other issues to then enable building tests.
                "BUILD_TEST": "0",
            }
        )
        print(
            f"  Flash attention enabled: {enable_flash_attention_windows or not is_windows}"
        )

    if not is_windows:
        # Prepend the ROCm sysdeps dir so that we use bundled libraries.
        # While a decent thing to be doing, this is presently required because:
        # TODO: include/rocm_smi/kfd_ioctl.h is included without its advertised
        # transitive includes. This triggers a compilation error for a missing
        # libdrm/drm.h.
        rocm_dir = _get_rocm_path("root")
        sysdeps_dir = rocm_dir / "lib" / "rocm_sysdeps"
        assert sysdeps_dir.exists(), f"No sysdeps directory found: {sysdeps_dir}"
        _add_env_compiler_flags(env, "CXXFLAGS", f"-I{sysdeps_dir / 'include'}")
        # Add correct include path for roctracer.h (for Kineto)
        _add_env_compiler_flags(
            env, "CXXFLAGS", f"-I{rocm_dir / 'include' / 'roctracer'}"
        )
        _add_env_compiler_flags(env, "LDFLAGS", f"-L{sysdeps_dir / 'lib'}")

        # needed to find liblzma packaged by rocm as sysdep to build aotriton
        os.environ["PKG_CONFIG_PATH"] = f"{sysdeps_dir / 'lib' / 'pkgconfig'}"
        os.environ["LD_LIBRARY_PATH"] = f"{sysdeps_dir / 'lib'}"

    print("+++ Uninstalling pytorch:")
    run_command(
        [sys.executable, "-m", "pip", "uninstall", "torch", "-y"],
        cwd=tempfile.gettempdir(),
    )

    print("+++ Installing pytorch requirements:")
    pip_install_args = []
    if pip_cache_dir:
        pip_install_args.extend(["--cache-dir", pip_cache_dir])
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            pytorch_dir / "requirements.txt",
        ]
        + pip_install_args,
        cwd=pytorch_dir,
    )

    if is_windows:
        # As of 2025-06-24, the 'ninja' package on pypi is trailing too far
        # behind upstream:
        # * https://pypi.org/project/ninja/#history
        # * https://github.com/ninja-build/ninja/releases
        # Version 1.11.1 is buggy on Windows (looping without making progress):
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "ninja",
                "-y",
            ],
            cwd=pytorch_dir,
        )

    return pytorch_build_version


def _copy_msvc_libomp_to_torch_lib(pytorch_dir: Path):
    # When USE_OPENMP is set (it is by default), torch_cpu.dll depends on OpenMP.
    #
    # Typically implementations of OpenMP are:
    #   * Intel OpenMP, `libiomp`, which PyTorch upstream uses
    #   * MSVC OpenMP, `libomp140`, which we'll use here since we have MSVC already
    #   * (?) LLVM OpenMP (https://openmp.llvm.org/)?
    #
    # Torch's CMake build selects which OpenMP to use in `FindOpenMP.cmake`,
    # then the relevant .dll files must be copied into the torch/lib/ folder or
    # torch will fail to initialize. This feels like something that could be
    # handled upstream as part of the centralized setup.py and/or CMake build
    # processes, but given the varied scripts and build workflows upstream and
    # multiple choices for where to source an implementation, we handle it here.
    #
    # If we wanted to switch to Intel OpenMP, we could:
    #   1. Install Intel OpenMP (and/or MKL?)
    #   2. Set CMAKE_INCLUDE_PATH and CMAKE_LIBRARY_PATH (?) so `FindOpenMP.cmake` finds them
    #   3. Copy `libiomp5md.dll` to torch/lib
    # Then remove the rest of the code from this function.

    vc_tools_redist_dir = os.environ.get("VCToolsRedistDir", "")
    if not vc_tools_redist_dir:
        raise RuntimeError("VCToolsRedistDir not set, can't copy libomp to torch lib")

    omp_name = "libomp140.x86_64.dll"
    dll_paths = sorted(Path(vc_tools_redist_dir).rglob(omp_name))
    if not dll_paths:
        raise RuntimeError(
            f"Did not find '{omp_name}' under '{vc_tools_redist_dir}', can't copy libomp to torch lib"
        )

    omp_path = dll_paths[0]
    target_lib = pytorch_dir / "torch" / "lib"
    print(f"Copying libomp from '{omp_path}' to '{target_lib}'")
    shutil.copy2(omp_path, target_lib)


def sanity_check_pytorch():
    """Verify that torch imports and can see CUDA devices."""
    print("+++ Sanity checking installed torch (unavailable is okay on CPU machines):")
    sanity_check_output = capture(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        cwd=tempfile.gettempdir(),
    )
    if not sanity_check_output:
        raise RuntimeError("torch package sanity check failed (see output above)")
    else:
        print(f"Sanity check output:\n{sanity_check_output}")
