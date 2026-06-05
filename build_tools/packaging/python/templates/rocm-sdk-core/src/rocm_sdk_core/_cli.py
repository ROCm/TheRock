# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Trampoline for console scripts."""

import importlib
import importlib.util
import os
import platform
import sys
from pathlib import Path

from ._dist_info import ALL_PACKAGES

CORE_PACKAGE = ALL_PACKAGES["core"]
CORE_PY_PACKAGE_NAME = CORE_PACKAGE.get_py_package_name()


def _get_core_module_path():
    # NOTE: dependent on there being an __init__.py in the core package.
    core_module = importlib.import_module(CORE_PY_PACKAGE_NAME)
    return Path(core_module.__file__).parent


DEVEL_PACKAGE = ALL_PACKAGES["devel"]
DEVEL_PURE_PY_PACKAGE_NAME = DEVEL_PACKAGE.pure_py_package_name
DEVEL_PY_PACKAGE_NAME = DEVEL_PACKAGE.get_py_package_name()


def _has_devel_module():
    return importlib.util.find_spec(DEVEL_PURE_PY_PACKAGE_NAME) is not None


def _is_devel_module_expanded():
    return importlib.util.find_spec(DEVEL_PY_PACKAGE_NAME) is not None


def _expand_devel_module():
    import subprocess

    try:
        subprocess.check_call([sys.executable, "-m", "rocm_sdk", "init", "--quiet"])
    except subprocess.CalledProcessError:
        print(
            "ERROR: Failed to expand rocm[devel] package. "
            "Try running `rocm-sdk init` manually for details.",
            file=sys.stderr,
        )
        sys.exit(1)


def _get_devel_module_path():
    # NOTE: dependent on there being an __init__.py in the devel package.
    try:
        devel_module = importlib.import_module(DEVEL_PY_PACKAGE_NAME)
    except ImportError:
        print(
            "WARNING: Failed to import devel module, falling back to core.",
            file=sys.stderr,
        )
        return _get_core_module_path()
    if devel_module.__file__ is None:
        print(
            "WARNING: Devel module has no __file__, falling back to core.",
            file=sys.stderr,
        )
        return _get_core_module_path()
    return Path(devel_module.__file__).parent


def _get_module_path(expand_devel: bool) -> Path:
    """Gets the module path, either from 'core' or 'devel'.

    If the 'devel' package IS NOT installed then 'core' is used, ignoring the input of `expand_devel`.
    If the 'devel' package IS installed AND already expanded then it is used.
    If the 'devel' package IS installed AND NOT already expanded then either
      A) System information tools like amd-smi can choose to run more quickly
         with 'core' by skipping the (compute-intensive) 'devel' expansion.
         These tools should pass `expand_devel=False`.
      B) Other tools that benefit from the extra files in the 'devel' package
         will expand expand it by passing `expand_devel=True`.
    """
    if _has_devel_module():
        if _is_devel_module_expanded():
            return _get_devel_module_path()
        elif expand_devel:
            _expand_devel_module()
            return _get_devel_module_path()
        else:
            # Passthrough. Fallback to core module.
            pass

    return _get_core_module_path()


is_windows = platform.system() == "Windows"
exe_suffix = ".exe" if is_windows else ""


def _core_wheel_path() -> Path:
    return _get_core_module_path().resolve()


def _host_has_stale_rocm_env(core_path: Path) -> bool:
    """True when host ROCM_PATH/HIP_PATH point outside the installed core wheel."""
    for var in ("ROCM_PATH", "HIP_PATH"):
        value = os.environ.get(var)
        if not value:
            continue
        try:
            if Path(value).resolve() != core_path:
                return True
        except OSError:
            return True
    return False


def _clear_stale_host_rocm_env() -> None:
    os.environ.pop("ROCM_PATH", None)
    os.environ.pop("HIP_PATH", None)


def _hip_path_flags(root: Path) -> list[str]:
    root_posix = root.as_posix()
    extra: list[str] = []
    if not any(arg.startswith("--rocm-path=") for arg in sys.argv[1:]):
        extra.append(f"--rocm-path={root_posix}")
    if not any(arg.startswith("--hip-path=") for arg in sys.argv[1:]):
        extra.append(f"--hip-path={root_posix}")
    return extra


def _prepare_hipcc_launch() -> tuple[str, list[str]]:
    """Launch hipcc from the core wheel tree.

    Clang/LLVM live under core even when console scripts expand devel. Stale
    host ROCM_PATH/HIP_PATH on Windows runners must not override the wheel;
    --rocm-path/--hip-path take precedence in hipcc even without llvm 0006/0007.
    """
    core_path = _core_wheel_path()
    full_path = core_path / ("bin/hipcc" + exe_suffix)
    launch_path = str(full_path)
    argv = [launch_path, *sys.argv[1:]]

    if not full_path.is_file():
        return launch_path, argv

    if is_windows or _host_has_stale_rocm_env(core_path):
        _clear_stale_host_rocm_env()
        argv = [launch_path, *_hip_path_flags(core_path), *sys.argv[1:]]
    return launch_path, argv


def _prepare_hipconfig_launch(expand_devel: bool) -> tuple[str, list[str]]:
    """Launch hipconfig from core or devel, dropping stale host env overrides."""
    core_path = _core_wheel_path()
    module_path = _get_module_path(expand_devel).resolve()
    full_path = module_path / ("bin/hipconfig" + exe_suffix)
    launch_path = str(full_path)
    argv = [launch_path, *sys.argv[1:]]

    if not full_path.is_file():
        return launch_path, argv

    if is_windows or _host_has_stale_rocm_env(core_path):
        _clear_stale_host_rocm_env()
    return launch_path, argv


def _exec(relpath: str, expand_devel=True):
    # Default is True because most CLI tools are compiler/build tools that
    # need the devel files. System info tools (amd-smi, rocminfo, etc.)
    # override with expand_devel=False to avoid the expansion cost.
    if relpath == "bin/hipcc":
        launch_path, argv = _prepare_hipcc_launch()
    elif relpath == "bin/hipconfig":
        launch_path, argv = _prepare_hipconfig_launch(expand_devel)
    else:
        module_path = _get_module_path(expand_devel)
        launch_path = str(module_path / (relpath + exe_suffix))
        argv = [launch_path, *sys.argv[1:]]
    if is_windows:
        # https://bugs.python.org/issue19124
        # prevent execution from occuring in the backround
        os._exit(os.spawnv(os.P_WAIT, launch_path, argv))
    os.execv(launch_path, argv)


def amdclang():
    _exec("lib/llvm/bin/amdclang")


def amdclang_cpp():
    _exec("lib/llvm/bin/amdclang-cpp")


def amdclang_cl():
    _exec("lib/llvm/bin/amdclang-cl")


def amdclangpp():
    _exec("lib/llvm/bin/amdclang++")


def amdflang():
    _exec("lib/llvm/bin/amdflang")


def amdlld():
    _exec("lib/llvm/bin/amdlld")


def amd_smi():
    _exec("bin/amd-smi", expand_devel=False)


def hipcc():
    _exec("bin/hipcc")


def hipconfig():
    _exec("bin/hipconfig")


def hipify_clang():
    _exec("bin/hipify-clang")


def hipify_perl():
    _exec("bin/hipify-perl")


def hipInfo():
    _exec("bin/hipInfo", expand_devel=False)


def offload_arch():
    _exec("lib/llvm/bin/offload-arch")


def rocm_agent_enumerator():
    _exec("bin/rocm_agent_enumerator", expand_devel=False)


def rocm_info():
    _exec("bin/rocminfo", expand_devel=False)


def rocm_smi():
    _exec("bin/rocm-smi", expand_devel=False)


def roccoremerge():
    _exec("bin/roccoremerge")


def rocgdb():
    _exec("bin/rocgdb")


def rocpd():
    _exec("bin/rocpd")


def rocpd2csv():
    _exec("bin/rocpd2csv")


def rocpd2otf2():
    _exec("bin/rocpd2otf2")


def rocpd2pftrace():
    _exec("bin/rocpd2pftrace")


def rocpd2summary():
    _exec("bin/rocpd2summary")


def rocprofv3():
    _exec("bin/rocprofv3")


def rocprofv3_attach():
    _exec("bin/rocprofv3-attach")


def rocprofv3_avail():
    _exec("bin/rocprofv3-avail")
