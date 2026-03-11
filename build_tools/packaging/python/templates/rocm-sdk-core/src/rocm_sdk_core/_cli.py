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


def _exec(relpath: str, expand_devel=True):
    # Default is True because most CLI tools are compiler/build tools that
    # need the devel files. System info tools (amd-smi, rocminfo, etc.)
    # override with expand_devel=False to avoid the expansion cost.
    full_path = _get_module_path(expand_devel) / (relpath + exe_suffix)
    os.execv(full_path, [str(full_path)] + sys.argv[1:])


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


def rocprofv3():
    _exec("bin/rocprofv3")


def rocprofv3_attach():
    _exec("bin/rocprofv3-attach")


def rocprofv3_avail():
    _exec("bin/rocprofv3-avail")
