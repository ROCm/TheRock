"""Trampoline for console scripts."""

import importlib
import os
import platform
import sys
from pathlib import Path

from ._dist_info import ALL_PACKAGES

CORE_PACKAGE = ALL_PACKAGES["core"]
CORE_PY_PACKAGE_NAME = CORE_PACKAGE.get_py_package_name()
# NOTE: dependent on there being an __init__.py in the core package.
CORE_MODULE = importlib.import_module(CORE_PY_PACKAGE_NAME)
CORE_MODULE_PATH = Path(CORE_MODULE.__file__).parent

DEVEL_PACKAGE = ALL_PACKAGES["devel"]
DEVEL_PY_PACKAGE_NAME = DEVEL_PACKAGE.get_py_package_name()
DEVEL_DIST_PACKAGE_NAME = DEVEL_PACKAGE.get_dist_package_name()

try:
    # NOTE: dependent on there being an __init__.py in the devel package.
    # The package must be initialized (e.g. with `rocm-sdk init`) first.
    # TODO(#1880): auto-init if DEVEL_DIST_PACKAGE_NAME (rocm_sdk_devel) exists?
    DEVEL_MODULE = importlib.import_module(DEVEL_PY_PACKAGE_NAME)
    DEVEL_MODULE_PATH = Path(DEVEL_MODULE.__file__).parent

    # Devel module is available, use it.
    MODULE_PATH = DEVEL_MODULE_PATH
except ModuleNotFoundError:
    # Fallback to the core module.
    MODULE_PATH = CORE_MODULE_PATH

is_windows = platform.system() == "Windows"
exe_suffix = ".exe" if is_windows else ""


def _exec(relpath: str):
    full_path = MODULE_PATH / (relpath + exe_suffix)
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
    _exec("bin/amd-smi")


def hipcc():
    _exec("bin/hipcc")


def hipconfig():
    _exec("bin/hipconfig")


def hipify_clang():
    _exec("bin/hipify-clang")


def hipify_perl():
    _exec("bin/hipify-perl")


def hipInfo():
    _exec("bin/hipInfo")


def offload_arch():
    _exec("lib/llvm/bin/offload-arch")


def rocm_agent_enumerator():
    _exec("bin/rocm_agent_enumerator")


def rocm_info():
    _exec("bin/rocminfo")


def rocm_smi():
    _exec("bin/rocm-smi")
