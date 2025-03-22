"""Trampoline for console scripts."""

import importlib
import os
import sys
from pathlib import Path

from ._dist_info import ALL_PACKAGES

CORE_PACKAGE = ALL_PACKAGES["core"]
PLATFORM_NAME = CORE_PACKAGE.get_py_package_name()
PLATFORM_MODULE = importlib.import_module(PLATFORM_NAME)
# NOTE: dependent on there being an __init__.py in the platform package.
PLATFORM_PATH = Path(PLATFORM_MODULE.__file__).parent


def _exec(relpath: str):
    full_path = PLATFORM_PATH / relpath
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


def hipcc():
    _exec("bin/hipcc")


def hipconfig():
    _exec("bin/hipconfig")


def rocm_agent_enumerator():
    _exec("bin/rocm_agent_enumerator")


def rocm_info():
    _exec("bin/rocminfo")


def rocm_smi():
    _exec("bin/rocm-smi")


def roc_obj():
    _exec("bin/roc-obj")


def roc_obj():
    _exec("bin/roc-obj")


def roc_obj_extract():
    _exec("bin/roc-obj-extract")


def roc_obj_ls():
    _exec("bin/roc-obj-ls")
