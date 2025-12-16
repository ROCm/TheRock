"""Trampoline for console scripts."""

import importlib
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


def _have_devel_module():
    return importlib.util.find_spec(DEVEL_PURE_PY_PACKAGE_NAME) is not None


def _is_devel_module_expanded():
    return importlib.util.find_spec(DEVEL_PY_PACKAGE_NAME) is not None


def _expand_devel_module():
    import subprocess

    # -P flag is only available in Python 3.11+
    cmd = [sys.executable]
    if sys.version_info >= (3, 11):
        cmd.append("-P")
    cmd.extend(["-m", "rocm_sdk", "init", "--quiet"])
    subprocess.check_call(cmd)


def _get_devel_module_path():
    # NOTE: dependent on there being an __init__.py in the devel package.
    devel_module = importlib.import_module(DEVEL_PY_PACKAGE_NAME)
    return Path(devel_module.__file__).parent


def _get_module_path(should_expand_devel: bool) -> Path:
    """Gets the module path, either from 'core' or 'devel'.

    If the 'devel' package IS NOT installed then 'core' is used.
    If the 'devel' package IS installed AND already expanded then it is used.
    If the 'devel' package IS installed AND NOT already expanded then either
      A) System information tools like amd-smi can choose to run more quickly
         with 'core' by skipping the (compute-intensive) 'devel' expansion.
         These tools should pass `should_expand_devel=False`.
      B) Other tools that benefit from the extra files in the 'devel' package
         will expand expand it by passing `should_expand_devel=True`.
    """
    if _have_devel_module():
        if _is_devel_module_expanded():
            return _get_devel_module_path()
        elif should_expand_devel:
            _expand_devel_module()
            return _get_devel_module_path()
        else:
            # Passthrough. Fallback to core module.
            pass

    return _get_core_module_path()


PLATFORM_NAME = CORE_PACKAGE.get_py_package_name()
PLATFORM_MODULE = importlib.import_module(PLATFORM_NAME)
# NOTE: dependent on there being an __init__.py in the platform package.
PLATFORM_PATH = Path(PLATFORM_MODULE.__file__).parent
CORE_PY_PACKAGE_NAME = CORE_PACKAGE.get_py_package_name()


def _get_core_module_path():
    # NOTE: dependent on there being an __init__.py in the core package.
    core_module = importlib.import_module(CORE_PY_PACKAGE_NAME)
    return Path(core_module.__file__).parent


DEVEL_PACKAGE = ALL_PACKAGES["devel"]
DEVEL_PURE_PY_PACKAGE_NAME = DEVEL_PACKAGE.pure_py_package_name
DEVEL_PY_PACKAGE_NAME = DEVEL_PACKAGE.get_py_package_name()


def _have_devel_module():
    return importlib.util.find_spec(DEVEL_PURE_PY_PACKAGE_NAME) is not None


def _is_devel_module_expanded():
    return importlib.util.find_spec(DEVEL_PY_PACKAGE_NAME) is not None


def _expand_devel_module():
    import subprocess

    # -P flag is only available in Python 3.11+
    cmd = [sys.executable]
    if sys.version_info >= (3, 11):
        cmd.append("-P")
    cmd.extend(["-m", "rocm_sdk", "init", "--quiet"])
    subprocess.check_call(cmd)


def _get_devel_module_path():
    # NOTE: dependent on there being an __init__.py in the devel package.
    devel_module = importlib.import_module(DEVEL_PY_PACKAGE_NAME)
    return Path(devel_module.__file__).parent


def _get_module_path(should_expand_devel: bool) -> Path:
    """Gets the module path, either from 'core' or 'devel'.

    If the 'devel' package IS NOT installed then 'core' is used.
    If the 'devel' package IS installed AND already expanded then it is used.
    If the 'devel' package IS installed AND NOT already expanded then either
      A) System information tools like amd-smi can choose to run more quickly
         with 'core' by skipping the (compute-intensive) 'devel' expansion.
         These tools should pass `should_expand_devel=False`.
      B) Other tools that benefit from the extra files in the 'devel' package
         will expand expand it by passing `should_expand_devel=True`.
    """
    if _have_devel_module():
        if _is_devel_module_expanded():
            return _get_devel_module_path()
        elif should_expand_devel:
            _expand_devel_module()
            return _get_devel_module_path()
        else:
            # Passthrough. Fallback to core module.
            pass

    return _get_core_module_path()


is_windows = platform.system() == "Windows"
exe_suffix = ".exe" if is_windows else ""


def _exec(relpath: str):
    full_path = PLATFORM_PATH / (relpath + exe_suffix)
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
