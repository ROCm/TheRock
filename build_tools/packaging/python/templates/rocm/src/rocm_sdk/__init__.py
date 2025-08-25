from typing import List, Optional
import importlib
import os
from pathlib import Path
import platform
import re

from ._dist_info import __version__

__all__ = [
    "__version__",
    "find_libraries",
    "initialize_process",
]


def find_libraries(*shortnames: str) -> list[Path]:
    """Finds absolute paths to dynamic libraries by shortname.

    See the list of LibraryEntry in _dist_info for the mapping of short names to
    dist package and path.

    Raises:
        ModuleNotFoundError if any packages are not installed which provide the
        requested libraries.
    """
    from . import _dist_info

    paths: list[Path] = []
    missing_extras: set[str] = set()
    is_windows = platform.system() == "Windows"
    for shortname in shortnames:
        try:
            lib_entry = _dist_info.ALL_LIBRARIES[shortname]
        except KeyError:
            raise ModuleNotFoundError(f"Unknown rocm library '{shortname}'")

        if is_windows and not lib_entry.dll_pattern:
            # Library is missing on Windows, skip it.
            # TODO(#827): Require callers to filter and error here instead?
            continue

        package = lib_entry.package
        target_family = None
        if package.is_target_specific:
            target_family = _dist_info.determine_target_family()
        py_package_name = package.get_py_package_name(target_family)
        try:
            py_module = importlib.import_module(py_package_name)
        except ModuleNotFoundError:
            missing_extras.add(package.logical_name)
        py_root = Path(py_module.__file__).parent  # Chop __init__.py
        if is_windows:
            relpath = py_root / lib_entry.windows_relpath
            entry_pattern = lib_entry.dll_pattern
        else:
            relpath = py_root / lib_entry.posix_relpath
            entry_pattern = lib_entry.so_pattern
        matching_paths = sorted(relpath.glob(entry_pattern))
        if len(matching_paths) == 0:
            raise FileNotFoundError(
                f"Could not find rocm library '{shortname}' at path '{relpath},' no match for pattern '{entry_pattern}'"
            )

        # If there are multiple paths matching the pattern, they are likely
        # versioned symlinks. For example:
        #   ['libhipblaslt.so', 'libhipblaslt.so.1', 'libhipblaslt.so.1.0']
        # Take whichever sorted first.
        path = matching_paths[0]

        paths.append(path)

    if missing_extras:
        raise ModuleNotFoundError(
            f"Missing required rocm libraries. Please refer to Python "
            f"setup instructions, reinstall your virtual environment, or attempt to "
            f"install manually via `pip install rocm[{','.join(missing_extras)}]`"
        )
    return paths


_ALL_CDLLS = {}


def preload_libraries(*shortnames: str, rtld_global: bool = True):
    """Preloads a list of library names, caching their handles globally.

    This is typically used in applications which depend on rocm runtime libraries
    prior to loading any of their own shared libraries that depend on them. By
    preloading into the linker namespace, it ensures that subsequent resolution of them
    by name should succeed.

    Library paths are resolved via `find_libraries`.
    """
    import ctypes

    paths = find_libraries(*shortnames)
    mode = ctypes.RTLD_GLOBAL if rtld_global is True else ctypes.RTLD_LOCAL
    for shortname, path in zip(shortnames, paths):
        if shortname in _ALL_CDLLS:
            continue
        cdll = ctypes.CDLL(str(path), mode=mode)
        _ALL_CDLLS[shortname] = cdll


def initialize_process(
    *,
    preload_shortnames: Optional[List[str]] = None,
    rtld_global: bool = True,
    env_override: bool = True,
    check_version: Optional[str | re.Pattern] = None,
    fail_on_version_mismatch: bool = False,
    **kwargs,
):
    """Global initialization of a python library which depends on ROCm native
    libraries via these packages. This is intended to be called by framework
    initialization code in a consistent and future proof way.

    Args:
        preload_shortnames: Library short-names to pass to preload_libraries.
        rtld_global: Whether to preload libraries with RTLD_GLOBAL (default True).
        env_override: If True, then also consult the `ROCM_SDK_PRELOAD_LIBRARIES`
          env variable and preload any libraries listed there (default True).
          Values are either comma or semi-colon delimitted.
        check_version: If present, checks that the rocm_sdk.__version__ matches
          what the caller expects. By default, issues a warning on mismatch.
          The version spec can contain '*' which expands to any number of
          characters.
        fail_on_version_mismatch: If True, then fail with a RuntimeError on
          version mismatch (default False).
    """
    if preload_shortnames:
        preload_libraries(*preload_shortnames, rtld_global=rtld_global)

    # Process environment variable overrides.
    if env_override:
        addl_preload_str = os.getenv("ROCM_SDK_PRELOAD_LIBRARIES")
        if addl_preload_str is not None:
            addl_preload_split = [s.strip() for s in re.split("[,;]", addl_preload_str)]
            addl_preload_split = [s for s in addl_preload_split if s]
            if addl_preload_split:
                try:
                    preload_libraries(*addl_preload_split, rtld_global=rtld_global)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not preload libraries from environment variable "
                        f"ROCM_SDK_PRELOAD_LIBRARIES='{addl_preload_str}'. Check this "
                        f"environment variable and unset it if not correct/needed."
                    ) from e

    # Version check.
    if check_version:
        if not isinstance(check_version, re.Pattern):
            pattern_str = re.escape(check_version).replace("\\*", ".*")
            check_version = re.compile(f"^{pattern_str}$")
        if not re.match(check_version, __version__):
            check_fail_message = (
                f"The program was compiled against a ROCm version matching "
                f"'{pattern_str}' but the installed ROCm version in this Python "
                f"environment is {__version__}."
            )
            if fail_on_version_mismatch:
                raise RuntimeError(check_fail_message)
            else:
                import warnings

                warnings.warn(
                    f"{check_fail_message} This incompatibility may result in "
                    f"unexpected behavior"
                )
