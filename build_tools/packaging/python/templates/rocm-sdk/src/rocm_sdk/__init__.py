import importlib
from pathlib import Path
import platform

from ._dist_info import __version__

__all__ = [
    "__version__",
    "find_libraries",
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
    is_windows = platform.platform() == "Windows"
    for shortname in shortnames:
        try:
            lib_entry = _dist_info.ALL_LIBRARIES[shortname]
        except KeyError:
            raise ModuleNotFoundError(f"Unknown rocm-sdk library '{shortname}'")
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
        assert not is_windows, "rocm_sdk.find_libraries not yet supported on Windows"
        path = py_root / lib_entry.posix_relpath / lib_entry.soname
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find rocm-sdk library '{shortname}' at path {path}"
            )
        paths.append(path)

    if missing_extras:
        raise ModuleNotFoundError(
            f"Missing required rocm-sdk libraries. Please refer to Python "
            f"setup instructions, reinstall your virtual environment, or attempt to "
            f"install manually via `pip install rocm-sdk[{','.join(missing_extras)}]`"
        )
    return paths


_ALL_CDLLS = []


def preload_libraries(*shortnames: str, rtld_global: bool = True):
    """Preloads a list of library names, caching their handles globally.

    This is typically used in applications which depend on rocm-sdk runtime libraries
    prior to loading any of their own shared libraries that depend on them. By
    preloading into the linker namespace, it ensures that subsequent resolution of them
    by name should succeed.

    Library paths are resolved via `find_libraries`.
    """
    import ctypes

    paths = find_libraries(*shortnames)
    mode = ctypes.RTLD_GLOBAL if rtld_global is True else ctypes.RTLD_LOCAL
    for path in paths:
        cdll = ctypes.CDLL(str(path), mode=mode)
        _ALL_CDLLS.append(cdll)
