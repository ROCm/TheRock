# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from typing import List, Optional
import functools
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
            if lib_entry.optional:
                # Optional libraries (e.g. WSL-only rocdxg) may be absent from
                # the distribution. Skip rather than failing.
                continue
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

_ASAN_RUNTIME_PREFIX = "libclang_rt.asan"


@functools.lru_cache(maxsize=1)
def _asan_runtime_loaded() -> bool:
    """Whether the ASan runtime is already mapped into this process.

    An ASan-instrumented library can only be loaded once its runtime is the first
    entry in the process' initial library list, which in practice means the
    process was started with `LD_PRELOAD=libclang_rt.asan*.so`. In that case the
    runtime shows up in `/proc/self/maps`. Only meaningful on Linux.
    """
    try:
        with open("/proc/self/maps", "r") as maps:
            return _ASAN_RUNTIME_PREFIX in maps.read()
    except OSError:
        return False


def _elf_needed_libraries(path: Path) -> list[str]:
    """Return the `DT_NEEDED` shared-library names of a 64-bit ELF file.

    Only the ELF header, program headers and the referenced dynamic-string
    entries are read, so this stays cheap even for very large libraries. Returns
    an empty list for non-ELF or unsupported files (best effort).
    """
    import struct

    try:
        with open(path, "rb") as f:
            ident = f.read(16)
            if ident[:4] != b"\x7fELF" or ident[4] != 2:  # 64-bit ELF only
                return []
            endian = "<" if ident[5] == 1 else ">"
            f.seek(0x20)
            (e_phoff,) = struct.unpack(endian + "Q", f.read(8))
            f.seek(0x36)
            e_phentsize, e_phnum = struct.unpack(endian + "HH", f.read(4))

            dyn_off = 0
            dyn_size = 0
            loads = []  # (p_vaddr, p_offset, p_filesz) for PT_LOAD segments.
            for i in range(e_phnum):
                f.seek(e_phoff + i * e_phentsize)
                phdr = f.read(56)  # sizeof(Elf64_Phdr)
                if len(phdr) < 56:
                    break
                (p_type,) = struct.unpack(endian + "I", phdr[0:4])
                p_offset, p_vaddr, _p_paddr, p_filesz = struct.unpack(
                    endian + "QQQQ", phdr[8:40]
                )
                if p_type == 2:  # PT_DYNAMIC
                    dyn_off, dyn_size = p_offset, p_filesz
                elif p_type == 1:  # PT_LOAD
                    loads.append((p_vaddr, p_offset, p_filesz))
            if not dyn_off:
                return []

            f.seek(dyn_off)
            dyn = f.read(dyn_size)
            needed_offsets = []
            strtab_vaddr = 0
            for off in range(0, len(dyn) - 15, 16):  # array of Elf64_Dyn
                d_tag, d_val = struct.unpack(endian + "qQ", dyn[off : off + 16])
                if d_tag == 0:  # DT_NULL
                    break
                if d_tag == 1:  # DT_NEEDED (d_val is an offset into DT_STRTAB)
                    needed_offsets.append(d_val)
                elif d_tag == 5:  # DT_STRTAB (virtual address)
                    strtab_vaddr = d_val
            if not needed_offsets or not strtab_vaddr:
                return []

            strtab_off = None
            for p_vaddr, p_offset, p_filesz in loads:
                if p_vaddr <= strtab_vaddr < p_vaddr + p_filesz:
                    strtab_off = p_offset + (strtab_vaddr - p_vaddr)
                    break
            if strtab_off is None:
                return []

            names = []
            for rel in needed_offsets:
                f.seek(strtab_off + rel)
                raw = f.read(256)
                end = raw.find(b"\x00")
                names.append(
                    raw[: end if end >= 0 else len(raw)].decode("utf-8", "replace")
                )
            return names
    except OSError:
        return []


def _library_needs_asan_runtime(path: Path) -> bool:
    """Whether an ELF library declares a `DT_NEEDED` on the ASan runtime."""
    return any(n.startswith(_ASAN_RUNTIME_PREFIX) for n in _elf_needed_libraries(path))


def _check_asan_preloaded(path: Path):
    """Raise an actionable error before loading an instrumented library unsafely.

    Loading an ASan-instrumented library via `dlopen` when the ASan runtime was
    not preloaded at process startup makes ASan `abort()` the whole process with
    "ASan runtime does not come first in initial library list" -- an uncatchable
    SIGABRT. Detect that situation up front and raise a clear error instead. See
    TheRock#6331.
    """
    if platform.system() != "Linux" or _asan_runtime_loaded():
        return
    if not _library_needs_asan_runtime(path):
        return
    asan_lib = f"{_ASAN_RUNTIME_PREFIX}-{platform.machine()}.so"
    raise RuntimeError(
        f"Cannot load '{path.name}': it was built with AddressSanitizer and "
        f"requires the ASan runtime to be the first library in the process. "
        f"Preload it at startup, e.g.:\n"
        f"    LD_PRELOAD=$(amdclang++ -print-file-name={asan_lib}) <command>\n"
        f"The ASan runtime cannot be loaded after the interpreter has started. "
        f"See TheRock#6331."
    )


def preload_libraries(*shortnames: str, rtld_global: bool = True):
    """Preloads a list of library names, caching their handles globally.

    This is typically used in applications which depend on rocm runtime libraries
    prior to loading any of their own shared libraries that depend on them. By
    preloading into the linker namespace, it ensures that subsequent resolution of them
    by name should succeed.

    Library paths are resolved via `find_libraries`.
    """
    import ctypes

    mode = ctypes.RTLD_GLOBAL if rtld_global is True else ctypes.RTLD_LOCAL
    for shortname in shortnames:
        if shortname in _ALL_CDLLS:
            continue
        # Resolve per shortname: find_libraries may skip entries (libraries
        # absent on the current platform or optional ones not present in the
        # distribution), so a positional zip against shortnames would mispair.
        paths = find_libraries(shortname)
        if not paths:
            continue
        _check_asan_preloaded(paths[0])
        cdll = ctypes.CDLL(str(paths[0]), mode=mode)
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
