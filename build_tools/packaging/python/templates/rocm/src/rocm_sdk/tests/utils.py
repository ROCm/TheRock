# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test utilities."""

from pathlib import Path
import base64
import ctypes
import hashlib
import importlib.metadata
import platform
import shlex
import subprocess
import sys
import sysconfig

is_windows = platform.system() == "Windows"
exe_suffix = ".exe" if is_windows else ""


def get_loaded_library_path(library: ctypes.CDLL) -> Path:
    """Ask the loader which file backs an already loaded library handle."""
    if is_windows:
        get_name = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleFileNameW
        get_name.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        get_name.restype = ctypes.c_uint32
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_name(library._handle, buffer, len(buffer))
        if not length or length == len(buffer):
            raise OSError("Cannot resolve loaded DLL path")
        return Path(buffer.value).resolve(strict=True)

    class LinkMap(ctypes.Structure):
        _fields_ = [("address", ctypes.c_void_p), ("name", ctypes.c_char_p)]

    # The Linux SDK supports glibc; RTLD_DI_LINKMAP is 2 in dlfcn.h.
    dlinfo = ctypes.CDLL("libdl.so.2").dlinfo
    dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    dlinfo.restype = ctypes.c_int
    link_map = ctypes.POINTER(LinkMap)()
    if dlinfo(library._handle, 2, ctypes.byref(link_map)) != 0 or not link_map:
        raise OSError("Cannot resolve loaded shared-library path")
    return Path(link_map.contents.name.decode()).resolve(strict=True)


def get_record_sha256(distributions: list[str], library_path: Path) -> str:
    """Read the wheel's expected digest, independently of the DLL/SO bytes."""
    expected_path = library_path.resolve(strict=True)
    for distribution in distributions:
        for entry in importlib.metadata.files(distribution) or ():
            if Path(entry.locate()).resolve() == expected_path:
                if entry.hash is None or entry.hash.mode != "sha256":
                    raise AssertionError(
                        f"Missing SHA-256 in wheel RECORD: {library_path}"
                    )
                return entry.hash.value
    raise AssertionError(f"Library is not in {distributions}' RECORDs: {library_path}")


def assert_loaded_library(
    library: ctypes.CDLL, expected_path: Path, expected_sha256: str
) -> None:
    """Check loader identity and on-disk bytes against the wheel RECORD.

    RECORD must come from a trusted wheel installation. This is an installation
    consistency check, not attestation of memory contents or a signed build.
    """
    actual_path = get_loaded_library_path(library)
    if actual_path != expected_path.resolve(strict=True):
        raise AssertionError(
            f"Unexpected loaded library: {actual_path}; expected {expected_path}"
        )
    digest = hashlib.sha256()
    with actual_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha256 = base64.urlsafe_b64encode(digest.digest()).decode().rstrip("=")
    if actual_sha256 != expected_sha256:
        raise AssertionError(
            f"Loaded library SHA-256 differs from wheel RECORD: {actual_path}"
        )
    print(f"Loaded library: {actual_path}; SHA-256: {digest.hexdigest()}", flush=True)


def run_command(args: list[str | Path], cwd: Path | None = None, capture: bool = False):
    args = [str(arg) for arg in args]
    if cwd is None:
        cwd = Path.cwd()
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    sys.stdout.flush()
    if capture:
        return subprocess.check_output(args, cwd=str(cwd), stdin=subprocess.DEVNULL)
    else:
        subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def assert_is_physical_package(mod):
    """Asserts that the given module is a non namespace module on disk defined
    by an __init__.py file."""
    # Test helper invariant.
    assert (  # nosec B101
        mod.__file__ is not None
    ), f"The `{mod.__name__}` module does not exist as a physical directory (__file__ is None)"
    # Test helper invariant.
    assert (  # nosec B101
        Path(mod.__file__).name == "__init__.py"
    ), f"Expected `{mod.__name__}` to be a non-namespace package"


def get_module_shared_libraries(mod) -> list[Path]:
    path = Path(mod.__file__).parent
    if is_windows:
        so_paths = [p for p in path.glob("**/*.dll") if p.is_file()]
    else:
        so_paths = [p for p in path.glob("**/*.so.*") if p.is_file()] + [
            p for p in path.glob("**/*.so") if p.is_file()
        ]

    return so_paths


def find_console_script(script_name: str) -> Path | None:
    scripts_paths = [
        Path(sys.executable).parent,
        sysconfig.get_path("scripts"),
    ]
    if is_windows:
        scripts_paths.append(sysconfig.get_path("scripts", "nt_user"))
    else:
        scripts_paths.append(sysconfig.get_path("scripts", "posix_user"))
    for scripts_path in scripts_paths:
        if scripts_path is None:
            continue
        script_path = (Path(scripts_path) / script_name).with_suffix(exe_suffix)
        if script_path.is_file():
            return script_path
    return None
