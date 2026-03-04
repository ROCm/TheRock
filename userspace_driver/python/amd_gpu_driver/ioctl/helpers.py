"""Low-level libc wrappers for ioctl() and mmap() via ctypes.

On Linux, this module loads libc and exposes ioctl/mmap/munmap/memcpy/memset
wrappers. On Windows, the libc-dependent parts are unavailable (libc is None),
but the pure-Python ioctl number encoding functions (_IOC, _IOR, _IOW, _IOWR)
and mmap constants are always importable.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Any

from amd_gpu_driver.errors import IoctlError

# --- libc loading (Linux only) ---

libc: ctypes.CDLL | None = None

if sys.platform != "win32":
    _libc_name = ctypes.util.find_library("c")
    if _libc_name is None:
        raise RuntimeError("Cannot find libc")
    libc = ctypes.CDLL(_libc_name, use_errno=True)

    # ioctl(int fd, unsigned long request, ...) -> int
    libc.ioctl.restype = ctypes.c_int
    libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]

    # mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) -> void*
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]

    # munmap(void *addr, size_t length) -> int
    libc.munmap.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    # memcpy(void *dest, const void *src, size_t n) -> void*
    libc.memcpy.restype = ctypes.c_void_p
    libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

    # memset(void *s, int c, size_t n) -> void*
    libc.memset.restype = ctypes.c_void_p
    libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]

# --- mmap constants (Linux values, used by ioctl encoding and KFD backend) ---

PROT_NONE = 0x0
PROT_READ = 0x1
PROT_WRITE = 0x2
PROT_EXEC = 0x4

MAP_SHARED = 0x01
MAP_PRIVATE = 0x02
MAP_FIXED = 0x10
MAP_ANONYMOUS = 0x20
MAP_NORESERVE = 0x4000

MAP_FAILED = ctypes.c_void_p(-1).value  # type: ignore[arg-type]


# --- ioctl number encoding (Linux) ---
# _IOC(dir, type, nr, size):
#   dir<<30 | size<<16 | type<<8 | nr

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

_IOC_DIRSHIFT = 30
_IOC_SIZESHIFT = 16
_IOC_TYPESHIFT = 8
_IOC_NRSHIFT = 0


def _IOC(direction: int, type_char: int, nr: int, size: int) -> int:
    return (direction << _IOC_DIRSHIFT) | (size << _IOC_SIZESHIFT) | (type_char << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT)


def _IO(type_char: int, nr: int) -> int:
    return _IOC(_IOC_NONE, type_char, nr, 0)


def _IOR(type_char: int, nr: int, size: int) -> int:
    return _IOC(_IOC_READ, type_char, nr, size)


def _IOW(type_char: int, nr: int, size: int) -> int:
    return _IOC(_IOC_WRITE, type_char, nr, size)


def _IOWR(type_char: int, nr: int, size: int) -> int:
    return _IOC(_IOC_READ | _IOC_WRITE, type_char, nr, size)


# --- Wrappers (Linux only) ---


def _require_libc() -> ctypes.CDLL:
    """Return libc, raising RuntimeError on Windows."""
    if libc is None:
        raise RuntimeError("libc wrappers are not available on Windows")
    return libc


def ioctl(fd: int, request: int, arg: Any, ioctl_name: str = "unknown") -> int:
    """Call ioctl, raising IoctlError on failure."""
    _libc = _require_libc()
    ret = _libc.ioctl(fd, request, ctypes.byref(arg) if hasattr(arg, '_fields_') else arg)
    if ret < 0:
        errno = ctypes.get_errno()
        raise IoctlError(ioctl_name, errno)
    return ret


def libc_mmap(
    addr: int | None,
    length: int,
    prot: int,
    flags: int,
    fd: int,
    offset: int,
) -> int:
    """Call mmap via libc, returning the mapped address as an int.

    Supports MAP_FIXED which Python's mmap module does not.
    """
    _libc = _require_libc()
    c_addr = ctypes.c_void_p(addr)
    result = _libc.mmap(c_addr, length, prot, flags, fd, offset)
    if result == MAP_FAILED:
        errno = ctypes.get_errno()
        raise OSError(errno, f"mmap failed: {os.strerror(errno)}")
    return result


def libc_munmap(addr: int, length: int) -> None:
    """Unmap memory."""
    _libc = _require_libc()
    ret = _libc.munmap(ctypes.c_void_p(addr), length)
    if ret < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"munmap failed: {os.strerror(errno)}")
