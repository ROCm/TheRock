#!/usr/bin/env python3
"""Run a command pinned to N randomly chosen logical CPUs (Windows linker launcher).

Works around ROCm/TheRock#9469: on AMD Turin (Zen 5) hosts, lld-link occasionally
drops an input object file without any error when its threads run on more than one
core. Pinning the whole link step to a single CPU avoids it. Child processes inherit
the affinity mask, so wrapping `clang++ ... -o foo.dll` also pins the lld-link it
spawns. The CPU is chosen at random so concurrent links spread across the machine.

Used as CMAKE_<LANG>_LINKER_LAUNCHER when THEROCK_WINDOWS_LINK_CPU_AFFINITY=ON.
Invoke with `python -S -E` to keep interpreter startup cost minimal.

On non-Windows hosts the command is run unchanged.

Usage:
  python -S -E affinity_launch.py [--cpus N] <program> [args...]
"""

import argparse
import os
import random
import subprocess
import sys


def pin_current_process(ncpu: int) -> int:
    """Pin this process to `ncpu` random CPUs of its processor group; returns the mask."""
    import ctypes
    import ctypes.wintypes as wt

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wt.HANDLE
    k32.GetProcessAffinityMask.argtypes = [
        wt.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    k32.SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
    proc = k32.GetCurrentProcess()
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(
        proc, ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        return 0
    cpus = [b for b in range(64) if process_mask.value >> b & 1]
    if not cpus:
        return 0
    start = random.randrange(len(cpus))
    mask = 0
    for i in range(min(ncpu, len(cpus))):
        mask |= 1 << cpus[(start + i) % len(cpus)]
    return mask if k32.SetProcessAffinityMask(proc, mask) else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command pinned to N random logical CPUs."
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=int(os.environ.get("THEROCK_LINK_AFFINITY_CPUS", "1")),
        help="Number of logical CPUs to allow (default: 1)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")
    args = parser.parse_args(argv)
    if not args.command or args.cpus < 1:
        parser.error("expected a command and --cpus >= 1")
    if sys.platform == "win32":
        pin_current_process(args.cpus)
    return subprocess.call(args.command)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
