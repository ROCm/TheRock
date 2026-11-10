#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runs a command pinned to N randomly chosen logical CPUs (Windows linker launcher).

Works around https://github.com/ROCm/TheRock/issues/9469: on AMD Turin (Zen 5)
hosts, lld-link occasionally drops an input object file without reporting an
error, and the link later fails with `undefined symbol` (or produces a DLL with
missing exports). This only happens when lld-link's threads run on more than one
core, so pinning the whole link step to a single CPU avoids it.

The launcher pins its own process and then starts the command, so the command and
everything it spawns (e.g. the lld-link started by `clang++ ... -o foo.dll`)
inherit the affinity mask. The CPU is picked at random so that concurrent links
spread across the machine. The command's exit code is returned unchanged.

On non-Windows hosts the command is run unchanged.

Used as the link rule launcher when THEROCK_WINDOWS_LINK_CPU_AFFINITY=ON (see
cmake/therock_subproject.cmake). Invoke with `python -S -E` to keep interpreter
startup cost low.

Usage:
  python -S -E build_tools/affinity_launch.py [--cpus N] <program> [args...]
"""

import argparse
import os
import random
import subprocess
import sys


def choose_affinity_mask(available_mask: int, ncpus: int, rng: random.Random) -> int:
    """Returns a mask of `ncpus` CPUs taken from `available_mask`.

    CPUs are taken in order starting from a randomly chosen available CPU. If
    fewer CPUs are available than requested, all available CPUs are used.
    """
    cpus = [
        bit for bit in range(available_mask.bit_length()) if available_mask >> bit & 1
    ]
    if not cpus:
        return 0
    start = rng.randrange(len(cpus))
    mask = 0
    for i in range(min(ncpus, len(cpus))):
        mask |= 1 << cpus[(start + i) % len(cpus)]
    return mask


def pin_current_process(ncpus: int) -> int:
    """Pins this process to `ncpus` random CPUs of its processor group.

    Returns the mask that was applied, or 0 if it could not be set.
    """
    import ctypes
    import ctypes.wintypes as wt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    kernel32.GetProcessAffinityMask.argtypes = [
        wt.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
    process = kernel32.GetCurrentProcess()
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        process, ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        return 0
    mask = choose_affinity_mask(process_mask.value, ncpus, random.Random())
    if not mask or not kernel32.SetProcessAffinityMask(process, mask):
        return 0
    return mask


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
