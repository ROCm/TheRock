#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compiler launcher that injects a stable, per-translation-unit ``-cuid`` before
handing off to sccache.

Why this exists (ROCm/TheRock#748)
----------------------------------
With ``-fgpu-rdc`` every HIP object carries a ``__hip_cuid_<id>`` symbol. clang
derives that id (``-fuse-cuid=hash``, the default) from ``md5(input-path + all
non-input args)`` -- which *includes* ``-o``. sccache's cache key deliberately
*excludes* ``-o`` and normalizes paths, so two translation units that differ only
in their output collapse to the same key. On a cache hit sccache returns the
first TU's object for the second, so both objects carry the *same*
``__hip_cuid_`` -> duplicate-symbol error at RDC link time.

Passing an explicit ``-cuid=<perTU>`` fixes both halves of the problem:
  * sccache keys on ``-cuid`` (it is a normal arg), so each TU gets a distinct
    cache entry -- no false cross-TU sharing.
  * clang uses the value verbatim as the CUID, so each object gets a distinct
    ``__hip_cuid_``.
Deriving the value deterministically from the (relative) output path keeps warm
cache hits working: the same TU always computes the same id across machines.

This wrapper is only meaningful for HIP device compiles, so injection is gated to
those (a bare ``-cuid`` on a host C/C++ compile would trip
``-Wunused-command-line-argument`` under ``-Werror``).

Invocation
----------
Set as a single executable path in either launcher position::

    CMAKE_C_COMPILER_LAUNCHER  = <this>     # -> <this> <compiler> <args...>
    CMAKE_CXX_COMPILER_LAUNCHER = <this>
    HIP_CLANG_LAUNCHER          = <this>     # hipcc: "<this>" "<clang>" <args...>

The real sccache binary is located via ``THEROCK_SCCACHE`` (preferred), then
``SCCACHE``, then ``PATH``. If none is found the compiler is exec'd directly
(caching disabled, build still succeeds).

Optional debugging: set ``THEROCK_CUID_LOG=<file>`` to append every injected
``-cuid`` (with its source output path) for post-build inspection.
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path

_HIP_ARCH_PREFIXES = ("--offload-arch", "--cuda-gpu-arch")
_HIP_INPUT_SUFFIXES = (".hip", ".cu")


def find_sccache() -> str | None:
    """Locate the real sccache binary (env override first, then PATH)."""
    for candidate in (os.environ.get("THEROCK_SCCACHE"), os.environ.get("SCCACHE")):
        if candidate and Path(candidate).exists():
            return candidate
    return shutil.which("sccache")


def output_path(args: list[str]) -> str | None:
    """Return the value of the compile output flag (``-o`` / ``--output``)."""
    for i, arg in enumerate(args):
        if arg in ("-o", "--output") and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("-o") and len(arg) > 2:
            return arg[2:]
        if arg.startswith("--output="):
            return arg[len("--output=") :]
    return None


def is_hip_compile(args: list[str]) -> bool:
    """True only for a HIP/CUDA object compile (``-c`` plus offload markers)."""
    if "-c" not in args:
        return False
    for i, arg in enumerate(args):
        if arg.startswith(_HIP_ARCH_PREFIXES):
            return True
        if arg == "-x" and i + 1 < len(args) and args[i + 1] in ("hip", "cuda"):
            return True
        if arg.endswith(_HIP_INPUT_SUFFIXES):
            return True
    return False


def has_explicit_cuid(args: list[str]) -> bool:
    return any(arg == "-cuid" or arg.startswith("-cuid=") for arg in args)


def stable_cuid(out: str) -> str:
    """Deterministic, machine-independent id derived from the output path.

    Relativized against the CWD so cold/warm builds on different machines (but
    the same build layout) compute the same value -- preserving warm cache hits.
    """
    try:
        rel = os.path.relpath(out, os.getcwd())
    except ValueError:
        rel = out
    rel = rel.replace(os.sep, "/")
    return hashlib.md5(rel.encode("utf-8")).hexdigest()[:16]


def build_command(argv: list[str]) -> list[str]:
    compiler, args = argv[1], argv[2:]

    extra: list[str] = []
    if is_hip_compile(args) and not has_explicit_cuid(args):
        out = output_path(args)
        if out:
            cuid = stable_cuid(out)
            extra = [f"-cuid={cuid}"]
            log = os.environ.get("THEROCK_CUID_LOG")
            if log:
                try:
                    with open(log, "a") as f:
                        f.write(f"{cuid}\t{out}\n")
                except OSError:
                    pass

    sccache = find_sccache()
    prefix = [sccache] if sccache else []
    return prefix + [compiler] + args + extra


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: cuid_launcher.py <compiler> <args...>", file=sys.stderr)
        return 2
    cmd = build_command(argv)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
