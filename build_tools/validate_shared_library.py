#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validates that a shared library can be loaded.

Optional Linux ELF gates (--forbid-needed, --forbid-dynsym) fail the process
if DT_NEEDED sonames or defined dynamic symbols match a regex. Used by
therock_test_validate_shared_lib for libamd_comgr.so so a dylib LLVM client
cannot ship.
"""

import argparse
import ctypes
import re
import subprocess
import sys
from pathlib import Path


def parse_needed_libraries(readelf_dynamic_output: str) -> list[str]:
    """Return DT_NEEDED sonames from `readelf -d` text.

    Matches lines of the form:
      0x0000000000000001 (NEEDED) Shared library: [libLLVM.so.23.0git]
    """
    needed: list[str] = []
    for line in readelf_dynamic_output.splitlines():
        if "(NEEDED)" not in line:
            continue
        start = line.find("[")
        end = line.rfind("]")
        if start == -1 or end <= start:
            continue
        needed.append(line[start + 1 : end])
    return needed


def parse_defined_dynsyms(nm_output: str) -> list[str]:
    """Return defined dynamic symbol names from `nm -D --defined-only` text.

    Note that this also returns GNU version *node* entries, which nm prints
    as absolute symbols (for example `LLVM_23.0`, `GLIBCXX_3.4`). They are
    harmless for an anchored pattern like `^LLVM[A-Z]`, but a broader
    pattern such as `^LLVM` would match them and false-positive on a
    correctly hidden static build.
    """
    names: list[str] = []
    for line in nm_output.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        names.append(parts[-1])
    return names


def matching_names(names: list[str], pattern: str) -> list[str]:
    """Return names that search(pattern) matches. Empty pattern matches nothing."""
    if not pattern:
        return []
    compiled = re.compile(pattern)
    return [name for name in names if compiled.search(name)]


def _run_tool(tool: str, tool_args: list[str], so_path: Path) -> str:
    try:
        completed = subprocess.run(
            [tool, *tool_args, str(so_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(f"error: required tool {tool!r} is not on PATH", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"error: {tool} failed on {so_path} with exit {e.returncode}: "
            f"{e.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return completed.stdout


def run(args: argparse.Namespace) -> None:
    linux_gate = bool(args.forbid_needed or args.forbid_dynsym)
    if linux_gate and sys.platform != "linux":
        print(
            "error: --forbid-needed and --forbid-dynsym are Linux-only",
            file=sys.stderr,
        )
        sys.exit(2)

    for shared_lib in args.shared_libs:
        print(f"Validating shared library: {shared_lib}", end="")
        so = ctypes.cdll.LoadLibrary(shared_lib)
        print(" :", so)
        so_path = Path(shared_lib)
        if args.forbid_needed:
            output = _run_tool("readelf", ["-d"], so_path)
            hits = matching_names(parse_needed_libraries(output), args.forbid_needed)
            if hits:
                print(
                    f"error: {shared_lib} DT_NEEDED matches "
                    f"{args.forbid_needed!r}: {hits}",
                    file=sys.stderr,
                )
                sys.exit(1)
        if args.forbid_dynsym:
            output = _run_tool("nm", ["-D", "--defined-only"], so_path)
            hits = matching_names(parse_defined_dynsyms(output), args.forbid_dynsym)
            if hits:
                print(
                    f"error: {shared_lib} defined dynsym matches "
                    f"{args.forbid_dynsym!r}: {hits}",
                    file=sys.stderr,
                )
                sys.exit(1)


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Load shared libraries and optionally gate ELF DT_NEEDED / dynsym."
        )
    )
    p.add_argument(
        "shared_libs",
        nargs="*",
        help="Shared libraries to validate",
    )
    p.add_argument(
        "--forbid-needed",
        metavar="REGEX",
        default="",
        help=(
            "Fail if any DT_NEEDED soname matches REGEX. Linux only. "
            "Uses readelf -d. A missing or failing readelf is a hard "
            "failure, not a skip."
        ),
    )
    p.add_argument(
        "--forbid-dynsym",
        metavar="REGEX",
        default="",
        help=(
            "Fail if any defined dynamic symbol matches REGEX. Linux only. "
            "Uses nm -D --defined-only. A missing or failing nm is a hard "
            "failure, not a skip. Green on a dylib client that imports LLVM "
            "C API but does not export it."
        ),
    )
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
