#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fix HeCBench Makefiles for SPIR-V builds.

The HeCBench *-hip benchmark Makefiles have a common link rule that passes
$(CFLAGS) to the linker invocation:

    $(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

The ``-x hip`` flag inside CFLAGS tells clang to treat inputs as HIP source,
which is correct for compilation but **breaks linking** (the linker tries to
recompile .o files as HIP source).  This script removes $(CFLAGS) from link
rules so only $(LDFLAGS) controls the link step.
"""

import argparse
from pathlib import Path


def fix_makefile(makefile: Path) -> bool:
    """Remove $(CFLAGS) from link rules in *makefile*.

    Returns True if the file was modified.
    """
    original = makefile.read_text()
    fixed_lines: list[str] = []
    changed = False

    for line in original.splitlines(keepends=True):
        stripped = line.lstrip()

        if (
            "$(CC)" in stripped
            and "$(CFLAGS)" in stripped
            and "-o" in stripped
            and "$@" in stripped
            and "$(LDFLAGS)" in stripped
            and "-c" not in stripped  # don't touch compile rules
        ):
            indent = line[: len(line) - len(stripped)]
            new_line = indent + stripped.replace("$(CFLAGS) ", "")
            if new_line != line:
                fixed_lines.append(new_line)
                changed = True
                continue

        fixed_lines.append(line)

    if changed:
        makefile.write_text("".join(fixed_lines))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix HeCBench Makefiles: remove $(CFLAGS) from link rules."
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Benchmark directories containing a Makefile to fix.",
    )
    args = parser.parse_args()

    fixed = 0
    skipped = 0
    for bench_dir in args.directories:
        makefile = bench_dir / "Makefile"
        if not makefile.is_file():
            skipped += 1
            continue
        if fix_makefile(makefile):
            fixed += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
