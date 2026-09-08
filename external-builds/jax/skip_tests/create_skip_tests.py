#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the pytest -k expression that skips known-bad JAX tests.

The entries live in data files next to this one, one per JAX version plus
generic.py, so that CI and a local run skip the same tests. See README.md for
the file layout and the entry format.

Example usage:

    python create_skip_tests.py --jax-version 0.10.2 --amdgpu-family gfx94X-dcgpu
"""

import argparse
import importlib.util
import pathlib
import sys
from typing import Any

THIS_DIR = pathlib.Path(__file__).resolve().parent

GENERIC_FILE = "generic.py"

# Applies regardless of which GPU family the tests run on.
COMMON_SECTION = "common"


def _load_data_file(path: pathlib.Path) -> dict[str, Any]:
    """Reads the skip_tests dict out of one data file."""
    spec = importlib.util.spec_from_file_location(f"skip_tests.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "skip_tests", {})


def data_files(jax_version: str) -> list[pathlib.Path]:
    """The data files that apply to jax_version, generic ones first.

    An unknown version contributes only the generic file, so adding a JAX version
    to CI is a no-op here.
    """
    files = [THIS_DIR / GENERIC_FILE]
    if jax_version == "all":
        files += sorted(THIS_DIR.glob("jax_*.py"))
    elif jax_version:
        version_file = THIS_DIR / f"jax_{jax_version}.py"
        if version_file.exists():
            files.append(version_file)
    return files


def _section_applies(section: str, amdgpu_family: str) -> bool:
    if section == COMMON_SECTION:
        return True
    # Substring match, so "gfx94" covers gfx94X-dcgpu and gfx942.
    return bool(amdgpu_family) and section in amdgpu_family


def keyword_filters(jax_version: str, amdgpu_family: str) -> list[dict[str, Any]]:
    """The keyword entries that apply to one configuration."""
    filters: list[dict[str, Any]] = []
    for path in data_files(jax_version):
        for section, entries in _load_data_file(path).items():
            if _section_applies(section, amdgpu_family):
                filters += entries.get("keywords", [])
    return filters


def keyword_expression(
    jax_version: str, amdgpu_family: str, debug: bool = False
) -> str:
    """The pytest -k expression for one configuration, empty when nothing applies.

    With debug set, it selects only the skipped tests instead, to reproduce a
    failure without editing the data files.
    """
    filters = keyword_filters(jax_version, amdgpu_family)
    if not filters:
        return ""

    if debug:
        return " or ".join(entry["deny"] for entry in filters)

    terms = []
    for entry in filters:
        deny = entry["deny"]
        # -k matches substrings, so denying "conv" also catches "convert".
        unless = entry.get("unless", [])
        if unless:
            kept = " or ".join(unless)
            terms.append(f"((not {deny}) or {kept})")
        else:
            terms.append(f"not {deny}")
    return " and ".join(terms)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="create_skip_tests.py")
    p.add_argument(
        "--jax-version",
        default="",
        help="JAX version being tested (e.g. 0.10.2), or 'all' for every file",
    )
    p.add_argument(
        "--amdgpu-family",
        default="",
        help="GPU family being tested (e.g. gfx94X-dcgpu)",
    )
    p.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print an expression selecting only the skipped tests",
    )
    args = p.parse_args(argv)

    print(keyword_expression(args.jax_version, args.amdgpu_family, args.debug))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
