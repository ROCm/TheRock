#!/usr/bin/env python
"""Applies TheRock-specific overrides to PyTorch's `requirements-ci.txt`.

We install PyTorch's own CI requirements file to test the wheels we build, but
we test interpreter versions that some of its pins predate. Where a pin has no
wheel for such an interpreter, pip falls back to building from source, which
drags in an unpinned build toolchain that can break without notice.

Rather than fork the file, this rewrites the affected requirement lines in the
checked out tree just before `pip install` runs. Overrides are keyed by
distribution name in `OVERRIDES` below; each one replaces every requirement
line for that distribution with the given block.

Usage:
    python patch_test_requirements.py path/to/requirements-ci.txt
"""

import argparse
import re
import sys
from pathlib import Path

# Requirement lines to substitute, keyed by (normalized) distribution name.
# Keep a comment on each entry explaining why the override exists, so that it
# can be dropped once the upstream pin moves.
OVERRIDES: dict[str, list[str]] = {
    # scikit-image 0.22.0 publishes no cp313/cp314 wheel, so pip builds it from
    # source on the Python versions we test. That build hardcodes
    # `cpp_std=c++14` in its meson.build and does not pass through pythran's own
    # compile flags, so it broke when pythran 0.19.0 (2026-08-17) raised its
    # baseline to C++17 and started using `std::is_integral_v` and friends.
    # 0.26.0 ships cp313 and cp314 wheels, so no compiler runs at all.
    # See https://github.com/ROCm/TheRock/issues/7448
    "scikit-image": [
        'scikit-image==0.22.0 ; python_version < "3.13"',
        'scikit-image==0.26.0 ; python_version >= "3.13"',
    ],
}

# Leading distribution name of a requirement line, per PEP 508.
_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:[=<>!~[;]|$)")


def normalize(name: str) -> str:
    """Normalizes a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str | None:
    """Returns the distribution a requirement line pins, or None if it is not one."""
    stripped = line.split("#", 1)[0].strip()
    if not stripped:
        return None
    match = _NAME_RE.match(stripped)
    return normalize(match.group(1)) if match else None


def patch(lines: list[str]) -> tuple[list[str], list[str]]:
    """Applies OVERRIDES to `lines`, returning the result and the names applied."""
    overrides = {normalize(name): block for name, block in OVERRIDES.items()}
    patched: list[str] = []
    applied: list[str] = []
    for line in lines:
        name = requirement_name(line)
        if name is None or name not in overrides:
            patched.append(line)
            continue
        # Emit the replacement block once, at the first line we match, and drop
        # any further lines for the same distribution (upstream sometimes splits
        # a pin across several marker-gated lines).
        if name not in applied:
            patched.extend(overrides[name])
            applied.append(name)
    return patched, applied


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "requirements",
        type=Path,
        help="Path to the requirements-ci.txt to rewrite in place.",
    )
    args = parser.parse_args(argv)

    original = args.requirements.read_text().splitlines()
    patched, applied = patch(original)

    for name in OVERRIDES:
        if normalize(name) not in applied:
            # Not fatal: the upstream pin may simply have been dropped, in which
            # case there is nothing to override and the install is fine as is.
            print(
                f"warning: no requirement for '{name}' in {args.requirements}, "
                "override not applied (the pin may have moved upstream)",
                file=sys.stderr,
            )

    if patched == original:
        print(f"{args.requirements}: no changes needed")
        return 0

    args.requirements.write_text("\n".join(patched) + "\n")
    print(f"{args.requirements}: applied overrides for {', '.join(applied)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
