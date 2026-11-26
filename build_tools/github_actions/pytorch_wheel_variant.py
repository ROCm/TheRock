#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide how a PyTorch wheel job treats a release build variant.

``asan``, ``asan-debug``, ``host-asan``, and ``host-asan-debug`` build torch
with AddressSanitizer and skip companion projects. ``release`` and an empty
value keep the normal checkouts, torchvision split, and tests. Any other
name, including ``tsan`` and ``lsan``, is rejected before the PyTorch
checkout starts.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Letters, digits, and . _ + - only, so the value can be written to GITHUB_ENV.
#   host-asan-debug -> accepted
#   asan debug      -> rejected
#   asan\nFOO=bar   -> rejected
_VARIANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")

# Names this wheel job builds. A typo or a sanitizer without a wheel build
# fails here, before the PyTorch checkout, instead of publishing as a release wheel.
_SUPPORTED_VARIANTS = frozenset(
    {
        "release",
        "asan",
        "asan-debug",
        "host-asan",
        "host-asan-debug",
    }
)


@dataclass(frozen=True)
class TorchWheelPlan:
    """What the wheel workflow should do for one build variant."""

    build_torch_only: bool
    build_variant_arg: str
    split_torchvision: bool
    torch_only_arg: str


def plan_for(variant: str) -> TorchWheelPlan:
    """Return the wheel-job plan for ``variant``.

    An empty variant is the normal release build. A value with a space or a
    newline is rejected so it cannot be written into ``GITHUB_ENV``.
    Only ``release`` and the four asan variants are accepted. ``tsan``,
    ``lsan``, and a misspelling raise ``ValueError``.
    """
    variant = variant.strip()
    if variant and _VARIANT_RE.fullmatch(variant) is None:
        raise ValueError(
            f"invalid build variant {variant!r}: "
            "use one word like host-asan-debug, with no spaces or newlines"
        )
    if not variant:
        variant = "release"
    if variant not in _SUPPORTED_VARIANTS:
        supported = ", ".join(sorted(_SUPPORTED_VARIANTS))
        raise ValueError(
            f"unsupported build variant {variant!r}: expected one of {supported}"
        )
    if "asan" in variant:
        return TorchWheelPlan(
            build_torch_only=True,
            build_variant_arg=f"--build-variant={variant}",
            split_torchvision=False,
            torch_only_arg="--torch-only",
        )
    return TorchWheelPlan(
        build_torch_only=False,
        build_variant_arg="",
        split_torchvision=True,
        torch_only_arg="",
    )


def github_env_lines(plan: TorchWheelPlan) -> list[str]:
    """Environment lines later workflow steps read."""
    torch_only = "true" if plan.build_torch_only else "false"
    split = "true" if plan.split_torchvision else "false"
    return [
        f"PYTORCH_BUILD_TORCH_ONLY={torch_only}",
        f"PYTORCH_BUILD_VARIANT_ARG={plan.build_variant_arg}",
        f"PYTORCH_SPLIT_TORCHVISION={split}",
        f"PYTORCH_TORCH_ONLY_ARG={plan.torch_only_arg}",
    ]


def write_github_env(plan: TorchWheelPlan, env_file: Path) -> None:
    """Append the plan to a GitHub Actions environment file."""
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(github_env_lines(plan)) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-variant",
        default="release",
        help="Release build variant (release, asan, asan-debug, host-asan, host-asan-debug).",
    )
    args = parser.parse_args(argv)
    try:
        plan = plan_for(args.build_variant)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    for line in github_env_lines(plan):
        print(line)
    env_file = os.environ.get("GITHUB_ENV")
    if not env_file:
        print("GITHUB_ENV is not set", file=sys.stderr)
        return 1
    write_github_env(plan, Path(env_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
