#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide how a PyTorch wheel job treats a release build variant.

``asan``, ``asan-debug``, ``host-asan``, and ``host-asan-debug`` build torch
with AddressSanitizer and skip companion projects. ``release`` keeps the
normal checkouts, torchvision split, and tests. TSan and LSan are unchanged
until they have a wheel build.

The rule is the same one ``setup_pytorch_sanitizer.selected_sanitizer`` uses:
a variant selects AddressSanitizer when it contains ``asan``.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass

# Letters, digits, and . _ + - only, so the value can be written to GITHUB_ENV.
#   host-asan-debug -> accepted
#   asan debug      -> rejected
#   asan\nFOO=bar   -> rejected
_VARIANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


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
    ``host-asan-debug`` is accepted. ``asan debug`` and ``asan\\nFOO=bar``
    are not.
    """
    variant = variant.strip()
    if variant and _VARIANT_RE.fullmatch(variant) is None:
        raise ValueError(
            f"invalid build variant {variant!r}: "
            "use one word like host-asan-debug, with no spaces or newlines"
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


def write_github_env(plan: TorchWheelPlan, env_file: str) -> None:
    """Append the plan to a GitHub Actions environment file."""
    with open(env_file, "a", encoding="utf-8") as handle:
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
    write_github_env(plan, env_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
