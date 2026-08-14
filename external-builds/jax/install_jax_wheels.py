#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installs a JAX ROCm stack: the plugin and PJRT wheels plus a matching jax.

The plugin and PJRT wheels come from wherever the build that made them put
them, since their names carry the ROCm major version they were built against
(jax_rocm7_plugin, jax_rocm10_plugin). A release publishes them to a package
index, so --index-url; PR CI uploads a run-scoped page of wheels instead, so
--find-links, which leaves PyPI in place for everything else. A build that
produced its own jaxlib installs that from the same place; otherwise jax and
jaxlib come from PyPI.

Installs are retried via build_tools/setup_venv.py.

Example usage:

    python install_jax_wheels.py \\
        --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \\
        --plugin-package jax_rocm10_plugin --pjrt-package jax_rocm10_pjrt \\
        --plugin-version 0.11.0 --pjrt-version 0.11.0 --jax-version 0.11.0
"""

import argparse
import os
import pathlib
import sys

THIS_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

sys.path.insert(0, os.fspath(THEROCK_DIR / "build_tools"))

from setup_venv import run_command_with_retries

RETRY_TIMEOUT_SECONDS = 180
RETRY_WAIT_BETWEEN_SECONDS = 15


def wheel_source(args: argparse.Namespace) -> list[str]:
    """Where pip should look for the wheels this build made.

    A find-links page adds to PyPI while an index replaces it, so a caller that
    gave both meant the page: an index that does not carry the run's wheels
    cannot install them. One of the two is required, which main() checks.
    """
    if args.find_links:
        return ["--find-links", args.find_links]
    if args.index_url:
        return ["--index-url", args.index_url]
    return []


def install_commands(args: argparse.Namespace) -> list[list[str]]:
    """The pip commands that install one JAX stack, in order."""
    python = sys.executable
    index = wheel_source(args)

    commands = [
        [
            python,
            "-m",
            "pip",
            "install",
            *index,
            f"{args.plugin_package}=={args.plugin_version}",
            f"{args.pjrt_package}=={args.pjrt_version}",
        ]
    ]

    if args.jaxlib_version:
        # Built alongside the plugin, so it only exists where the plugin does.
        commands.append(
            [python, "-m", "pip", "install", *index, f"jaxlib=={args.jaxlib_version}"]
        )
        commands.append([python, "-m", "pip", "install", f"jax=={args.jax_version}"])
    else:
        commands.append(
            [
                python,
                "-m",
                "pip",
                "install",
                f"jax=={args.jax_version}",
                f"jaxlib=={args.jax_version}",
            ]
        )

    return commands


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="install_jax_wheels.py")
    p.add_argument(
        "--index-url",
        default=os.getenv("WHEEL_INDEX_URL", ""),
        help="Package index holding the plugin, PJRT and built jaxlib wheels",
    )
    p.add_argument(
        "--find-links",
        default=os.getenv("WHEEL_FIND_LINKS_URL", ""),
        help="Page of wheels to install from, on top of PyPI; wins over --index-url",
    )
    p.add_argument(
        "--plugin-package",
        default=os.getenv("JAX_PLUGIN_PACKAGE", ""),
        help="Plugin package name (e.g. jax_rocm10_plugin)",
    )
    p.add_argument(
        "--pjrt-package",
        default=os.getenv("JAX_PJRT_PACKAGE", ""),
        help="PJRT package name (e.g. jax_rocm10_pjrt)",
    )
    p.add_argument(
        "--plugin-version",
        default=os.getenv("JAX_PLUGIN_VERSION", ""),
        help="Plugin version to install",
    )
    p.add_argument(
        "--pjrt-version",
        default=os.getenv("JAX_PJRT_VERSION", ""),
        help="PJRT version to install",
    )
    p.add_argument(
        "--jax-version",
        default=os.getenv("JAX_VERSION", ""),
        help="jax version to install",
    )
    p.add_argument(
        "--jaxlib-version",
        default=os.getenv("JAXLIB_VERSION", ""),
        help="jaxlib version to install alongside the plugin, if one was built",
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the commands without running them",
    )
    args = p.parse_args(argv)

    required = {
        "--plugin-package": args.plugin_package,
        "--pjrt-package": args.pjrt_package,
        "--plugin-version": args.plugin_version,
        "--pjrt-version": args.pjrt_version,
        "--jax-version": args.jax_version,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        p.error(f"missing required argument(s): {', '.join(missing)}")

    if not args.find_links and not args.index_url:
        # jax_rocm<major>_plugin is a published name, so leaving pip on PyPI
        # would install a release in place of the wheels this run built.
        p.error(
            "missing required argument: --find-links or --index-url"
            " (https://pypi.org/simple to install published wheels)"
        )

    for command in install_commands(args):
        if args.dry_run:
            print(f"++ Would exec $ {' '.join(command)}")
            continue
        run_command_with_retries(
            command,
            retry_timeout_seconds=RETRY_TIMEOUT_SECONDS,
            retry_wait_between_seconds=RETRY_WAIT_BETWEEN_SECONDS,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
