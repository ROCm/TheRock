#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PyTorch depends on a pinned build of Triton.

This script runs after `pytorch_torch_repo.py` and checks out the proper triton
repository based on pins in the torch repo.

On Windows, uses triton-windows (https://github.com/triton-lang/triton-windows).
The commit pin is stored in ci_commit_pins/triton-windows.txt.

This procedure is adapted from `pytorch/.github/scripts/build_triton_wheel.py`
"""
import argparse
import json
import platform
from pathlib import Path
import sys

import repo_management

THIS_MAIN_REPO_NAME = "triton"
THIS_DIR = Path(__file__).resolve().parent
COMMIT_PINS_DIR = THIS_DIR / "ci_commit_pins"

# Platform detection
IS_WINDOWS = platform.system() == "Windows"

# Repository URLs
ROCM_TRITON_ORIGIN = "https://github.com/ROCm/triton.git"
TRITON_WINDOWS_ORIGIN = "https://github.com/triton-lang/triton-windows.git"


def get_triton_pin(torch_dir: Path) -> str:
    pin_file = torch_dir / ".ci" / "docker" / "ci_commit_pins" / "triton.txt"
    return pin_file.read_text().strip()


def get_triton_version(torch_dir: Path) -> str:
    version_file = torch_dir / ".ci" / "docker" / "triton_version.txt"
    return version_file.read_text().strip()


def get_triton_windows_pin() -> str | None:
    pin_file = COMMIT_PINS_DIR / "triton-windows.txt"
    if pin_file.exists():
        return pin_file.read_text().strip()
    return None


def do_checkout(args: argparse.Namespace):
    repo_dir: Path = args.checkout_dir
    torch_dir: Path = args.torch_dir

    build_env = {}

    if IS_WINDOWS:
        print("Using triton-windows repository (Windows build)")

        if args.gitrepo_origin == ROCM_TRITON_ORIGIN:
            args.gitrepo_origin = TRITON_WINDOWS_ORIGIN

        if args.repo_hashtag is None:
            triton_windows_pin = get_triton_windows_pin()
            if triton_windows_pin:
                args.repo_hashtag = triton_windows_pin
                print(f"Triton-windows commit pin: {args.repo_hashtag}")
            else:
                args.repo_hashtag = "main-windows"
                print("No triton-windows pin found, using main-windows branch")

        args.hipify = False
    else:
        print("Using ROCm/triton repository (Linux build)")

        if not torch_dir.exists():
            raise ValueError(
                f"Could not find torch dir: {torch_dir} (did you check out torch first)"
            )

        if args.repo_hashtag is None:
            if args.release:
                # Derive the commit pin based on --release.
                pin_version = get_triton_version(torch_dir)
                pin_major, pin_minor, *_ = pin_version.split(".")
                args.repo_hashtag = f"release/{pin_major}.{pin_minor}.x"
                print(
                    f"Triton version pin: {args.triton_version} -> {args.repo_hashtag}"
                )
            else:
                # Derive the commit pin base on ci commit.
                args.repo_hashtag = get_triton_pin(torch_dir)
                # Latest triton calculates its own git hash and TRITON_WHEEL_VERSION_SUFFIX
                # goes after the "+". Older versions must supply their own "+". We just
                # leave it out entirely to avoid version errors.
                build_env["TRITON_WHEEL_VERSION_SUFFIX"] = ""
                print(f"Triton CI commit pin: {args.repo_hashtag}")

    def _do_hipify(args: argparse.Namespace):
        print("Applying local modifications...")
        with open(repo_dir / "build_env.json", "w") as f:
            json.dump(build_env, f, indent=2)

    repo_management.do_checkout(args, custom_hipify=_do_hipify)


def main(cl_args: list[str]):
    if IS_WINDOWS:
        default_origin = TRITON_WINDOWS_ORIGIN
        default_hipify = False
    else:
        default_origin = ROCM_TRITON_ORIGIN
        default_hipify = True

    def add_common(command_parser: argparse.ArgumentParser):
        command_parser.add_argument(
            "--checkout-dir",
            type=Path,
            default=THIS_DIR / THIS_MAIN_REPO_NAME,
            help=f"Directory path where the git repo is cloned into. Default is {THIS_DIR / THIS_MAIN_REPO_NAME}",
        )
        command_parser.add_argument(
            "--repo-name",
            type=Path,
            default=THIS_MAIN_REPO_NAME,
            help="Subdirectory name in which to checkout repo",
        )
        command_parser.add_argument(
            "--repo-hashtag",
            help="Git repository ref/tag to checkout",
        )

    p = argparse.ArgumentParser("pytorch_triton_repo.py")
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser("checkout", help="Clone Triton locally and checkout")
    add_common(checkout_p)
    checkout_p.add_argument(
        "--torch-dir",
        type=Path,
        default=THIS_DIR / "pytorch",
        help="Directory of the torch checkout",
    )
    checkout_p.add_argument(
        "--gitrepo-origin",
        default=default_origin,
        help=f"git repository url (default: {default_origin})",
    )
    checkout_p.add_argument(
        "--release",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Build a release Triton (vs nightly pin)",
    )
    checkout_p.add_argument("--depth", type=int, help="Fetch depth")
    checkout_p.add_argument("--jobs", type=int, help="Number of fetch jobs")
    checkout_p.add_argument(
        "--hipify",
        action=argparse.BooleanOptionalAction,
        default=default_hipify,
        help="Run hipify",
    )
    checkout_p.set_defaults(func=do_checkout)

    args = p.parse_args(cl_args)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
