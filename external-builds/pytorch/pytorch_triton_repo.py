#!/usr/bin/env python
"""PyTorch depends on a pinned build of Triton.

This script runs after `pytorch_torch_repo.py` and checks out the proper triton
repository based on pins in the torch repo.

This procedure is adapted from `pytorch/.github/scripts/build_triton_wheel.py`
"""
import argparse
import json
from pathlib import Path
import sys

import repo_management

THIS_MAIN_REPO_NAME = "triton"
THIS_DIR = Path(__file__).resolve().parent
THIS_PATCHES_DIR = THIS_DIR / "patches" / THIS_MAIN_REPO_NAME


def get_triton_pin(torch_dir: Path) -> str:
    pin_file = torch_dir / ".ci" / "docker" / "ci_commit_pins" / "triton.txt"
    return pin_file.read_text().strip()


def get_triton_version(torch_dir: Path) -> str:
    version_file = torch_dir / ".ci" / "docker" / "triton_version.txt"
    return version_file.read_text().strip()


def do_checkout(args: argparse.Namespace):
    repo_dir: Path = args.repo
    torch_dir: Path = args.torch_dir
    if not torch_dir.exists():
        raise ValueError(
            f"Could not find torch dir: {torch_dir} (did you check out torch first)"
        )

    build_env = {"TRITON_WHEEL_NAME": "pytorch-triton-rocm"}
    if args.repo_hashtag is None:
        if args.release:
            # Derive the commit pin based on --release.
            pin_version = get_triton_version(torch_dir)
            pin_major, pin_minor, *_ = pin_version.split(".")
            args.repo_hashtag = f"release/{pin_major}.{pin_minor}.x"
            print(f"Triton version pin: {args.triton_version} -> {args.repo_hashtag}")
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
    def add_common(command_parser: argparse.ArgumentParser):
        command_parser.add_argument(
            "--repo",
            type=Path,
            default=THIS_DIR / THIS_MAIN_REPO_NAME,
            help="Git repository path",
        )
        command_parser.add_argument(
            "--patch-dir",
            type=Path,
            default=THIS_PATCHES_DIR,
            help="Git repository patch path",
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
        command_parser.add_argument(
            "--patchset",
            help="patch dir subdirectory (defaults to mangled --repo-hashtag)",
        )

    p = argparse.ArgumentParser("pytorch_triton_repo.py")
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser("checkout", help="Clone Triton locally and checkout")
    add_common(checkout_p)
    checkout_p.add_argument(
        "--torch-dir",
        default=THIS_DIR / "pytorch",
        help="Directory of the torch checkout",
    )
    checkout_p.add_argument(
        "--gitrepo-origin",
        default="https://github.com/ROCm/triton.git",
        help="git repository url",
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
        default=True,
        help="Run hipify",
    )
    checkout_p.add_argument(
        "--patch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply patches for the repo-hashtag",
    )
    checkout_p.set_defaults(func=do_checkout)

    save_patches_p = sub_p.add_parser(
        "save-patches", help="Save local commits as patch files for later application"
    )
    add_common(save_patches_p)
    save_patches_p.set_defaults(func=repo_management.do_save_patches)

    args = p.parse_args(cl_args)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
