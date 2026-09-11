#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Checks out PyTorch.

There is nothing that this script does which you couldn't do by hand, but because of
the following, getting PyTorch sources ready to build with ToT TheRock built SDKs
consists of multiple steps:

* Sources must be pre-processed with HIPIFY, creating dirty git trees that are hard
  to develop on further.
* Both the ROCM SDK and PyTorch are moving targets that are eventually consistent.

Primary usage:

    ./pytorch_torch_repo.py checkout

The checkout process combines the following activities:

* Clones the pytorch repository into `THIS_MAIN_REPO_NAME` with a requested `--repo-hashtag`
  tag (default to latest release).
* Configures PyTorch submodules to be ignored for any local changes.
* Runs `hipify` to prepare sources for AMD GPU and commits the result to the
  main repo and any modified submodules.
* Records tag information for tracking upstream and hipify commits.
"""
import argparse
from pathlib import Path
import sys
import subprocess

import repo_management

THIS_MAIN_REPO_NAME = "pytorch"
THIS_DIR = Path(__file__).resolve().parent

DEFAULT_ORIGIN = "https://github.com/pytorch/pytorch.git"
DEFAULT_HASHTAG = "nightly"


def apply_strict_target_patches(checkout_dir: Path, targets: str):
    if "gfx1250-strict" not in targets.replace(",", ";").split(";"):
        return
    # PyTorch's private CK source excludes gfx1250. Its compiler variant must
    # share that exclusion, while global PyTorch compilation keeps both targets.
    patch_dir = THIS_DIR / "patches" / "pytorch"
    patches = [
        patch_dir / "gfx1250-strict-ck.patch",
        patch_dir / "gfx1250-strict-ck-2.12.patch",
    ]
    for patch in patches:
        command = ["git", "apply", "--check", str(patch)]
        if (
            subprocess.run(
                command + ["--reverse"],
                cwd=checkout_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        ):
            return
        if (
            subprocess.run(
                command,
                cwd=checkout_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        ):
            subprocess.run(["git", "apply", str(patch)], cwd=checkout_dir, check=True)
            print(f"Applied strict target CK exclusion: {patch.name}")
            return
    raise RuntimeError(
        "PyTorch source does not match the gfx1250-strict CK exclusion patches"
    )


def main(cl_args: list[str]):
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
            default=DEFAULT_HASHTAG,
            help="Git repository ref/tag to checkout",
        )

    p = argparse.ArgumentParser("pytorch_torch_repo.py")
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser("checkout", help="Clone PyTorch locally and checkout")
    add_common(checkout_p)
    checkout_p.add_argument(
        "--gitrepo-origin",
        default=DEFAULT_ORIGIN,
        help="git repository url",
    )
    repo_management.add_checkout_options(checkout_p, default_hipify=True)
    checkout_p.set_defaults(jobs=10)
    checkout_p.set_defaults(func=repo_management.do_checkout)

    hipify_p = sub_p.add_parser("hipify", help="Run HIPIFY on the project")
    add_common(hipify_p)
    hipify_p.set_defaults(func=repo_management.do_hipify)

    args = p.parse_args(cl_args)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
