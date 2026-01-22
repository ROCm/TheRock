#!/usr/bin/env python
"""Checks out PyTorch Vision.

There is nothing that this script does which you couldn't do by hand, but because of
the following, getting PyTorch sources ready to build with ToT TheRock built SDKs
consists of multiple steps:

* Sources must be pre-processed with HIPIFY, creating dirty git trees that are hard
  to develop on further.
* Both the ROCM SDK and PyTorch are moving targets that are eventually consistent.

Primary usage:

    ./pytorch_vision_repo.py checkout

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

import repo_management

THIS_MAIN_REPO_NAME = "pytorch_vision"
THIS_DIR = Path(__file__).resolve().parent

DEFAULT_ORIGIN = "https://github.com/pytorch/vision.git"
DEFAULT_HASHTAG = "nightly"


def main(cl_args: list[str]):
    def add_common(command_parser: argparse.ArgumentParser):
        command_parser.add_argument(
            "--checkout-dir",
            type=Path,
            default=THIS_DIR / THIS_MAIN_REPO_NAME,
            help=f"Directory path where the git repo is cloned into. Default is {THIS_DIR / THIS_MAIN_REPO_NAME}",
        )
        command_parser.add_argument(
            "--gitrepo-origin",
            type=str,
            default=None,
            help=f"Git repository url. Defaults to the origin in torch/related_commits (see --torch-dir), or '{DEFAULT_ORIGIN}'",
        )
        command_parser.add_argument(
            "--repo-name",
            type=Path,
            default=THIS_MAIN_REPO_NAME,
            help="Subdirectory name in which to checkout repo",
        )
        command_parser.add_argument(
            "--repo-hashtag",
            type=str,
            default=None,
            help=f"Git repository ref/tag to checkout. Defaults to the ref in torch/related_commits (see --torch-dir), or '{DEFAULT_HASHTAG}'",
        )
        command_parser.add_argument(
            "--require-related-commit",
            action=argparse.BooleanOptionalAction,
            help="Require that a related commit was found from --torch-dir",
        )
        command_parser.add_argument(
            "--torch-dir",
            type=Path,
            default=THIS_DIR / "pytorch",
            help="Directory of the torch checkout, for loading the related_commits file that can populate alternate default values for --gitrepo-origin and --repo-hashtag. If missing then fallback/upstream defaults will be used",
        )

    p = argparse.ArgumentParser("pytorch_vision_repo.py")
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser(
        "checkout", help="Clone PyTorch Vision locally and checkout"
    )
    add_common(checkout_p)
    checkout_p.add_argument("--depth", type=int, help="Fetch depth")
    checkout_p.add_argument("--jobs", type=int, help="Number of fetch jobs")
    checkout_p.add_argument(
        "--hipify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run hipify",
    )
    checkout_p.set_defaults(func=repo_management.do_checkout)

    hipify_p = sub_p.add_parser("hipify", help="Run HIPIFY on the project")
    add_common(hipify_p)
    hipify_p.set_defaults(func=repo_management.do_hipify)

    args = p.parse_args(cl_args)
    # Set default values based on the pin file in the pytorch repo.
    (
        default_git_origin,
        default_git_hashtag,
        has_related_commit,
    ) = repo_management.read_pytorch_rocm_pins(
        args.torch_dir,
        os="centos",  # Read pins for "centos" on Linux and Windows
        project="torchvision",
        default_origin=DEFAULT_ORIGIN,
        default_hashtag=DEFAULT_HASHTAG,
    )

    if args.require_related_commit and not has_related_commit:
        raise ValueError(
            f"Could not find torchvision in '{args.torch_dir}/related_commits' (did you mean to set a different --torch-dir?)"
        )

    # Priority order:
    #   1. Explicitly set values
    #   2. Values loaded from the pin in the torch repo
    #   3. Fallback default values
    args.gitrepo_origin = args.gitrepo_origin or default_git_origin
    args.repo_hashtag = args.repo_hashtag or default_git_hashtag

    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
