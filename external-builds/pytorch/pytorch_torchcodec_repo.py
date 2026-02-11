#!/usr/bin/env python
"""Checks out PyTorch torchcodec.

There is nothing that this script does which you couldn't do by hand, but because of
the following, getting PyTorch sources ready to build with ToT TheRock built SDKs
consists of multiple steps:

* Sources must be pre-processed with HIPIFY, creating dirty git trees that are hard
  to develop on further.
* Both the ROCM SDK and PyTorch are moving targets that are eventually consistent.

Primary usage:

    ./pytorch_torchcodec_repo.py checkout

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

THIS_MAIN_REPO_NAME = "pytorch_torchcodec"
THIS_DIR = Path(__file__).resolve().parent

DEFAULT_ORIGIN = "https://github.com/meta-pytorch/torchcodec.git"
DEFAULT_HASHTAG = "nightly"


def get_pytorch_version(torch_dir: Path) -> str:
    version_file = torch_dir / "version.txt"
    return version_file.read_text().strip()


def do_checkout(args: argparse.Namespace):
    print("do_checkout torchcodec")
    repo_dir: Path = args.checkout_dir
    torch_dir: Path = args.torch_dir
    if not torch_dir.exists():
        raise ValueError(
            f"do_checkout torchcodec, Could not find torch dir: {torch_dir} (did you check out torch first)"
        )
    if args.repo_hashtag is None:
        pin_version = get_pytorch_version(torch_dir)
        pin_major, pin_minor, *_ = pin_version.split(".")
        pin_major = int(pin_major)
        pin_minor = int(pin_minor)
        print(f"pytorch <major>.<minor>: {pin_major}.{pin_minor}")
        if pin_major == 2:
            if pin_minor == 7:
                args.repo_hashtag = "v0.5.0"
            elif pin_minor == 8:
                args.repo_hashtag = "v0.7.0"
            elif pin_minor == 9:
                args.repo_hashtag = "v0.9.1"
            elif pin_minor == 10:
                args.repo_hashtag = "v0.10.0"
            elif pin_minor == 11:
                args.repo_hashtag = "nightly"
            else:
                raise ValueError(
                    f"do_checkout torchcodec, Unsupported torch minor version: {pin_minor}, torch version: {pin_major}.{pin_minor} (did you check out torch first)"
                )
        else:
            raise ValueError(
                f"do_checkout torchcodec, Unsupported torch major version: {pin_major}, torch version: {pin_major}.{pin_minor} (did you check out torch first)"
            )
    print(f"torchcodec version: {args.repo_hashtag}")
    repo_management.do_checkout(args)


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

    p = argparse.ArgumentParser("pytorch_audio_repo.py")
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser(
        "checkout", help="Clone PyTorch Audio locally and checkout"
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
    checkout_p.set_defaults(func=do_checkout)

    hipify_p = sub_p.add_parser("hipify", help="Run HIPIFY on the project")
    add_common(hipify_p)
    hipify_p.set_defaults(func=repo_management.do_hipify)

    args = p.parse_args(cl_args)

    # torchcodec has not pin-mapping in file pytorch/.github/ci_commit_pins
    default_git_origin = DEFAULT_ORIGIN

    # Priority order:
    #   1. Explicitly set values
    #   2. Values loaded from the pin in the torch repo
    #   3. Fallback default values
    args.gitrepo_origin = args.gitrepo_origin or default_git_origin
    args.repo_hashtag = args.repo_hashtag

    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
