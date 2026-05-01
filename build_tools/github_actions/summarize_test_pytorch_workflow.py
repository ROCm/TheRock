#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
This summarizes the environment setup steps for the
.github/workflows/test_pytorch_wheels.yml workflow.

It is intended to be run from within that workflow and writes markdown to the
GITHUB_STEP_SUMMARY file.

The script can be tested locally with inputs like this:

    # Per-family mode:
    python ./build_tools/github_actions/summarize_test_pytorch_workflow.py \
      --pytorch-git-ref=release/2.7 \
      --index-url=https://rocm.nightlies.amd.com/v2-staging \
      --index-subdir=gfx110X-dgpu \
      --torch-version=2.7.1+rocm7.10.0a20251120

    # Multi-arch mode:
    python ./build_tools/github_actions/summarize_test_pytorch_workflow.py \
      --pytorch-git-ref=release/2.10 \
      --index-url=https://rocm.nightlies.amd.com/whl-staging-multi-arch/ \
      --device-extras=device-gfx942 \
      --torch-version=2.10.0+rocm7.12.0a20260501
"""

import argparse
import platform
import sys

from github_actions_api import *


def is_windows() -> bool:
    return platform.system() == "Windows"


LINE_CONTINUATION_CHAR = "^" if is_windows() else "\\"
LINE_CONTINUATION = f" {LINE_CONTINUATION_CHAR}\n  "


def run(args: argparse.Namespace):
    pytorch_repo_org = "pytorch" if args.pytorch_git_ref == "nightly" else "ROCm"
    pytorch_origin_args = "" if args.pytorch_git_ref == "nightly" else "--origin rocm"
    pytorch_remote_url = f"https://github.com/{pytorch_repo_org}/pytorch.git"
    pytorch_web_url = f"https://github.com/{pytorch_repo_org}/pytorch"
    pytorch_web_url_with_branch = f"{pytorch_web_url}/tree/{args.pytorch_git_ref}"

    # Determine install mode based on provided arguments.
    if args.device_extras:
        # Multi-arch mode: single index URL, device selected via pip extras.
        index_url = args.index_url.rstrip("/") + "/"
        package_spec = f'"torch[{args.device_extras}]'
        package_spec += f"=={args.torch_version}" if args.torch_version else ""
        package_spec += '"'
        gpu_label = args.device_extras
    else:
        # Per-family mode: index URL includes the family subdir.
        index_url = f"{args.index_url}/{args.index_subdir}/"
        package_spec = "torch"
        package_spec += f"=={args.torch_version}" if args.torch_version else ""
        gpu_label = args.index_subdir

    # This report should be as brief as possible while still conveying what
    # is unique to the given arguments.

    summary = ""
    summary += "## PyTorch Test Report\n\n"

    # Summary information.
    summary += f"* Torch version: `{args.torch_version}`\n"
    summary += f"* Python version: `{args.python_version}`\n"
    summary += f"* GPU target: `{gpu_label}`\n"
    summary += f"* Package index: {index_url}\n"
    summary += f"* PyTorch source code: {pytorch_web_url_with_branch}\n"

    # Link to detailed documentation.
    summary += "\n"
    summary += "To reproduce, see [Running/testing PyTorch](https://github.com/ROCm/TheRock/tree/main/external-builds/pytorch#runningtesting-pytorch) and setup with:\n"

    # Simple to copy/paste instructions to get the code and packages.
    summary += "\n"
    summary += "```bash\n"
    summary += "# Fetch pytorch source files, including tests:\n"
    summary += f"git clone --branch {args.pytorch_git_ref} {pytorch_origin_args} {pytorch_remote_url}\n"
    summary += "\n"
    summary += "# Install torch and test requirements\n"
    summary += "pip install" + LINE_CONTINUATION
    summary += f"--index-url={index_url}" + LINE_CONTINUATION
    summary += package_spec
    summary += "\n"
    summary += "pip install -r pytorch/.ci/docker/requirements-ci.txt\n"
    summary += "```\n\n"

    gha_append_step_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize test pytorch")
    parser.add_argument(
        "--torch-version",
        type=str,
        help="torch package version to install (e.g. '2.7.1+rocm7.10.0a20251120'), or empty for latest",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        default=f"{sys.version_info[0]}.{sys.version_info[1]}",
        help="Python version to used for tests (defaults to sys.version as X.Y)",
    )
    parser.add_argument(
        "--pytorch-git-ref",
        type=str,
        default="nightly",
        help="PyTorch ref to checkout test sources from",
    )
    parser.add_argument(
        "--index-url",
        type=str,
        default="https://rocm.nightlies.amd.com/v2-staging",
        help="Full URL for a release index to use with 'pip install --index-url='",
    )
    # Per-family mode: --index-subdir selects the GPU family subdirectory.
    parser.add_argument(
        "--index-subdir",
        type=str,
        help="Index subdirectory (e.g. gfx110X-dgpu). Used for per-family installs.",
    )
    # Multi-arch mode: --device-extras selects GPU-specific device packages.
    parser.add_argument(
        "--device-extras",
        type=str,
        help="Comma-separated device extras (e.g. 'device-gfx942'). Used for multi-arch installs.",
    )
    args = parser.parse_args()

    if not args.index_subdir and not args.device_extras:
        parser.error("one of --index-subdir or --device-extras is required")

    run(args)
