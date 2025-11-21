#!/usr/bin/env python3

"""
This summarizes the environment setup steps for the
.github/workflows/test_pytorch_wheels.yml workflow.

It is intended to be run from within that workflow and writes markdown to the
GITHUB_STEP_SUMMARY file.

The script can be tested locally with inputs like this:

    python ./build_tools/github_actions/summarize_test_pytorch_workflow.py \
      --pytorch-ref=release/2.7 \
      --index-url=https://rocm.nightlies.amd.com/v2-staging \
      --index-subdir=gfx110X-dgpu \
      --torch-version=2.7.1+rocm7.10.0a20251120
"""

import argparse
import os
import platform

from github_actions_utils import *


def is_windows() -> bool:
    return platform.system() == "Windows"


LINE_CONTINUATION_CHAR = "^" if is_windows() else "\\"
LINE_CONTINUATION = f" {LINE_CONTINUATION_CHAR}\n  "


def run(args: argparse.Namespace):
    summary = ""

    summary += "## PyTorch test environment reproduction instructions\n\n"

    if not is_windows():
        summary += "(Optional) Run under Docker\n"
        summary += "```bash\n"
        summary += "sudo docker run -it" + LINE_CONTINUATION
        summary += "--device=/dev/kfd --device=/dev/dri" + LINE_CONTINUATION
        summary += (
            "--ipc=host --group-add=video --group-add=render --group-add=110"
            + LINE_CONTINUATION
        )
        summary += "ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest\n"
        summary += "```\n\n"
        summary += "```bash\n"
        summary += "# Install extra packages\n"
        summary += "sudo apt install python3.12-venv -y\n"
        summary += "```\n\n"

    pytorch_remote_name = "upstream" if args.pytorch_git_ref == "nightly" else "rocm"
    pytorch_repo_org = "pytorch" if args.pytorch_git_ref == "nightly" else "ROCm"
    summary += "Fetch pytorch source files, including tests\n\n"
    summary += "* (A) Clone pytorch if starting fresh\n\n"
    summary += "    ```bash\n"
    summary += f"    git clone --branch {args.pytorch_git_ref} --origin {pytorch_remote_name} https://github.com/{pytorch_repo_org}/pytorch.git\n"
    summary += "    ```\n\n"
    summary += "* (B) Switch to pytorch ref using an existing repository\n\n"
    summary += "    ```bash\n"
    summary += "    cd pytorch\n"
    summary += f"    git remote add {pytorch_remote_name} https://github.com/{pytorch_repo_org}/pytorch.git\n"
    summary += f"    git fetch {pytorch_remote_name} {args.pytorch_git_ref} && "
    summary += f"git checkout {pytorch_remote_name}/{args.pytorch_git_ref}\n"
    summary += "    ```\n\n"

    summary += "Install torch and test requirements into a venv\n\n"
    summary += "```bash\n"
    if is_windows():
        summary += "python -m venv .venv && .venv\Scripts\Activate.bat\n"
    else:
        summary += "python3 -m venv .venv && source .venv/bin/activate\n"
    summary += "pip install" + LINE_CONTINUATION
    summary += f"--index-url={args.index_url}/{args.index_subdir}" + LINE_CONTINUATION
    summary += "torch"
    summary += f"=={args.torch_version}" if args.torch_version else ""
    summary += "\n"
    summary += "pip install -r pytorch/.ci/docker/requirements-ci.txt\n"
    summary += "```\n\n"

    summary += "## PyTorch testing instructions\n\n"
    summary += "See [Running/testing PyTorch](https://github.com/ROCm/TheRock/tree/main/external-builds/pytorch#runningtesting-pytorch). "
    summary += "For example:\n\n"
    summary += "```bash\n"
    summary += "PYTORCH_TEST_WITH_ROCM=1 python pytorch/test/run_test.py --include test_torch\n"
    summary += "```\n"

    gha_append_step_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize test pytorch")
    parser.add_argument(
        "--torch-version",
        type=str,
        help="torch package version to install (e.g. '2.7.1+rocm7.10.0a20251120'), or empty for latest",
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
    # TODO: default the index subdir based on the current GPU somehow?
    #       (share that logic with setup_venv.py if so)
    parser.add_argument(
        "--index-subdir",
        type=str,
        required=True,
        help="Index subdirectory (e.g. gfx110X-dgpu)",
    )
    args = parser.parse_args()

    run(args)
