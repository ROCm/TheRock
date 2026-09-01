#!/usr/bin/env python3

"""
Checks out llama.cpp.

Automates preparing the llama.cpp repository. While all steps could be performed manually with git,
this script simplifies the process and ensures consistency.

Primary activities include:

* Cloning the llama.cpp repository into `THIS_MAIN_REPO_NAME` if it does not
  exist locally.
* Fetching the specified `--repo-hashtag` (default: `amd-integration`) from
  the remote repository.
* Checking out the exact commit fetched using `FETCH_HEAD`.
* Tagging the checked-out commit as `THEROCK_UPSTREAM_DIFFBASE` for reference.

Unlike PyTorch, llama.cpp does not currently require:

* Submodule initialization or updates.
* HIPIFY preprocessing.
* Patch application (the patch-related logic is currently not used).

This script can be used for:

* One-shot builds: simply run the script to have the correct repository state.
* Development: you can make changes to the repository locally, and the
  `THEROCK_UPSTREAM_DIFFBASE` tag serves as a known reference point for
  integration with TheRock workflows.

Primary usage:

    ./llamacpp_repo.py

Optional arguments:

* `--repo`        : Local path to clone or use the repository (default: ./llama.cpp)
* `--repo-name`   : Subdirectory name for the repo (default: llama.cpp)
* `--repo-hashtag`: Branch, tag, or commit to fetch (default: amd-integration)
* `--gitrepo-origin`: URL of the remote repository (default: https://github.com/ROCm/llama.cpp.git)
* `--depth`       : Optional shallow fetch depth
* `--jobs`        : Number of parallel fetch jobs (default: 10)

After running, the repository is ready for local development or building
with TheRock.
"""

import argparse
from pathlib import Path
import sys

import repo_management

THIS_MAIN_REPO_NAME = "llama.cpp"
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_HASHTAG = "master"
GIT_REPO_ORIGIN = "https://github.com/ggml-org/llama.cpp.git"
FETCH_JOBS = 10


def main():
    # Create a simple argparse parser for optional arguments
    p = argparse.ArgumentParser("llamacpp_repo.py")
    p.add_argument(
        "--repo",
        type=Path,
        default=THIS_DIR / THIS_MAIN_REPO_NAME,
        help="Git repository path",
    )
    p.add_argument(
        "--repo-name",
        default=THIS_MAIN_REPO_NAME,
        help="Subdirectory name in which to checkout repo",
    )
    p.add_argument(
        "--repo-hashtag",
        default=DEFAULT_REPO_HASHTAG,
        help="Git repository ref/tag to checkout",
    )
    p.add_argument(
        "--gitrepo-origin",
        default=GIT_REPO_ORIGIN,
        help="Git repository URL",
    )
    p.add_argument("--depth", type=int, default=None, help="Fetch depth")
    p.add_argument("--jobs", type=int, default=FETCH_JOBS, help="Number of fetch jobs")

    args = p.parse_args()
    repo_management.do_checkout(args)


if __name__ == "__main__":
    main()
