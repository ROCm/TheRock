#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Checks out PyTorch torchcodec.

Primary usage:

    ./pytorch_torchcodec_repo.py checkout

The checkout process clones https://github.com/meta-pytorch/torchcodec into
`THIS_MAIN_REPO_NAME` with the selected ref.

Unlike torchaudio and torchvision, ROCm/pytorch's `related_commits` file and
pytorch's `.github/ci_commit_pins` have no torchcodec entry, so there is no pin
to read. The ref is selected here instead:

* `--repo-hashtag <ref>` always wins. Nightly builds pass `--repo-hashtag
  nightly`, which is the branch upstream updates daily to track torch nightly.
* Otherwise the ref comes from the torch version, read from `--torch-version`
  or from `version.txt` in the torch checkout (`--torch-dir`):
  `TORCH_VERSION_TO_TAG` has an optional override per torch `MAJOR.MINOR`, and
  every other torch version at or above `MIN_TORCH_FOR_DEFAULT_STABLE_TAG` uses
  `DEFAULT_STABLE_TAG`. Torch versions below that need an explicit override and
  otherwise fail with an error.
* If no torch version is available at all, `DEFAULT_STABLE_TAG` is used.

Upstream torchcodec has no ROCm/HIP support, so the build is CPU-only and no
HIPIFY step is run by default.
"""
import argparse
from pathlib import Path
import re
import sys

import repo_management

THIS_MAIN_REPO_NAME = "pytorch_torchcodec"
THIS_DIR = Path(__file__).resolve().parent

DEFAULT_ORIGIN = "https://github.com/meta-pytorch/torchcodec.git"

# Upstream branch that tracks torch nightly.
NIGHTLY_HASHTAG = "nightly"

# Latest upstream torchcodec release used for stable (release) torch builds.
# torchcodec 0.12 and newer are ABI stable against torch and are listed as
# compatible with `torch>=2.11` in the table at
# https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions
# Update this tag after checking that table and testing the new release.
DEFAULT_STABLE_TAG = "v0.17.0"

# Lowest torch (major, minor) that DEFAULT_STABLE_TAG is compatible with.
MIN_TORCH_FOR_DEFAULT_STABLE_TAG = (2, 11)

# Optional per torch version override of the torchcodec tag, keyed by the torch
# "MAJOR.MINOR" version, for example:
#
#     TORCH_VERSION_TO_TAG = {"2.10": "v0.10.0"}
#
# Use it when a torch version needs another torchcodec release than
# DEFAULT_STABLE_TAG, either because DEFAULT_STABLE_TAG does not support it
# (older torch) or because a newer torchcodec release stops supporting it.
TORCH_VERSION_TO_TAG: dict[str, str] = {}


def parse_torch_major_minor(torch_version: str) -> tuple[int, int]:
    """Returns (major, minor) of a torch version like '2.13.0a0' or '2.13'."""
    match = re.match(r"^\s*(\d+)\.(\d+)", torch_version)
    if not match:
        raise ValueError(f"Cannot parse torch version '{torch_version}'")
    return int(match.group(1)), int(match.group(2))


def select_hashtag(torch_version: str | None) -> str:
    """Selects the torchcodec release tag to build for a torch version.

    A torch version of None means that no torch version is known (for example no
    torch checkout exists) and selects DEFAULT_STABLE_TAG.
    """
    if torch_version is None:
        return DEFAULT_STABLE_TAG
    major, minor = parse_torch_major_minor(torch_version)
    override = TORCH_VERSION_TO_TAG.get(f"{major}.{minor}")
    if override:
        return override
    if (major, minor) >= MIN_TORCH_FOR_DEFAULT_STABLE_TAG:
        return DEFAULT_STABLE_TAG
    min_major, min_minor = MIN_TORCH_FOR_DEFAULT_STABLE_TAG
    raise ValueError(
        f"No torchcodec version is configured for torch {major}.{minor}: "
        f"{DEFAULT_STABLE_TAG} needs torch>={min_major}.{min_minor}. Add an entry "
        f"for '{major}.{minor}' to TORCH_VERSION_TO_TAG in {Path(__file__).name} "
        "or pass --repo-hashtag"
    )


def read_torch_version(torch_dir: Path) -> str | None:
    """Reads the torch version from version.txt in a torch checkout, if any."""
    version_file = torch_dir / "version.txt"
    if not version_file.exists():
        return None
    return version_file.read_text().strip()


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
            help=f"Git repository url. Defaults to '{DEFAULT_ORIGIN}'",
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
            help=f"Git repository ref/tag to checkout, for example '{NIGHTLY_HASHTAG}'. Defaults to a release tag selected from the torch version, see --torch-version",
        )
        command_parser.add_argument(
            "--torch-version",
            type=str,
            default=None,
            help="Torch version (for example 2.13 or 2.13.0a0) to select the torchcodec release for. Defaults to the version in version.txt of --torch-dir",
        )
        command_parser.add_argument(
            "--torch-dir",
            type=Path,
            default=THIS_DIR / "pytorch",
            help="Directory of the torch checkout. Its version.txt is used when --torch-version is not set. If missing then the default torchcodec release is used",
        )

    p = argparse.ArgumentParser("pytorch_torchcodec_repo.py")
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser(
        "checkout", help="Clone PyTorch torchcodec locally and checkout"
    )
    add_common(checkout_p)
    repo_management.add_checkout_options(checkout_p, default_hipify=False)
    checkout_p.set_defaults(func=repo_management.do_checkout)

    args = p.parse_args(cl_args)

    if not args.repo_hashtag:
        torch_version = args.torch_version or read_torch_version(args.torch_dir)
        if torch_version is None:
            print(
                f"No torch version given or found in '{args.torch_dir}', "
                f"using the default torchcodec release {DEFAULT_STABLE_TAG}"
            )
        try:
            args.repo_hashtag = select_hashtag(torch_version)
        except ValueError as e:
            p.error(str(e))
        print(f"Selected torchcodec {args.repo_hashtag} for torch {torch_version}")
    args.gitrepo_origin = args.gitrepo_origin or DEFAULT_ORIGIN

    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
