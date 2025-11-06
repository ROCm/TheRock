#!/usr/bin/env python
"""Checks out UCCL.

Helper script to checkout upstream UCCL sources without cloning the full repo.

Primary usage:

    ./uccl_repo.py checkout

The checkout process combines the following activities:

* Clones the uccl repository into `THIS_MAIN_REPO_NAME` with a requested `--repo-hashtag`
  tag (default to latest release).
* Configures UCCL submodules to be ignored for any local changes (so that
  the result is suitable for development with local patches).
* Applies "base" patches to the uccl repo and any submodules (by using
  `git am` with patches from `patches/uccl_ref_to_THIS_PATCHES_DIR_name(<repo-hashtag>)/<repo-name>/base`).
* Records some tag information for subsequent activities.

For one-shot builds and CI use, the above is sufficient. But this tool can also
be used to develop. Any commits made to UCCL or any of its submodules can
be saved locally in TheRock by running `./uccl_repo.py save-patches`. If checked
in, CI runs for that revision will incorporate them the same as anyone
interactively using this tool.
"""
import argparse
from pathlib import Path
import sys

import repo_management

THIS_MAIN_REPO_NAME = "uccl"
THIS_DIR = Path(__file__).resolve().parent
THIS_PATCHES_DIR = THIS_DIR / "patches" / THIS_MAIN_REPO_NAME


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
            default=default_repo_hashtag,
            help="Git repository ref/tag to checkout",
        )
        command_parser.add_argument(
            "--patchset",
            help="patch dir subdirectory (defaults to mangled --repo-hashtag)",
        )

    p = argparse.ArgumentParser("uccl_repo.py")
    default_repo_hashtag = "main"
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser("checkout", help="Clone UCCL locally and checkout")
    add_common(checkout_p)
    checkout_p.add_argument(
        "--gitrepo-origin",
        default="https://github.com/uccl-project/uccl.git",
        help="git repository url",
    )
    checkout_p.add_argument("--depth", type=int, help="Fetch depth")
    checkout_p.add_argument("--jobs", default=10, type=int, help="Number of fetch jobs")
    checkout_p.add_argument(
        "--patch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply patches for the repo-hashtag",
    )
    checkout_p.set_defaults(func=repo_management.do_checkout)

    save_patches_p = sub_p.add_parser(
        "save-patches", help="Save local commits as patch files for later application"
    )
    add_common(save_patches_p)
    save_patches_p.set_defaults(func=repo_management.do_save_patches)

    args = p.parse_args(cl_args)

    # Hard-set an "option" as disabled (post-parse) to keep repo_management.py happy
    setattr(args, "hipify", False)

    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
