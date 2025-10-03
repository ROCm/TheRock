#!/usr/bin/env python
"""Checks out PyTorch Audio.

There is nothing that this script does which you couldn't do by hand, but because of
the following, getting PyTorch sources ready to build with ToT TheRock built SDKs
consists of multiple steps:

* Sources must be pre-processed with HIPIFY, creating dirty git trees that are hard
  to develop on further.
* Both the ROCM SDK and PyTorch are moving targets that are eventually consistent.
  We maintain patches for recent PyTorch revisions to adapt to packaging and library
  compatibility differences until all releases are done and available.

Primary usage:

    ./pytorch_audio_repo.py checkout

The checkout process combines the following activities:

* Clones the pytorch repository into `THIS_MAIN_REPO_NAME` with a requested `--repo-hashtag`
  tag (default to latest release).
* Configures PyTorch submodules to be ignored for any local changes (so that
  the result is suitable for development with local patches).
* Applies "base" patches to the pytorch repo and any submodules (by using
  `git am` with patches from `patches/pytorch_ref_to_patches_dir_name(<repo-hashtag>)/<repo-name>/base`).
* Runs `hipify` to prepare sources for AMD GPU and commits the result to the
  main repo and any modified submodules.
* Applies "hipified" patches to the pytorch repo and any submodules (by using
  `git am` with patches from `patches/<repo-hashtag>/<repo-name>/hipified`).
* Records some tag information for subsequent activities.

For one-shot builds and CI use, the above is sufficient. But this tool can also
be used to develop. Any commits made to PyTorch or any of its submodules can
be saved locally in TheRock by running `./pybuild.py save-patches`. If checked
in, CI runs for that revision will incorporate them the same as anyone
interactively using this tool.
"""
import argparse
from pathlib import Path
import sys

import repo_management

THIS_MAIN_REPO_NAME = "pytorch_audio"
THIS_DIR = Path(__file__).resolve().parent

DEFAULT_ORIGIN = "https://github.com/pytorch/audio.git"
DEFAULT_HASHTAG = "main"
DEFAULT_PATCHES_DIR = THIS_DIR / "patches" / THIS_MAIN_REPO_NAME
DEFAULT_PATCHSET = None


def main(cl_args: list[str]):
    def add_common(command_parser: argparse.ArgumentParser):
        command_parser.add_argument(
            "--repo",
            type=Path,
            default=THIS_DIR / THIS_MAIN_REPO_NAME,
            help="Git repository path",
        )
        command_parser.add_argument(
            "--gitrepo-origin",
            type=str,
            default=None,
            help="Git repository url (default is set based on --use-related-commit)",
        )
        command_parser.add_argument(
            "--patch-dir",
            type=Path,
            default=DEFAULT_PATCHES_DIR,
            help="Git repository patch path",
        )
        command_parser.add_argument(
            "--repo-name",
            type=Path,
            default=THIS_MAIN_REPO_NAME,
            help="Subdirectory name in which to checkout repo",
        )

        commit_group = command_parser.add_mutually_exclusive_group()
        commit_group.add_argument(
            "--repo-hashtag",
            type=str,
            default=None,
            help="Git repository ref/tag to checkout",
        )
        commit_group.add_argument(
            "--use-related-commit",
            action=argparse.BooleanOptionalAction,
            help="Use Git repository related commit from --torch-repo",
        )

        command_parser.add_argument(
            "--torch-repo",
            type=Path,
            default=THIS_DIR / "pytorch",
            help="Git repository path for torch, if using --use-related-commit",
        )
        command_parser.add_argument(
            "--patchset",
            default=None,
            help="patch dir subdirectory (defaults to mangled --repo-hashtag)",
        )

    p = argparse.ArgumentParser("pytorch_audio_repo.py")
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser("checkout", help="Clone PyTorch locally and checkout")
    add_common(checkout_p)
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
        default=True,
        help="Apply patches for the repo-hashtag",
    )
    checkout_p.set_defaults(func=repo_management.do_checkout)

    hipify_p = sub_p.add_parser("hipify", help="Run HIPIFY on the project")
    add_common(hipify_p)
    hipify_p.set_defaults(func=repo_management.do_hipify)

    save_patches_p = sub_p.add_parser(
        "save-patches", help="Save local commits as patch files for later application"
    )
    add_common(save_patches_p)
    save_patches_p.set_defaults(func=repo_management.do_save_patches)

    args = p.parse_args(cl_args)

    if args.use_related_commit:
        # Set default values based on the pin file in the pytorch repo.
        (
            git_origin,
            git_hashtag,
            patchset,
            has_related_commit,
        ) = repo_management.read_pytorch_rocm_pins(
            args.torch_repo,
            os="centos",
            project="torchaudio",
            default_origin=DEFAULT_ORIGIN,
            default_hashtag=DEFAULT_HASHTAG,
            default_patchset=DEFAULT_PATCHSET,
        )
        if not has_related_commit:
            raise ValueError(
                f"Could not find torchaudio in '{args.torch_repo}/related_commits' (did you mean to set a different --torch-repo?)"
            )
        print("Found pytorch rocm pins:")
        print(f"  git_origin: {git_origin}")
        print(f"  git_hashtag: {git_hashtag}")
        print(f"  patchset: {patchset}")
        args.gitrepo_origin = args.gitrepo_origin or git_origin
        args.repo_hashtag = args.repo_hashtag or git_hashtag
        args.patchset = args.patchset or patchset
    else:
        # Otherwise use the usual defaults.
        args.gitrepo_origin = args.gitrepo_origin or DEFAULT_ORIGIN
        args.repo_hashtag = args.repo_hashtag or DEFAULT_HASHTAG
        args.patchset = args.patchset or DEFAULT_PATCHSET

    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
