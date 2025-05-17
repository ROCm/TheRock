#!/usr/bin/env python
"""Checks out and builds PyTorch against a built from source ROCm SDK.

There is nothing that this script does which you couldn't do by hand, but because of
the following, getting PyTorch sources ready to build with ToT TheRock built SDKs
consists of multiple steps:

* Sources must be pre-processed with HIPIFY, creating dirty git trees that are hard
  to develop on further.
* Both the ROCm SDK and PyTorch are moving targets that are eventually consistent.
  We maintain patches for recent PyTorch revisions to adapt to packaging and library
  compatibility differences until all releases are done and available.

Primary usage:

    ./ptbuild.py checkout
    (TODO) ./ptbuild.py develop

The checkout process combines the following activities:

* Clones pytorch (pytorch, vision, audio) repositories into `src/` with
  requested `--[repo]-ref` tag (default to latest release).
* Configures PyTorch submodules to be ignored for any local changes (so that
  the result is suitable for development with local patches).
* Applies "base" patches to each repo and any submodules (by using
  `git am` with patches from `patches/<repo>/<repo-ref>)/<repo-name>/base`).
* Runs `hipify` to prepare sources for AMD GPU and commits the result to the
  main repo and any modified submodules.
* Applies "hipified" patches to the pytorch repo and any submodules (by using
  `git am` with patches from `patches/<repo>/<repo-ref>/<repo-name>/hipified`).
* Records some tag information for subsequent activities.

For one-shot builds and CI use, the above is sufficient. But this tool can also
be used to develop. Any commits made to any repositories or any of their
submodules can be saved locally in TheRock by running `./pybuild.py save-patches`.
If checked in, CI runs for that revision will incorporate them the same as
anyone interactively using this tool.
"""

import argparse
from pathlib import Path
import sys

import repo_management

THIS_DIR = Path(__file__).resolve().parent


def do_checkout_for_all_repos(args: argparse.Namespace):
    if args.pytorch:
        print(f"Performing checkout steps for pytorch")
        repo_management.do_checkout(
            src_dir=args.pytorch_src_dir,
            patch_dir=args.pytorch_patch_dir,
            remote_url=args.pytorch_remote_url,
            ref=args.pytorch_ref,
            depth=args.depth,
            jobs=args.jobs,
            patch=args.patch,
            hipify=args.hipify,
        )
    if args.audio:
        print(f"Performing checkout steps for audio")
        repo_management.do_checkout(
            src_dir=args.audio_src_dir,
            patch_dir=args.audio_patch_dir,
            remote_url=args.audio_remote_url,
            ref=args.audio_ref,
            depth=args.depth,
            jobs=args.jobs,
            patch=args.patch,
            hipify=args.hipify,
        )
    if args.vision:
        print(f"Performing checkout steps for vision")
        repo_management.do_checkout(
            src_dir=args.vision_src_dir,
            patch_dir=args.vision_patch_dir,
            remote_url=args.vision_remote_url,
            ref=args.vision_ref,
            depth=args.depth,
            jobs=args.jobs,
            patch=args.patch,
            hipify=args.hipify,
        )


def do_hipify_for_all_repos(args: argparse.Namespace):
    if args.pytorch:
        repo_management.do_hipify(src_dir=args.pytorch_src_dir)
    if args.audio:
        repo_management.do_hipify(src_dir=args.audio_src_dir)
    if args.vision:
        repo_management.do_hipify(src_dir=args.vision_src_dir)


def do_save_patches_for_all_repos(args: argparse.Namespace):
    if args.pytorch:
        repo_management.do_save_patches(
            src_dir=args.pytorch_src_dir,
            patch_dir=args.pytorch_patch_dir,
            ref=args.pytorch_ref,
        )
    if args.audio:
        repo_management.do_save_patches(
            src_dir=args.audio_src_dir,
            patch_dir=args.audio_patch_dir,
            ref=args.audio_ref,
        )
    if args.vision:
        repo_management.do_save_patches(
            src_dir=args.vision_src_dir,
            patch_dir=args.vision_patch_dir,
            ref=args.vision_ref,
        )


def main(cl_args: list[str]):
    def add_common(command_parser: argparse.ArgumentParser):
        # https://github.com/pytorch/pytorch
        group_pytorch = command_parser.add_argument_group("pytorch")
        group_pytorch.add_argument(
            "--pytorch",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Controls whether pytorch repository operations are enabled",
        )
        group_pytorch.add_argument(
            "--pytorch-src-dir",
            type=Path,
            default=THIS_DIR / "src" / "pytorch",
            help="PyTorch repository source directory path to clone into",
        )
        group_pytorch.add_argument(
            "--pytorch-patch-dir",
            type=Path,
            default=THIS_DIR / "patches" / "pytorch",
            help="PyTorch repository patches root directory path",
        )
        group_pytorch.add_argument(
            "--pytorch-remote-url",
            default="https://github.com/pytorch/pytorch.git",
            help="PyTorch repository git origin remote URL",
        )
        group_pytorch.add_argument(
            "--pytorch-ref",
            default="v2.6.0",
            help="PyTorch git ref/tag to checkout",
        )

        # https://github.com/pytorch/audio
        group_audio = command_parser.add_argument_group("audio")
        group_audio.add_argument(
            "--audio",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Controls whether torchaudio repository operations are enabled",
        )
        group_audio.add_argument(
            "--audio-src-dir",
            type=Path,
            default=THIS_DIR / "src" / "audio",
            help="torchaudio repository source directory path to clone into",
        )
        group_audio.add_argument(
            "--audio-patch-dir",
            type=Path,
            default=THIS_DIR / "patches" / "audio",
            help="torchaudio repository patches root directoryp ath",
        )
        group_audio.add_argument(
            "--audio-remote-url",
            default="https://github.com/pytorch/audio.git",
            help="torchaudio repository git origin remote URL",
        )
        group_audio.add_argument(
            "--audio-ref",
            default="v2.6.0",
            help="torchaudio git ref/tag to checkout",
        )

        # https://github.com/pytorch/vision
        group_vision = command_parser.add_argument_group("vision")
        group_vision.add_argument(
            "--vision",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Controls whether torchvision repository operations are enabled",
        )
        group_vision.add_argument(
            "--vision-src-dir",
            type=Path,
            default=THIS_DIR / "src" / "vision",
            help="torchvision repository source directory path to clone into",
        )
        group_vision.add_argument(
            "--vision-patch-dir",
            type=Path,
            default=THIS_DIR / "patches" / "vision",
            help="torchvision repository patches root directoryp ath",
        )
        group_vision.add_argument(
            "--vision-remote-url",
            default="https://github.com/pytorch/vision.git",
            help="torchvision repository git origin remote URL",
        )
        group_vision.add_argument(
            "--vision-ref",
            default="v0.21.0",
            help="torchvision git ref/tag to checkout",
        )

    p = argparse.ArgumentParser("ptbuild.py")
    sub_p = p.add_subparsers(required=True)
    checkout_p = sub_p.add_parser(
        "checkout",
        help="Clone repositories locally and checkout",
    )
    add_common(checkout_p)
    checkout_p.add_argument("--depth", type=int, help="Fetch depth")
    checkout_p.add_argument("--jobs", type=int, help="Number of fetch jobs")
    checkout_p.add_argument(
        "--hipify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run HIPIFY after checkout on enabled repositories",
    )
    checkout_p.add_argument(
        "--patch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply patches for enabled repositories",
    )
    checkout_p.set_defaults(func=do_checkout_for_all_repos)

    hipify_p = sub_p.add_parser(
        "hipify",
        help="Run HIPIFY on checked out repositories",
    )
    add_common(hipify_p)
    hipify_p.set_defaults(func=do_hipify_for_all_repos)

    save_patches_p = sub_p.add_parser(
        "save-patches",
        help="Save local commits as patch files for later application",
    )
    add_common(save_patches_p)
    save_patches_p.set_defaults(func=do_save_patches_for_all_repos)

    args = p.parse_args(cl_args)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
