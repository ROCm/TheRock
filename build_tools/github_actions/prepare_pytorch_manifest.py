#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare one PyTorch source manifest for a build workflow.

The build workflow always consumes a manifest URL: an HTTP URL to a JSON source
manifest that pins the exact repository URLs, commits, branches, and expected
package versions for one PyTorch build. Passing this URL between jobs lets the
workflow freeze source discovery once, then have the build job check out exactly
the same sources even when floating refs such as ``nightly`` move later.

This script supports two input modes:

* Pass-through mode: ``--manifest-url`` is already known. The script emits that
  URL as a GitHub Actions output. It does not write a step summary because the
  caller that generated or selected the manifests owns that summary.
* Generation mode: all other arguments are forwarded to
  ``generate_pytorch_source_manifest.py --upload``. The generator validates
  those arguments, writes and uploads the manifest, emits the manifest URL
  output, and writes a step summary.
"""

import argparse
import subprocess
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import gha_set_output


GENERATOR_SCRIPT = (
    Path(__file__).resolve().with_name("generate_pytorch_source_manifest.py")
)


def _split_words(value: str) -> list[str]:
    return value.replace(";", " ").split() if value else []


def run_manifest_generator(generator_args: list[str]) -> None:
    command = [sys.executable, str(GENERATOR_SCRIPT), "--upload", *generator_args]
    subprocess.check_call(command)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-url",
        default="",
        help="Already-uploaded manifest URL to pass through.",
    )
    parser.add_argument(
        "--pytorch-git-refs",
        default="",
        help=("PyTorch ref for this manifest. Required unless --manifest-url is set."),
    )
    args, generator_args = parser.parse_known_args(argv)

    # This singular wrapper forwards most generator arguments unchanged, but it
    # must prepare exactly one manifest URL. Validate --pytorch-git-refs here
    # before forwarding it so an empty direct run does not expand to the
    # generator's full default ref set.
    refs = _split_words(args.pytorch_git_refs)
    if len(refs) > 1:
        parser.error(
            "This workflow prepares one manifest; pass exactly one PyTorch ref"
        )
    if not args.manifest_url and not refs:
        parser.error("--pytorch-git-refs is required unless --manifest-url is set")
    if args.pytorch_git_refs:
        generator_args.extend(["--pytorch-git-refs", args.pytorch_git_refs])

    return args, generator_args


def main(argv: list[str]) -> None:
    args, generator_args = parse_args(argv)
    if args.manifest_url:
        # Pass-through mode: the caller already selected or generated the
        # manifest, so this job only forwards the URL to downstream jobs.
        gha_set_output({"manifest_url": args.manifest_url})
        return

    # Generation mode: invoke the manifest generator, upload its output, then
    # pass the uploaded URL to downstream jobs through the generator's outputs.
    run_manifest_generator(generator_args)


if __name__ == "__main__":
    main(sys.argv[1:])
