#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write PyTorch package versions from a manifest to GITHUB_OUTPUT."""

import argparse
import json
import sys
from pathlib import Path

from github_actions_api import gha_set_output


PACKAGE_VERSION_OUTPUTS = {
    "pytorch": "torch_version",
    "pytorch_audio": "torchaudio_version",
    "pytorch_vision": "torchvision_version",
    "triton": "triton_version",
    "apex": "apex_version",
}


def collect_versions(manifest: dict[str, object]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for project, output_name in PACKAGE_VERSION_OUTPUTS.items():
        source_info = manifest.get(project)
        if source_info is None:
            continue
        if not isinstance(source_info, dict):
            raise TypeError(f"Manifest entry '{project}' must be an object")
        version = source_info.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError(f"Manifest entry '{project}' is missing a version")
        outputs[output_name] = version
    if not outputs:
        raise ValueError("Manifest did not contain any PyTorch package versions")
    return outputs


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Write PyTorch package versions from a manifest to GITHUB_OUTPUT"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a PyTorch manifest JSON file.",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gha_set_output(collect_versions(manifest))


if __name__ == "__main__":
    main(sys.argv[1:])
