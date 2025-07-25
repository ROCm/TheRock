#!/usr/bin/env python3

"""Writes torch_version to GITHUB_OUTPUT."""

import os
import glob

from github_actions_utils import *


def main(argv: list[str]):
    # Get the torch version from the first torch wheel in PACKAGE_DIST_DIR.
    package_dist_dir = os.getenv("PACKAGE_DIST_DIR")
    torch_version = glob.glob("torch-*.whl", root_dir=package_dist_dir)[0].split("-")[1]
    gha_set_output({"torch_version": torch_version})

    torchaudio_version_tmp = glob.glob("torchaudio-*.whl", root_dir=package_dist_dir)
    if torchaudio_version_tmp:
        torchaudio_version = torchaudio_version_tmp[0].split("-")[1]
        gha_set_output({"torchaudio_version": torchaudio_version})

    torchvision_version_tmp = glob.glob("torchvision-*.whl", root_dir=package_dist_dir)
    if torchvision_version_tmp:
        torchvision_version = torchvision_version_tmp[0].split("-")[1]
        gha_set_output({"torchaudio_version": torchaudio_version})


if __name__ == "__main__":
    main(sys.argv[1:])
