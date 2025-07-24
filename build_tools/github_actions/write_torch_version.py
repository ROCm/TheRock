#!/usr/bin/env python3

"""Writes torch_version to GITHUB_OUTPUT."""

import os
import glob

from github_actions_utils import *


def main(argv: list[str]):
    # Get the torch version from the first torch wheel in PACKAGE_DIST_DIR.
    package_dist_dir = os.getenv("PACKAGE_DIST_DIR")
    version = glob.glob("torch-*.whl", root_dir=package_dist_dir)[0].split("-")[1]
    gha_set_output({"torch_version": version})


if __name__ == "__main__":
    main(sys.argv[1:])
