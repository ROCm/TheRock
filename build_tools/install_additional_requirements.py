#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""Script to install additional requirements.txt files for projects that require additional files for testing.

Requires a requirements-file input parameter that is a list of comma separated paths to requirements.txt files.
This path will always be relative to the absolute path of the OUTPUT_ARTIFACTS_DIR.

Usage:
python install_additional_requirements.py
    (--requirements-files REQUIREMENTS_FILES)

Examples:

- Install a single requirements.txt file
    ```
    python install_additional_requirements.py \
        --requirements-files path/to/requirements.txt
    ```

- Install multiple requirements.txt files
    ```
    python install_additional_requirements.py \
        --requirements-files path/to/requirements.txt,path/to/requirements-test.txt
    ```

"""

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent
THEROCK_OUTPUT_DIR = str(
    THEROCK_DIR / os.getenv("OUTPUT_ARTIFACTS_DIR").removeprefix("./")
)


def install_requirements(req_files_list: str):
    environ_vars = os.environ.copy()

    requirements_files = req_files_list.split(",")

    for file in requirements_files:
        cmd = [
            "uv",
            "pip",
            "install",
            "-r",
            f"{THEROCK_OUTPUT_DIR}/{file}",
        ]
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requirements-files",
        type=str,
        default="",
        help="A comma separated list of requirements.txt files to install",
        required=True,
    )
    args = parser.parse_args(argv)
    install_requirements(str(args.requirements_files))


if __name__ == "__main__":
    main(sys.argv[1:])
