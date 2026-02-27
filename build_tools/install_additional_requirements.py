#!/usr/bin/env python

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent
THEROCK_OUTPUT_DIR = str(THEROCK_DIR / "build")


def install_requirements(input: str):
    environ_vars = os.environ.copy()

    requirements_files = input.split(",")

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
    )
    args = parser.parse_args(argv)
    if not args.requirements_files:
        logging.info(
            "No requirements file(s) provided. Exiting install_additional_requirements.py..."
        )
        sys.exit(0)

    install_requirements(str(args.requirements_files))


if __name__ == "__main__":
    main(sys.argv[1:])
