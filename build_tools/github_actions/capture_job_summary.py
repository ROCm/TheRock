#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Publish the accumulated job summary as a step output.

A reusable workflow can only read another job's data through needs, which
carries job outputs. This script reads the per-job summary mirror written by
gha_append_step_summary and publishes it under a step output so a downstream
notify job can forward it via toJSON(needs).
"""

import argparse
import sys

from github_actions_api import gha_set_job_summary_output


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-name",
        default="summary",
        help="Step-output name to set (default: summary).",
    )
    args = parser.parse_args(argv)

    gha_set_job_summary_output(output_name=args.output_name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
