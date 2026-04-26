#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Writes `bucket`, `iam_role`, and `aws_region` for the artifacts S3 bucket to GITHUB_OUTPUT.

Used by .github/actions/configure_aws_artifacts_credentials/action.yml.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _therock_utils.s3_buckets import get_artifacts_bucket_config_for_workflow_run
from github_actions_api import gha_set_output


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Determine IAM role ARN and region for the artifacts S3 bucket"
    )
    parser.add_argument(
        "--release-type",
        type=str,
        default="",
        help='Release type: "" for CI, or "dev", "nightly", "prerelease".',
    )
    args = parser.parse_args(argv)

    repository = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
    config = get_artifacts_bucket_config_for_workflow_run(
        github_repository=repository,
        release_type=args.release_type,
    )

    gha_set_output(
        {
            "bucket": config.name,
            "iam_role": config.write_access_iam_role or "",
            "aws_region": config.region,
        }
    )


if __name__ == "__main__":
    main()
