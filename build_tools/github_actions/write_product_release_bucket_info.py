#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write final product publication bucket metadata to GITHUB_OUTPUT."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _therock_utils.s3_buckets import (
    get_product_release_bucket_config,
    get_release_package_index_url,
)
from github_actions_api import gha_set_output


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Determine IAM role ARN and region for a product release bucket"
    )
    parser.add_argument(
        "--release-type",
        required=True,
        choices=["dev", "dev-bkc", "nightly", "nightly-bkc", "prerelease"],
        help="Release type used to select the final publication stream.",
    )
    parser.add_argument(
        "--product",
        required=True,
        choices=["core", "pytorch", "jax"],
        help="Product bucket to publish to.",
    )
    args = parser.parse_args(argv)

    config = get_product_release_bucket_config(
        release_type=args.release_type,
        product=args.product,
    )

    gha_set_output(
        {
            "bucket": config.name,
            "iam_role": config.write_access_iam_role or "",
            "aws_region": config.region,
            "package_index_url": get_release_package_index_url(args.release_type),
        }
    )


if __name__ == "__main__":
    main()
