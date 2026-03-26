#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AWS Lambda handler for S3-triggered index generation.

Triggered by S3 PutObject events. For each uploaded object, determines the
first-level subdirectory under the run prefix and regenerates its index.html
by calling generate_s3_index.generate_index_for_directory().

Expected trigger: S3 event notification on PutObject for all keys under the
therock-ci-artifacts and therock-ci-artifacts-external buckets.

S3 key structure:

  {run_prefix}/logs/...         -- build logs
  {run_prefix}/manifests/...   -- manifests
  {run_prefix}/python/...      -- python packages
  {run_prefix}/...             -- any other subdirectory

where run_prefix is: [{external_repo}/]{run_id}-{platform}
  e.g. "12345678901-linux" or "Fork-TheRock/12345678901-linux"

For each uploaded object at {run_prefix}/{subdir}/..., this handler
regenerates {run_prefix}/{subdir}/index.html with a recursive file listing.
Objects uploaded directly at the run root regenerate {run_prefix}/index.html.

Handler entry point: s3_index_handler.handler
Runtime:            Python 3.12+
Required IAM:       s3:GetObject, s3:PutObject, s3:ListBucket on target buckets

-------------------------------------------------------------------------------
DEPLOYMENT PACKAGE
-------------------------------------------------------------------------------
The Lambda deployment package must be a flat zip containing this file and the
following files copied manually from the TheRock repository:

  File in deployment package            Source path in TheRock repo
  ------------------------------------  ----------------------------------------
  generate_s3_index.py                  build_tools/generate_s3_index.py
  _therock_utils/storage_backend.py     build_tools/_therock_utils/storage_backend.py
  _therock_utils/storage_location.py   build_tools/_therock_utils/storage_location.py

Resulting zip layout:

  s3_index_handler.py
  generate_s3_index.py
  _therock_utils/
      storage_backend.py
      storage_location.py

Third-party dependencies (install into the package root before zipping):

  pip install boto3
-------------------------------------------------------------------------------
"""

import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# All co-located files (generate_s3_index.py, _therock_utils/) live alongside
# this handler in the deployment package. For local development (repo checkout)
# the same files are found relative to build_tools/.

_THIS_DIR = Path(__file__).resolve().parent

for _candidate in [
    _THIS_DIR,         # Lambda: generate_s3_index.py alongside handler
    _THIS_DIR.parent,  # repo checkout: build_tools/generate_s3_index.py
]:
    if (_candidate / "generate_s3_index.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

for _candidate in [
    _THIS_DIR,         # Lambda: _therock_utils/ alongside handler
    _THIS_DIR.parent,  # repo checkout: build_tools/_therock_utils/
]:
    if (_candidate / "_therock_utils").is_dir():
        sys.path.insert(0, str(_candidate))
        break

import generate_s3_index
from _therock_utils.storage_backend import create_storage_backend


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------


def _get_dir_prefix(key: str) -> str | None:
    """Return the directory containing this object, or None if the key should be skipped.

    Returns None for index.html files (to avoid regeneration loops) or keys
    with no directory component (files at the bucket root).

    Examples:
        "12345-linux/logs/gfx94X-dcgpu/build.log"  -> "12345-linux/logs/gfx94X-dcgpu"
        "12345-linux/core_lib.tar.xz"               -> "12345-linux"
        "Fork/12345-linux/logs/gfx94X/build.log"   -> "Fork/12345-linux/logs/gfx94X"
        "12345-linux/logs/gfx94X/index.html"        -> None (skip)
    """
    if key.rsplit("/", 1)[-1] == "index.html":
        return None
    if "/" not in key:
        return None  # file at bucket root, nothing meaningful to index
    return key.rsplit("/", 1)[0]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _process_record(record: dict[str, Any]) -> None:
    """Process a single S3 event record."""
    s3_info = record["s3"]
    bucket = s3_info["bucket"]["name"]
    # S3 event keys are URL-encoded.
    key = unquote_plus(s3_info["object"]["key"])

    logger.info("Processing S3 event: s3://%s/%s", bucket, key)

    dir_prefix = _get_dir_prefix(key)
    if dir_prefix is None:
        logger.info("Skipping key (index file or unrecognized structure): %s", key)
        return

    import boto3
    s3_client = boto3.client("s3")
    backend = create_storage_backend()

    logger.info("Generating index for: %s/%s", bucket, dir_prefix)
    generate_s3_index.generate_index_for_directory(
        bucket=bucket,
        dir_prefix=dir_prefix,
        backend=backend,
        s3_client=s3_client,
    )


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point.

    Args:
        event: S3 event payload containing a list of Records.
        context: Lambda context object (unused).

    Returns:
        Dict with statusCode 200 on success.

    Raises:
        RuntimeError: After logging if any record fails, so Lambda can retry
            via its configured retry policy.
    """
    records = event.get("Records", [])
    logger.info("Received %d S3 event record(s)", len(records))

    errors: list[tuple[str, Exception]] = []
    for record in records:
        key = record.get("s3", {}).get("object", {}).get("key", "<unknown>")
        try:
            _process_record(record)
        except Exception as exc:
            logger.exception("Failed to process record for key %s: %s", key, exc)
            errors.append((key, exc))

    if errors:
        keys = ", ".join(k for k, _ in errors)
        raise RuntimeError(
            f"Failed to process {len(errors)}/{len(records)} record(s): {keys}"
        )

    return {"statusCode": 200, "processed": len(records)}
