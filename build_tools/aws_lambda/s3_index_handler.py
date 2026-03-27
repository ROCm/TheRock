#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AWS Lambda handler for S3-triggered index generation.

Triggered by S3 PutObject events via an SQS queue with a batch window.
Batching allows multiple uploads to the same directory to be deduplicated
within one invocation, avoiding redundant S3 list+put calls.

Expected trigger: SQS queue fed by S3 event notifications on PutObject for
all keys under the therock-ci-artifacts and therock-ci-artifacts-external
buckets. The handler also accepts direct S3 event payloads (useful for local
testing without SQS infrastructure).

S3 key structure:

  {run_prefix}/logs/...         -- build logs
  {run_prefix}/manifests/...   -- manifests
  {run_prefix}/python/...      -- python packages
  {run_prefix}/...             -- any other subdirectory

where run_prefix is: [{external_repo}/]{run_id}-{platform}
  e.g. "12345678901-linux" or "ROCm-TheRock/12345678901-linux"

For each uploaded object the handler regenerates index.html for its
directory and all ancestor directories up to (but not including) the run
prefix. Across a batch, each unique directory is indexed exactly once.

Handler entry point: s3_index_handler.lambda_handler
Runtime:            Python 3.12+

See README.md in this directory for deployment and IAM instructions.
"""

import json
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
        "12345-linux/python/foo.whl"                -> None (skip)
    """
    if key.rsplit("/", 1)[-1] == "index.html":
        return None
    if "/" not in key:
        return None  # file at bucket root, nothing meaningful to index
    dir_prefix = key.rsplit("/", 1)[0]
    if _EXCLUDED_SUBDIRS.intersection(dir_prefix.split("/")):
        return None
    return dir_prefix


# ---------------------------------------------------------------------------
# Per-bucket configuration
# ---------------------------------------------------------------------------

# Subdirectory names (directly under the run prefix) that should never be
# indexed. These directories are excluded regardless of bucket.
_EXCLUDED_SUBDIRS: frozenset[str] = frozenset({"python"})

# Number of path segments in the run prefix for each bucket. The ancestor
# walk stops at this depth so the run root is not re-indexed from deep
# uploads (it is already indexed by its own direct file upload events).
# Default is 1 (e.g. "12345-linux"). External buckets add one org-level
# segment (e.g. "ROCm-TheRock/12345-linux").
_RUN_PREFIX_DEPTH: dict[str, int] = {
    "therock-ci-artifacts-external": 2,
}
_DEFAULT_RUN_PREFIX_DEPTH = 1


# ---------------------------------------------------------------------------
# Event parsing and directory collection
# ---------------------------------------------------------------------------


def _extract_s3_records(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract S3 event records from a direct S3 trigger or SQS-wrapped S3 events.

    When triggered directly by S3, event["Records"] contains S3 records with
    an "s3" key. When triggered via SQS, event["Records"] contains SQS records
    whose "body" is a JSON-encoded S3 event.
    """
    raw_records = event.get("Records", [])
    if not raw_records:
        return []
    if "body" in raw_records[0]:
        # SQS trigger: unwrap each message body as an S3 event.
        s3_records = []
        for sqs_record in raw_records:
            s3_event = json.loads(sqs_record["body"])
            s3_records.extend(s3_event.get("Records", []))
        return s3_records
    return raw_records


def _collect_dirs_to_index(
    s3_records: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Collect unique directories to index from a batch of S3 event records.

    For each record, adds the leaf directory and all ancestors up to (but not
    including) the run prefix. Deduplicates across records so each directory
    is indexed at most once per batch regardless of how many files landed there.

    Returns:
        Dict mapping bucket name to a set of dir_prefixes to index.
    """
    dirs: dict[str, set[str]] = {}
    for record in s3_records:
        s3_info = record["s3"]
        bucket = s3_info["bucket"]["name"]
        key = unquote_plus(s3_info["object"]["key"])
        logger.info("Processing S3 event: s3://%s/%s", bucket, key)
        dir_prefix = _get_dir_prefix(key)
        if dir_prefix is None:
            logger.info("Skipping key (index file or bucket root): %s", key)
            continue
        bucket_dirs = dirs.setdefault(bucket, set())
        bucket_dirs.add(dir_prefix)
        run_prefix_depth = _RUN_PREFIX_DEPTH.get(bucket, _DEFAULT_RUN_PREFIX_DEPTH)
        parts = dir_prefix.split("/")
        for i in range(len(parts) - 1, run_prefix_depth, -1):
            bucket_dirs.add("/".join(parts[:i]))
    return dirs


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point.

    Args:
        event: SQS event (preferred) or direct S3 event payload.
        context: Lambda context object (unused).

    Returns:
        Dict with statusCode 200 and the number of directories indexed.

    Raises:
        RuntimeError: After logging if any directory fails to index, so the
            SQS message is not deleted and can be retried or sent to the DLQ.
    """
    s3_records = _extract_s3_records(event)
    logger.info("Received %d S3 event record(s)", len(s3_records))

    dirs_to_index = _collect_dirs_to_index(s3_records)
    total_dirs = sum(len(d) for d in dirs_to_index.values())
    logger.info(
        "Indexing %d unique director%s (deduplicated from %d record(s))",
        total_dirs,
        "y" if total_dirs == 1 else "ies",
        len(s3_records),
    )

    import boto3
    s3_client = boto3.client("s3")
    backend = create_storage_backend()

    errors: list[tuple[str, str, Exception]] = []
    indexed = 0
    for bucket, dir_prefixes in dirs_to_index.items():
        # Index deeper paths first so ancestor indexes reflect children.
        run_prefix_depth = _RUN_PREFIX_DEPTH.get(bucket, _DEFAULT_RUN_PREFIX_DEPTH)
        for dir_prefix in sorted(dir_prefixes, key=lambda p: p.count("/"), reverse=True):
            try:
                logger.info("Generating index for: %s/%s", bucket, dir_prefix)
                parent_href = (
                    None
                    if dir_prefix.count("/") < run_prefix_depth
                    else "../index.html"
                )
                generate_s3_index.generate_index_for_directory(
                    bucket=bucket,
                    dir_prefix=dir_prefix,
                    backend=backend,
                    s3_client=s3_client,
                    parent_href=parent_href,
                )
                indexed += 1
            except Exception as exc:
                logger.exception(
                    "Failed to generate index for %s/%s: %s", bucket, dir_prefix, exc
                )
                errors.append((bucket, dir_prefix, exc))

    if errors:
        details = ", ".join(f"{b}/{p}" for b, p, _ in errors)
        raise RuntimeError(
            f"Failed to index {len(errors)}/{total_dirs} director(ies): {details}"
        )

    return {"statusCode": 200, "indexed": indexed}
