# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pytest configuration for build_tools tests."""

import sys
from pathlib import Path

import pytest

# Add build_tools to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from _therock_utils.s3_buckets import set_bucket_config_file


@pytest.fixture(autouse=True)
def _reset_bucket_registry():
    """Rebuild the S3 bucket registry around every test.

    The registry is cached after first use, so a test that points
    THEROCK_S3_BUCKETS_FILE (or --bucket-config-file) at a fixture would
    otherwise leak its buckets into every test that runs after it in the same
    process. Reset on both sides so the leak cannot travel in either direction.
    """
    set_bucket_config_file(None)
    yield
    set_bucket_config_file(None)
