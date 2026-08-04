# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pytest configuration for build_tools/github_actions tests.

This is a separate pytest testpath from build_tools/tests (see the ``testpaths``
list in build_tools/pyproject.toml), so it needs its own copy of the fixtures
that must apply to every test in the process.
"""

import sys
from pathlib import Path

import pytest

# Add build_tools to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from _therock_utils.s3_buckets import reset_bucket_registry


@pytest.fixture(autouse=True)
def _reset_bucket_registry():
    """Rebuild the S3 bucket registry around every test.

    The registry is cached after first use, so a test that points
    THEROCK_S3_BUCKETS_FILE at a fixture would otherwise leak its buckets into
    every test that runs after it in the same process. Reset on both sides so
    the leak cannot travel in either direction.
    """
    reset_bucket_registry()
    yield
    reset_bucket_registry()
