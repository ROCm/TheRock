#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify the permanent publication regression rejects the older Windows implementations."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import unittest

from publication_stress import load_fetcher, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.modules["fetch_dvc_artifacts"] = load_fetcher("fixed")
    spec = importlib.util.spec_from_file_location(
        "permanent_tests", ROOT / "fixed/fetch_dvc_artifacts_test.py"
    )
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    records = []
    for variant in ("baseline", "candidate", "fixed"):
        tests.fda = load_fetcher(variant)
        result = unittest.TestResult()
        tests.TestDirectCachePublication(
            "test_concurrent_publication_keeps_cache_readable"
        ).run(result)
        errors = [stack for test, stack in result.failures + result.errors]
        records.append(
            {
                "variant": variant,
                "passed": result.wasSuccessful(),
                "failure_count": len(errors),
                "examples": errors[:3],
            }
        )
    correct = records[-1]["passed"]
    if sys.platform == "win32":
        correct &= all(not record["passed"] for record in records[:-1])
    metadata = {
        "python": platform.python_version(),
        "platform": sys.platform,
        "tests_sha256": hashlib.sha256(
            (ROOT / "fixed/fetch_dvc_artifacts_test.py").read_bytes()
        ).hexdigest(),
        "records": records,
        "success": correct,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
