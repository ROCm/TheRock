# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard against false success when lit drops or skips device tests."""

import json
from pathlib import Path
import tempfile
import unittest

from test_tools.dasan_smoke import HIP_TESTS, HOST_TESTS, check_lit_results


class LitCoverageTest(unittest.TestCase):
    def check_results(self, results: dict[str, str], expected: set[str], allowed=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lit.json"
            path.write_text(
                json.dumps(
                    {
                        "tests": [
                            {"name": "DASAN-x86_64-linux :: " + name, "code": code}
                            for name, code in results.items()
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return check_lit_results(path, expected, allowed or set())

    def test_host_only_success_cannot_satisfy_device_scope(self):
        with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
            self.check_results({name: "PASS" for name in HOST_TESTS}, HIP_TESTS)

    def test_missing_device_test_is_rejected(self):
        results = {name: "PASS" for name in HIP_TESTS}
        results.pop("AMDGPU/heap-overflow.hip")
        with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
            self.check_results(results, HIP_TESTS)

    def test_only_explicit_multi_device_skip_is_allowed(self):
        results = {name: "PASS" for name in HIP_TESTS}
        results["AMDGPU/multi-device.hip"] = "UNSUPPORTED"
        allowed = {"AMDGPU/multi-device.hip"}
        self.assertEqual(self.check_results(results, HIP_TESTS, allowed), results)
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            self.check_results(results, HIP_TESTS)
        results["AMDGPU/heap-overflow.hip"] = "UNSUPPORTED"
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            self.check_results(results, HIP_TESTS, allowed)

    def test_expected_failure_does_not_count_as_pass(self):
        results = {name: "PASS" for name in HOST_TESTS}
        results["initialize.c"] = "XFAIL"
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            self.check_results(results, HOST_TESTS)


if __name__ == "__main__":
    unittest.main()
