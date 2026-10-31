# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-only transaction fixture; no ROCm compilation or performance measurement."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PrivateTransactionTests(unittest.TestCase):
    def test_real_sqlite_guards_durability_commit_failure_and_rollback(self):
        # This fixture deliberately uses a declared SQLite build, not a mock DB.
        # The source and exact SHA3 are documented in rocfft_aot_profile.md.
        compiler = os.environ.get("ROCFFT_PROFILE_TEST_CXX")
        include = os.environ.get("ROCFFT_AOT_TEST_SQLITE_DIR")
        library = os.environ.get("ROCFFT_AOT_TEST_SQLITE_LIB")
        if not all([compiler, include, library]):
            self.skipTest(
                "Set declared host compiler and real SQLite include/library paths"
            )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            executable = root / "fixture.exe"
            subprocess.run(
                [
                    *json.loads(compiler),
                    "-std=c++17",
                    "-I",
                    include,
                    str(ROOT / "build_tools/tests/rocfft_aot_transaction_fixture.cpp"),
                    library,
                    "-o",
                    str(executable),
                ],
                check=True,
            )
            result = subprocess.run(
                [str(executable), str(root / "private.db")],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("commit failure and rollback passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
