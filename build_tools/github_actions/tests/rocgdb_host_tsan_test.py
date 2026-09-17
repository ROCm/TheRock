#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import test_rocgdb_host_tsan


class RocgdbHostTsanTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / f"_rocgdb_host_tsan_{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_inventory_is_exact_and_drift_fails_closed(self):
        testsuite = self.root / "gdb.dwarf2"
        testsuite.mkdir()
        names = ["a.exp", "b.exp"]
        for name in names:
            (testsuite / name).touch()

        with mock.patch.object(test_rocgdb_host_tsan, "EXPECTED_DWARF2_COUNT", 2), mock.patch.object(
            test_rocgdb_host_tsan,
            "EXPECTED_DWARF2_SHA256",
            test_rocgdb_host_tsan._inventory_digest(names),
        ):
            self.assertEqual(
                test_rocgdb_host_tsan.validate_dwarf2_inventory(self.root), names
            )
            (testsuite / "new.exp").touch()
            with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                test_rocgdb_host_tsan.validate_dwarf2_inventory(self.root)

    def test_ignore_list_must_match_exact_contract(self):
        path = self.root / "ignore.json"
        expected = {
            "Generic": [],
            "GCC": [],
            "LLVM": list(test_rocgdb_host_tsan.EXPECTED_LLVM_XFAILS),
        }
        path.write_text(json.dumps(expected), encoding="utf-8")
        test_rocgdb_host_tsan.validate_ignore_list(path)

        expected["LLVM"].append("gdb.dwarf2/new.exp")
        path.write_text(json.dumps(expected), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "expected-failure contract changed"):
            test_rocgdb_host_tsan.validate_ignore_list(path)

    def test_discovers_every_python_variant_and_requires_one(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        (bin_dir / "rocgdb").touch()
        (bin_dir / "rocgdb-pynone").touch()
        (bin_dir / "rocgdb-py3.12").touch()
        self.assertEqual(
            [path.name for path in test_rocgdb_host_tsan.rocgdb_host_binaries(self.root)],
            ["rocgdb-py3.12", "rocgdb-pynone"],
        )
        (bin_dir / "rocgdb-pynone").unlink()
        (bin_dir / "rocgdb-py3.12").unlink()
        with self.assertRaisesRegex(RuntimeError, "no ROCgdb host ELF binaries"):
            test_rocgdb_host_tsan.rocgdb_host_binaries(self.root)

    def test_command_is_cpu_only_and_disables_failure_retries(self):
        command = test_rocgdb_host_tsan.test_command(Path("test_rocgdb.py"))
        self.assertEqual(command[-2:], ["--tests", "gdb.dwarf2"])
        self.assertIn("--max-failed-retries", command)
        retry_index = command.index("--max-failed-retries")
        self.assertEqual(command[retry_index + 1], "0")
        self.assertNotIn("gdb.rocm", command)
        self.assertNotIn("--gpu-tests", command)
        self.assertNotIn("--sanity-check", command)

    @staticmethod
    def _valid_outcome_log():
        lines = []
        for compiler, counts in test_rocgdb_host_tsan.EXPECTED_OUTCOME_COUNTS.items():
            lines.append(f"---------------- {compiler} ----------------")
            for status, count in counts.items():
                lines.append(f"  [!] {status}: {count}")
        lines.extend(
            [
                "================ FINAL TEST STATUS ================",
                "[✓] GCC: PASS",
                "[✓] LLVM: PASS",
                "[X] Total Failed Tests: 8",
                "[!] Ignored Failures (LLVM) (8):",
                *test_rocgdb_host_tsan.EXPECTED_USED_LLVM_XFAILS,
                "",
                "[!] Unused Ignored Failures (1):",
                *test_rocgdb_host_tsan.EXPECTED_UNUSED_LLVM_XFAILS,
                "",
                "[✓] OVERALL STATUS: PASS",
            ]
        )
        return "\n".join(f"2026-09-16 00:00:00,000 - INFO: {line}" for line in lines)

    def test_exact_outcome_profile_is_admitted(self):
        test_rocgdb_host_tsan.validate_outcome_profile(self._valid_outcome_log())

    def test_outcome_count_drift_fails_closed(self):
        output = self._valid_outcome_log().replace("PASS: 2336", "PASS: 2335")
        with self.assertRaisesRegex(RuntimeError, "outcome profile changed"):
            test_rocgdb_host_tsan.validate_outcome_profile(output)

    def test_expected_failure_use_drift_fails_closed(self):
        output = self._valid_outcome_log().replace(
            "gdb.dwarf2/pr13961.exp", "gdb.dwarf2/new-failure.exp"
        )
        with self.assertRaisesRegex(RuntimeError, "approved failure use changed"):
            test_rocgdb_host_tsan.validate_outcome_profile(output)

    def test_matrix_harness_exception_requires_rocgdb_validation_marker(self):
        harness = (
            Path(__file__).parents[3] / "run_host_tsan_phase1_validation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('[[ "$component" == "rocgdb-cpu" ]]', harness)
        self.assertIn(test_rocgdb_host_tsan.OUTCOME_VALIDATION_MARKER, harness)
        self.assertIn(
            "Preserve strict generic skip rejection for every other component",
            harness,
        )


if __name__ == "__main__":
    unittest.main()
