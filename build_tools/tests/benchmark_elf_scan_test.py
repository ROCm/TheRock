# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "packaging"
    / "linux"
    / "benchmark_elf_scan.py"
)


class BenchmarkElfScanTest(unittest.TestCase):
    def test_abba_preserves_inputs_and_classification_with_bounded_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inputs"
            root.mkdir()
            payloads = {
                "elf": b"\x7fELF" + b"x" * 100000,
                "text": b"text" + b"y" * 100000,
                "short": b"\x7fEL",
                "empty": b"",
            }
            for name, content in payloads.items():
                (root / name).write_bytes(content)
            before = {name: (root / name).stat().st_mtime_ns for name in payloads}
            output = Path(temporary) / "diagnostics"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                    "--blocks",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = json.loads((output / "passes.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual([row["method"] for row in rows], list("ABBA"))
            self.assertTrue(summary["classification_and_manifest_match"])
            self.assertTrue(summary["complete_input_scan"])
            self.assertTrue(all(row["elf_count"] == 1 for row in rows))
            for row in rows:
                expected = (
                    sum(map(len, payloads.values())) if row["method"] == "A" else 11
                )
                self.assertEqual(row["application_returned_bytes"], expected)
            for name, content in payloads.items():
                self.assertEqual((root / name).read_bytes(), content)
                self.assertEqual((root / name).stat().st_mtime_ns, before[name])

    def test_empty_input_cannot_validate_a_speedup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inputs"
            root.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root",
                    str(root),
                    "--output",
                    str(Path(temporary) / "diagnostics"),
                    "--blocks",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Empty or incomplete", result.stderr)


if __name__ == "__main__":
    unittest.main()
