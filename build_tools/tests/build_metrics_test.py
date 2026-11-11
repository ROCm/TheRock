#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for per-component resource measurement and artifact collection."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_metrics import binary_sizes, ninja_stages, report

SCRIPT = Path(__file__).resolve().parents[1] / "build_metrics.py"


class BuildMetricsTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Linux build metrics")
    def test_cmake_stage_hooks(self):
        import importlib.util

        harness_path = (
            SCRIPT.parents[1] / "cmake/tests/therock_subproject_prebuilt_test.py"
        )
        spec = importlib.util.spec_from_file_location("prebuilt_harness", harness_path)
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness.write_harness(root / "src")
            build = root / "build"
            subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(root / "src"),
                    "-B",
                    str(build),
                    "-G",
                    "Ninja",
                    "-DTHEROCK_BUILD_METRICS=ON",
                    "-DTHEROCK_HARDENING_POLICY=OFF",
                    f"-DPython3_EXECUTABLE={sys.executable}",
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    "main_project",
                ],
                check=True,
                capture_output=True,
            )
            records = [
                json.loads(p.read_text())
                for p in (build / "logs/metrics").glob("*.json")
            ]
            for component in ("dep_project", "main_project"):
                self.assertEqual(
                    {r["stage"] for r in records if r["component"] == component},
                    {"configure", "build", "install", "dist"},
                )
            count = len(records)
            subprocess.run(
                [
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    "main_project",
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(len(list((build / "logs/metrics").glob("*.json"))), count)

    def test_ninja_aliases_and_incremental_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / ".ninja_log"
            output = "core/hip/stamp/build.stamp"
            log.write_text(
                "# ninja log v5\n"
                f"0\t100\t1\t{output}\tabc\n"
                f"0\t100\t1\t{root / output}\tabc\n"
                "0\t100\t1\tcore/other/stamp/build.stamp\tabc\n"
                f"0\t50\t2\t{output}\tabc\n"
                f"0\tbad\t3\t{output}\tabc\n"
                "0\tunfinished"
            )
            rows = ninja_stages(log)
            self.assertEqual(len(rows), 3)
            self.assertEqual([r["wall_seconds"] for r in rows], [0.1, 0.1, 0.05])

    @unittest.skipUnless(sys.platform == "linux", "Linux build metrics")
    def test_failure_resources_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "run",
                    "--output-dir",
                    str(root / "logs/metrics"),
                    "--component",
                    "fixture",
                    "--stage",
                    "build",
                    "--interval",
                    "0.02",
                    "--",
                    sys.executable,
                    "-c",
                    "import time; x=bytearray(8000000); sum(range(1000000)); time.sleep(.12); raise SystemExit(7)",
                ]
            )
            self.assertEqual(result.returncode, 7)
            report(root, root / "metrics")
            records = json.loads((root / "metrics/stages.json").read_text())
            self.assertEqual(records[0]["exit_code"], 7)
            self.assertGreater(records[0]["user_cpu_seconds"], 0)
            self.assertGreater(records[0]["sampled_peak_process_tree_rss_bytes"], 0)
            self.assertGreater(records[0]["max_child_rss_bytes"], 0)
            self.assertTrue(list((root / "metrics/raw").glob("*.csv")))

    @unittest.skipUnless(sys.platform == "linux", "Symlink support")
    def test_binary_inventory_avoids_aliases_and_dist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / "core/hip/stage"
            stage.mkdir(parents=True)
            (stage.parent / "stamp").mkdir()
            (stage / "library.so.1").write_bytes(b"\x7fELF01234567")
            (stage / "library.so").symlink_to("library.so.1")
            (stage / "library.a").write_bytes(b"!<arch>\n0123")
            (stage / "README").write_text("text")
            dist = root / "dist"
            dist.mkdir()
            (dist / "other.so").write_bytes(b"\x7fELF01234567")
            self.assertEqual(len(binary_sizes(root)), 2)
            report(root, root / "metrics")
            self.assertIn("24", (root / "metrics/binary_totals.csv").read_text())


if __name__ == "__main__":
    unittest.main()
