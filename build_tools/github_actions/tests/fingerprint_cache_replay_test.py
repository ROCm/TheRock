# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for fingerprint_cache.replay (point-in-time reuse measurement)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprint_cache.replay import ReplayRun, replay
from fingerprint_cache.topology_view import DictTopologyView

TARGETS = ["gfx94X-dcgpu"]


def _topology() -> DictTopologyView:
    return DictTopologyView(
        stages=[
            ("compiler", [("amd-llvm", False)]),
            ("math", [("blas", True), ("fft", True)]),
        ]
    )


class ReplayTest(unittest.TestCase):
    def test_first_run_all_miss_then_same_commit_all_hit(self):
        units = (
            ("amd-llvm", "generic"),
            ("blas", "gfx94X-dcgpu"),
            ("fft", "gfx94X-dcgpu"),
        )
        scenarios = [
            ReplayRun("1", "c1", "2026-09-01T00:00:00Z", produced_units=units),
            ReplayRun("2", "c1", "2026-09-01T01:00:00Z", produced_units=units),
        ]
        report = replay(scenarios, topology=_topology(), target_families=TARGETS)
        first, second = report.results
        self.assertEqual(first.cache_hits, 0)  # empty cache
        self.assertEqual(second.cache_hits, second.units)  # same commit, all cached

    def test_cross_commit_reuse_beats_naive_baseline(self):
        units = (
            ("amd-llvm", "generic"),
            ("blas", "gfx94X-dcgpu"),
            ("fft", "gfx94X-dcgpu"),
        )
        # Per-unit fingerprints model real .fprint: unchanged subprojects keep a
        # stable fingerprint across commits; only fft changes at c2.
        fp_c1 = {
            ("amd-llvm", "generic"): "llvm-rev1",
            ("blas", "gfx94X-dcgpu"): "blas-rev1",
            ("fft", "gfx94X-dcgpu"): "fft-rev1",
        }
        fp_c2 = dict(fp_c1)
        fp_c2[("fft", "gfx94X-dcgpu")] = "fft-rev2"  # only fft changed

        scenarios = [
            ReplayRun(
                "1",
                "c1",
                "2026-09-01T00:00:00Z",
                produced_units=units,
                unit_fingerprints=fp_c1,
            ),
            ReplayRun(
                "2",
                "c2",
                "2026-09-02T00:00:00Z",
                produced_units=units,
                unit_fingerprints=fp_c2,
            ),
        ]
        report = replay(scenarios, topology=_topology(), target_families=TARGETS)
        second = report.results[1]
        # c2 is a new commit: naive same-commit baseline reuses nothing.
        self.assertEqual(second.baseline_hits, 0)
        # Fingerprint cache reuses the 2 unchanged units (llvm + blas); fft misses.
        self.assertEqual(second.cache_hits, 2)
        self.assertEqual(second.units, 3)
        # Aggregate: cache strictly beats the naive baseline.
        self.assertGreater(report.cache_hit_rate, report.baseline_hit_rate)

    def test_report_serializes(self):
        scenarios = [
            ReplayRun(
                "1",
                "c1",
                "2026-09-01T00:00:00Z",
                produced_units=(("blas", "gfx94X-dcgpu"),),
            ),
        ]
        report = replay(scenarios, topology=_topology(), target_families=TARGETS)
        d = report.to_dict()
        self.assertIn("cache_hit_rate", d)
        self.assertIn("per_run", d)
        self.assertEqual(len(d["per_run"]), 1)


if __name__ == "__main__":
    unittest.main()
