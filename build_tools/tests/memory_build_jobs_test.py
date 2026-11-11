#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resource budgets for memory-heavy nested CSAN builds."""

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import memory_build_jobs as jobs


class MemoryBuildJobsTest(unittest.TestCase):
    def test_observed_machine_budgets(self):
        # 32 GiB/job exceeds the observed 19.6 GiB link; half stays reserved.
        for memory_gib, cpus, expected in ((256, 64, 4), (240, 64, 3), (1536, 192, 24)):
            with self.subTest(memory_gib=memory_gib):
                self.assertEqual(
                    jobs.choose_build_jobs(jobs.Resources(memory_gib * jobs.GIB, cpus)),
                    expected,
                )

    def test_memory_and_cpu_caps(self):
        self.assertEqual(jobs.choose_build_jobs(jobs.Resources(2048 * jobs.GIB, 2)), 2)
        self.assertEqual(jobs.choose_build_jobs(jobs.Resources(None, 64)), 1)
        for memory in (0, -1, 32 * jobs.GIB):
            self.assertEqual(jobs.choose_build_jobs(jobs.Resources(memory, 64)), 1)
        for quota, period, expected in (
            ("250000", "100000", 2),
            ("50000", "100000", 1),
            ("max", "100000", 64),
            ("-1", "100000", 64),
            ("200000", "0", 64),
            (None, None, 64),
        ):
            with self.subTest(quota=quota):
                self.assertEqual(jobs.available_cpus(64, quota, period), expected)
        self.assertEqual(jobs.available_cpus(2, "6400000", "100000"), 2)
        self.assertEqual(jobs.available_cpus(None, None, None), 1)

    def test_cgroup_memory_headroom_and_unlimited_limits(self):
        self.assertEqual(jobs.cgroup_memory_available("100", "30"), 70)
        self.assertEqual(jobs.cgroup_memory_available("100", "120"), 0)
        self.assertEqual(jobs.cgroup_memory_available("100", None), 0)
        for limit in ("max", "9223372036854771712", "-1", None):
            with self.subTest(limit=limit):
                self.assertIsNone(jobs.cgroup_memory_available(limit, "100"))

    def test_host_and_container_bounds(self):
        host = "MemTotal: 999999 kB\nMemAvailable: 100 kB\n"
        self.assertEqual(jobs.available_memory(host, 80 * 1024), 80 * 1024)
        self.assertEqual(jobs.available_memory(host, 200 * 1024), 100 * 1024)
        self.assertEqual(jobs.available_memory(host, None), 100 * 1024)
        self.assertEqual(jobs.available_memory("", 1024), 1024)
        self.assertIsNone(jobs.available_memory("MemAvailable: invalid kB", None))

    def test_detect_v2_and_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = root / "proc"
            cgroup = root / "cgroup"
            proc.mkdir()
            cgroup.mkdir()
            (proc / "meminfo").write_text(f"MemAvailable: {1024 * 1024 * 1024} kB\n")
            with mock.patch.object(
                jobs.os, "cpu_count", return_value=64
            ), mock.patch.object(
                jobs.os, "sched_getaffinity", return_value=set(range(64)), create=True
            ):
                (cgroup / "memory.max").write_text(str(256 * jobs.GIB))
                (cgroup / "memory.current").write_text(str(16 * jobs.GIB))
                (cgroup / "cpu.max").write_text("200000 100000")
                resources = jobs.detect_resources(proc, cgroup)
                self.assertEqual(resources, jobs.Resources(240 * jobs.GIB, 2))
                self.assertEqual(jobs.choose_build_jobs(resources), 2)
                for name in ("memory.max", "memory.current", "cpu.max"):
                    (cgroup / name).unlink()
                (cgroup / "memory").mkdir()
                (cgroup / "cpu").mkdir()
                (cgroup / "memory/memory.limit_in_bytes").write_text(
                    "9223372036854771712"
                )
                (cgroup / "memory/memory.usage_in_bytes").write_text("123")
                (cgroup / "cpu/cpu.cfs_quota_us").write_text("300000")
                (cgroup / "cpu/cpu.cfs_period_us").write_text("100000")
                self.assertEqual(
                    jobs.detect_resources(proc, cgroup),
                    jobs.Resources(1024 * jobs.GIB, 3),
                )

    def test_github_environment_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "github_env"
            stdout = io.StringIO()
            with mock.patch.object(
                jobs,
                "detect_resources",
                return_value=jobs.Resources(256 * jobs.GIB, 64),
            ), mock.patch.dict(
                os.environ, {"GITHUB_ENV": str(env_file)}
            ), contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(jobs.main(["--github-env"]), 0)
            self.assertEqual(env_file.read_text(), "CMAKE_BUILD_PARALLEL_LEVEL=4\n")
            self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
