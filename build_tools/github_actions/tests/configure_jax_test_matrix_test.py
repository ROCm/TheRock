# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from datetime import date
from pathlib import Path
from unittest import mock
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_jax_test_matrix as matrix_script

# gfx94x is the outer key used to construct workflow pipelines, while
# gfx94X-dcgpu is the inner key, which we use for package names. Workflows pass
# the inner key, but a manually dispatched run may pass either.
GFX94X_SINGLE_GPU = "linux-gfx942-1gpu-ccs-csp-ossci-rocm"
GFX94X_MULTI_GPU = "linux-gfx942-8gpu-ossci-rocm"

SUNDAY = date(2026, 8, 9)
TUESDAY = date(2026, 8, 11)


def subsets(matrix: dict) -> list[str]:
    return [job["test_subset"] for job in matrix["include"]]


def job_for(matrix: dict, subset: str) -> dict:
    for job in matrix["include"]:
        if job["test_subset"] == subset:
            return job
    raise AssertionError(f"no {subset} job in {matrix}")


def runner_for(matrix: dict, subset: str) -> str:
    return job_for(matrix, subset)["test_runs_on"]


class WantsMultiGpuTest(unittest.TestCase):
    """Which sizes are worth a slot in the shared multi-GPU pool."""

    def test_small_never_takes_one(self):
        # It blocks pull requests, so it stays on the 1-GPU runner.
        for day in [SUNDAY, TUESDAY]:
            with self.subTest(day=day):
                self.assertFalse(matrix_script.wants_multi_gpu("small", day))

    def test_medium_takes_one_a_day_a_week(self):
        self.assertTrue(matrix_script.wants_multi_gpu("medium", SUNDAY))
        self.assertFalse(matrix_script.wants_multi_gpu("medium", TUESDAY))

    def test_large_always_takes_one(self):
        for day in [SUNDAY, TUESDAY]:
            with self.subTest(day=day):
                self.assertTrue(matrix_script.wants_multi_gpu("large", day))

    def test_the_weekly_day_is_the_one_named(self):
        self.assertEqual(SUNDAY.weekday(), matrix_script.WEEKLY_MULTI_GPU_WEEKDAY)


class BuildTestMatrixTest(unittest.TestCase):
    def matrix(
        self, target="gfx94X-dcgpu", platform="linux", size="small", today=TUESDAY
    ) -> dict:
        return matrix_script.build_test_matrix(
            target=target, platform=platform, size=size, today=today
        )

    def test_a_pull_request_runs_only_the_single_gpu_job(self):
        matrix = self.matrix(size="small")

        self.assertEqual(subsets(matrix), ["all"])
        self.assertEqual(runner_for(matrix, "all"), GFX94X_SINGLE_GPU)

    def test_a_nightly_off_the_weekly_day(self):
        matrix = self.matrix(size="medium", today=TUESDAY)

        self.assertEqual(subsets(matrix), ["all"])

    def test_a_nightly_on_the_weekly_day_adds_the_multi_gpu_job(self):
        matrix = self.matrix(size="medium", today=SUNDAY)

        self.assertEqual(subsets(matrix), ["all", "multi"])
        self.assertEqual(runner_for(matrix, "multi"), GFX94X_MULTI_GPU)

    def test_a_release_adds_it_whatever_day_it_is(self):
        matrix = self.matrix(size="large", today=TUESDAY)

        self.assertEqual(subsets(matrix), ["all", "multi"])
        self.assertEqual(runner_for(matrix, "all"), GFX94X_SINGLE_GPU)
        self.assertEqual(runner_for(matrix, "multi"), GFX94X_MULTI_GPU)

    def test_the_outer_family_key_resolves_too(self):
        matrix = self.matrix(target="gfx94x", size="large")

        self.assertEqual(runner_for(matrix, "all"), GFX94X_SINGLE_GPU)
        self.assertEqual(runner_for(matrix, "multi"), GFX94X_MULTI_GPU)

    def test_a_family_without_a_multi_gpu_runner_skips_that_job(self):
        # Those tests need several GPUs, and a 1-GPU runner would skip every one
        # of them while reporting a pass.
        matrix = self.matrix(target="gfx1151", platform="windows", size="large")

        self.assertEqual(subsets(matrix), ["all"])
        self.assertEqual(runner_for(matrix, "all"), "windows-gfx1151-gpu-rocm")

    def test_a_family_without_any_test_runner_runs_nothing(self):
        # A family that has a build but no test hardware carries an empty label,
        # so this is a configuration a run has to survive, loudly.
        with contextlib.redirect_stdout(io.StringIO()) as out:
            matrix = self.matrix(target="gfx90a", platform="windows", size="large")

        self.assertEqual(matrix["include"], [])
        self.assertIn("::warning::", out.getvalue())

    def test_an_unknown_family_is_an_error(self):
        with self.assertRaises(ValueError):
            self.matrix(target="gfx-not-a-family")

    def test_only_small_narrows_the_single_gpu_job(self):
        small = job_for(self.matrix(size="small"), "all")
        medium = job_for(self.matrix(size="medium"), "all")

        self.assertEqual(small["test_list"], matrix_script.SMALL_TEST_LIST)
        self.assertEqual(small["test_timeout_minutes"], 45)
        self.assertEqual(medium["test_list"], "")
        self.assertEqual(medium["test_timeout_minutes"], 120)

    def test_the_multi_gpu_job_runs_all_of_its_subset(self):
        # There is no reduced form of the multi-accelerator script, so a size
        # only decides whether this job runs, not what it runs.
        job = job_for(self.matrix(size="large"), "multi")

        self.assertEqual(job["test_list"], "")
        self.assertEqual(job["test_timeout_minutes"], 90)


class OutputsTest(unittest.TestCase):
    def outputs(self, argv: list[str], today=TUESDAY) -> dict[str, str]:
        with open(os.environ["GITHUB_OUTPUT"], "w") as f:
            f.truncate()
        with mock.patch.object(matrix_script, "today_utc", lambda: today):
            matrix_script.main(argv)
        written = Path(os.environ["GITHUB_OUTPUT"]).read_text()
        return dict(line.split("=", 1) for line in written.splitlines() if "=" in line)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.dict(
            os.environ, {"GITHUB_OUTPUT": os.path.join(self.tmp.name, "out")}
        )
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_matrix_a_workflow_consumes(self):
        outputs = self.outputs(["--target", "gfx94X-dcgpu", "--test-size", "large"])

        self.assertEqual(outputs["enabled"], "true")
        self.assertEqual(
            json.loads(outputs["matrix"]),
            {
                "include": [
                    {
                        "test_subset": "all",
                        "test_runs_on": GFX94X_SINGLE_GPU,
                        "test_timeout_minutes": 120,
                        "test_list": "",
                    },
                    {
                        "test_subset": "multi",
                        "test_runs_on": GFX94X_MULTI_GPU,
                        "test_timeout_minutes": 90,
                        "test_list": "",
                    },
                ]
            },
        )

    def test_a_pull_request_gets_one_job_on_the_single_gpu_runner(self):
        outputs = self.outputs(["--target", "gfx94X-dcgpu", "--test-size", "small"])

        self.assertEqual(
            json.loads(outputs["matrix"]),
            {
                "include": [
                    {
                        "test_subset": "all",
                        "test_runs_on": GFX94X_SINGLE_GPU,
                        "test_timeout_minutes": 45,
                        "test_list": matrix_script.SMALL_TEST_LIST,
                    }
                ]
            },
        )

    def test_the_nightly_picks_up_the_multi_gpu_job_on_its_day(self):
        weekday = self.outputs(
            ["--target", "gfx94X-dcgpu", "--test-size", "medium"], today=TUESDAY
        )
        weekly = self.outputs(
            ["--target", "gfx94X-dcgpu", "--test-size", "medium"], today=SUNDAY
        )

        self.assertEqual(len(json.loads(weekday["matrix"])["include"]), 1)
        self.assertEqual(len(json.loads(weekly["matrix"])["include"]), 2)

    def test_a_nightly_off_the_day_says_how_to_get_those_tests(self):
        # A nightly re-run onto another day loses the week of multi-accelerator
        # coverage, so the run says so and names the way back.
        argv = ["--target", "gfx94X-dcgpu", "--test-size", "medium"]
        with contextlib.redirect_stdout(io.StringIO()) as weekday:
            self.outputs(argv, today=TUESDAY)
        with contextlib.redirect_stdout(io.StringIO()) as weekly:
            self.outputs(argv, today=SUNDAY)

        self.assertIn("::notice::", weekday.getvalue())
        self.assertIn("Sunday", weekday.getvalue())
        self.assertNotIn("::notice::", weekly.getvalue())

    def test_a_size_is_required_and_checked(self):
        # Defaulting it would quietly drop the multi-accelerator job.
        for argv in [
            ["--target", "gfx94X-dcgpu"],
            ["--target", "gfx94X-dcgpu", "--test-size", "meduim"],
        ]:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                self.outputs(argv)


class SmallTestSelectionTest(unittest.TestCase):
    def test_pr_small_selection_includes_rocprofiler_smoke(self):
        repo_root = Path(__file__).resolve().parents[3]
        listed = (repo_root / matrix_script.SMALL_TEST_LIST).read_text().splitlines()
        paths = {line.partition("#")[0].strip() for line in listed if line.strip()}
        self.assertIn("tests/logging_test.py", paths)
        self.assertIn("tests/profiler_test.py", paths)


if __name__ == "__main__":
    unittest.main()
