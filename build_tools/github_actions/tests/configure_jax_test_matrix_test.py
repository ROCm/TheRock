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


def runner_for(matrix: dict, subset: str) -> str:
    for job in matrix["include"]:
        if job["test_subset"] == subset:
            return job["test_runs_on"]
    raise AssertionError(f"no {subset} job in {matrix}")


class ResolveScopeTest(unittest.TestCase):
    def test_a_prerelease_tests_everything_whatever_day_it_is(self):
        for day in [SUNDAY, TUESDAY]:
            with self.subTest(day=day):
                self.assertEqual(
                    matrix_script.resolve_scope("auto", "prerelease", day), "full"
                )

    def test_a_nightly_tests_everything_one_day_a_week(self):
        # The multi-GPU pool is small and that subset moves slowly, so a slot
        # every night is not worth what it finds.
        self.assertEqual(matrix_script.resolve_scope("auto", "nightly", SUNDAY), "full")
        self.assertEqual(
            matrix_script.resolve_scope("auto", "nightly", TUESDAY), "short"
        )

    def test_release_types_that_run_per_change(self):
        # An 8-GPU queue slot per job is too much for these.
        for release_type in ["ci", "dev"]:
            for day in [SUNDAY, TUESDAY]:
                with self.subTest(release_type=release_type, day=day):
                    self.assertEqual(
                        matrix_script.resolve_scope("auto", release_type, day), "short"
                    )

    def test_an_explicit_scope_wins(self):
        # So a workflow, or a person, can ask for either on any day.
        self.assertEqual(
            matrix_script.resolve_scope("short", "prerelease", SUNDAY), "short"
        )
        self.assertEqual(matrix_script.resolve_scope("full", "ci", TUESDAY), "full")

    def test_the_weekly_day_is_the_one_named(self):
        self.assertEqual(SUNDAY.weekday(), matrix_script.WEEKLY_FULL_WEEKDAY)


class BuildTestMatrixTest(unittest.TestCase):
    def matrix(self, target="gfx94X-dcgpu", platform="linux", scope="short") -> dict:
        return matrix_script.build_test_matrix(
            target=target, platform=platform, scope=scope
        )

    def test_short_scope_runs_only_the_single_gpu_job(self):
        # On one GPU the suite is the single-accelerator tests.
        matrix = self.matrix(scope="short")

        self.assertEqual(subsets(matrix), ["all"])
        self.assertEqual(runner_for(matrix, "all"), GFX94X_SINGLE_GPU)

    def test_full_scope_adds_the_multi_accelerator_subset(self):
        matrix = self.matrix(scope="full")

        self.assertEqual(subsets(matrix), ["all", "multi"])
        self.assertEqual(runner_for(matrix, "all"), GFX94X_SINGLE_GPU)
        self.assertEqual(runner_for(matrix, "multi"), GFX94X_MULTI_GPU)

    def test_the_outer_family_key_resolves_too(self):
        matrix = self.matrix(target="gfx94x", scope="full")

        self.assertEqual(runner_for(matrix, "all"), GFX94X_SINGLE_GPU)
        self.assertEqual(runner_for(matrix, "multi"), GFX94X_MULTI_GPU)

    def test_a_family_without_a_multi_gpu_runner_skips_that_subset(self):
        # Those tests need several GPUs, and a 1-GPU runner would skip every one
        # of them while reporting a pass.
        matrix = self.matrix(target="gfx1151", platform="windows", scope="full")

        self.assertEqual(subsets(matrix), ["all"])
        self.assertEqual(runner_for(matrix, "all"), "windows-gfx1151-gpu-rocm")

    def test_a_family_without_any_test_runner_runs_nothing(self):
        # A family that has a build but no test hardware carries an empty label,
        # so this is a configuration a run has to survive, loudly.
        with contextlib.redirect_stdout(io.StringIO()) as out:
            matrix = self.matrix(target="gfx90a", platform="windows", scope="full")

        self.assertEqual(matrix["include"], [])
        self.assertIn("::warning::", out.getvalue())

    def test_an_unknown_family_is_an_error(self):
        with self.assertRaises(ValueError):
            self.matrix(target="gfx-not-a-family")


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
        outputs = self.outputs(
            ["--target", "gfx94X-dcgpu", "--release-type", "prerelease"]
        )

        self.assertEqual(outputs["enabled"], "true")
        self.assertEqual(
            json.loads(outputs["matrix"]),
            {
                "include": [
                    {"test_subset": "all", "test_runs_on": GFX94X_SINGLE_GPU},
                    {"test_subset": "multi", "test_runs_on": GFX94X_MULTI_GPU},
                ]
            },
        )

    def test_a_pull_request_gets_one_job_on_the_single_gpu_runner(self):
        outputs = self.outputs(["--target", "gfx94X-dcgpu", "--release-type", "ci"])

        self.assertEqual(
            json.loads(outputs["matrix"]),
            {"include": [{"test_subset": "all", "test_runs_on": GFX94X_SINGLE_GPU}]},
        )

    def test_a_nightly_picks_up_the_multi_gpu_job_on_its_day(self):
        weekday = self.outputs(
            ["--target", "gfx94X-dcgpu", "--release-type", "nightly"], today=TUESDAY
        )
        weekly = self.outputs(
            ["--target", "gfx94X-dcgpu", "--release-type", "nightly"], today=SUNDAY
        )

        self.assertEqual(len(json.loads(weekday["matrix"])["include"]), 1)
        self.assertEqual(len(json.loads(weekly["matrix"])["include"]), 2)

    def test_a_rerun_repeats_the_matrix_of_the_run_it_repeats(self):
        # The workflow passes the run's creation date, so re-running a Sunday
        # nightly on a Tuesday still takes the multi-GPU runner.
        outputs = self.outputs(
            [
                "--target",
                "gfx94X-dcgpu",
                "--release-type",
                "nightly",
                "--run-date",
                SUNDAY.isoformat(),
            ],
            today=TUESDAY,
        )

        self.assertEqual(len(json.loads(outputs["matrix"])["include"]), 2)

    def test_an_unknown_release_type_is_rejected(self):
        # Defaulting it would quietly drop the multi-accelerator job.
        with self.assertRaises(SystemExit):
            self.outputs(["--target", "gfx94X-dcgpu", "--release-type", "nighlty"])


if __name__ == "__main__":
    unittest.main()
