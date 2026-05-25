# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import baseline_runs

RequiredArtifact = baseline_runs.RequiredArtifact


def _workflow_run(
    run_id: str,
    *,
    status: str = "completed",
    conclusion: str | None = "success",
    head_sha: str | None = None,
) -> dict:
    return {
        "id": run_id,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/ROCm/TheRock/actions/runs/{run_id}",
        "head_sha": head_sha or f"sha-{run_id}",
        "head_branch": "main",
    }


class FakeBackend:
    def __init__(self, artifacts: list[str]):
        self._artifacts = artifacts

    def list_artifacts(self):
        return list(self._artifacts)


class BaselineRunsTest(unittest.TestCase):
    def test_is_successful_workflow_run(self):
        self.assertTrue(baseline_runs.is_successful_workflow_run(_workflow_run("1")))
        self.assertFalse(
            baseline_runs.is_successful_workflow_run(
                _workflow_run("1", conclusion="failure")
            )
        )
        self.assertFalse(
            baseline_runs.is_successful_workflow_run(
                _workflow_run("1", status="in_progress", conclusion=None)
            )
        )

    def test_query_successful_workflow_runs(self):
        with mock.patch.object(
            baseline_runs,
            "gha_send_request",
            return_value={
                "workflow_runs": [
                    _workflow_run("1"),
                    _workflow_run("2"),
                    _workflow_run("3"),
                ]
            },
        ) as mock_send_request:
            runs = baseline_runs.query_successful_workflow_runs(
                github_repository="ROCm/TheRock",
                workflow_name="multi_arch_ci.yml",
                branch="main",
                max_runs=2,
            )

        self.assertEqual([run["id"] for run in runs], ["1", "2"])
        url = mock_send_request.call_args.args[0]
        self.assertIn(
            "/repos/ROCm/TheRock/actions/workflows/multi_arch_ci.yml/runs", url
        )
        self.assertIn("status=success", url)
        self.assertIn("branch=main", url)
        self.assertIn("per_page=2", url)

    def test_validate_required_artifacts_available(self):
        backend = FakeBackend(
            [
                "base_lib_generic.tar.zst",
                "blas_lib_gfx94X-dcgpu.tar.zst",
                "blas_dev_gfx94X-dcgpu.tar.xz",
                "unrelated_lib_generic.tar.zst",
            ]
        )

        availability = baseline_runs.validate_required_artifacts_available(
            backend=backend,
            required_artifacts=[
                RequiredArtifact("base", "generic"),
                RequiredArtifact("blas", "gfx94X-dcgpu"),
            ],
        )

        self.assertTrue(availability.is_valid)
        self.assertEqual(availability.missing_artifacts, ())
        self.assertEqual(
            availability.matched_filenames,
            (
                "base_lib_generic.tar.zst",
                "blas_lib_gfx94X-dcgpu.tar.zst",
                "blas_dev_gfx94X-dcgpu.tar.xz",
            ),
        )

    def test_validate_required_artifacts_available_reports_missing_names(self):
        backend = FakeBackend(["blas_lib_gfx94X-dcgpu.tar.zst"])

        availability = baseline_runs.validate_required_artifacts_available(
            backend=backend,
            required_artifacts=[
                RequiredArtifact("blas", "gfx94X-dcgpu"),
                RequiredArtifact("rand", "gfx94X-dcgpu"),
            ],
        )

        self.assertFalse(availability.is_valid)
        self.assertEqual(
            availability.missing_artifacts,
            (RequiredArtifact("rand", "gfx94X-dcgpu"),),
        )
        self.assertEqual(
            availability.matched_filenames,
            ("blas_lib_gfx94X-dcgpu.tar.zst",),
        )

    def test_validate_required_artifacts_requires_nonempty_requirements(self):
        backend = FakeBackend(["blas_lib_gfx94X-dcgpu.tar.zst"])

        with self.assertRaisesRegex(ValueError, "required_artifacts"):
            baseline_runs.validate_required_artifacts_available(
                backend=backend,
                required_artifacts=[],
            )

        with self.assertRaisesRegex(ValueError, "non-empty"):
            baseline_runs.validate_required_artifacts_available(
                backend=backend,
                required_artifacts=[RequiredArtifact("blas", "")],
            )

    def test_validate_required_artifacts_checks_each_target_family(self):
        backend = FakeBackend(["blas_lib_gfx94X-dcgpu.tar.zst"])

        availability = baseline_runs.validate_required_artifacts_available(
            backend=backend,
            required_artifacts=[
                RequiredArtifact("blas", "gfx94X-dcgpu"),
                RequiredArtifact("blas", "gfx120X-all"),
            ],
        )

        self.assertFalse(availability.is_valid)
        self.assertEqual(
            availability.missing_artifacts,
            (RequiredArtifact("blas", "gfx120X-all"),),
        )

    def test_select_baseline_run_uses_first_successful_run_with_artifacts(self):
        runs = [
            _workflow_run("current"),
            _workflow_run("missing-artifacts"),
            _workflow_run("failed", conclusion="failure"),
            _workflow_run("usable"),
        ]
        artifacts_by_run_id = {
            "missing-artifacts": ["base_lib_generic.tar.zst"],
            "failed": [
                "base_lib_generic.tar.zst",
                "blas_lib_gfx94X-dcgpu.tar.zst",
            ],
            "usable": [
                "base_lib_generic.tar.zst",
                "blas_lib_gfx94X-dcgpu.tar.zst",
            ],
        }

        def backend_factory(workflow_run, github_repository, platform):
            return FakeBackend(artifacts_by_run_id.get(workflow_run["id"], []))

        baseline = baseline_runs.select_baseline_run(
            required_artifacts=[
                RequiredArtifact("base", "generic"),
                RequiredArtifact("blas", "gfx94X-dcgpu"),
            ],
            github_repository="ROCm/TheRock",
            workflow_name="multi_arch_ci.yml",
            branch="main",
            platform="linux",
            exclude_run_ids=["current"],
            workflow_runs=runs,
            backend_factory=backend_factory,
        )

        self.assertIsNotNone(baseline)
        assert baseline is not None
        self.assertEqual(baseline.run_id, "usable")
        self.assertEqual(baseline.head_sha, "sha-usable")
        self.assertEqual(baseline.platform, "linux")
        self.assertEqual(baseline.artifact_availability.missing_artifacts, ())

    def test_select_baseline_run_returns_none_when_no_candidate_is_valid(self):
        runs = [
            _workflow_run("failed", conclusion="failure"),
            _workflow_run("missing-artifacts"),
        ]

        def backend_factory(workflow_run, github_repository, platform):
            return FakeBackend([])

        baseline = baseline_runs.select_baseline_run(
            required_artifacts=[RequiredArtifact("base", "generic")],
            platform="linux",
            workflow_runs=runs,
            backend_factory=backend_factory,
        )

        self.assertIsNone(baseline)


if __name__ == "__main__":
    unittest.main()
