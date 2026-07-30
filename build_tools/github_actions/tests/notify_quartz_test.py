# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for notify_quartz reporting-workflow path attribution."""

import argparse
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import notify_quartz
from notify_quartz import (
    _GithubApiResponse,
    _build_payload,
    _load_payload,
    _normalize_reporting_path,
    _step_outputs_to_needs_shape,
)


class NormalizeReportingPathTest(unittest.TestCase):
    def test_empty_is_passthrough_signal(self):
        self.assertEqual(_normalize_reporting_path(""), "")
        self.assertEqual(_normalize_reporting_path("   "), "")

    def test_bare_filename(self):
        self.assertEqual(
            _normalize_reporting_path("multi_arch_build_portable_linux.yml"),
            ".github/workflows/multi_arch_build_portable_linux.yml",
        )

    def test_full_workflow_ref_strips_owner_and_ref(self):
        self.assertEqual(
            _normalize_reporting_path(
                "ROCm/TheRock/.github/workflows/build_tarballs.yml@refs/heads/main"
            ),
            ".github/workflows/build_tarballs.yml",
        )

    def test_plain_path_passthrough(self):
        self.assertEqual(
            _normalize_reporting_path(".github/workflows/x.yml"),
            ".github/workflows/x.yml",
        )


def _run_obj() -> dict:
    return {
        "id": 999,
        "name": "Multi-Arch Release",
        "path": ".github/workflows/multi_arch_release.yml",
        "workflow_id": 1,
        "status": "in_progress",
    }


class BuildPayloadPathOverrideTest(unittest.TestCase):
    def _build(self, reporting_workflow: str) -> dict:
        with (
            mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "999"}),
            mock.patch.object(
                notify_quartz,
                "_github_api_request",
                return_value=_GithubApiResponse(body=_run_obj(), headers={}),
            ),
        ):
            return _build_payload(
                token="t",
                repo="ROCm/rockrel",
                embedded_inputs={},
                captured_outputs={},
                run_conclusion="",
                run_phase="started",
                reporting_workflow=reporting_workflow,
            )

    def test_override_sets_child_path_but_keeps_shared_run_id(self):
        payload = self._build("multi_arch_build_portable_linux.yml")
        wr = payload["workflow_run"]
        self.assertEqual(
            wr["path"], ".github/workflows/multi_arch_build_portable_linux.yml"
        )
        # run_id stays the shared entry run -- it links the leaf to its parent.
        self.assertEqual(wr["id"], 999)

    def test_empty_reporting_keeps_api_path(self):
        payload = self._build("")
        self.assertEqual(
            payload["workflow_run"]["path"],
            ".github/workflows/multi_arch_release.yml",
        )


class StepOutputsToNeedsShapeTest(unittest.TestCase):
    """The folded-completed case: `${{ toJSON(steps) }}` -> `toJSON(needs)`."""

    def test_collapses_steps_into_single_job_entry(self):
        step_outputs = {
            "build": {"outcome": "success", "conclusion": "success", "outputs": {}},
            "publish": {
                "outcome": "success",
                "conclusion": "success",
                "outputs": {"artifact_url": "https://example/x"},
            },
        }
        shaped = _step_outputs_to_needs_shape(
            step_outputs, job_name="build_job", run_conclusion="success"
        )
        self.assertEqual(
            shaped,
            {
                "build_job": {
                    "result": "success",
                    "outputs": {"artifact_url": "https://example/x"},
                }
            },
        )

    def test_result_comes_from_run_conclusion_not_step_conclusions(self):
        step_outputs = {
            "flaky": {"outcome": "failure", "conclusion": "success", "outputs": {}},
        }
        shaped = _step_outputs_to_needs_shape(
            step_outputs, job_name="j", run_conclusion="failure"
        )
        self.assertEqual(shaped["j"]["result"], "failure")

    def test_last_step_wins_on_output_key_collision(self):
        step_outputs = {
            "a": {"outputs": {"k": "first"}},
            "b": {"outputs": {"k": "second"}},
        }
        shaped = _step_outputs_to_needs_shape(
            step_outputs, job_name="j", run_conclusion="success"
        )
        self.assertEqual(shaped["j"]["outputs"]["k"], "second")

    def test_missing_or_null_outputs_tolerated(self):
        step_outputs = {
            "a": {"outcome": "success"},
            "b": {"outputs": None},
            "c": "not-a-dict",
        }
        shaped = _step_outputs_to_needs_shape(
            step_outputs, job_name="j", run_conclusion="success"
        )
        self.assertEqual(shaped, {"j": {"result": "success", "outputs": {}}})

    def test_empty_steps_yield_empty_outputs(self):
        shaped = _step_outputs_to_needs_shape(
            {}, job_name="j", run_conclusion="cancelled"
        )
        self.assertEqual(shaped, {"j": {"result": "cancelled", "outputs": {}}})


class LoadPayloadStepOutputsTest(unittest.TestCase):
    """`_load_payload` folds --step-outputs into `captured_outputs`."""

    def _args(self, **overrides) -> argparse.Namespace:
        base = dict(
            embedded_inputs="",
            captured_outputs="",
            step_outputs="",
            job_name="",
            run_conclusion="",
            run_phase="completed",
            reporting_workflow="",
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def _load(self, args: argparse.Namespace) -> dict:
        with (
            mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "999"}),
            mock.patch.object(
                notify_quartz,
                "_github_api_request",
                return_value=_GithubApiResponse(body=_run_obj(), headers={}),
            ),
            mock.patch.object(notify_quartz, "_fetch_jobs", return_value=[]),
        ):
            return _load_payload(args, token="t", repo="ROCm/rockrel")

    def test_step_outputs_populate_captured_outputs(self):
        step_outputs = json.dumps(
            {"publish": {"outputs": {"url": "https://example/x"}}}
        )
        args = self._args(
            step_outputs=step_outputs,
            job_name="build_job",
            run_conclusion="success",
        )
        payload = self._load(args)
        self.assertEqual(
            payload["workflow_run"]["captured_outputs"],
            {"build_job": {"result": "success", "outputs": {"url": "https://example/x"}}},
        )

    def test_step_outputs_take_precedence_over_captured_outputs(self):
        # `main()` guards against passing both, but `_load_payload` itself
        # deterministically prefers step-outputs when present.
        args = self._args(
            step_outputs=json.dumps({"a": {"outputs": {}}}),
            captured_outputs=json.dumps({"other_job": {"result": "failure"}}),
            job_name="build_job",
            run_conclusion="success",
        )
        payload = self._load(args)
        self.assertEqual(
            payload["workflow_run"]["captured_outputs"],
            {"build_job": {"result": "success", "outputs": {}}},
        )


class MainStepOutputsValidationTest(unittest.TestCase):
    """`main()` guards + job-name fallback for the folded-completed path."""

    def _run_main(self, argv, env=None) -> int:
        full_env = {"GITHUB_REPOSITORY": "ROCm/rockrel", "GITHUB_RUN_ID": "999"}
        full_env.update(env or {})
        captured = {}

        def _fake_load(args, token, repo):
            captured["job_name"] = args.job_name
            captured["run_conclusion"] = args.run_conclusion
            return {"event_type": "x", "repository": repo, "workflow_run": {}}

        with (
            mock.patch.dict(os.environ, full_env, clear=False),
            mock.patch.object(notify_quartz, "_load_payload", side_effect=_fake_load),
            mock.patch.object(notify_quartz, "dispatch_to_quartz", return_value=None),
        ):
            rc = notify_quartz.main(argv)
        self._captured = captured
        return rc

    def test_step_and_captured_outputs_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run_main(
                [
                    "--token", "t",
                    "--run-phase", "completed",
                    "--run-conclusion", "success",
                    "--step-outputs", json.dumps({"a": {"outputs": {}}}),
                    "--captured-outputs", json.dumps({"j": {"result": "success"}}),
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_step_outputs_require_run_conclusion(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run_main(
                [
                    "--token", "t",
                    "--run-phase", "completed",
                    "--step-outputs", json.dumps({"a": {"outputs": {}}}),
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_job_name_falls_back_to_github_job_env(self):
        rc = self._run_main(
            [
                "--token", "t",
                "--run-phase", "completed",
                "--run-conclusion", "success",
                "--step-outputs", json.dumps({"a": {"outputs": {}}}),
            ],
            env={"GITHUB_JOB": "build_linux"},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(self._captured["job_name"], "build_linux")

    def test_job_name_falls_back_to_reporting_workflow_basename(self):
        rc = self._run_main(
            [
                "--token", "t",
                "--run-phase", "completed",
                "--run-conclusion", "success",
                "--step-outputs", json.dumps({"a": {"outputs": {}}}),
                "--reporting-workflow", "multi_arch_build_portable_linux.yml",
            ],
            env={"GITHUB_JOB": ""},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            self._captured["job_name"], "multi_arch_build_portable_linux"
        )

    def test_explicit_job_name_wins_over_env(self):
        rc = self._run_main(
            [
                "--token", "t",
                "--run-phase", "completed",
                "--run-conclusion", "success",
                "--step-outputs", json.dumps({"a": {"outputs": {}}}),
                "--job-name", "explicit_job",
            ],
            env={"GITHUB_JOB": "runner_job"},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(self._captured["job_name"], "explicit_job")


if __name__ == "__main__":
    unittest.main()
