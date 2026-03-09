# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

import pytest

from workflow_summary import (
    evaluate_results,
    main,
    parse_needs_json,
)

# ---------------------------------------------------------------------------
# Fixtures: realistic needs JSON blobs
# ---------------------------------------------------------------------------

ALL_SUCCESS = {
    "setup": {"result": "success", "outputs": {}},
    "linux_build_and_test": {"result": "success", "outputs": {}},
    "windows_build_and_test": {"result": "success", "outputs": {}},
}

ONE_FAILURE = {
    "setup": {"result": "success", "outputs": {}},
    "linux_build_and_test": {"result": "failure", "outputs": {}},
    "windows_build_and_test": {"result": "success", "outputs": {}},
}

FAILURE_WITH_CONTINUE_ON_ERROR = {
    "setup": {"result": "success", "outputs": {}},
    "linux_build_and_test": {
        "result": "failure",
        "outputs": {"continue_on_error": "true"},
    },
    "windows_build_and_test": {"result": "success", "outputs": {}},
}

ONE_CANCELLED = {
    "setup": {"result": "success", "outputs": {}},
    "linux_build_and_test": {"result": "cancelled", "outputs": {}},
}

ONE_SKIPPED = {
    "setup": {"result": "success", "outputs": {}},
    "windows_build_and_test": {"result": "skipped", "outputs": {}},
}


# ---------------------------------------------------------------------------
# parse_needs_json
# ---------------------------------------------------------------------------


class TestParseNeedsJson:
    def test_all_success(self):
        jobs = parse_needs_json(json.dumps(ALL_SUCCESS))
        assert len(jobs) == 3
        assert all(j.result == "success" for j in jobs)
        assert all(not j.continue_on_error for j in jobs)

    def test_continue_on_error_parsed(self):
        jobs = parse_needs_json(json.dumps(FAILURE_WITH_CONTINUE_ON_ERROR))
        by_name = {j.name: j for j in jobs}
        assert by_name["linux_build_and_test"].continue_on_error is True
        assert by_name["setup"].continue_on_error is False

    def test_missing_outputs_key(self):
        """Jobs with no outputs key should still parse (continue_on_error=False)."""
        needs = {"build": {"result": "success"}}
        jobs = parse_needs_json(json.dumps(needs))
        assert len(jobs) == 1
        assert jobs[0].continue_on_error is False

    def test_null_outputs(self):
        """Jobs with null outputs should still parse."""
        needs = {"build": {"result": "success", "outputs": None}}
        jobs = parse_needs_json(json.dumps(needs))
        assert len(jobs) == 1
        assert jobs[0].continue_on_error is False

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_needs_json("not json")

    def test_non_dict_top_level_raises(self):
        with pytest.raises(AssertionError, match="Expected a JSON object"):
            parse_needs_json("[]")

    def test_non_dict_job_raises(self):
        with pytest.raises(AssertionError, match="JSON object for job"):
            parse_needs_json(json.dumps({"build": "not a dict"}))


# ---------------------------------------------------------------------------
# evaluate_results
# ---------------------------------------------------------------------------


class TestEvaluateResults:
    def test_all_success(self):
        jobs = parse_needs_json(json.dumps(ALL_SUCCESS))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 0
        assert len(ok) == 3

    def test_one_failure(self):
        jobs = parse_needs_json(json.dumps(ONE_FAILURE))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 1
        assert failed[0].name == "linux_build_and_test"
        assert len(ok) == 2

    def test_continue_on_error_not_failed(self):
        jobs = parse_needs_json(json.dumps(FAILURE_WITH_CONTINUE_ON_ERROR))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 0
        assert len(ok) == 3

    def test_cancelled_is_failure(self):
        jobs = parse_needs_json(json.dumps(ONE_CANCELLED))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 1
        assert failed[0].name == "linux_build_and_test"
        assert failed[0].result == "cancelled"

    def test_skipped_is_ok(self):
        jobs = parse_needs_json(json.dumps(ONE_SKIPPED))
        failed, ok = evaluate_results(jobs)
        assert len(failed) == 0
        assert len(ok) == 2

    def test_empty_needs(self):
        failed, ok = evaluate_results([])
        assert len(failed) == 0
        assert len(ok) == 0


# ---------------------------------------------------------------------------
# main (integration)
# ---------------------------------------------------------------------------


class TestMain:
    def test_all_success_returns_zero(self, capsys):
        rc = main(["--needs-json", json.dumps(ALL_SUCCESS)])
        assert rc == 0
        assert "succeeded" in capsys.readouterr().out

    def test_failure_returns_one(self, capsys):
        rc = main(["--needs-json", json.dumps(ONE_FAILURE)])
        assert rc == 1
        assert "failed" in capsys.readouterr().out

    def test_continue_on_error_returns_zero(self, capsys):
        rc = main(["--needs-json", json.dumps(FAILURE_WITH_CONTINUE_ON_ERROR)])
        assert rc == 0
        assert "succeeded" in capsys.readouterr().out
