# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Evaluate rocddi reusable-job authorization before runner allocation."""

import ast
import json
import re

import pytest

from workflow_utils import WORKFLOWS_DIR, get_workflow_job, load_workflow


JOBS = ("build", "cts", "hip")
AUTHORIZED_ORIGINS = (
    ("workflow_dispatch", "refs/heads/main"),
    ("workflow_dispatch", "refs/heads/shared/rocddi-cts-dev"),
    ("workflow_dispatch", "refs/heads/users/example/rocddi"),
    ("workflow_dispatch", "refs/tags/rocddi-dev"),
    ("push", "refs/heads/shared/rocddi-cts-dev"),
)


def _evaluate(node: ast.AST, context: dict) -> object:
    """Evaluate the expression syntax used by rocddi job guards and run names."""
    match node:
        case ast.Constant(value=value) if isinstance(value, (str, bool, int)):
            return value
        case ast.Attribute(value=parent, attr=attribute):
            if ast.unparse(node) in context:
                return context[ast.unparse(node)]
            value = _evaluate(parent, context)
            return value.get(attribute) if isinstance(value, dict) else None
        case ast.Subscript(value=parent, slice=index):
            value = _evaluate(parent, context)
            index = _evaluate(index, context)
            return (
                value[index] if isinstance(value, list) and index < len(value) else None
            )
        case ast.BoolOp(op=operator, values=values):
            result = _evaluate(values[0], context)
            for value in values[1:]:
                if isinstance(operator, ast.And) and not result:
                    break
                if isinstance(operator, ast.Or) and result:
                    break
                result = _evaluate(value, context)
            return result
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return not _evaluate(operand, context)
        case ast.Compare(left=left, ops=[operator], comparators=[right]):
            left_value = _evaluate(left, context)
            right_value = _evaluate(right, context)
            # GitHub Actions compares strings without case sensitivity.
            if isinstance(left_value, str) and isinstance(right_value, str):
                left_value = left_value.casefold()
                right_value = right_value.casefold()
            if isinstance(operator, ast.Eq):
                return left_value == right_value
            if isinstance(operator, ast.NotEq):
                return left_value != right_value
        case ast.Call(func=ast.Name(id=name), args=[], keywords=[]):
            return context[f"{name}()"]
        case ast.Call(func=ast.Name(id="fromJSON"), args=[value], keywords=[]):
            return json.loads(_evaluate(value, context))
    raise AssertionError(f"Unsupported job condition syntax: {ast.dump(node)}")


@pytest.fixture(scope="module")
def workflow() -> dict:
    return load_workflow(WORKFLOWS_DIR / "rocddi_dev_test.yml")


def evaluate_expression(
    expression: str,
    context: dict[str, str | bool],
    *,
    success: bool = True,
    cancelled: bool = False,
) -> bool:
    assert expression.startswith("${{") and expression.endswith("}}")
    expression = expression[3:-2].strip()
    context = context | {
        "cancelled()": cancelled,
        "success()": success and not cancelled,
        "failure()": not success and not cancelled,
        "always()": True,
    }
    # GitHub adds success() unless the condition contains a status function.
    if not re.search(r"\b(?:always|cancelled|failure|success)\s*\(", expression):
        if not context["success()"]:
            return False
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    return bool(_evaluate(ast.parse(f"({expression})", mode="eval").body, context))


def job_runs(
    workflow: dict,
    job_name: str,
    *,
    repository: str = "ROCm/TheRock",
    event_name: str = "workflow_dispatch",
    ref: str = "refs/heads/main",
    cancelled: bool = False,
    build_result: str = "success",
    test_runs_on: str = '"gpu-runner"',
    cts_ready: str = "true",
    hip_ready: str = "true",
) -> bool:
    job = get_workflow_job(workflow, job_name)
    # A step-level guard cannot prevent allocating this job's runner.
    expression = job["if"]
    context = {
        "github.repository": repository,
        "github.event_name": event_name,
        "github.ref": ref,
        "inputs.test_runs_on": test_runs_on,
        "needs.build.outputs.cts_ready": cts_ready,
        "needs.build.outputs.hip_ready": hip_ready,
    }
    return evaluate_expression(
        expression, context, success=build_result == "success", cancelled=cancelled
    )


@pytest.mark.parametrize("job_name", JOBS)
@pytest.mark.parametrize("event_name, ref", AUTHORIZED_ORIGINS)
def test_authorized_origins_can_allocate_runners(
    workflow: dict, job_name: str, event_name: str, ref: str
) -> None:
    assert job_runs(workflow, job_name, event_name=event_name, ref=ref)


@pytest.mark.parametrize("job_name", JOBS)
@pytest.mark.parametrize(
    "event_name, ref",
    (
        ("pull_request", "refs/pull/17/merge"),
        ("pull_request", "refs/heads/shared/rocddi-cts-dev"),
        ("pull_request_target", "refs/heads/shared/rocddi-cts-dev"),
        ("workflow_run", "refs/heads/shared/rocddi-cts-dev"),
        ("schedule", "refs/heads/shared/rocddi-cts-dev"),
        ("push", "refs/heads/main"),
        ("push", "refs/heads/users/example/rocddi"),
        ("push", "refs/tags/shared/rocddi-cts-dev"),
    ),
)
def test_other_events_and_push_refs_cannot_allocate_runners(
    workflow: dict, job_name: str, event_name: str, ref: str
) -> None:
    assert not job_runs(workflow, job_name, event_name=event_name, ref=ref)


@pytest.mark.parametrize("job_name", JOBS)
@pytest.mark.parametrize("event_name, ref", AUTHORIZED_ORIGINS)
@pytest.mark.parametrize("repository", ("example/TheRock", "ROCm/rocm-systems"))
def test_foreign_repositories_cannot_allocate_runners(
    workflow: dict, job_name: str, event_name: str, ref: str, repository: str
) -> None:
    assert not job_runs(
        workflow, job_name, repository=repository, event_name=event_name, ref=ref
    )


@pytest.mark.parametrize("job_name", JOBS)
def test_cancelled_runs_do_not_allocate_runners(workflow: dict, job_name: str) -> None:
    assert not job_runs(workflow, job_name, cancelled=True)


@pytest.mark.parametrize("job_name", ("cts", "hip"))
@pytest.mark.parametrize("package_ready", ("false", ""))
def test_native_jobs_require_their_package(
    workflow: dict, job_name: str, package_ready: str
) -> None:
    assert not job_runs(workflow, job_name, **{f"{job_name}_ready": package_ready})


@pytest.mark.parametrize("cts_ready, hip_ready", (("true", "false"), ("false", "true")))
def test_native_package_readiness_is_independent(
    workflow: dict, cts_ready: str, hip_ready: str
) -> None:
    for job_name, package_ready in (("cts", cts_ready), ("hip", hip_ready)):
        assert job_runs(
            workflow, job_name, cts_ready=cts_ready, hip_ready=hip_ready
        ) == (package_ready == "true")


@pytest.mark.parametrize("job_name", ("cts", "hip"))
def test_native_jobs_can_use_ready_packages_after_build_failure(
    workflow: dict, job_name: str
) -> None:
    assert get_workflow_job(workflow, job_name)["needs"] == "build"
    assert job_runs(workflow, job_name, build_result="failure")


def test_host_only_profile_builds_without_allocating_native_runners(
    workflow: dict,
) -> None:
    host_profile = {"test_runs_on": '""', "cts_ready": "true", "hip_ready": "false"}
    assert job_runs(workflow, "build", **host_profile)
    assert not job_runs(workflow, "cts", **host_profile)
    assert not job_runs(workflow, "hip", **host_profile)


@pytest.mark.parametrize("job_name", ("cts", "hip"))
def test_native_jobs_use_the_selected_runner(workflow: dict, job_name: str) -> None:
    assert (
        get_workflow_job(workflow, job_name)["runs-on"]
        == "${{ fromJSON(inputs.test_runs_on) }}"
    )
