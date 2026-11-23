# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ast
from collections import Counter
import json
from pathlib import Path
import re
import sys

import pytest

from workflow_utils import WORKFLOWS_DIR, load_workflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import configure_multi_arch_ci as ci
import fetch_test_configurations as configurations


FAMILIES = [
    ("gfx90a", "gfx90a"),
    ("gfx94X-dcgpu", "gfx942"),
    ("gfx950-dcgpu", "gfx950"),
]


def _dispatch_inputs(job: dict) -> dict[str, str]:
    step = next(
        step["with"]
        for step in job["steps"]
        if step.get("with", {}).get("workflow") == "test_artifacts.yml"
    )
    return json.loads(step["inputs"])


def _family_scope(value: str | None, *, has_generic_policy: bool = True) -> str | None:
    if value and value.startswith("${{"):
        assert value.endswith(
            ".generic_test_label_overrides != null && 'family' || 'all' }}"
        )
        return "family" if has_generic_policy else "all"
    return value


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scope: str | None = None,
    family: str = "",
    target: str = "",
    platform: str = "linux",
    **env: str,
) -> tuple[list[dict], dict | None]:
    output: dict[str, str] = {}
    inputs = {
        "AMDGPU_FAMILIES": family,
        "AMDGPU_TARGETS": target,
        "BUILD_VARIANT": "release",
        "CI_CONFIG_PATH": "",
        "GENERIC_TEST_LABEL_OVERRIDES": "",
        "GITHUB_ACTOR": "",
        "GITHUB_EVENT_NAME": "",
        "PROJECTS_TO_TEST": "rocgdb-cpu,miopen-dbsync",
        "SKIP_STAGES": "",
        "TEST_LABELS": "[]",
        "TEST_RUNS_ON": "",
        "TEST_RUNS_ON_CPU": "linux-cpu",
        "TEST_TYPE": "standard",
        **env,
    }
    argv = ["fetch_test_configurations.py", f"--platform={platform}"]
    if scope is not None:
        argv += ["--test-scope", scope]
    with monkeypatch.context() as patch:
        for key, value in inputs.items():
            patch.setenv(key, value)
        patch.setattr(sys, "argv", argv)
        patch.setattr(configurations, "gha_set_output", output.update)
        configurations.run()
    return json.loads(output["components"]), json.loads(output["sanity_component"])


@pytest.mark.parametrize(
    "workflow, generic_job, family_jobs",
    [
        (
            "multi_arch_ci_linux.yml",
            "test_artifacts_generic",
            {"test_artifacts_per_family": FAMILIES},
        ),
        (
            "multi_arch_ci_linux.yml",
            "test_artifacts_generic",
            {"test_artifacts_per_family": FAMILIES[:1]},
        ),
        (
            "multi_arch_ci_linux.yml",
            "test_artifacts_generic",
            {"test_artifacts_per_family": FAMILIES[::-1]},
        ),
        (
            "multi_arch_release_linux.yml",
            "trigger_test_artifacts_generic",
            {"trigger_test_artifacts_per_family": FAMILIES},
        ),
        (
            "compiler_bump.yml",
            "trigger-linux-generic-tests",
            {
                "trigger-linux-tests": FAMILIES[1:2],
                "trigger-linux-tests-extra": FAMILIES[::2],
            },
        ),
    ],
    ids=["ci-three", "ci-single", "ci-reversed", "release", "compiler"],
)
def test_linux_test_plan_runs_generic_cpu_tests_once(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    generic_job: str,
    family_jobs: dict[str, list[tuple[str, str]]],
) -> None:
    """Drive the generator with the scope and labels from the parent jobs."""
    jobs = load_workflow(WORKFLOWS_DIR / workflow)["jobs"]
    components = []
    projects = "rocgdb-cpu,miopen-dbsync,hipfile"
    for job_name, families in family_jobs.items():
        family_call = jobs[job_name]
        if "steps" in family_call:
            family_inputs = _dispatch_inputs(family_call)
        else:
            family_inputs = family_call["with"]
        for family, target in families:
            emitted, _ = _configure(
                monkeypatch,
                scope=_family_scope(family_inputs.get("test_scope")),
                family=family,
                target=target,
                PROJECTS_TO_TEST=projects,
            )
            dbsync = next(c for c in emitted if c["job_name"] == "miopen-dbsync")
            assert dbsync["expect_failure"] is (target == "gfx90a")
            components.extend(emitted)
    if generic_job in jobs:
        job = jobs[generic_job]
        generic_inputs = job.get("with") or _dispatch_inputs(job)
        assert "sanity_check_only_for_family" not in generic_inputs
        emitted, sanity = _configure(
            monkeypatch,
            scope=generic_inputs["test_scope"],
            PROJECTS_TO_TEST=projects,
        )
        assert sanity is None
        components.extend(emitted)

    assert Counter(c["job_name"] for c in components) == {
        "rocgdb-cpu": 1,
        "hipfile": 1,
        "miopen-dbsync": sum(len(families) for families in family_jobs.values()),
    }
    for component in components:
        assert component["test_runner"] == "linux-cpu"
        assert component["test_type"] == "standard"
        assert component["total_shards"] == 1
        assert component["shard_arr"] == [1]


def _linux_ci_plan(monkeypatch: pytest.MonkeyPatch, **inputs: object) -> ci.BuildConfig:
    monkeypatch.setenv("CI_CONFIG_PATH", "")
    monkeypatch.setattr(ci, "get_cpu_test_runner", lambda platform: "linux-cpu")
    plan = ci.configure(
        ci.CIInputs(
            run_id="12345",
            event_name="pull_request",
            commit_ref="test",
            base_ref=None,
            build_variant="release",
            build_python_packages=False,
            build_pytorch=False,
            build_native_linux=False,
            build_native_windows=False,
            **inputs,
        ),
        ci.GitContext(),
    )
    assert plan.builds.linux is not None
    return plan.builds.linux


@pytest.mark.parametrize(
    "families, overrides, global_label, expected",
    [
        (["gfx94x"], {"gfx94x": ["test:rocgdb"]}, "test:rocblas", {"rocgdb-cpu": 1}),
        (
            ["gfx94x", "gfx90a"],
            {"gfx94x": ["test:rocgdb"], "gfx90a": ["test:rocgdb"]},
            "test:hipfile",
            {"rocgdb-cpu": 1},
        ),
        (
            ["gfx94x", "gfx90a"],
            {"gfx94x": ["test:rocgdb"]},
            "test:hipfile",
            {"rocgdb-cpu": 1, "hipfile": 1},
        ),
        (
            ["gfx94x", "gfx90a"],
            {"gfx94x": ["test:rocgdb"], "gfx90a": []},
            "test:rocblas",
            {"rocgdb-cpu": 1, "hipfile": 1},
        ),
        (["gfx94x"], {"gfx94x": ["test:rocblas"]}, "test:rocgdb", {}),
    ],
    ids=[
        "single-override",
        "two-matching",
        "global-fallback",
        "empty-override",
        "excluded",
    ],
)
def test_linux_ci_generic_uses_effective_family_labels(
    monkeypatch: pytest.MonkeyPatch,
    families: list[str],
    overrides: dict[str, list[str]],
    global_label: str,
    expected: dict[str, int],
) -> None:
    monkeypatch.setenv(
        "EXTERNAL_FAMILY_OVERRIDES",
        json.dumps(
            {
                family: {"linux": {"test_labels_for_family": labels}}
                for family, labels in overrides.items()
            }
        ),
    )
    plan = _linux_ci_plan(
        monkeypatch,
        linux_amdgpu_families=families,
        linux_test_labels=[global_label],
    )
    assert plan.generic_test_runner == "linux-cpu"
    jobs = load_workflow(WORKFLOWS_DIR / "multi_arch_ci_linux.yml")["jobs"]
    projects = "rocgdb-cpu,rocgdb-gpu,hipfile"
    # The pre-fix planner can still be used to reproduce the missing component.
    label_overrides = getattr(plan, "generic_test_label_overrides", None)
    generic, _ = _configure(
        monkeypatch,
        scope=jobs["test_artifacts_generic"]["with"]["test_scope"],
        PROJECTS_TO_TEST=projects,
        TEST_RUNS_ON_CPU=plan.generic_test_runner,
        TEST_LABELS=json.dumps([global_label]),
        GENERIC_TEST_LABEL_OVERRIDES=(
            "" if label_overrides is None else json.dumps(label_overrides)
        ),
    )
    components = list(generic)
    for entry in plan.per_family_info:
        family, _ = _configure(
            monkeypatch,
            scope=_family_scope(
                jobs["test_artifacts_per_family"]["with"]["test_scope"]
            ),
            family=entry["amdgpu_family"],
            target=entry["amdgpu_targets"],
            PROJECTS_TO_TEST=projects,
            TEST_RUNS_ON=entry["test-runs-on"],
            TEST_LABELS=json.dumps(entry.get("test_labels_for_family", [global_label])),
        )
        assert all(component["job_name"] == "rocgdb-gpu" for component in family)
        if families == ["gfx94x"] and expected:
            assert [component["job_name"] for component in family] == ["rocgdb-gpu"]
        components.extend(family)
    assert (
        Counter(c["job_name"] for c in components if c.get("linux_cpu_runner"))
        == expected
    )


@pytest.mark.parametrize(
    "labels, families, runs_generic",
    [
        (["ci:build:gfx906"], ["gfx906"], False),
        (["ci:build:gfx906", "ci:test:gfx906"], ["gfx906"], True),
        (["ci:build:gfx906", "ci:gfx94x"], ["gfx906", "gfx94X-dcgpu"], True),
        (["ci:gfx94x", "ci:cpu-test-only"], ["gfx94X-dcgpu"], True),
    ],
    ids=["only-build", "test-override", "mixed", "cpu-only"],
)
def test_linux_ci_generic_obeys_architecture_test_policy(
    monkeypatch: pytest.MonkeyPatch,
    labels: list[str],
    families: list[str],
    runs_generic: bool,
) -> None:
    monkeypatch.setenv("EXTERNAL_FAMILY_OVERRIDES", "")
    plan = _linux_ci_plan(
        monkeypatch, pr_labels=["ci:exactly", "ci:platform:linux", *labels]
    )
    assert {entry["amdgpu_family"] for entry in plan.per_family_info} == set(families)
    assert plan.generic_test_runner == ("linux-cpu" if runs_generic else "")
    assert plan.generic_test_label_overrides == ([None] if runs_generic else [])
    if "ci:cpu-test-only" in labels:
        assert all(not entry["test-runs-on"] for entry in plan.per_family_info)


def _runs_after_artifact_jobs(
    job: dict,
    build_config: dict,
    copy_result: str,
    build_result: str,
    *,
    workflow_cancelled: bool = False,
) -> bool:
    """Evaluate the expression subset used by the two Linux artifact test jobs."""
    condition = job["if"]
    if isinstance(condition, bool):
        return condition
    assert condition.startswith("${{") and condition.endswith("}}")
    expression = " ".join(condition[3:-2].split())
    results = (copy_result, build_result)
    values = {
        "failure()": "failure" in results,
        "cancelled()": workflow_cancelled,
        "contains(needs.*.result, 'cancelled')": "cancelled" in results,
        "needs.copy_prebuilt_stages.result": copy_result,
        "needs.build_multi_arch_stages.result": build_result,
        "fromJSON(inputs.build_config).build_rocm": build_config["build_rocm"],
        "fromJSON(inputs.build_config).generic_test_runner": build_config[
            "generic_test_runner"
        ],
    }
    for reference, value in values.items():
        expression = expression.replace(reference, repr(value))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", "not ", expression)
    expression = re.sub(r"\btrue\b", "True", expression)
    tree = ast.parse(expression, mode="eval")
    allowed = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
    )
    assert all(isinstance(node, allowed) for node in ast.walk(tree))
    return bool(eval(compile(tree, "<workflow if>", "eval"), {"__builtins__": {}}))


def test_linux_ci_generic_job_uses_build_status_and_test_policy() -> None:
    jobs = load_workflow(WORKFLOWS_DIR / "multi_arch_ci_linux.yml")["jobs"]
    job = jobs["test_artifacts_generic"]
    assert job["uses"] == "./.github/workflows/test_artifacts.yml"
    assert job["needs"] == ["copy_prebuilt_stages", "build_multi_arch_stages"]
    config = {"build_rocm": True, "generic_test_runner": "linux-cpu"}
    assert _runs_after_artifact_jobs(job, config, "skipped", "success")
    assert not _runs_after_artifact_jobs(job, config, "skipped", "skipped")
    assert not _runs_after_artifact_jobs(
        job, {**config, "generic_test_runner": ""}, "success", "skipped"
    )
    assert "strategy" not in job
    assert job["with"]["test_labels"] == "${{ inputs.test_labels }}"
    assert job["with"]["generic_test_label_overrides"] == (
        "${{ toJSON(fromJSON(inputs.build_config).generic_test_label_overrides) }}"
    )
    assert (
        job["with"]["test_runs_on_cpu"]
        == "${{ fromJSON(inputs.build_config).generic_test_runner }}"
    )


@pytest.mark.parametrize(
    "copy_result, build_result, workflow_cancelled, expected",
    [
        ("success", "skipped", False, {"hipfile": 1, "rocgdb-cpu": 1, "rocgdb-gpu": 1}),
        ("failure", "skipped", False, {}),
        ("cancelled", "skipped", False, {}),
        ("success", "failure", False, {}),
        ("success", "cancelled", False, {}),
        ("success", "skipped", True, {}),
    ],
    ids=[
        "all-stages-reused",
        "copy-failed",
        "copy-cancelled",
        "build-failed",
        "build-cancelled",
        "workflow-cancelled",
    ],
)
def test_linux_ci_plan_tests_successfully_reused_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    copy_result: str,
    build_result: str,
    workflow_cancelled: bool,
    expected: dict[str, int],
) -> None:
    monkeypatch.setenv("STAGE_REUSE_MODE", "dry-run")
    monkeypatch.delenv("EXTERNAL_FAMILY_OVERRIDES", raising=False)
    labels = ["test:rocgdb", "test:hipfile"]
    plan = _linux_ci_plan(
        monkeypatch,
        linux_amdgpu_families=["gfx94x"],
        linux_test_labels=labels,
        prebuilt_stages="all",
        baseline_run_id="12345",
    )
    assert plan.prebuilt_stages and plan.baseline_run_id == "12345"
    assert plan.generic_test_runner == "linux-cpu"
    jobs = load_workflow(WORKFLOWS_DIR / "multi_arch_ci_linux.yml")["jobs"]
    generic = jobs["test_artifacts_generic"]
    family = jobs["test_artifacts_per_family"]
    config = plan.to_dict()
    env = {
        "PROJECTS_TO_TEST": "*",
        "TEST_LABELS": json.dumps(labels),
        "SKIP_STAGES": config["skip_stages"],
    }
    emitted = []
    if _runs_after_artifact_jobs(
        generic,
        config,
        copy_result,
        build_result,
        workflow_cancelled=workflow_cancelled,
    ):
        components, _ = _configure(
            monkeypatch,
            scope=generic["with"]["test_scope"],
            GENERIC_TEST_LABEL_OVERRIDES=json.dumps(plan.generic_test_label_overrides),
            TEST_RUNS_ON_CPU=plan.generic_test_runner,
            **env,
        )
        emitted.extend(components)
    if _runs_after_artifact_jobs(
        family, config, copy_result, build_result, workflow_cancelled=workflow_cancelled
    ):
        (entry,) = plan.per_family_info
        components, _ = _configure(
            monkeypatch,
            scope=_family_scope(family["with"]["test_scope"]),
            family=entry["amdgpu_family"],
            target=entry["amdgpu_targets"],
            TEST_RUNS_ON=entry["test-runs-on"],
            TEST_RUNS_ON_CPU=entry["test-runs-on-cpu"],
            **env,
        )
        emitted.extend(components)
    assert Counter(c["job_name"] for c in emitted) == expected


def test_compiler_bump_linux_tests_use_the_built_source_ref() -> None:
    jobs = load_workflow(WORKFLOWS_DIR / "compiler_bump.yml")["jobs"]
    built_ref = jobs["build-linux"]["with"]["ref"]
    assert built_ref == "${{ needs.setup-linux.outputs.ref }}"
    for name in (
        "trigger-linux-generic-tests",
        "trigger-linux-tests",
        "trigger-linux-tests-extra",
    ):
        assert _dispatch_inputs(jobs[name])["ref"] == built_ref, name


@pytest.mark.parametrize(
    "workflow, job_name, build_jobs, summary",
    [
        (
            "multi_arch_release_linux.yml",
            "trigger_test_artifacts_generic",
            ["build_artifacts"],
            "notify_quartz_completed",
        ),
        (
            "compiler_bump.yml",
            "trigger-linux-generic-tests",
            ["setup-linux", "build-linux"],
            "summary",
        ),
    ],
)
def test_release_and_compiler_bump_record_generic_dispatch_without_waiting(
    workflow: str, job_name: str, build_jobs: list[str], summary: str
) -> None:
    jobs = load_workflow(WORKFLOWS_DIR / workflow)["jobs"]
    job = jobs[job_name]
    assert job["needs"] == build_jobs
    assert job["permissions"]["actions"] == "write"
    assert "uses" not in job and "strategy" not in job
    dispatch = next(step for step in job["steps"] if step.get("id") == "trigger-tests")
    assert dispatch["uses"].startswith("benc-uk/workflow-dispatch@")
    assert dispatch["with"]["wait-for-completion"] is False
    payload = _dispatch_inputs(job)
    assert payload["test_scope"] == "generic"
    assert payload["artifact_run_id"] == "${{ github.run_id }}"
    assert not {"amdgpu_families", "amdgpu_targets", "test_runs_on"} & payload.keys()
    assert (
        job["outputs"]["test_run_url"]
        == "${{ steps.trigger-tests.outputs.runUrlHtml }}"
    )
    assert job_name in jobs[summary]["needs"]
    condition = " ".join(job["if"].split())
    if workflow == "compiler_bump.yml":
        assert condition == (
            "${{ inputs.dry_run != true && inputs.build_only != true && "
            "needs.build-linux.result == 'success' && "
            "fromJSON(needs.setup-linux.outputs.linux_build_config || '{}').generic_test_runner != '' }}"
        )
        summary_step = next(
            step
            for step in jobs[summary]["steps"]
            if step["name"] == "Generate summary"
        )
        assert summary_step["env"]["GENERIC_TEST_RUN_URL"] == (
            "${{ needs.trigger-linux-generic-tests.outputs.test_run_url }}"
        )
    else:
        assert (
            condition
            == "${{ fromJSON(inputs.build_config).generic_test_runner != '' }}"
        )
        assert job["steps"][1]["env"]["TEST_RUN_URL"] == job["outputs"]["test_run_url"]


@pytest.mark.parametrize(
    "platform, env",
    [
        ("linux", {"PROJECTS_TO_TEST": "miopen-dbsync"}),
        ("linux", {"TEST_LABELS": '["test:miopen-dbsync"]'}),
        ("linux", {"SKIP_STAGES": "debug-tools"}),
        ("linux", {"SKIP_STAGES": "compiler-runtime"}),
        ("linux", {"PROJECTS_TO_TEST": "hipfile", "SKIP_STAGES": "storage-libs"}),
        ("linux", {"PROJECTS_TO_TEST": "hipfile", "SKIP_STAGES": "compiler-runtime"}),
        ("windows", {}),
        ("wsl", {}),
    ],
)
def test_generic_scope_keeps_component_filters(
    monkeypatch: pytest.MonkeyPatch, platform: str, env: dict[str, str]
) -> None:
    assert _configure(monkeypatch, scope="generic", platform=platform, **env) == (
        [],
        None,
    )


def test_generic_scope_skips_tests_without_a_cpu_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(configurations, "get_cpu_test_runner", lambda platform: "")
    assert _configure(monkeypatch, scope="generic", TEST_RUNS_ON_CPU="") == ([], None)


def test_generic_scope_uses_global_tier_and_preserves_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rocgdb = configurations.test_matrix["rocgdb-cpu"]
    monkeypatch.setitem(
        configurations.test_matrix,
        "rocgdb-cpu",
        {**rocgdb, "test_types": ["standard"], "total_shards_dict": {"linux": 3}},
    )
    # gfx125X forces quick; the accompanying generic call uses the global tier.
    family_components, _ = _configure(
        monkeypatch,
        scope="family",
        family="gfx125X-dcgpu",
        target="gfx1250",
        PROJECTS_TO_TEST="hip-tests",
        TEST_RUNS_ON="linux-gpu",
    )
    assert [c["test_type"] for c in family_components] == ["quick"]
    components, _ = _configure(monkeypatch, scope="generic")
    (component,) = components
    assert component["job_name"] == "rocgdb-cpu"
    assert component["test_type"] == "standard"
    assert component["shard_arr"] == [1, 2, 3]
    assert component["total_shards"] == 3
    assert _configure(monkeypatch, scope="generic", TEST_TYPE="quick") == ([], None)


@pytest.mark.parametrize(
    "variant", ["release", "asan", "asan-debug", "host-asan", "host-asan-debug"]
)
def test_test_plan_keeps_variant_filters(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    family = _configure(
        monkeypatch,
        scope="family",
        family="gfx94X-dcgpu",
        target="gfx942",
        BUILD_VARIANT=variant,
    )[0]
    generic = _configure(monkeypatch, scope="generic", BUILD_VARIANT=variant)[0]
    assert [c["job_name"] for c in family] == (
        ["miopen-dbsync"] if variant == "release" else []
    )
    assert [c["job_name"] for c in generic] == ["rocgdb-cpu"]


@pytest.mark.parametrize(
    "family, target, test_type",
    [("gfx110X-all", "gfx1100", "standard"), ("gfx94X-dcgpu", "gfx942", "quick")],
)
def test_family_scope_keeps_dbsync_family_and_tier_filters(
    monkeypatch: pytest.MonkeyPatch, family: str, target: str, test_type: str
) -> None:
    assert _configure(
        monkeypatch, scope="family", family=family, target=target, TEST_TYPE=test_type
    ) == ([], None)


def test_family_scope_keeps_gpu_tests_and_sanity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, sanity = _configure(
        monkeypatch,
        scope="family",
        family="gfx94X-dcgpu",
        target="gfx942",
        PROJECTS_TO_TEST="rocgdb-cpu,rocgdb-gpu",
        TEST_RUNS_ON="linux-gpu",
    )
    assert [c["job_name"] for c in components] == ["rocgdb-gpu"]
    assert components[0]["test_runner"] == "linux-gpu"
    assert sanity is not None and sanity["test_runner"] == "linux-gpu"


def test_scope_defaults_to_all_for_standalone_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, _ = _configure(monkeypatch, family="gfx94X-dcgpu", target="gfx942")
    assert [c["job_name"] for c in components] == ["rocgdb-cpu", "miopen-dbsync"]


@pytest.mark.parametrize(
    "options",
    [
        {"scope": "unknown"},
        {"scope": "family"},
        {"scope": "family", "family": " \t "},
        {"scope": "generic", "family": "gfx90a"},
        {"scope": "generic", "target": "gfx90a"},
        {"scope": "generic", "TEST_RUNS_ON": "linux-gpu"},
        {"scope": "generic", "GENERIC_TEST_LABEL_OVERRIDES": "not-json"},
        {"scope": "generic", "GENERIC_TEST_LABEL_OVERRIDES": '["test:rocgdb"]'},
        {"scope": "generic", "GENERIC_TEST_LABEL_OVERRIDES": "[[42]]"},
        {"scope": "family", "family": "gfx90a", "GENERIC_TEST_LABEL_OVERRIDES": "[]"},
    ],
)
def test_invalid_scope_inputs_fail(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _configure(monkeypatch, **options)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("include_family", {"linux": ["gfx90a"]}),
        ("expect_failure", {"linux": ["gfx90a"]}),
        ("linux_cpu_runner", False),
    ],
)
def test_generic_component_rejects_family_specific_settings(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    entry = {**configurations.test_matrix["rocgdb-cpu"], field: value}
    monkeypatch.setitem(configurations.test_matrix, "rocgdb-cpu", entry)
    with pytest.raises(ValueError, match="rocgdb-cpu: run_per_family=False"):
        _configure(monkeypatch, scope="generic")
