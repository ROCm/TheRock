# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Executes a YAML scenario against the real CI configuration pipeline.

The run mirrors what ``configure_multi_arch_ci.main()`` does in a workflow,
with exactly one substitution: ``GitContext`` is built from the scenario's
``git:`` block instead of by shelling out to the repository. Everything else -
input parsing, the skip gate, target selection, job decisions, config
expansion, and output serialization - is the production code path.

Assertions run against the parsed ``GITHUB_OUTPUT`` file rather than the
in-memory ``CIOutputs``. That file is the actual contract with
``setup_multi_arch.yml``; asserting on it catches serialization regressions
that a dataclass comparison would miss.
"""

import json
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import configure_multi_arch_ci as cm
from mock_family_matrix import mock_get_all_families_for_trigger_types
from scenario_schema import DERIVED_PLATFORM_KEYS, Scenario

# Must match the delimiter gha_set_output uses for multi-line values.
_HEREDOC_DELIMITER = "EOF_mag1c"

_JOB_GROUPS = (
    "build_rocm",
    "test_rocm",
    "build_rocm_python",
    "build_pytorch",
    "test_pytorch",
    "build_jax",
)


@dataclass
class ScenarioResult:
    """What a scenario produced, in the shapes assertions care about."""

    outputs: dict[str, str] = field(default_factory=dict)
    linux: dict[str, Any] | None = None
    windows: dict[str, Any] | None = None
    jobs: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    ci_outputs: Any = None


def parse_github_output(path: Path) -> dict[str, str]:
    """Parse a ``GITHUB_OUTPUT`` file, including the multi-line heredoc form.

    Mirrors how the Actions runner reads the file back, so scenarios assert on
    the values a workflow would actually see.
    """
    outputs: dict[str, str] = {}
    if not path.exists():
        return outputs

    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        heredoc_marker = f"<<{_HEREDOC_DELIMITER}"
        if heredoc_marker in line:
            key = line.split(heredoc_marker, 1)[0]
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i] != _HEREDOC_DELIMITER:
                body.append(lines[i])
                i += 1
            outputs[key] = "\n".join(body)
        elif "=" in line:
            key, _, value = line.partition("=")
            outputs[key] = value
        i += 1
    return outputs


def _scenario_env(scenario: Scenario, tmp_path: Path) -> dict[str, str]:
    """GitHub Actions context for this scenario, layered over the base env."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(scenario.payload), encoding="utf-8")

    env = {
        "GITHUB_EVENT_NAME": scenario.event,
        "GITHUB_REF_NAME": scenario.ref,
        "GITHUB_EVENT_PATH": os.fspath(event_path),
        "GITHUB_OUTPUT": os.fspath(tmp_path / "github_output"),
        "GITHUB_STEP_SUMMARY": os.fspath(tmp_path / "step_summary.md"),
        "RUNNER_TEMP": os.fspath(tmp_path),
    }
    # Scenario env wins, so a scenario can override the defaults above.
    env.update(scenario.env)
    return env


def run_scenario(scenario: Scenario, tmp_path: Path) -> ScenarioResult:
    """Run one scenario end to end and collect its outputs."""
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, _scenario_env(scenario, tmp_path)))

        if scenario.families == "mock":
            stack.enter_context(
                patch.object(
                    cm,
                    "get_all_families_for_trigger_types",
                    mock_get_all_families_for_trigger_types,
                )
            )

        ci_inputs = cm.CIInputs.from_environ()

        # The one substitution: the scenario's declared diff stands in for the
        # repository. A scenario that omits `git:` gets GitContext.empty(),
        # which is what schedule and workflow_dispatch runs use in production.
        git_context = cm.GitContext(
            changed_files=scenario.changed_files,
            submodule_paths=scenario.submodule_paths,
        )

        outputs = cm.configure(ci_inputs, git_context)
        cm.write_outputs(ci_inputs=ci_inputs, outputs=outputs)

    raw_outputs = parse_github_output(tmp_path / "github_output")
    summary_path = tmp_path / "step_summary.md"

    return ScenarioResult(
        outputs=raw_outputs,
        linux=json.loads(raw_outputs["linux_build_config"])
        if raw_outputs.get("linux_build_config")
        else None,
        windows=json.loads(raw_outputs["windows_build_config"])
        if raw_outputs.get("windows_build_config")
        else None,
        jobs=_job_actions(outputs),
        summary=(
            summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        ),
        ci_outputs=outputs,
    )


def _job_actions(outputs) -> dict[str, str]:
    """Flatten JobDecisions into ``{job_group: action}`` plus the test-type reason."""
    if outputs.jobs is None:
        return {}
    actions = {name: getattr(outputs.jobs, name).action.value for name in _JOB_GROUPS}
    actions["test_type_reason"] = outputs.jobs.test_rocm.test_type_reason
    return actions


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def _derived_platform_values(config: dict[str, Any]) -> dict[str, list[str]]:
    """Convenience views over per_family_info that scenarios assert on."""
    per_family = config.get("per_family_info", [])
    return {
        "families": [f["amdgpu_family"] for f in per_family],
        "tested_families": [
            f["amdgpu_family"] for f in per_family if f.get("test-runs-on")
        ],
        "wsl_families": [
            f["amdgpu_family"] for f in config.get("wsl_per_family_info", [])
        ],
    }


def _check(failures: list[str], label: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        failures.append(
            f"  {label}:\n"
            f"      expected: {expected!r}\n"
            f"      actual:   {actual!r}"
        )


def _check_platform(
    failures: list[str],
    platform: str,
    expected: dict[str, Any] | None,
    actual: dict[str, Any] | None,
) -> None:
    if expected is None:
        if actual is not None:
            failures.append(
                f"  {platform}: expected no build config, but one was produced "
                f"(families={_derived_platform_values(actual)['families']})"
            )
        return

    if actual is None:
        failures.append(
            f"  {platform}: expected a build config, but the platform was skipped"
        )
        return

    derived = _derived_platform_values(actual)
    for key, want in expected.items():
        if key in DERIVED_PLATFORM_KEYS:
            _check(failures, f"{platform}.{key}", want, derived[key])
        elif key not in actual:
            failures.append(
                f"  {platform}.{key}: not a build config field. "
                f"Available: {sorted(actual)}"
            )
        else:
            _check(failures, f"{platform}.{key}", want, actual[key])


def assert_scenario(scenario: Scenario, result: ScenarioResult) -> None:
    """Check a scenario's expectations, reporting every mismatch at once.

    Collecting all failures rather than stopping at the first means a scenario
    that drifts shows the full picture in one run instead of one field per
    edit-and-rerun cycle.
    """
    expect = scenario.expect
    failures: list[str] = []

    for key, want in expect.get("outputs", {}).items():
        if key not in result.outputs:
            failures.append(
                f"  outputs.{key}: not written to GITHUB_OUTPUT. "
                f"Written keys: {sorted(result.outputs)}"
            )
        else:
            _check(failures, f"outputs.{key}", want, result.outputs[key])

    if "linux" in expect:
        _check_platform(failures, "linux", expect["linux"], result.linux)
    if "windows" in expect:
        _check_platform(failures, "windows", expect["windows"], result.windows)

    for key, want in expect.get("jobs", {}).items():
        if key not in result.jobs:
            failures.append(
                f"  jobs.{key}: unknown job group. Available: {sorted(result.jobs)}"
            )
        else:
            _check(failures, f"jobs.{key}", want, result.jobs[key])

    for needle in expect.get("summary_contains", []):
        if needle not in result.summary:
            failures.append(f"  summary: missing expected text {needle!r}")

    if failures:
        raise AssertionError(
            f"Scenario {scenario.id!r} did not match:\n" + "\n".join(failures)
        )
