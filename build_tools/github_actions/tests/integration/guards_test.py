# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the hermeticity guards themselves.

The integration scenarios are only trustworthy if the guards in ``conftest.py``
actually fire. A guard that silently stopped working would turn the whole suite
into tests that pass while quietly reading the developer's environment, the
real git checkout, or the live GitHub API.

So the guards get tested too.
"""

import os
import subprocess

import configure_ci_path_filters
import github_actions_api
import pytest
from conftest import BASE_ENV
from scenario_runner import run_scenario
from scenario_schema import Scenario


def test_environment_is_isolated():
    """Only the declared base context is visible to the pipeline."""
    # pytest sets PYTEST_CURRENT_TEST after fixtures run; it is pytest's own
    # bookkeeping and nothing in the pipeline reads it.
    visible = set(os.environ) - {"PYTEST_CURRENT_TEST"}
    assert visible == set(BASE_ENV)
    assert os.environ["STAGE_REUSE_MODE"] == "off"


def test_developer_environment_does_not_leak(monkeypatch):
    """A variable set outside the suite is not visible inside it.

    ``CI_CONFIG_PATH`` is the dangerous one: it makes
    ``amdgpu_family_matrix`` overlay an external runner config, which would
    change scenario results based on what the developer has checked out.
    """
    assert "CI_CONFIG_PATH" not in os.environ
    assert "GITHUB_EVENT_PATH" not in os.environ


def test_network_guard_blocks_http():
    with pytest.raises(AssertionError, match="attempted an HTTP request"):
        github_actions_api.urlopen("https://api.github.com/repos/ROCm/TheRock")


def test_network_guard_blocks_the_gh_cli_transport():
    """The gh CLI path bypasses urlopen entirely, so it needs its own block.

    When the developer has an authenticated gh CLI, GitHubAPI shells out to
    `gh api` instead of making an HTTP request. Guarding only urlopen would
    leave scenarios free to hit the live API on exactly the machines where
    nobody would notice.
    """
    with pytest.raises(AssertionError, match="attempted to run"):
        github_actions_api.subprocess.run(["gh", "api", "/repos/ROCm/TheRock"])


def test_github_auth_method_is_pinned():
    """Auth detection must not vary with what else ran first in the session."""
    method = github_actions_api._default_github_api.get_auth_method()
    assert method is github_actions_api.GitHubAPI.AuthMethod.UNAUTHENTICATED


def test_git_guard_blocks_subprocess():
    with pytest.raises(AssertionError, match="attempted to run"):
        configure_ci_path_filters.subprocess.run(["git", "rev-parse", "HEAD"])


def test_git_guard_still_exposes_module_constants():
    """Exception handlers around the blocked call sites must keep working."""
    assert configure_ci_path_filters.subprocess.PIPE is subprocess.PIPE
    assert (
        configure_ci_path_filters.subprocess.CalledProcessError
        is subprocess.CalledProcessError
    )


def test_stage_reuse_dry_run_would_reach_the_network(tmp_path):
    """Proves the network guard is load-bearing, not decorative.

    STAGE_REUSE_MODE defaults to dry-run rather than off, so the baseline
    lookup is live unless a scenario disables it. This scenario re-enables the
    default and asserts the guard catches the resulting API call. If this ever
    starts passing, the suite has silently become non-hermetic.

    Two details matter for reproducing the call:

    * The change must leave some stage reusable. A broad change (say
      core/CMakeLists.txt) forces a full rebuild, and the planner skips the
      baseline lookup entirely because there is nothing to reuse.
    * Families must be real. Stage reuse resolves required artifact names via
      expand_amdgpu_families, which validates against the production family
      list and rejects the frozen mock names.
    """
    scenario = Scenario(
        name="stage reuse reaches for a baseline",
        source=tmp_path / "probe.yaml",
        event="pull_request",
        ref="my-feature",
        payload={"pull_request": {"base": {"ref": "main"}}},
        env={"STAGE_REUSE_MODE": "dry-run", "STAGE_REUSE_CURRENT_SHA": "a" * 40},
        changed_files=["rocm-libraries/projects/rocblas/src/foo.cpp"],
        submodule_paths=[],
        families="real",
        expect={"outputs": {}},
    )

    with pytest.raises(AssertionError, match="HTTP request|artifact \\(S3\\) backend"):
        run_scenario(scenario, tmp_path)


def test_runner_selection_is_deterministic(tmp_path):
    """Weighted runner picks must not vary between runs.

    select_weighted_label uses random.SystemRandom().choices(), so without the
    determinism guard build_runs_on_* would flake across runs.
    """
    scenario = Scenario(
        name="baseline pull request",
        source=tmp_path / "probe.yaml",
        event="pull_request",
        ref="my-feature",
        payload={"pull_request": {"base": {"ref": "main"}}},
        changed_files=["core/CMakeLists.txt"],
        submodule_paths=[],
        expect={"outputs": {}},
    )

    runs = [
        run_scenario(scenario, tmp_path / f"run{i}").linux["build_runs_on_large"]
        for i in range(5)
    ]
    assert len(set(runs)) == 1, f"build runner selection varied across runs: {runs}"
    assert runs[0], "expected a non-empty build runner label"
