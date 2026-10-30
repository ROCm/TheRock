# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hermeticity guards for the CI integration suite.

These scenarios run the *real* pipeline - no internal function is mocked. That
only produces trustworthy results if the pipeline cannot reach anything outside
the test, and if its answer does not depend on who is running it or where.

Four autouse guards enforce that:

``_isolated_env``
    Wipes the environment. ``amdgpu_family_matrix`` reads ``CI_CONFIG_PATH``
    and ``GITHUB_REPOSITORY``; ``stage_reuse_decision`` reads seven
    ``STAGE_REUSE_*`` variables. Without this, a scenario's result depends on
    the developer's shell, and running the suite inside TheRock's own CI (where
    ``GITHUB_*`` is fully populated) gives different answers than running it
    locally.

``_no_network``
    Fails any HTTP request. ``STAGE_REUSE_MODE`` defaults to ``dry-run``, not
    ``off``, so the baseline-lookup path is live by default - a scenario that
    forgets to disable it would otherwise silently hit the GitHub API and
    produce results that vary with repository state.

``_no_git``
    Fails any git subprocess. Scenarios supply their diff through
    ``GitContext`` directly; shelling out would make results depend on the
    checkout the suite happens to run in.

``_deterministic_runner_selection``
    ``select_weighted_label`` picks runner labels with
    ``random.SystemRandom().choices()``. Left alone, ``build_runs_on_*`` varies
    between runs and any scenario asserting on it would flake.

Note that the TOML files (``BUILD_TOPOLOGY.toml``, ``skip-ci-*.toml``) are
deliberately *not* stubbed. Reading the real ones is the point: a stage added
to the topology that breaks job decisions should fail these tests.
"""

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[2]))

import amdgpu_family_matrix
import baseline_runs
import configure_ci_path_filters
import github_actions_api

# The minimum GitHub Actions context the pipeline needs to start. Scenarios
# layer their own env on top of this; anything not listed here is absent.
BASE_ENV = {
    "GITHUB_RUN_ID": "1000000001",
    "GITHUB_REPOSITORY": "ROCm/TheRock",
    # Stage reuse reaches for the network under its default (dry-run) mode.
    # Scenarios that exercise reuse opt back in explicitly.
    "STAGE_REUSE_MODE": "off",
    # Keep PATH so pytest and any genuinely-needed tooling still resolve.
    "PATH": os.environ.get("PATH", ""),
}


class _NetworkBlocked(AssertionError):
    """Raised when a scenario tries to make an HTTP request."""


class _GitBlocked(AssertionError):
    """Raised when a scenario tries to shell out to git."""


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Replace the environment with a known-minimal GitHub Actions context."""
    for name in list(os.environ):
        monkeypatch.delenv(name, raising=False)
    for name, value in BASE_ENV.items():
        monkeypatch.setenv(name, value)


class _GuardedSubprocess:
    """Stands in for the ``subprocess`` module, allowing everything but execution.

    The module's constants (``PIPE``, ``CalledProcessError``, ...) still resolve
    so exception handlers around the call sites keep working; only the calls
    that would actually spawn a process are blocked.
    """

    def __init__(self, error: type[AssertionError], hint: str):
        self._error = error
        self._hint = hint

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def run(self, cmd, *args, **kwargs):
        raise self._error(f"Scenario attempted to run {cmd!r}.\n{self._hint}")

    check_output = run
    call = run
    check_call = run
    Popen = run


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly on GitHub or S3 access instead of silently reaching them.

    GitHubAPI has two transports, and both have to be blocked. ``urlopen`` is
    the obvious one. The other is ``gh api`` over subprocess, chosen when the
    gh CLI is authenticated - which is the common case on a developer machine
    and invisible in CI, so blocking only urlopen would leave a hole that opens
    up exactly where it is hardest to notice.

    The cached auth method on the shared singleton is pinned for the same
    reason. It is detected once per process by shelling out to ``gh auth
    status``, so without pinning, whether a scenario takes the token path or
    the CLI path depends on what else ran first in the session.
    """

    def blocked_urlopen(request, *args, **kwargs):
        url = getattr(request, "full_url", request)
        raise _NetworkBlocked(
            f"Scenario attempted an HTTP request to {url!r}.\n"
            f"Integration scenarios must be hermetic. If this scenario is "
            f"meant to exercise stage reuse, give it a canned baseline rather "
            f"than letting it call the GitHub API."
        )

    def blocked_backend(*args, **kwargs):
        raise _NetworkBlocked(
            "Scenario attempted to reach the artifact (S3) backend. "
            "Integration scenarios must be hermetic."
        )

    monkeypatch.setattr(github_actions_api, "urlopen", blocked_urlopen)
    monkeypatch.setattr(
        github_actions_api,
        "subprocess",
        _GuardedSubprocess(
            _NetworkBlocked,
            "This is the `gh` CLI transport for the GitHub API. Integration "
            "scenarios must be hermetic.",
        ),
    )
    monkeypatch.setattr(
        github_actions_api._default_github_api,
        "_auth_method",
        github_actions_api.GitHubAPI.AuthMethod.UNAUTHENTICATED,
    )
    monkeypatch.setattr(github_actions_api._default_github_api, "_github_token", None)
    monkeypatch.setattr(
        baseline_runs,
        "create_artifact_backend_for_workflow_run",
        blocked_backend,
        raising=False,
    )


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    """Fail loudly on git subprocesses instead of reading the real checkout."""
    monkeypatch.setattr(
        configure_ci_path_filters,
        "subprocess",
        _GuardedSubprocess(
            _GitBlocked,
            "Scenarios declare their diff via the 'git:' block, which becomes "
            "a GitContext directly. Reaching for the real repository would "
            "make results depend on the checkout under test.",
        ),
    )


class _FirstChoice(random.SystemRandom):
    """A SystemRandom whose weighted pick is the first candidate, every time."""

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        return [population[0]] * k


@pytest.fixture(autouse=True)
def _deterministic_runner_selection(monkeypatch):
    """Pin weighted runner-label selection so build runners don't flake."""
    pinned = random
    monkeypatch.setattr(
        amdgpu_family_matrix,
        "random",
        type(
            "_PinnedRandom",
            (),
            {
                "SystemRandom": staticmethod(_FirstChoice),
                "__getattr__": staticmethod(lambda name: getattr(pinned, name)),
            },
        )(),
    )
