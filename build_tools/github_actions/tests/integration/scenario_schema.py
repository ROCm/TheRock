# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema and loader for YAML CI scenarios.

A scenario is one end-to-end run of the CI configuration pipeline: a GitHub
event plus a git diff goes in, a set of GitHub Actions outputs comes out. The
YAML is deliberately close to what GitHub itself provides, so a scenario can be
read next to a real workflow run without translation.

    - name: docs-only pull request skips CI
      trigger:
        event: pull_request
        ref: my-feature
        payload:
          pull_request:
            base: {ref: main}
            labels: [{name: "test:rccl"}]
      env:
        BUILD_VARIANT: release
      git:
        changed_files: ["README.md"]
        submodule_paths: []
      expect:
        outputs:
          enable_build_jobs: "false"

Expectations are *partial*: only the keys a scenario names are checked. Adding
a field to ``BuildConfig`` therefore does not invalidate every scenario, while
a scenario that cares about the new field can opt in by naming it.

Unknown keys are rejected rather than ignored. A silently-ignored typo in an
``expect`` block is a test that passes without asserting anything, which is
worse than no test at all.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

_TRIGGER_KEYS = {"event", "ref", "payload"}
_GIT_KEYS = {"changed_files", "submodule_paths"}
_EXPECT_KEYS = {"outputs", "linux", "windows", "jobs", "summary_contains"}
_SCENARIO_KEYS = {
    "name",
    "description",
    "trigger",
    "env",
    "git",
    "families",
    "expect",
    "expect_error",
}

# Derived conveniences in an `expect.linux` / `expect.windows` block. Everything
# else in those blocks is matched directly against the build config dict.
DERIVED_PLATFORM_KEYS = {"families", "tested_families", "wsl_families"}

VALID_FAMILY_MODES = {"mock", "real"}


class ScenarioError(Exception):
    """A scenario file is malformed. Raised at collection time, not run time."""


@dataclass(frozen=True)
class Scenario:
    """One end-to-end CI configuration scenario."""

    name: str
    source: Path
    event: str
    ref: str = "main"
    payload: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    changed_files: list[str] | None = None
    submodule_paths: list[str] | None = None
    families: str = "mock"
    expect: dict[str, Any] = field(default_factory=dict)
    expect_error: str | None = None
    description: str = ""

    @property
    def id(self) -> str:
        """Pytest test id: ``pull_request.yaml::docs-only pull request skips CI``."""
        return f"{self.source.stem}::{self.name}"


def _require_mapping(value: Any, what: str, source: Path) -> dict:
    if not isinstance(value, dict):
        raise ScenarioError(
            f"{source.name}: {what} must be a mapping, got {type(value).__name__}"
        )
    return value


def _reject_unknown(value: dict, allowed: set[str], what: str, source: Path) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ScenarioError(
            f"{source.name}: unknown {what} key(s) {unknown}. "
            f"Allowed: {sorted(allowed)}"
        )


def _parse_scenario(raw: Any, source: Path, index: int) -> Scenario:
    raw = _require_mapping(raw, f"scenario #{index}", source)
    _reject_unknown(raw, _SCENARIO_KEYS, "scenario", source)

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(
            f"{source.name}: scenario #{index} needs a non-empty 'name'"
        )

    trigger = _require_mapping(raw.get("trigger", {}), f"{name!r} trigger", source)
    _reject_unknown(trigger, _TRIGGER_KEYS, "trigger", source)
    event = trigger.get("event")
    if not event:
        raise ScenarioError(f"{source.name}: {name!r} needs 'trigger.event'")

    git = _require_mapping(raw.get("git", {}), f"{name!r} git", source)
    _reject_unknown(git, _GIT_KEYS, "git", source)

    expect = _require_mapping(raw.get("expect", {}), f"{name!r} expect", source)
    _reject_unknown(expect, _EXPECT_KEYS, "expect", source)

    expect_error = raw.get("expect_error")
    if expect and expect_error:
        raise ScenarioError(
            f"{source.name}: {name!r} sets both 'expect' and 'expect_error'; "
            f"a scenario either produces outputs or raises, not both"
        )
    if not expect and not expect_error:
        raise ScenarioError(
            f"{source.name}: {name!r} has no 'expect' or 'expect_error' block, "
            f"so it would assert nothing"
        )

    families = raw.get("families", "mock")
    if families not in VALID_FAMILY_MODES:
        raise ScenarioError(
            f"{source.name}: {name!r} has families={families!r}, "
            f"expected one of {sorted(VALID_FAMILY_MODES)}"
        )

    for platform in ("linux", "windows"):
        block = expect.get(platform)
        # `linux: null` is meaningful - it asserts the platform was skipped.
        if block is not None:
            _require_mapping(block, f"{name!r} expect.{platform}", source)

    # Env values go into os.environ, which only holds strings. Coerce here so a
    # scenario can write `PYTHON_VERSION: 3.12` without tripping putenv.
    env = {
        str(k): "" if v is None else str(v)
        for k, v in _require_mapping(
            raw.get("env", {}), f"{name!r} env", source
        ).items()
    }

    return Scenario(
        name=name,
        source=source,
        event=event,
        ref=trigger.get("ref", "main"),
        payload=trigger.get("payload") or {},
        env=env,
        changed_files=git.get("changed_files"),
        submodule_paths=git.get("submodule_paths"),
        families=families,
        expect=expect,
        expect_error=expect_error,
        description=raw.get("description", ""),
    )


def load_scenario_file(path: Path) -> list[Scenario]:
    """Parse one YAML file into scenarios, validating as we go."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScenarioError(
            f"{path.name}: expected a top-level list of scenarios, "
            f"got {type(raw).__name__}"
        )

    scenarios = [_parse_scenario(entry, path, i) for i, entry in enumerate(raw)]

    seen: set[str] = set()
    duplicates = {s.name for s in scenarios if s.name in seen or seen.add(s.name)}
    if duplicates:
        raise ScenarioError(
            f"{path.name}: duplicate scenario name(s) {sorted(duplicates)}; "
            f"names must be unique within a file so test ids stay stable"
        )
    return scenarios


def load_scenarios(group: str) -> list[Scenario]:
    """Load every scenario in ``scenarios/<group>/``, sorted for stable ordering."""
    group_dir = SCENARIOS_DIR / group
    if not group_dir.is_dir():
        raise ScenarioError(f"No scenario directory at {group_dir}")

    scenarios: list[Scenario] = []
    for path in sorted(group_dir.glob("*.yaml")):
        scenarios.extend(load_scenario_file(path))

    if not scenarios:
        raise ScenarioError(f"No scenarios found in {group_dir}")
    return scenarios
