# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Plans a pull request code coverage run for multi_arch_ci_coverage_pr.yml.

Decides which coverage projects a pull request measures and what the run can
reuse; multi_arch_ci_coverage_linux.yml then builds, tests, and reports on them.

Project selection:
  - PROJECTS_TO_TEST, when set, is the selection, validated as strictly as the
    nightly's (configure_coverage_ci.parse_projects).
  - Otherwise the candidates are the onboarded projects named in
    CHANGED_PROJECTS (an external repo's changed subtrees, e.g.
    "projects/hiprand"); a pull request coverage run measures only what the
    change touches, so with none there are no candidates. `test:<project>` PR
    labels narrow the candidates further. A candidate that cannot be measured
    is skipped with its reason instead of failing the run.
  An empty selection sets enabled=false, and nothing is built.

Baseline:
  BASELINE_RUN_ID when set, else the newest multi_arch_ci.yml run on
  THEROCK_REPOSITORY's main branch whose Linux compiler-runtime and math-libs
  builds for the selected families passed and which has the selected projects'
  artifacts. Tests install the baseline and overlay only the project under test,
  so with a baseline the instrumented build is limited to the selected projects'
  artifacts. Without one, the whole math-libs stage is built and tested, as in
  the self-contained nightly.

  compiler-runtime is copied from the baseline instead of being rebuilt only when
  the baseline comes from the running repository (artifacts embed the build
  workspace path, so a stage cannot configure against another repository's
  artifacts) and either it was named explicitly or the PR's changed files leave
  compiler-runtime unaffected.

Environment variables:
  - PROJECTS_TO_TEST, CHANGED_PROJECTS: comma-separated, as above.
  - AMDGPU_FAMILIES: comma-separated GPU families (default gfx94X-dcgpu).
  - TEST_TYPE: quick, standard, comprehensive, or full (default standard). A
    `test_filter:<type>` PR label overrides it.
  - BASELINE_RUN_ID, BASELINE_REPOSITORY: an explicit baseline; an empty
    repository means GITHUB_REPOSITORY.
  - THEROCK_REPOSITORY: repository searched for a baseline (default
    GITHUB_REPOSITORY).
  - COVERAGE_BASELINE_MAX_AGE_HOURS: age limit for the baseline search
    (default 72).

Outputs (GITHUB_OUTPUT): enabled, projects_to_test, amdgpu_families, test_type,
baseline_run_id, baseline_repository, rebuild_artifacts, and
prebuilt_compiler_runtime.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))
sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from _therock_utils.build_topology import get_topology
from baseline_runs import BaselineRun, RequiredArtifact, select_baseline_run
from configure_ci_path_filters import get_git_modified_paths
from configure_coverage_ci import (
    COVERAGE_PROJECTS,
    DEFAULT_AMDGPU_FAMILIES,
    DEFAULT_PROJECTS,
    STAGE_COMPILER_RUNTIME,
    STAGE_MATH_LIBS,
    parse_amdgpu_families,
    parse_projects,
    resolve_build_stages,
)
from github_actions_api import (
    gha_append_step_summary,
    gha_load_github_event,
    gha_set_output,
)
from stage_reuse_decision import _required_artifacts_for_stages, plan_stage_reuse

PLATFORM = "linux"
TEST_TYPES = ("quick", "standard", "comprehensive", "full")
DEFAULT_TEST_TYPE = "standard"
DEFAULT_BASELINE_MAX_AGE_HOURS = 72.0
# Checking a candidate costs several API calls; bound the search when none fit.
BASELINE_MAX_RUNS = 10
# How multi_arch_ci.yml names its Linux stage build jobs. Matching only these
# keeps a failed package, framework, or Windows build from ruling a run out,
# while a healthy stage job vouches for every artifact it publishes.
LINUX_STAGE_JOB = "Linux::release / Build Multi-Arch Stages / {stage}"

TEST_LABEL_PREFIX = "test:"
TEST_FILTER_LABEL_PREFIX = "test_filter:"

BaselineFinder = Callable[[], Optional[BaselineRun]]
ChangedFilesProvider = Callable[[], Optional[Sequence[str]]]


@dataclass
class Selection:
    """The projects a run measures, and why any candidates were left out."""

    projects: list[str]
    # Requested name (changed subtree or label) -> reason it is not measured.
    skipped: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselinePlan:
    """Where non-instrumented dependencies come from."""

    run_id: str = ""
    repository: str = ""
    html_url: str = ""
    explicit: bool = False
    prebuilt_compiler_runtime: bool = False
    notes: tuple[str, ...] = ()


def changed_project_key(path: str) -> str:
    """Maps a changed subtree such as 'projects/hipblaslt/tensilelite' to 'hipblaslt'."""
    key = path.strip().strip("/").lower().removeprefix("projects/")
    return key.split("/", 1)[0]


def skip_reason(key: str) -> Optional[str]:
    """Why a project picked on the caller's behalf is not measured, or None."""
    if key in DEFAULT_PROJECTS:
        return None
    project = COVERAGE_PROJECTS.get(key)
    if project is None:
        return "not onboarded to coverage"
    if not project.coverage_option:
        return project.unsupported_reason or "no LLVM coverage option"
    if project.blocked_reason:
        return f"blocked: {project.blocked_reason}"
    return f"no coverage build job for the {project.stage} stage"


def select_projects(
    projects_to_test: str, changed_projects: str, pr_labels: Sequence[str]
) -> Selection:
    """Resolves the projects a run measures; see the module docstring."""
    if projects_to_test.strip():
        projects = parse_projects(projects_to_test)
        resolve_build_stages(projects)
        return Selection(projects=projects)

    skipped: dict[str, str] = {}
    candidates: set[str] = set()
    for path in [p.strip() for p in changed_projects.split(",") if p.strip()]:
        key = changed_project_key(path)
        reason = skip_reason(key)
        if reason:
            skipped[path] = reason
        else:
            candidates.add(key)

    labelled = {
        label.removeprefix(TEST_LABEL_PREFIX).strip().lower()
        for label in pr_labels
        if label.startswith(TEST_LABEL_PREFIX)
    }
    if labelled:
        for key in sorted(labelled - candidates):
            skipped[f"{TEST_LABEL_PREFIX}{key}"] = (
                skip_reason(key) or "not among the changed projects"
            )
        candidates &= labelled

    return Selection(projects=sorted(candidates), skipped=skipped)


def resolve_test_type(raw_test_type: str, pr_labels: Sequence[str]) -> str:
    test_type = raw_test_type.strip().lower() or DEFAULT_TEST_TYPE
    for label in pr_labels:
        if label.startswith(TEST_FILTER_LABEL_PREFIX):
            test_type = label.removeprefix(TEST_FILTER_LABEL_PREFIX).strip().lower()
    if test_type not in TEST_TYPES:
        raise ValueError(
            f"Unknown test type '{test_type}'; expected one of {', '.join(TEST_TYPES)}."
        )
    return test_type


def plan_baseline(
    *,
    baseline_run_id: str,
    baseline_repository: str,
    running_repository: str,
    therock_repository: str,
    find_baseline: BaselineFinder,
    changed_files: ChangedFilesProvider,
) -> BaselinePlan:
    """Picks the baseline run and whether compiler-runtime is copied from it."""
    if baseline_run_id:
        repository = baseline_repository or running_repository
        copy = repository == running_repository
        notes = (
            ()
            if copy
            else (f"compiler-runtime is built: the baseline is from {repository}.",)
        )
        return BaselinePlan(
            run_id=baseline_run_id,
            repository=repository,
            explicit=True,
            prebuilt_compiler_runtime=copy,
            notes=notes,
        )

    run = find_baseline()
    if run is None:
        return BaselinePlan(
            notes=(
                "No healthy baseline run found, so the whole math-libs stage is "
                "built instrumented and tests run against this run's own stack.",
            )
        )

    copy = False
    if therock_repository != running_repository:
        notes = (
            f"compiler-runtime is built: the baseline is from {therock_repository}.",
        )
    else:
        plan = plan_stage_reuse(changed_files=changed_files(), platform=PLATFORM)
        copy = STAGE_COMPILER_RUNTIME in plan.candidate_stages
        notes = () if copy else (f"compiler-runtime is built: {_why(plan.reasons)}.",)
    return BaselinePlan(
        run_id=str(run.run_id),
        repository=therock_repository,
        html_url=run.html_url or "",
        prebuilt_compiler_runtime=copy,
        notes=notes,
    )


def _why(reasons: Sequence[str]) -> str:
    """Stage impact gives a reason per changed file; one is enough to explain."""
    if not reasons:
        return "the changes affect it"
    more = len(reasons) - 1
    return reasons[0] + (f" (and {more} more)" if more else "")


def rebuild_artifacts_for(projects: Sequence[str]) -> list[str]:
    """The artifacts an instrumented build needs to produce for these projects."""
    names: set[str] = set()
    for key in projects:
        names.update(COVERAGE_PROJECTS[key].artifact_names)
    return sorted(names)


def baseline_build_jobs(amdgpu_families: Sequence[str]) -> list[str]:
    """Job name substrings whose jobs must all have passed in a baseline run."""
    math_libs = LINUX_STAGE_JOB.format(stage=STAGE_MATH_LIBS)
    return [
        LINUX_STAGE_JOB.format(stage=STAGE_COMPILER_RUNTIME) + " /",
        *(f"{math_libs} ({family}," for family in amdgpu_families),
    ]


def baseline_artifacts(
    projects: Sequence[str], amdgpu_families: Sequence[str]
) -> list[RequiredArtifact]:
    """The selected projects' artifacts, for every target their families expand to."""
    names = set(rebuild_artifacts_for(projects))
    stages = sorted({COVERAGE_PROJECTS[key].stage for key in projects})
    return [
        required
        for required in _required_artifacts_for_stages(
            get_topology(), stages, amdgpu_families, platform=PLATFORM
        )
        if required.name in names
    ]


def find_baseline_run(
    projects: Sequence[str],
    amdgpu_families: Sequence[str],
    *,
    github_repository: str,
    max_age_hours: Optional[float],
) -> Optional[BaselineRun]:
    return select_baseline_run(
        required_artifacts=baseline_artifacts(projects, amdgpu_families),
        required_successful_job_name_substrings=baseline_build_jobs(amdgpu_families),
        github_repository=github_repository,
        platform=PLATFORM,
        max_runs=BASELINE_MAX_RUNS,
        max_age_hours=max_age_hours,
    )


def _safe_finder(finder: BaselineFinder) -> BaselineFinder:
    """A lookup failure degrades to the self-contained shape instead of failing."""

    def find() -> Optional[BaselineRun]:
        try:
            return finder()
        except Exception as e:
            print(f"::warning::Baseline lookup failed, continuing without one: {e}")
            return None

    return find


def _pr_labels() -> list[str]:
    if not os.getenv("GITHUB_EVENT_PATH"):
        return []
    pull_request = gha_load_github_event().get("pull_request") or {}
    return [label["name"] for label in pull_request.get("labels", [])]


def _pr_changed_files() -> Optional[Sequence[str]]:
    if os.getenv("GITHUB_EVENT_NAME") != "pull_request":
        return None
    # The setup job checks out the PR merge commit with its base as HEAD^1.
    try:
        files = get_git_modified_paths("HEAD^1")
    except subprocess.CalledProcessError as e:
        print(
            f"::warning::Could not diff the pull request; building compiler-runtime: {e}"
        )
        return None
    return None if files is None else list(files)


def render_summary(
    selection: Selection,
    baseline: BaselinePlan,
    rebuild_artifacts: Sequence[str],
    amdgpu_families: Sequence[str],
    test_type: str,
) -> str:
    lines = ["## Pull request coverage plan", ""]
    if selection.projects or selection.skipped:
        lines += ["| Project | Measured |", "| --- | --- |"]
        lines += [f"| `{key}` | yes |" for key in selection.projects]
        lines += [
            f"| `{name}` | no: {reason} |"
            for name, reason in sorted(selection.skipped.items())
        ]
        lines.append("")
    if not selection.projects:
        lines.append("No project to measure; nothing is built.")
        return "\n".join(lines)

    if baseline.run_id:
        run = (
            f"[{baseline.run_id}]({baseline.html_url})"
            if baseline.html_url
            else baseline.run_id
        )
        how = "as requested" if baseline.explicit else "newest healthy main run"
        lines.append(f"- Baseline: run {run} from `{baseline.repository}` ({how}).")
    else:
        lines.append("- Baseline: none.")
    lines.append(
        "- compiler-runtime: "
        + (
            "copied from the baseline."
            if baseline.prebuilt_compiler_runtime
            else "built."
        )
    )
    lines.append(
        "- Instrumented math-libs build: "
        + (
            "limited to " + ", ".join(f"`{a}`" for a in rebuild_artifacts) + "."
            if rebuild_artifacts
            else "whole stage."
        )
    )
    lines.append(
        f"- GPU families: {', '.join(amdgpu_families)}; test type: {test_type}."
    )
    lines += [f"- {note}" for note in baseline.notes]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_outputs",
        help="Print the outputs to stdout instead of writing GITHUB_OUTPUT",
    )
    args = parser.parse_args(argv)

    labels = _pr_labels()
    selection = select_projects(
        os.getenv("PROJECTS_TO_TEST", ""), os.getenv("CHANGED_PROJECTS", ""), labels
    )
    amdgpu_families = parse_amdgpu_families(
        os.getenv("AMDGPU_FAMILIES", "") or DEFAULT_AMDGPU_FAMILIES
    )
    test_type = resolve_test_type(os.getenv("TEST_TYPE", ""), labels)

    baseline = BaselinePlan()
    rebuild_artifacts: list[str] = []
    if selection.projects:
        running_repository = os.getenv("GITHUB_REPOSITORY", "")
        therock_repository = os.getenv("THEROCK_REPOSITORY") or running_repository
        max_age_raw = os.getenv("COVERAGE_BASELINE_MAX_AGE_HOURS", "")
        max_age_hours = (
            float(max_age_raw) if max_age_raw else DEFAULT_BASELINE_MAX_AGE_HOURS
        )
        baseline = plan_baseline(
            baseline_run_id=os.getenv("BASELINE_RUN_ID", "").strip(),
            baseline_repository=os.getenv("BASELINE_REPOSITORY", "").strip(),
            running_repository=running_repository,
            therock_repository=therock_repository,
            find_baseline=_safe_finder(
                lambda: find_baseline_run(
                    selection.projects,
                    amdgpu_families,
                    github_repository=therock_repository,
                    max_age_hours=max_age_hours,
                )
            ),
            changed_files=_pr_changed_files,
        )
        if baseline.run_id:
            rebuild_artifacts = rebuild_artifacts_for(selection.projects)

    outputs = {
        "enabled": "true" if selection.projects else "false",
        "projects_to_test": ",".join(selection.projects),
        "amdgpu_families": ",".join(amdgpu_families),
        "test_type": test_type,
        "baseline_run_id": baseline.run_id,
        "baseline_repository": baseline.repository,
        "rebuild_artifacts": ",".join(rebuild_artifacts),
        "prebuilt_compiler_runtime": (
            "true" if baseline.prebuilt_compiler_runtime else "false"
        ),
    }
    summary = render_summary(
        selection, baseline, rebuild_artifacts, amdgpu_families, test_type
    )

    if args.print_outputs:
        print(summary)
        print(json.dumps(outputs, indent=2))
    else:
        gha_append_step_summary(summary)
        gha_set_output(outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
