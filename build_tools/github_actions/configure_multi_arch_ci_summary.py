"""Formats the GITHUB_STEP_SUMMARY markdown for configure_multi_arch_ci.py.

Produces human-readable markdown explaining what CI will do and why.
See reviews/summary_format_v5.md in the claude-rocm-workspace for the
design rationale and example outputs.
"""

import json

from configure_multi_arch_ci import (
    CIInputs,
    CIOutputs,
    GitContext,
)

_DAG = """\
```
# Build graph
build-rocm ──┬── test-rocm
             └── build-rocm-python ── build-pytorch
```"""

_PATH_FILTERS_URL = (
    "https://github.com/ROCm/TheRock/blob/main/"
    "build_tools/github_actions/configure_ci_path_filters.py"
)


def format_summary(
    ci_inputs: CIInputs,
    git_context: GitContext,
    outputs: CIOutputs,
) -> str:
    """Generate the full step summary markdown."""
    lines = ["## Multi-Arch CI Configuration", ""]

    if not outputs.is_ci_enabled:
        return _format_skipped(lines, git_context)

    if not outputs.jobs:
        return "\n".join(lines)

    # One-liner: trigger, branch, variant
    lines.append(
        f"Trigger: `{ci_inputs.event_name}` on `{ci_inputs.branch_name}` branch, "
        f"`{ci_inputs.build_variant}` variant."
    )
    lines.append("")

    # Nothing to build (e.g. workflow_dispatch with no families selected)
    if outputs.builds.linux is None and outputs.builds.windows is None:
        lines.append("No GPU families selected — nothing to build or test.")
        return "\n".join(lines)

    # Non-default callout
    callouts = _non_default_callouts(ci_inputs, outputs)
    if callouts:
        lines.append("> [!NOTE]")
        lines.append("> **Non-default configuration:**")
        for callout in callouts:
            lines.append(f"> - {callout}")
        lines.append("")

    # Fixed DAG
    lines.append(_DAG)
    lines.append("")

    # build-rocm
    lines.append("### build-rocm")
    lines.append("")
    _append_build_rocm(lines, outputs, ci_inputs.build_variant)

    # test-rocm
    lines.append("### test-rocm")
    lines.append("")
    _append_test_rocm(lines, outputs)

    return "\n".join(lines)


def _format_skipped(lines: list[str], git_context: GitContext) -> str:
    lines.append(
        f"CI was **skipped**: no CI-relevant files changed "
        f"(see [configure_ci_path_filters.py]({_PATH_FILTERS_URL}) "
        f"for skip patterns)."
    )
    if git_context.changed_files:
        lines.append("")
        lines.append("Changed files:")
        lines.append("```")
        for path in git_context.changed_files:
            lines.append(path)
        lines.append("```")
    return "\n".join(lines)


def _non_default_callouts(ci_inputs: CIInputs, outputs: CIOutputs) -> list[str]:
    callouts: list[str] = []
    jobs = outputs.jobs

    # Explicit family selection (workflow_dispatch)
    if ci_inputs.is_workflow_dispatch:
        if ci_inputs.linux_amdgpu_families or ci_inputs.windows_amdgpu_families:
            parts = []
            if ci_inputs.linux_amdgpu_families:
                fams = ", ".join(ci_inputs.linux_amdgpu_families)
                parts.append(f"Linux: `[{fams}]`")
            if ci_inputs.windows_amdgpu_families:
                fams = ", ".join(ci_inputs.windows_amdgpu_families)
                parts.append(f"Windows: `[{fams}]`")
            callouts.append(f"Explicit family selection — {', '.join(parts)}")

    # PR labels that affect behavior
    for label in ci_inputs.pr_labels:
        if label.startswith("gfx"):
            callouts.append(
                f"Label `{label}`: added family `{label}` "
                f"(not in default presubmit set)"
            )
        elif label.startswith("test_filter:"):
            callouts.append(
                f"Label `{label}`: overrode test level " f"(default would be `quick`)"
            )
        elif label.startswith("test:"):
            callouts.append(f"Label `{label}`: requested component tests")

    # Prebuilt stages
    if jobs and jobs.build_rocm.prebuilt_stages:
        stage_list = ", ".join(jobs.build_rocm.prebuilt_stages)
        run_id = jobs.build_rocm.baseline_run_id
        repo = _repo_slug()
        callouts.append(
            f"Prebuilt stages: `[{stage_list}]` from run "
            f"[{run_id}](https://github.com/{repo}/actions/runs/{run_id})"
        )

    return callouts


def _append_build_rocm(
    lines: list[str], outputs: CIOutputs, build_variant: str
) -> None:
    jobs = outputs.jobs

    # Prebuilt info
    prebuilt = jobs.build_rocm.prebuilt_stages
    if prebuilt:
        stage_list = ", ".join(prebuilt)
        run_id = jobs.build_rocm.baseline_run_id
        repo = _repo_slug()
        lines.append(
            f"Using prebuilt artifacts for stages: `[{stage_list}]` "
            f"from run [{run_id}]"
            f"(https://github.com/{repo}/actions/runs/{run_id}). "
            f"Remaining stages build from source."
        )
    else:
        lines.append("Building all stages from source.")
    lines.append("")

    # Platform table
    lines.append("| Platform | Families | Artifact Group |")
    lines.append("|----------|----------|----------------|")
    for platform, config in [
        ("Linux", outputs.builds.linux),
        ("Windows", outputs.builds.windows),
    ]:
        if config is None:
            lines.append(f"| {platform} | — | — |")
        else:
            families = ", ".join(
                f"`{f}`" for f in config.dist_amdgpu_families.split(";")
            )
            lines.append(f"| {platform} | {families} | `{config.artifact_group}` |")
    lines.append("")


def _append_test_rocm(lines: list[str], outputs: CIOutputs) -> None:
    jobs = outputs.jobs
    test_rocm = jobs.test_rocm

    lines.append(
        f"Test level: **{test_rocm.test_type}** ({test_rocm.test_type_reason})"
    )

    # Component test labels
    test_labels = []
    if outputs.linux_test_labels:
        test_labels.append(outputs.linux_test_labels)
    if outputs.windows_test_labels:
        test_labels.append(outputs.windows_test_labels)
    if test_labels:
        labels_str = ", ".join(f"`{t}`" for t in test_labels)
        lines.append(f"Component tests: {labels_str}")
    lines.append("")

    # Per-family test runner table
    lines.append("| Platform | Family | Runner Label | Scope |")
    lines.append("|----------|--------|--------------|-------|")
    for platform, config in [
        ("Linux", outputs.builds.linux),
        ("Windows", outputs.builds.windows),
    ]:
        if config is None:
            continue
        per_family = config.per_family_info
        for entry in per_family:
            family = f"`{entry['amdgpu_family']}`"
            runner = f"`{entry['test-runs-on']}`" if entry["test-runs-on"] else "—"
            if entry.get("sanity_check_only_for_family"):
                scope = "sanity check only"
            else:
                scope = test_rocm.test_type
            lines.append(f"| {platform} | {family} | {runner} | {scope} |")
    lines.append("")


def _repo_slug() -> str:
    """Return OWNER/REPO from GITHUB_REPOSITORY, or a placeholder."""
    import os

    return os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
