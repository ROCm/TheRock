"""Formats the GITHUB_STEP_SUMMARY markdown for configure_multi_arch_ci.py.

Produces human-readable markdown explaining what CI will do and why.
"""

from configure_multi_arch_ci import (
    CIInputs,
    CIOutputs,
)
from pathlib import Path
import sys

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

sys.path.insert(0, str(THEROCK_DIR / "build_tools"))
from _therock_utils.workflow_outputs import WorkflowOutputRoot

# Hardcoded for now — prebuilt artifacts are always fetched from ROCm/TheRock
# workflow runs. TODO(#3399): when baseline_run_id carries a repo qualifier,
# pass the repo slug through from CIInputs instead of hardcoding.
_REPO_SLUG = "ROCm/TheRock"


def format_summary(
    ci_inputs: CIInputs,
    outputs: CIOutputs,
) -> str:
    """Generate the full step summary markdown."""
    lines = []
    lines.append(
        "## Multi-Arch CI Configuration (tips: [ci_behavior_manipulation.md](https://github.com/ROCm/TheRock/blob/main/docs/development/ci_behavior_manipulation.md))"
    )
    lines.append("")

    if not outputs.is_ci_enabled:
        return _format_skipped_ci(lines, ci_inputs)

    if not outputs.jobs:
        return "\n".join(lines)

    # One-liner: trigger, branch, variant
    lines.append(
        f"Trigger: `{ci_inputs.event_name}` on `{ci_inputs.commit_ref}`, "
        f"`{ci_inputs.build_variant}` variant."
    )
    lines.append("")

    # Nothing to build (e.g. workflow_dispatch with no families selected)
    if outputs.builds.linux is None and outputs.builds.windows is None:
        lines.append("No GPU families selected — nothing to build or test.")
        return "\n".join(lines)

    # Highlight noteworthy non-default settings ahead of the standard output.
    highlights = _non_default_highlights(ci_inputs)
    if highlights:
        lines.append("> [!NOTE]")
        lines.append("> **Non-default configuration:**")
        for callout in highlights:
            lines.append(f"> - {callout}")
        lines.append("")

    lines.append("### build-rocm")
    lines.append("")
    _append_build_rocm(lines, ci_inputs, outputs)

    lines.append("### test-rocm")
    lines.append("")
    _append_test_rocm(lines, outputs)

    return "\n".join(lines)


def _format_skipped_ci(lines: list[str], ci_inputs: CIInputs) -> str:
    # Determine skip reason (same priority order as should_skip_ci).
    if "ci:skip" in ci_inputs.pr_labels:
        reason = "`ci:skip` PR label"
    elif ci_inputs.is_pull_request and "ci:run-multi-arch" not in ci_inputs.pr_labels:
        reason = "PR does not have `ci:run-multi-arch` label"
    else:
        reason = "no CI-relevant files changed"

    lines.append(f"CI was **skipped**: {reason}. See logs for details.")
    return "\n".join(lines)


def _non_default_highlights(ci_inputs: CIInputs) -> list[str]:
    highlights: list[str] = []

    # Explicit family selection (workflow_dispatch)
    if ci_inputs.is_workflow_dispatch:
        parts = []
        if ci_inputs.linux_amdgpu_families:
            families = ", ".join(ci_inputs.linux_amdgpu_families)
            parts.append(f"Linux: `[{families}]`")
        if ci_inputs.windows_amdgpu_families:
            families = ", ".join(ci_inputs.windows_amdgpu_families)
            parts.append(f"Windows: `[{families}]`")
        if parts:
            highlights.append(f"Explicit family selection — {', '.join(parts)}")

    # PR labels that affect behavior
    for label in ci_inputs.pr_labels:
        if label.startswith("gfx"):
            highlights.append(
                f"Label `{label}`: added family `{label}` "
                f"(not in default presubmit set)"
            )
        elif label.startswith("test_filter:"):
            highlights.append(
                f"Label `{label}`: overrode test level (default would be `quick`)"
            )
        elif label.startswith("test_runner:"):
            kernel = label.split(":")[1]
            highlights.append(
                f"Label `{label}`: using `{kernel}` kernel-specific test runners"
            )
        elif label.startswith("test:"):
            highlights.append(f"Label `{label}`: requested component tests")
        elif label.startswith("ci:"):
            highlights.append(f"Label `{label}`")

    # Explicit test labels (workflow_dispatch)
    if ci_inputs.is_workflow_dispatch:
        if ci_inputs.linux_test_labels:
            highlights.append(
                f"Explicit Linux test labels: `{ci_inputs.linux_test_labels}`"
            )
        if ci_inputs.windows_test_labels:
            highlights.append(
                f"Explicit Windows test labels: `{ci_inputs.windows_test_labels}`"
            )

    return highlights


def _append_build_rocm(
    lines: list[str], ci_inputs: CIInputs, outputs: CIOutputs
) -> None:
    # Note: this assumes that the build_rocm job is never skipped.
    # We may decide to skip it under certain conditions in the future
    # (e.g. only editing pytorch-related files, no ROCm-related files).
    # This code will need to adapt then.

    jobs = outputs.jobs

    # Prebuilt info
    prebuilt = jobs.build_rocm.prebuilt_stages
    if prebuilt:
        stage_list = ", ".join(prebuilt)
        run_id = jobs.build_rocm.baseline_run_id
        repo = _REPO_SLUG
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

    # Link to log and artifact index pages
    lines.extend(
        [
            "## Build outputs",
            "",
            "Platform | 📋 Logs | 📦 Artifacts",
            "-- | -- | --",
        ]
    )
    for platform_name in ["linux", "windows"]:
        output_root = WorkflowOutputRoot.from_workflow_run(
            run_id=ci_inputs.run_id, platform=platform_name
        )
        log_url = output_root.log_root_index().https_url
        artifact_url = output_root.artifact_index().https_url
        lines.append(f"{platform_name.capitalize()} | {log_url} | {artifact_url}")


def _append_test_rocm(lines: list[str], outputs: CIOutputs) -> None:
    # Note: this assumes that the test_rocm job is never skipped.
    # We may decide to skip it under certain conditions in the future
    # (e.g. only editing pytorch-related files, no ROCm-related files).
    # This code will need to adapt then.

    jobs = outputs.jobs
    test_rocm = jobs.test_rocm

    lines.append(
        f"Test level: **{test_rocm.test_type}** ({test_rocm.test_type_reason})"
    )

    # Component test labels (per platform)
    if outputs.linux_test_labels:
        lines.append(f"Component tests (Linux): `{outputs.linux_test_labels}`")
    if outputs.windows_test_labels:
        lines.append(f"Component tests (Windows): `{outputs.windows_test_labels}`")
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
