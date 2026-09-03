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
from _therock_utils.build_topology import get_topology

# Hardcoded for now — prebuilt artifacts are always fetched from ROCm/TheRock
# workflow runs. TODO(#3399): when baseline_run_id carries a repo qualifier,
# pass the repo slug through from CIInputs instead of hardcoding.
_REPO_SLUG = "ROCm/TheRock"

# Stages that emit a per-stage build_observability.html report, keyed by the
# `platform` value used to build the WorkflowOutputRoot log prefix.
#
# The report is produced by the "Analyze build times" step, which today only
# lives in .github/workflows/multi_arch_build_portable_linux_artifacts.yml. Only
# the Linux stages routed through that reusable workflow generate a report, so
# only they get a link here. `wsl-rocdxg` (a separate reusable workflow) and the
# Windows build workflow are not wired for the report yet and are intentionally
# omitted to avoid permanently-404 links.
#
# Keep this in sync with the stage jobs in
# .github/workflows/multi_arch_build_portable_linux.yml. Stage names must match
# `[build_stages.*]` in BUILD_TOPOLOGY.toml (used for per-arch fan-out).
_OBSERVABILITY_STAGES: dict[str, list[str]] = {
    "linux": [
        "compiler-runtime",
        "runtime-tests",
        "math-libs",
        "comm-libs",
        "storage-libs",
        "debug-tools",
        "dctools-core",
        "profiler-apps",
        "cv-libs",
        "media-libs",
    ],
}


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

    _append_build_observability(lines, ci_inputs, outputs)

    lines.append("### test-rocm")
    lines.append("")
    _append_test_rocm(lines, outputs)

    lines.append("### build-pytorch")
    lines.append("")
    _append_build_pytorch(lines, outputs)

    lines.append("### build-jax")
    lines.append("")
    _append_build_jax(lines, outputs)

    return "\n".join(lines)


def _format_skipped_ci(lines: list[str], ci_inputs: CIInputs) -> str:
    # Determine skip reason (same priority order as should_skip_ci).
    if "ci:skip" in ci_inputs.pr_labels:
        reason = "`ci:skip` PR label"
    else:
        reason = "no CI-relevant files changed"

    lines.append(f"CI was **skipped**: {reason}. See logs for details.")
    return "\n".join(lines)


def _non_default_highlights(ci_inputs: CIInputs) -> list[str]:
    highlights: list[str] = []

    if ci_inputs.release_type:
        highlights.append(f"Release type: {ci_inputs.release_type}")

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
            filter_type = label.split(":")[1]
            highlights.append(
                f"Label `{label}`: overrode test level to `{filter_type}`"
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

    # Link to log, artifact, and manifest index pages
    lines.extend(
        [
            "## Build outputs",
            "",
            "Platform | 📋 Logs | 📦 Artifacts | 📄 Manifests",
            "-- | -- | -- | --",
        ]
    )
    linux_output_root = None
    for platform_name in ["linux", "windows"]:
        output_root = WorkflowOutputRoot.from_workflow_run(
            run_id=ci_inputs.run_id, platform=platform_name
        )
        if platform_name == "linux":
            linux_output_root = output_root
        log_url = output_root.log_root_index().https_url
        artifact_url = output_root.artifact_index().https_url
        manifest_url = output_root.manifests_index().https_url
        lines.append(
            f"{platform_name.capitalize()} | {log_url} | {artifact_url} | {manifest_url}"
        )
    manifest_diff_url = linux_output_root.log_file(
        "manifest-diff", "index.html"
    ).https_url
    lines.append(f"Manifest diff *(if produced)* | {manifest_diff_url} | — | —")
    lines.append("")
    lines.append(
        "> The manifest-diff report compares submodules across the CI commit range. "
        "Expect no changes when this run does not advance any submodule pointers. "
        "The report is generated after the setup job completes and the link may be "
        "unavailable until then; it is not produced on ASAN workflows."
    )


def _append_build_observability(
    lines: list[str], ci_inputs: CIInputs, outputs: CIOutputs
) -> None:
    """Append one consolidated table of per-stage build-observability links.

    Previously each stage job appended its own `[Build Observability]` link to
    that job's step summary, scattering ~10 links across the run and bloating the
    aggregated summary page. This gathers them into a single table in the
    top-level configure summary. Links are deterministic (derived from the stage
    log layout) so they can be rendered before the stages finish; a link 404s
    until its stage uploads its logs.
    """
    linux_config = outputs.builds.linux
    if linux_config is None:
        return

    # Stages fetched from a baseline run (prebuilt) or excluded (skipped) do not
    # produce a fresh report in this run's output tree, so omit them.
    omit = set(linux_config.prebuilt_stages) | set(linux_config.skip_stages)

    stage_types = {s.name: s.type for s in get_topology().get_build_stages()}
    families = [f["amdgpu_family"] for f in linux_config.per_family_info]

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=ci_inputs.run_id, platform="linux"
    )

    rows: list[str] = []
    for stage in _OBSERVABILITY_STAGES["linux"]:
        if stage in omit:
            continue
        # Per-arch stages produce one report per family; generic stages one.
        stage_families = families if stage_types.get(stage) == "per-arch" else [""]
        for family in stage_families:
            url = output_root.build_observability_stage(stage, family).https_url
            family_cell = f"`{family}`" if family else "—"
            rows.append(f"`{stage}` | {family_cell} | {url}")

    if not rows:
        return

    lines.append("## Build Observability")
    lines.append("")
    lines.append("Stage | Family | 📈 Report")
    lines.append("-- | -- | --")
    lines.extend(rows)
    lines.append("")

    profiling_on = _resource_profiling_enabled(ci_inputs)
    status = "**ON**" if profiling_on else "**OFF**"
    lines.append(
        f"> **Resource profiling: {status}** for this run — "
        "when ON, each report embeds a CPU/memory usage timeline "
        "(`resource_info.py` replaces ccache as the compiler launcher); when OFF, "
        "reports contain ninja build timings only. Profiling defaults ON for "
        "`nightly`/`release` builds and OFF otherwise (it sits in the compile hot "
        "path). Force it via the `force_resource_profiling` workflow_dispatch "
        'input: `"true"` to force on, `"false"` to force off, empty to use the '
        "default gate."
    )
    lines.append("")
    lines.append(
        "> Per-stage build-time reports. Each link becomes available once that "
        "stage finishes uploading its logs. Linux only for now; prebuilt and "
        "skipped stages are omitted."
    )
    lines.append("")


def _resource_profiling_enabled(ci_inputs: CIInputs) -> bool:
    """Compute whether resource-usage profiling is active for this run.

    Mirrors the ENABLE_RESOURCE_PROFILING gate in
    multi_arch_build_portable_linux_artifacts.yml so the summary reports the
    effective state:
      force == "true"  -> on
      force == "false" -> off
      otherwise        -> on for release_type nightly/release, else off
    """
    force = (ci_inputs.force_resource_profiling or "").strip().lower()
    if force == "true":
        return True
    if force == "false":
        return False
    return ci_inputs.release_type in ("nightly", "release")


def _append_build_pytorch(lines: list[str], outputs: CIOutputs) -> None:
    lines.append("| Platform | Python | PyTorch ref | Families |")
    lines.append("|----------|--------|-------------|----------|")

    rows = 0
    for platform, config in [
        ("Linux", outputs.builds.linux),
        ("Windows", outputs.builds.windows),
    ]:
        if config is None:
            continue
        for row in config.pytorch_build_matrix:
            families = ", ".join(
                f"`{family}`" for family in row["amdgpu_families"].split(";")
            )
            lines.append(
                f"| {platform} | `{row['python_version']}` | "
                f"`{row['pytorch_git_ref']}` | {families} |"
            )
            rows += 1

    if rows == 0:
        lines.append("| — | — | — | — |")
    lines.append("")


def _append_build_jax(lines: list[str], outputs: CIOutputs) -> None:
    lines.append("| Platform | Python | JAX ref | Repository | GFX arch |")
    lines.append("|----------|--------|---------|------------|----------|")

    rows = 0
    for platform, config in [
        ("Linux", outputs.builds.linux),
        ("Windows", outputs.builds.windows),
    ]:
        if config is None:
            continue
        for row in config.jax_build_matrix:
            gfx_arch = row["gfx_arch"] or "—"
            lines.append(
                f"| {platform} | `{row['python_version']}` | "
                f"`{row['jax_ref']}` | `{row['jax_repository']}` | {gfx_arch} |"
            )
            rows += 1

    if rows == 0:
        lines.append("| — | — | — | — | — |")
    lines.append("")


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
