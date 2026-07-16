# Stage Reuse

Stage reuse lets a multi-arch CI run **skip build stages that a change does not
affect** and instead copy those stages' artifacts from a previous, compatible
workflow run. On a change that only touches one component, the unaffected
lower/parallel stages (for example `compiler-runtime`) can be satisfied from a
baseline run instead of rebuilt, cutting build time.

There are two independent mechanisms:

- **Manual reuse** (`prebuilt_stages` + `baseline_run_id`) — you explicitly name
  the stages to reuse and the run to copy them from.
- **Automatic reuse** (`stage_reuse_mode`) — the setup job computes which stages
  are unaffected, verifies their artifacts exist in a healthy baseline run, and
  reuses them for you.

Both feed the same underlying `prebuilt_stages` decision consumed by the build
jobs; automatic reuse simply *adds* to whatever you set manually.

## How it works

[`setup_multi_arch.yml`](https://github.com/ROCm/TheRock/.github/workflows/setup_multi_arch.yml) runs
[`configure_multi_arch_ci.py`](https://github.com/ROCm/TheRock/build_tools/github_actions/configure_multi_arch_ci.py),
which calls into
[`stage_reuse_decision.py`](https://github.com/ROCm/TheRock/build_tools/github_actions/stage_reuse_decision.py).
For every stage the change leaves unaffected, two gates must both pass before the
stage is reused:

1. **Impact gate** (`stage_impact.analyze_stage_impact`) — the changed files do
   not affect the stage, so it is a *candidate* for reuse.
1. **Availability gate** (`baseline_runs.select_baseline_run`) — a single
   healthy, commit-compatible baseline run actually contains the artifacts that
   stage would produce. Availability is verified **independently for every
   platform being built** (Linux and/or Windows): a stage is only reused when
   its artifacts are present for *all* of those platforms. A stage available in
   the Linux baseline but missing from Windows is rebuilt.

The availability gate reuses the existing baseline-selection logic, so a
candidate run must have healthy `Build` jobs *and* contain all required
artifacts. A run with no artifacts is never
selected.

### Modes

`stage_reuse_mode` controls the automatic layer:

- `dry-run` (default) — compute the analysis and **report** which stages would
  be reused (console log + job step summary), but change nothing. Every stage
  still builds. Use this to observe decisions.
- `reuse-stage` — actually reuse the eligible stages: they are merged into
  `prebuilt_stages`, so the orchestrator copies their artifacts and skips the
  build.

Automatic reuse is **disabled** when no build platform is selected.

### Inputs

| Input                        | Default   | Purpose                                                                                                |
| ---------------------------- | --------- | ------------------------------------------------------------------------------------------------------ |
| `stage_reuse_mode`           | `dry-run` | `dry-run` (report only) or `reuse-stage` (auto-reuse unaffected stages).                               |
| `stage_reuse_max_age_hours`  | `72`      | Reject baseline runs older than this many hours (recency window).                                      |
| `stage_reuse_commit_history` | `50`      | Number of recent branch commits to fetch when establishing ancestry for the commit-compatibility rule. |
| `prebuilt_stages`            | `""`      | Manual, comma-separated stages to reuse (or `all`). Always honored, independent of `stage_reuse_mode`. |
| `baseline_run_id`            | `""`      | Run ID to copy manually-listed `prebuilt_stages` artifacts from.                                       |

Example `workflow_call` from a component CI:

```yaml
jobs:
  therock:
    uses: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@main
    with:
      linux_amdgpu_families: "gfx94X"
      stage_reuse_mode: "dry-run"
      # stage_reuse_mode: "reuse-stage"
      # Optional tuning:
      # stage_reuse_max_age_hours: "72"
      # stage_reuse_commit_history: "50"
```

### What "compatible baseline" means

A baseline run is only used when it is:

- on the configured baseline branch (default `main`) of the configured workflow
  (default `multi_arch_ci.yml`),
- recent enough (`stage_reuse_max_age_hours`),
- commit-compatible — its commit is the same as, or an ancestor of, the current
  commit (established from the last `stage_reuse_commit_history` commits), and
- healthy — its `Build` jobs succeeded and it contains every required artifact
  for every platform being built.

If no run satisfies all of these, nothing is reused and every candidate stage is
rebuilt — reuse fails safe toward a full build.

## Reading the report

Both modes emit a `Stage reuse analysis` section to the job step summary and
`[STAGE-REUSE]` lines to the console, including:

- the platforms verified,
- unaffected candidate stages,
- which stages are available in the baseline (per platform), and for
  unavailable ones, which platform is missing the artifacts,
- the baseline run used, and
- in `dry-run`, an explicit note that no build steps were skipped.

## Further features

- Reuse is scoped to a single build configuration/variant. Threading build
  flags through for superrepo builds that vary configuration is tracked in
  [#3399](https://github.com/ROCm/TheRock/issues/3399); until then, do not rely
  on reuse across builds with differing build flags.

## Related

- [ci_overview.md](ci_overview.md) — overall CI pipeline and stages.
- [ci_behavior_manipulation.md](ci_behavior_manipulation.md) — other CI inputs
  and labels.
- `build_tools/github_actions/stage_reuse_decision.py` — implementation and the
  full `STAGE_REUSE_*` environment contract.
