# CI Behavior Manipulation

**Multi-Arch CI** ([`multi_arch_ci.yml`](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml)) is configured by [`configure_multi_arch_ci.py`](../../build_tools/github_actions/configure_multi_arch_ci.py) and reads GPU family definitions from [`amdgpu_family_matrix.py`](../../build_tools/github_actions/amdgpu_family_matrix.py).

## Trigger behavior

The CI pipelines test a growing set of GPU targets depending on trigger type/frequency:

| Trigger type   | Included family groups                                                                                                                             | Notes                                           |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `pull_request` | <ul><li>`amdgpu_family_info_matrix_presubmit`</li></ul>                                                                                            | Common targets with the most test runners       |
| `push`         | <ul><li>`amdgpu_family_info_matrix_presubmit`</li><li>`amdgpu_family_info_matrix_postsubmit`</li></ul>                                             | High priority targets with limited test runners |
| `schedule`     | <ul><li>`amdgpu_family_info_matrix_presubmit`</li><li>`amdgpu_family_info_matrix_postsubmit`</li><li>`amdgpu_family_info_matrix_nightly`</li></ul> | All targets, even those that fail to build      |

### Pull request

CI runs on pull requests if modified files pass the filters in
[`configure_ci_path_filters.py`](../../build_tools/github_actions/configure_ci_path_filters.py).

The following labels may be added to a pull request to modify CI behavior:

| Label or group     | Description                                                                                                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ci:skip`          | Skip all builds and tests                                                                                                                                                                         |
| `ci:run-all-archs` | Build and test all possible architectures                                                                                                                                                         |
| `ci:asan`          | Enable ASAN CI builds and tests. ASAN CI is skipped by default on PRs unless this label is present.                                                                                               |
| `ci:host-asan`     | Alias for `ci:asan`. Enable ASAN CI builds and tests.                                                                                                                                             |
| `gfx...`           | Opt-in to building and testing the specified gfx family (e.g. `gfx120X`, `gfx950`)                                                                                                                |
| `test:...`         | Run tests only for the specified projects (e.g. `test:rocthrust`, `test:hipblaslt`). Sets test level to `full` unless overridden by `test_filter:`. Multiple `test:` labels can be combined.      |
| `test_runner:...`  | Run tests on only custom test machines (e.g. `test_runner:oem`). Single-arch CI only.                                                                                                             |
| `test_filter:...`  | Override the test level (e.g. `test_filter:comprehensive`, `test_filter:quick`). Takes priority over all other test level logic. See [test_filtering.md](./test_filtering.md) for allowed values. |

Pull requests in the component repositories that call Multi-Arch CI may also
carry labels that change the cmake configuration of the build. See
[Label-gated cmake flags](#label-gated-cmake-flags) below.

### Push

CI runs on pushes to `main` if modified files pass the filters in
[`configure_ci_path_filters.py`](../../build_tools/github_actions/configure_ci_path_filters.py).

### Schedule

The
[`multi_arch_release.yml`](https://github.com/ROCm/rockrel/blob/main/.github/workflows/multi_arch_release.yml)
workflow in https://github.com/ROCm/rockrel runs once a day. It selects _all_
families, builds release artifacts, and runs comprehensive tests.

### Workflow dispatch

The Multi-Arch CI pipeline can be triggered manually from its GitHub
Actions workflow page:
\[\[ [Multi-Arch CI workflow dispatch](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml) \]\]
Inputs allow per-platform family selection, test label filtering, and prebuilt
stage configuration.

## Label-gated cmake flags

A label on an *external repository's* pull request can make the multi-arch
presubmit build TheRock with a non-default cmake flag, for that pull request
only. This is how you get CI coverage — including GPU tests — for a code path
that is off by default.

The mechanism applies to runs that an external repository (for example
[`ROCm/rocm-libraries`](https://github.com/ROCm/rocm-libraries)) starts by
calling [`setup_multi_arch.yml`](../../.github/workflows/setup_multi_arch.yml)
with an `external_repo` input. A called reusable workflow runs in the caller's
context, so the labels read are the ones on *that* repository's pull request; no
token and no API call are involved, and it works for pull requests from forks.

Labels are honored only for `pull_request` events. A `workflow_dispatch`, push,
nightly or release run always builds with the default configuration.

The map lives in
[`label_gated_flags.py`](../../build_tools/github_actions/label_gated_flags.py)
and is keyed by repository, then by label. **No labels are mapped today**; the
entry below is an example of the shape:

```python
LABEL_GATED_FLAGS: dict[str, dict[str, list[str]]] = {
    "rocm-libraries": {
        "ci:miopen-hipdnn-wrapper": [
            "-DTHEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER=ON",
        ],
    },
}
```

The repository key is the short name from the `external_repo` payload's
`repository` field, lowercased (`ROCm/rocm-libraries` → `rocm-libraries`).

List each label as a bullet here when you add it to the map.

### Only `-DTHEROCK_FLAG_*` options are allowed

The resolver rejects anything else, and the prefix is not a formality.
`therock_declare_flag` *adds* it: declaring `NAME MIOPEN_ENABLE_HIPDNN_WRAPPER`
creates the cache variable `THEROCK_FLAG_MIOPEN_ENABLE_HIPDNN_WRAPPER`. That
prefixed name is the superbuild knob you set; the flag machinery forwards the
unprefixed name down into the subprojects listed in `SUB_PROJECTS`. Setting the
unprefixed name at the top level does nothing at all — subproject arguments are
an explicit allowlist — and you get a green build with the flag off. See
[flags.md](./flags.md).

`-DTHEROCK_ENABLE_*` is rejected for a different reason: TheRock generates its
own `THEROCK_ENABLE_*` options *after* these are spliced in, and cmake takes the
last `-D`, so such an option would be silently overridden.

Adding a new gated flag therefore takes two pull requests, in order:

1. Declare the flag in [`FLAGS.cmake`](../../FLAGS.cmake) with
   `therock_declare_flag(... SUB_PROJECTS <project>)`. If the flag is
   Linux-only, make it a no-op on Windows here — the Windows build leg gets the
   same options and there is no per-platform key.
1. Add the map entry in `label_gated_flags.py`, a bullet under this section, and
   the label itself to the calling repository's label list.

### A label has an effect if and only if it is a key in the map

There is no naming convention and none is enforced. `ci:<project>-<feature>` is
suggested for new labels, matching the existing style, but nothing checks it.
Avoid bare `ci:` names that collide with the already-crowded namespace
(`ci:skip`, `ci:asan`, `ci:run-all-archs`).

What decides the configuration is the set of labels on the pull request at the
moment the run is triggered — never which label happened to change. A label
already applied keeps taking effect on later pushes, and removing one needs no
special handling because it is simply absent from the next run's label set.

GitHub only lets users with triage or write permission label a pull request, so
a fork contributor cannot self-apply one. That permission boundary is what makes
it safe to inject cmake options from a label at all.

### Caveats

**Any label change restarts the multi-arch presubmit, mapped or not.** The
concurrency group cancels a pull request's in-progress run the moment a new one
starts, and that happens before anything can inspect the label. Declining to
rebuild for an unmapped label would not save the build — it is already dead — it
would only leave the pull request with no build until the next push. Batch label
edits while a build is running if you care about the wasted runner time.

**The gated build replaces the normal one.** A labeled pull request has no green
flag-off signal, because both configurations would collide in the same artifact
store. If you need the baseline too, open a second pull request; separate runs
are namespaced by run ID and never collide.

**A flag-on run builds every stage itself.** Nothing that carries artifacts
between runs is keyed on cmake flags, so a flag-on run would otherwise inherit
stages that were built flag-off. `configure_multi_arch_ci.py` therefore drops
`prebuilt_stages` and `baseline_run_id` and downgrades the stage reuse mode to
`dry-run` whenever a label flag is active, naming what it dropped in the log.
Expect a full build's wall time, not an incremental one.

**Re-running an old run replays its original payload.** The labels a re-run sees
are the ones that were on the pull request when that run was *first* triggered,
so applying a label and then re-running a stale failure gives you a flag-off
build that looks fine. Push a commit or re-apply the label instead. The step
summary always names the exact label set and the flags the run acted on, which
is the way to confirm what a given run actually built.

**Removing a label does not always rebuild.** A workflow only reacts to the
label events it subscribes to; the rocm-libraries ASAN workflow, for instance,
subscribes to `labeled` but not `unlabeled`. Remove the label and push.

**The pinned TheRock ref decides which labels exist.** A calling repository pins
a TheRock commit, so a label added to the map after that pin is unknown to the
run and the build goes green with the flag off. The step summary is the check:
it names the matched labels, so an empty match on a labeled pull request means
the pin predates the map entry.

## Prebuilt stages

> [!NOTE]
> This feature is under active development and will evolve as
> automatic stage selection and baseline run lookup are added.
>
> See https://github.com/ROCm/TheRock/issues/3399 for details and
> [`stage_reuse.md`](stage_reuse.md) for the automatic stage-reuse layer
> that builds on the manual inputs described below.

The [Multi-Arch CI](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml)
workflow supports skipping individual build stages by copying their artifacts
from a previous workflow run. This will be used in a few scenarios. For example:

- Changes to the rocm-libraries project will use prebuilt artifacts for
  `compiler-runtime`
- Changes to just test scripts or python packages will use prebuilt artifacts for
  all stages

Two workflow inputs control this:

- **`prebuilt_stages`**: Comma-separated list of stage names to skip
  (e.g. `compiler-runtime,runtime-tests,math-libs`). Artifacts for these stages are copied
  from the baseline run instead of being built. Applied to both Linux and
  Windows; stages not present on a platform are ignored.
- **`baseline_run_id`**: The workflow run ID to copy prebuilt artifacts from.
  Required when `prebuilt_stages` is set. Find this in the URL of a previous
  successful Multi-Arch CI run
  (e.g. https://github.com/ROCm/TheRock/actions/runs/22777631940).

> [!IMPORTANT]
> The baseline run must have built the GPU families you want for the current
> run, otherwise the copy will find no matching artifacts.

### Stage names

Stage names come from [`BUILD_TOPOLOGY.toml`](/BUILD_TOPOLOGY.toml).

Currently, stage names must be explicitly specified. In the future these may
be computed based on dependencies and a special "all" option may be available.

<!-- TODO: The workflows currently use `contains(prebuilt_stages, 'name')` for
     substring matching, which would break if a stage name is a prefix of
     another. When configure_multi_arch_ci.py generates the stage list
     automatically, switch to a JSON array and use `fromJSON()` + `contains()`
     for exact matching. -->

For now, these are the common configurations used for testing:

```
compiler-runtime
compiler-runtime,runtime-tests,math-libs,comm-libs,debug-tools,dctools-core,profiler-apps,cv-libs,media-libs
```
