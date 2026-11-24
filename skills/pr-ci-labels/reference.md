# PR CI labels — Reference

Sources, selection rules, and run mechanics for the `pr-ci-labels` skill. The label list itself is
not copied here; it lives in TheRock's docs, so it stays current.

## Contents

- [Sources](#sources-read-these-do-not-copy-them)
- [Narrowing ladder](#narrowing-ladder), [widening list](#widening-list), and
  [combination hazards](#combination-hazards)
- [Recording the choice](#recording-the-choice)
- [Which labels did a run use?](#which-labels-did-a-run-use) and
  [a passing run is not always a tested run](#a-passing-run-is-not-always-a-tested-run)
- [The `ci_labels.py` report](#the-ci_labelspy-report)
- [Repository specifics](#repository-specifics)

______________________________________________________________________

## Sources (read these; do not copy them)

`tools/pr_facts.py` reads most of these for you in one call (skip patterns, the repository's
labels, and the default test selection). Go to the sources below directly when a label's behavior
is in question or the tool cannot reach them.

- **TheRock's label list:**
  [`ci_behavior_manipulation.md`](https://github.com/ROCm/TheRock/blob/main/docs/development/ci_behavior_manipulation.md)
  (the "Pull request" table and its subsections), and
  [`test_filtering.md`](https://github.com/ROCm/TheRock/blob/main/docs/development/test_filtering.md)
  for `test_filter:` values. Repositories that call TheRock's CI, such as rocm-libraries and
  rocm-systems, use the same labels.
- **The auto-skip patterns:** TheRock's
  [`skip-ci-base.toml`](https://github.com/ROCm/TheRock/blob/main/build_tools/github_actions/skip-ci-base.toml)
  applies to every repository, and each repository extends it with its own `skip-ci-config.toml`
  (TheRock's is in `build_tools/github_actions/`; rocm-libraries' is `.github/skip-ci-config.toml`).
  When every changed path matches one of these `fnmatch` patterns (where `*` also matches `/`), CI
  does not build or test at all.
- **The labels this repository has created:** `gh label list -R <owner/repo> --search "ci:"` (and
  `--search "test:"`). A documented label that the repository has not created cannot be applied.
- **What default CI would run:** in a TheRock checkout,
  `python test_tools/determine_rocm_test_dependencies.py --changed-projects <path>` prints the test
  jobs a changed project selects (it accepts paths such as `projects/hipblaslt`), and
  `--explain <component>` shows why. Use it to say what a narrowing label drops.
- **Repository-specific label docs**, where a repository has its own (see
  [Repository specifics](#repository-specifics)).

______________________________________________________________________

## Narrowing ladder

Choose the smallest label set that still covers everything the diff can reach, then widen it where
the change needs more. When unsure, test more. Take the first step that applies, then check the
widening list.

| Step | Question                                                                           | Label                                          | Example                                                                                                                                                                             |
| ---- | ---------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Does every changed path match an auto-skip pattern?                                | None. CI already skips; `ci:skip` adds nothing | A docs typo; a skill change in TheRock                                                                                                                                              |
| 2    | Can the change affect no build and no test, though CI will not skip it on its own? | `ci:skip`                                      | A comment-only edit in a source file, where the diff is its own proof                                                                                                               |
| 3    | Can the change reach only the changed projects, and none of their consumers?       | `ci:no-downstream-test`                        | Test or benchmark client code (for example `projects/hipblaslt/clients/`). Never library code, public headers, kernels, code generation, or build files that consumers link against |
| 4    | Can nothing in the change affect code that runs on a GPU?                          | `ci:cpu-test-only`                             | Host-only tooling whose tests run on the CPU                                                                                                                                        |
| 5    | Is the change confined to one platform?                                            | `ci:platform:linux` or `ci:platform:windows`   | A Linux-only script. Never code that is compiled on both platforms                                                                                                                  |
| 6    | Do only some projects' tests matter?                                               | `test:<project>`                               | A change whose effect is confined to one project's tests                                                                                                                            |

Steps 3 to 6 can combine; step 2 replaces everything else. Labels choose what runs. They are not a
substitute for tests the change needs.

## Widening list

Add these where the diff needs them, even on a narrowed run:

- Code specific to a GPU family that default PR CI does not test (`pr_facts.py` lists which
  families PR CI builds and tests by default, from `amdgpu_family_matrix.py`; do not hardcode
  them): add that family's `ci:gfx...` label. In current code
  (`_apply_arch_labels` in `build_tools/github_actions/configure_multi_arch_ci.py`) a `ci:gfx...`
  label builds and tests the family, bypassing trigger-based gating. The docs' build/test table
  still pairs it with `ci:test:gfx...` to force tests. Until the two agree, add both when a family
  that otherwise tests only nightly must be tested on the PR; the pair is correct under either
  reading. Tests run only where the family has runners. Add `ci:exactly` when only the labeled
  families matter.
- A change to a default path or to which devices are supported: `ci:run-all-archs`.
- Tests that live only in a deeper tier: `test_filter:comprehensive` (or the tier that holds them).
- Memory safety: `ci:asan` (ASAN build plus its tests on the PR) when the diff fixes or could
  introduce a double free, use-after-free, leak, buffer overrun, or uninitialized read, or changes
  who owns or frees memory (allocation pools, caches, smart-pointer or RAII changes, early
  release). A fix for one of these is the strongest signal: it should be tested under ASAN.
- Opt-in features such as `ci:run-wsl` or `ci:run-multi-gpu`: only when the diff touches that
  feature (its code or the workflow that runs it), and say which file needs it.

## Combination hazards

- `test:` labels intersect with the changed-project selection, so they can only narrow it. With
  `ci:no-downstream-test` the intersection can be empty: on a nested project such as TensileLite,
  "changed project only" may not include the project named by the `test:` label.
- `ci:exactly` needs at least one `ci:gfx...` or `ci:build:gfx...` label; a `ci:test:gfx...` label
  alone is an error.
- `ci:skip` wins over `ci:run-on-draft`.
- Draft PRs skip CI, and the Multi-Arch summary check fails on a draft skip so it never passes a
  required check without a build. Iterate in draft, and add `ci:run-on-draft` only when a run is
  wanted. When a PR is a draft, report that CI is not running yet (a red summary on a draft is
  expected, not a defect); `ci:run-on-draft` is about when CI runs, not what it covers, so do not
  suggest it as coverage.
- Adding or removing a label starts a new run. Multi-Arch CI setup reads the PR's labels and draft
  state when it starts, not from the triggering event. Apply labels before marking the PR ready (or
  before the push that should be tested), so the first full run is the right one. Adding several
  labels one at a time starts one run per label; concurrency cancels all but the last.

______________________________________________________________________

## Recording the choice

Put the labels in the PR description with a one-line reason for each, and say what default CI would
have run that the labels drop. For example: "`ci:no-downstream-test`: only hipBLASLt client code
changed (no library, header, or kernel code), so the rocBLAS, MIOpen, hipBLAS, and hipSPARSELt test
jobs cannot see it." With no labels, say "default CI," or, when every changed path is auto-skipped,
"no CI labels needed: CI skips these paths." A reviewer can then check the reasoning, not just the
label names.

______________________________________________________________________

## Which labels did a run use?

Multi-Arch CI setup reads the PR's labels from the GitHub API when the setup job starts. **Re-run
all jobs** runs setup again, so that re-run uses the labels present at the re-run. **Re-run failed
jobs** keeps the earlier setup and the build matrix it chose, so labels added between attempts are
not used. The API does not record which labels setup saw, so reconstruct them: compare the PR's
label events (`gh api repos/<owner>/<repo>/issues/<n>/events --paginate`, the `labeled` and
`unlabeled` entries with their timestamps) against the setup job's `started_at`
(`gh api repos/<owner>/<repo>/actions/runs/<id>/jobs?filter=latest`; a setup job kept by Re-run
failed jobs reports its original start time). The result is exact only when no label changed
within about a minute of that time. If the setup job can't be read, a re-run's labels are uncertain
anywhere between the run's creation and its latest attempt (`created_at` and `run_started_at` in
the Actions runs API). Confirm uncertain labels in the run's setup log, which prints the labels it
read (`pr_labels`) and a `Label '...'` line for each label it acted on.

## A passing run is not always a tested run

Under `ci:skip`, or when every changed path is auto-skipped, the setup and summary jobs still
succeed and the run concludes `success` with nothing built or tested. Count a run as coverage only
when a build or test job actually succeeded. In TheRock's Multi-Arch CI those jobs are named
`<platform>::<variant> / Build ...` or `<platform>::<variant> / Test ...`; a skip-only run has only
setup, configure, and summary jobs.

______________________________________________________________________

## The `ci_labels.py` report

`tools/ci_labels.py --pr <number|url> [--repo owner/name] [--repo-path <clone>] [--json]` follows
the PR's main CI workflow (`Multi-Arch CI` in TheRock, `TheRock Multi-Arch CI` in repositories that
call it). It:

1. Replays the PR's label events to give each run the labels it used, as of its setup job's start
   (read from the run's jobs when it checks them; otherwise the latest attempt's start), and flags
   labels that changed within 60 seconds of that time as uncertain. For a re-run whose setup job it
   did not read, every label change between the run's creation and its latest attempt is uncertain.
   If the label history cannot be read, it uses the PR's current labels, marks every run's labels as
   unknown, and never gives a go that depends on them.
1. Reads the jobs of passing runs, newest first (up to 10), to find the last one with a successful
   build or test job. The jobs decide, not the replayed labels: a `ci:skip` run has no build or test
   jobs, so it never counts, and neither does a run whose jobs cannot be read; the report says how
   many it could not check.
1. Lists every commit after that run with its changed files (all pages), classified with the
   repository's skip patterns. A commit whose files cannot be read counts as code.
1. Gives a verdict:
   - **go:** the head commit passed a run that built or tested it; or every later commit touches
     only CI-skippable paths; or no run built anything and every commit in the PR is CI-skippable.
     Coverage labels added since the last tested pass block a go even when the later commits are
     docs-only, because no run has used them.
   - **hold:** code changed after a narrowing label was added (for example `ci:skip`), and no tested
     run covers it.
   - **caution:** anything else, such as code that no tested run covers yet, coverage labels added
     after the last run, a go that rests on an uncertain coverage label or an unreadable label
     history, or history rewritten since the last tested pass.

The repository comes from the PR URL when one is given, then the current checkout. The report is
advisory; whether a label set was safe for the diff is a judgment the tool does not make.

______________________________________________________________________

## Repository specifics

**TheRock**

- The label docs and skip patterns are local: `docs/development/ci_behavior_manipulation.md`,
  `docs/development/test_filtering.md`, and `build_tools/github_actions/skip-ci-*.toml`.
- Submodule-bump PRs from the bump bots carry `ci:run-all-archs` and `ci:asan` (plus some GPU-family
  labels), because a bump is the first place many GPU families see the new code. Do not suggest
  narrowing them.
- A change to a reusable workflow needs a run that exercises that workflow on the platforms and
  families that call it, so do not narrow past it.
- Changes confined to framework files (PyTorch, JAX) can skip the ROCm build on their own; see
  "Building frameworks with published ROCm" in the label doc before suggesting `ci:skip-rocm`.

**rocm-libraries**

- The skip patterns are in `.github/skip-ci-config.toml`.
- A change under `projects/<name>/` selects that project and, by default, its direct consumers
  (TheRock's `test_tools/test_policies.toml` sets the depth per component). Client and test code
  usually reaches only its own project, so `ci:no-downstream-test` is often safe there. Library
  code, kernels, and code generators (for example TensileLite and StinkyTofu) reach consumers.
- Component CI and Math CI have their own labels (for example `ci:gpu:<gfx>`); check the component's
  overlay skill or `TESTING.md` for them.
