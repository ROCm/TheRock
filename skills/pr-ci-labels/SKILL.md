---
name: pr-ci-labels
description: Suggest the CI labels a pull request should carry (the smallest set that safely tests the change, plus GPU families it needs), or check a PR's labels and which labels its CI runs used. Use when opening, updating, reviewing, or merging a PR in TheRock or a repository that runs TheRock's CI (rocm-libraries, rocm-systems).
argument-hint: '[suggest | check] [PR URL | branch:<name> | local]'
allowed-tools: Bash, Read, Grep, Glob, WebFetch
---

# PR CI labels

CI labels change what a pull request's CI runs: skip it, narrow it to the changed projects or one
platform, or widen it to more GPU families and a deeper test tier. Most PRs carry none, so they run
whatever CI selects by default. That is often far more than the change can reach, and sometimes
less than it needs. This skill picks labels from the diff and checks the labels a PR already has.

It has two actions. Pick the one the user asked for; if it is unclear, ask.

- **Suggest:** "which CI labels should this PR use?"
- **Check:** "are these labels right, and what did CI actually run?"

The rules, sources, and mechanics are in [`reference.md`](reference.md) in this directory. Read it
before either action. Evaluation cases are in [`evals.md`](evals.md).

This skill stands alone, and other skills can call it. A PR-quality skill, for example, can use it
in author, review, and pre-merge work and decide for itself which label choices are acceptable.

______________________________________________________________________

## Operating principles

1. **Read the live docs; never copy the label list.** Labels and their behavior change. Read
   TheRock's label docs, the repository's skip patterns, and the labels the repository has created
   at run time (see "Sources" in `reference.md`).
1. **Never write to GitHub without approval.** Applying or removing a label starts a CI run. Draft
   the `gh pr edit` commands, show them, and run them only after the user approves.
1. **Smallest safe set; when unsure, test more.** Narrow only when the diff clearly cannot reach
   what a label drops. Widen wherever the change needs coverage default CI does not give.
1. **Evidence over assertion.** Ground each label in the files the diff changes, and ground claims
   about what CI ran in the runs and jobs themselves, not in a green summary.

______________________________________________________________________

## Action 1 — Suggest

**Inputs:** the PR or branch (diff and repository). Ask only for what cannot be discovered.

**Workflow:**

1. Run `tools/pr_facts.py --pr <number|url>` (or `--local [--base <ref>]` for a branch without a
   PR). In one call it prints the draft state and current CI labels, the changed files with their
   auto-skip classification, the touched projects and the test jobs default CI would select, the
   GPU families PR CI builds and tests by default, and the CI labels this repository has created.
1. **Fast path:** if every changed file is auto-skipped, the answer is "no labels needed: CI skips
   these paths." Stop there.
1. Otherwise walk the narrowing ladder in `reference.md`, then the widening list (including its
   `ci:asan` triggers), and apply the repository specifics. Use only labels the report lists as
   existing in the repository.
1. Check the combination hazards for the set you chose. Read the rest of `reference.md` or the
   label docs only when a label's behavior is in question.
1. If the PR is a draft, say so: CI does not run on drafts until the PR is marked ready or carries
   `ci:run-on-draft`. That is a fact about when CI runs, not a coverage label to recommend.

**Output:**

- The proposed labels, each with a one-line reason tied to the diff, or "no labels needed" with the
  reason (for example, every path is auto-skipped).
- What default CI would have run that the labels drop.
- A ready-to-paste line for the PR description, for example: "CI labels: `ci:no-downstream-test`:
  only hipBLASLt client code changed, so the rocBLAS, MIOpen, hipBLAS, and hipSPARSELt test jobs
  cannot see it."
- The drafted `gh pr edit <n> --add-label ...` commands, and a note to apply them before the PR is
  marked ready (or before the push that should be tested), so the first full run uses them.

______________________________________________________________________

## Action 2 — Check

**Inputs:** a PR (number or URL).

**Workflow:**

1. Run `tools/pr_facts.py --pr <number|url>` for the PR's current labels and changed files.
1. Judge each label against the diff with the same rules as Suggest. Report labels that drop
   coverage the diff can reach, coverage the diff needs that no label adds, and labels that run more
   than the diff needs.
1. Run `tools/ci_labels.py --pr <number|url>` (add `--json` for machine-readable output). It
   reconstructs the labels each CI run used, flags the uncertain ones, reads whether passing runs
   actually built or tested anything, and lists the commits and label changes since the last tested
   pass. Its verdict (go, caution, or hold) answers "did anything since the last tested pass escape
   CI?"

**Output:** per label, safe / drops reachable coverage / more than needed, with the reason; any
missing coverage; the tool's timeline and verdict; and suggested label changes as drafted commands.
This action reports facts and suggestions. Whether a finding blocks a merge is for the calling
skill or the reviewer to decide.

______________________________________________________________________

## Operational notes

- Use `gh` for PR data, labels, runs, and jobs. `tools/pr_facts.py` and `tools/ci_labels.py` need
  only `gh` (authenticated) and Python 3.11 or later (for `tomllib`; older versions fall back to a
  docs heuristic). `pr_facts.py` takes the default test selection from TheRock's
  `test_tools/determine_rocm_test_dependencies.py`, next to this skill.
- Speed matters: the skill is meant to answer in tens of seconds, and to be cheap enough to run on
  every PR eventually. Start from `pr_facts.py`, take the fast path when it applies, and read
  further only when a decision needs it.
- The tools' tests (`tools/*_test.py`) run in TheRock's `Unit Tests` workflow, which picks up every
  `*_test.py` under `skills/`. Run them locally with `python -m pytest skills/pr-ci-labels`.
