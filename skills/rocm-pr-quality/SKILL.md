---
name: rocm-pr-quality
description: Help an engineer author, review, or pre-merge-gate a pull request to a ROCm library so it is traceable, tested, and safe to merge. Use when preparing a PR, reviewing a PR, or deciding whether an approved PR is safe to merge right now, or when the user provides a GitHub PR URL or branch and asks for help with PR quality, description, testing, review, or merge-readiness. Library-agnostic base; component overlays extend it.
argument-hint: '[author | review | pre-merge] [PR URL | branch:<name> | local] [base:<branch>]'
allowed-tools: Bash, Read, Grep, Glob, Task, WebFetch
---

# ROCm PR Quality (base)

This is the library-agnostic base skill for PR quality across ROCm. It exposes three
actions the user invokes with a single sentence; the skill does the legwork through the
tools already on the desk (`gh`, the repo, and Jira/GitHub MCPs when available).

- **Author assist** — "help me write a PR worth reviewing."
- **Review assist** — "help me review this PR well."
- **Pre-merge gate** — "is it actually safe to press merge right now?"

Pick the action from the user's request. If it is ambiguous, ask which one they want.

Detailed rubric tables, the PR description template, change classes, the test-level map,
waiver codes, and the MUST set live in `reference.md` in this directory. Read it before
running an action.

______________________________________________________________________

## Operating principles

1. **Paved road, not compliance cop.** The goal is to make the high-quality path the easy
   path. Be the helpful senior reviewer, not the gate guard. "Easy = used" is the point.
1. **Never write to external systems without explicit human approval.** Do not post PR
   descriptions, review comments, issue/ticket comments, labels, or merges to GitHub or
   Jira on your own. Draft the text, show it, and wait for the user to approve or edit.
   This applies even when the user says "review the PR" — you produce the review, the human
   posts it.
1. **Advisory first.** This skill raises the floor; it does not approve and it does not
   block a merge by itself. Hard enforcement (a required CI status check, a merge queue) is a
   separate, DevOps-owned lane. Do not assume it exists.
1. **Conform to the machine gate; it outranks this skill.** The repo's contributing guide is the
   source of truth for PR policy, but when a machine gate enforces it — a policy bot, a required
   status check that parses the branch/title/description, a merge queue — that gate is the hoop a
   PR must actually clear, so make the PR conform to it before anything else. Both the guide and
   the gate outrank this skill; it defers to them and never works around the gate. A machine gate
   honors none of this skill's waivers or self-evident exemptions, so where they disagree the gate
   wins. When you advise the author to do something solely to satisfy the gate that is **not** in
   the contributing guide, say so explicitly so they know the requirement came from the gate, not
   the guide.
1. **Follow the repo's own standards.** When a component repo ships its own
   testing/contributing/standards docs (e.g. `CONTRIBUTING.md`, a `docs/testing*.md`, a per-repo
   best-practices file), read and apply them rather than inventing guidance. The skill checks that
   repo standards are followed; it does not replace them.
1. **Discover, do not hardcode.** Supported architectures, CI labels, test lanes, and
   tracker prefixes drift. Discover them from the repo and CI config at run time.
1. **Evidence over assertion.** Ground every finding in a `file:line`, a CI link, a diff
   hunk, or a resolved tracker. Do not invent test results or completed runs.

______________________________________________________________________

## How overlays extend this base

Each ROCm library may publish a thin overlay skill (e.g. `hipblaslt-pr-quality`) that
depends on this base. Overlays may **add** rules, **tighten** thresholds, and bind rules to
component paths. Overlays may **not** weaken a base MUST-rule. On any conflict, the base
MUST-rule wins.

When invoked through an overlay, read this base first, then apply the overlay's supplements
on top.

______________________________________________________________________

## Shared concepts (used by all three actions)

### Change classification (extensible; multiple tags allowed)

Map touched paths + intent to one or more change classes. A PR may carry more than one tag
(e.g. `defect-fix` + `kernel/tuning`). When ambiguous, pick the **stricter** class.

Base classes: `new-public-api`, `new-op/dtype/path`, `heuristic/default-selection`,
`kernel/tuning`, `build/ci`, `docs`, `revert`, `defect-fix`, `regression-test-only`, `other`.
Overlays may add component-specific classes. The class drives the test/flag bar (see
`reference.md`).

### The MUST set (M1–M5, non-overridable by overlays)

- **M1** Defect-fix PRs include a regression test, or a tracked two-PR known-bug plan.
- **M2** Product-code changes carry tests, a safe-default flag, or a written waiver.
- **M3** Never disable, skip, or weaken tests solely to green CI. *(hard MUST — no waiver.)*
- **M4** Non-trivial PRs carry work tracking (ticket, public issue, or credible no-tracker reason).
- **M5** PRs link the artifacts they relate to (work-tracking item, the defect they fix,
  directly related PRs), and those links must resolve.

M2, M4, M5 are **escape-hatch MUSTs**: waiver-able with an explicit written reason that a
reviewer adjudicates. M1 is waiver-able only via the tracked two-PR plan. M3 has no waiver.

### Waivers (author-declared, reviewer-adjudicated)

An exception is never a silent self-override. The author declares a waiver code + a one-to-
two-sentence reason (and any required approver); the reviewer accepts (`APPROVED`, with the waiver
recorded) or rejects a weak one (`CHANGES REQUESTED`). A bare "N/A" is not a waiver — "why" is mandatory even
when the rule is waived. Higher-risk waivers need a named approver. Waiver codes are in
`reference.md`.

Waivers govern the human review floor only. They do **not** override a machine-enforced policy
gate (see "Conform to the machine gate" in the operating principles): conform to that gate
regardless of any waiver this skill would otherwise allow.

### Test substance — the mutation question (and the "why")

Presence of tests is not enough. For 2–3 sampled tests ask: *what specific change to the
source would make this test fail?* If the answer is hand-wavy, it is coverage padding — a
blocker, not a nit. Smell scan: assert-callable-only, `hasattr`-guarded bodies, lone
`is not None`/`.called` assertions, copy-paste that should be parametrized, mock-only
assertions that pass against a no-op, and phantom methods that do not exist in the source.

Also ask for the **rationale** behind non-obvious test choices: where did this tolerance
value come from, why this parameter combination and not another? A short paper trail prevents
the "author is long gone, nobody knows why this test exists" problem. Prompt the author to
record it; flag its absence on non-obvious numeric tolerances or parameter sets.

### Blast radius & device/architecture coverage

Judge from what the diff actually changes, not the file's path. Decide whether the change is
architecture-independent (wiring/plumbing with no kernel-selection/default/support-surface
change → standard CI is enough), behavior-shifting (defaults/dispatch → multi-arch run
warranted), arch-scoped (only the affected devices), or support-surface-expanding (full
sweep). Kernel/device code generally needs cross-device coverage; host-only code often does
not. Discover the supported architecture/label set from the repo's CI config; do not hardcode
it. Do not over-escalate: an arch-independent change covered by passing CI needs no sweep, and
saying so is a valid outcome.

### Risk level (1–5)

Take any user-provided hint, then independently evaluate the diff. 1 = docs/comments/metadata
only; 2 = narrow low-blast-radius change, good coverage; 3 = core subsystem / new feature path
/ schema-API addition / dispatch-build-test infra; 4 = broad behavior change, compat-sensitive
public API, perf-critical path, incomplete coverage; 5 = cross-project/architectural, default
behavior change in a critical path, ABI break, large unproven refactor, known unresolved
failures. Residual coverage gaps (a required sweep that has not passed) raise the level.

### Interlinking — every PR is an entry point

A PR should never be a dead end. From any artifact a reader should follow links outward to the
why, the fix, the dependencies, and the predecessors. Forward links should resolve into
reverse edges (GitHub back-references; Jira dev-panel auto-linking when the issue key is in the
branch/PR title). M5 makes the core links mandatory; broader context (design docs, loosely
related PRs) is a SHOULD.

______________________________________________________________________

## Action 1 — Author assist

Goal: the author pushes a PR a reviewer can evaluate without archaeology, that already passes
the obvious quality bar.

**Required inputs** (ask only for what cannot be discovered from the PR/branch/diff/repo):
work-tracking reference (ticket / issue / prior PR / RFC / credible none), a risk hint, testing
performed (groups, exact commands when known, status, devices for hardware tests, run/CI links),
and the blast-radius/device-coverage picture. Accept `N/A`/`none`/`not run` as answers, but do
not render empty `N/A` fields in the body.

**Workflow:**

1. Detect any machine-enforced PR policy on this repo (a policy bot, required status checks) and
   conform the branch name, title, description, and test layout to it first, since it is the gate
   the PR must clear and it outranks this skill's defaults and waivers. Flag any gate requirement
   not found in the contributing guide so the author knows where it came from.
1. Inspect the diff: `gh pr view`, `gh pr diff`, `git diff --stat`, targeted file reads.
1. Classify the change (one or more classes; stricter when unsure).
1. Scan branch name, commit messages, and diff for tracker keys, issue numbers, and referenced
   PRs; pre-fill the Related section; prompt for anything obviously missing (e.g. "this fixes a
   defect — link the defect ticket").
1. Check the test/flag/waiver obligation for the class (see `reference.md`). If product code
   changed, require at least one of: tests exercising the change, a safe-default flag guard plus
   a follow-up tracker, or a documented waiver. "Disable/skip/weaken tests to green CI" is never
   valid.
1. Draft a complete PR description from the template in `reference.md`. New PRs are **draft by
   default**; open ready-for-review only when the user explicitly asks.
1. Prompt for the things authors skip: work tracking, flags for default-path/experimental
   behavior, and adjacent tests ("what else exercises this code path that you should run?").

**Output:** a suggested PR title + body (Markdown), a per-step checklist (pass / warn / fail),
CI labels to consider, and open questions for the author — all before the push. Do not create
or edit the PR on GitHub unless the user explicitly approves the exact text.

______________________________________________________________________

## Action 2 — Review assist

Goal: a consistent, substance-focused review floor, so every reviewer checks the same
fundamentals. Lead with findings ordered by severity, each grounded in `file:line`.

**Setup:** determine the repo root (`git rev-parse --show-toplevel`); inspect changed files
(`gh pr view --json ...`, `gh pr diff --name-only`); save the full diff to a temp file rather
than pasting it into the conversation; prefer local source for cross-reference (a review from a
web diff alone has lower confidence). Do not modify files during review.

**Workflow:**

1. Classify changed files into scope buckets (overlays define component buckets; generic buckets:
   public API, core/runtime, build/CI, tests, docs/tooling).
1. Run the rubric (`reference.md`): scope, change class, work tracking, test/flag obligation,
   work type, and any defect-specific extras (regression test + evidence, or a documented two-PR
   plan).
1. Review for correctness, resource/memory ownership (leaks, RAII, lifetime on failure paths),
   code reuse (flag copy-paste where a helper exists), and build/packaging where touched.
1. Testing review is required every time, even when no test files changed. Do not equate "tests
   added" with "behavior covered" — read assertions, run the mutation question, run the smell
   scan, and check the test "why" for non-obvious choices.
1. Assess blast radius and device/arch coverage; reconcile what the content warrants against what
   the PR actually tested/claimed. Flag gaps; do not over-escalate.
1. Answer the four review questions explicitly: (1) what new/changed functionality lands,
   (2) what test level is appropriate, (3) if tests/flags are omitted, is that acceptable
   (waiver?), (4) what adjacent tests should have been considered.
1. Adjudicate any author-declared waiver: accept a well-justified one, reject a weak one.

**Evidence-based review (CI validation).** Treat CI as data, not a binary. Look at *all* CI runs on
the PR, not only the one linked in the description — a green rollup can hide a skipped lane, a flaky
retry that masked a real failure, or a required check that never ran. Where a baseline exists (the
target branch's latest run, or the PR's own earlier run), compare against it: new failures,
newly-skipped tests, and large step-timing deltas (a job that suddenly runs much longer or shorter)
are signals worth a finding. Calibrate severity against this data instead of asserting risk — "this
lane was green on the base and now fails here" is `BLOCKING`; "the build step is 3× slower" is at
least `IMPORTANT` to ask about. Ground each finding in the specific run/step link.

**Finding tiers** (severity-ordered): `BLOCKING` / `IMPORTANT` / `SUGGESTION` / `FUTURE WORK`
(see `reference.md`). Quick decision framework: correctness / security / incomplete cleanup of
code being modified → `BLOCKING`; will bite users or developers soon → `IMPORTANT`; nice-to-have
on code already being touched → `SUGGESTION`; genuinely out of scope for this PR → `FUTURE WORK`.

**Overall assessment** for the PR: `APPROVED` (policy met — may still carry documented,
reviewer-accepted waivers and optional recommendations), `CHANGES REQUESTED` (one or more
`BLOCKING` items: missing tests/flags/waiver, bad/absent tracking, a defect without a regression
plan, or an unresolved correctness/security finding), or `REJECTED` (fundamental problem with the
approach that needs rework). A valid, documented waiver does not block: that is `APPROVED` with the
waiver recorded.

**Optional delegation:** for broad PRs, you may split the review by scope bucket via subagents
and add cross-cutting testing/reuse passes; otherwise do a single-pass review.

**Output:** the overall assessment + the four answers + a severity-ordered findings list + a
`BLOCKING`-items list. Only on request, and only after the user approves the wording, produce a
draft request-changes comment. Never post it yourself.

______________________________________________________________________

## Action 3 — Pre-merge gate

This is genuinely different from review: a PR can be approved and green and still be the wrong
thing to merge *at this moment*. Run against an already-approved PR, immediately before merge.
This is a final red/green sanity check the dev (or a team's merge group) can lean on — **not** a
merge queue, and it does not force a CI rerun by default.

**Criteria:**

- **Timing / risky moment.** Discourage merging large or default-path changes when nobody is
  around to babysit fallout — end of week, right before a holiday or freeze. This is
  timezone/region-aware: use the owning team's region (overlays configure it, e.g. a Calgary-based
  team) to decide "going into the weekend / end of day." Optionally, when calendar/Teams
  availability is reachable, check that a minimal set of people are online with enough hours left
  in the day. Advisory-with-teeth: a configurable "are you sure?", except where a freeze is in
  effect. Small/low-risk/revert PRs can be exempted.
- **CI currency.** An approved green can be hours or days old. Report when this PR's CI last
  completed against a freshness policy (default SHOULD: flag beyond ~3 days). Whether a rebase +
  re-run is *required* vs *recommended* is a component SHOULD (overlays set it); the base does not
  force a rerun. Also confirm the green is *real*, not just a passing rollup summary (see
  "Evidence-based review" under Action 2): every required lane actually ran and passed, with no
  skipped/cancelled required check hiding under the green.
- **Adjacent-file / stale-base check on develop.** Compare the PR's changed files against the
  files the base branch changed since the PR diverged. No overlap → do not tax velocity, merge is
  fine. Overlap on high-coupling files (shared generators, register/lifetime code, shared
  components — overlays name them) → strong-recommend rebase + re-run. This catches the
  "both PRs individually green, the bug only exists combined" class that nothing compiled until
  after merge.
- **Impacted open PRs (advisory).** Optionally identify other open PRs that touch the same
  high-coupling files and would be affected by this merge. You may *draft* a courtesy
  "consider rebasing" comment for those PRs — but never post it without explicit human approval.

**Output:** a go / caution / hold summary with the specific reason and the recommended action
(e.g. "overlap on a high-coupling file — rebase + re-run before merge"). The decision stays with
the human.

______________________________________________________________________

## Operational notes

- Use `gh` for PR data, diffs, files, and CI status. Use the Jira/GitHub MCPs to look up and
  validate tickets, issues, and related PRs.
- Save large diffs to a temp file; read focused sections rather than pasting everything.
- The optional shared checker `tools/pr_quality_check.py` performs the deterministic subset of
  the MUST checks (description sections, work-tracking presence, link resolution, test-file-touched
  heuristic, CI currency, adjacent-file overlap). It is advisory; semantic judgments (test
  substance, M3) stay with the agent and the human.
