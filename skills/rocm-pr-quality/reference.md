# ROCm PR Quality — Reference

Rubric tables, templates, and vocabularies for the `rocm-pr-quality` base skill. This is the
single source of truth the author/review actions and the shared checker share. Overlays add to
or tighten these; they never relax a MUST.

## Contents

- [MUST set](#must-set-verbatim--non-overridable-by-overlays) and [waiver-ability](#waiver-ability)
- [Rule strengths](#rule-strengths-how-overlays-may-change-a-rule), [change classes](#change-classes-extensible-multiple-tags-allowed), [test-level mapping](#test-level-mapping-should)
- [The four review questions](#the-four-review-questions)
- [Test-substance smell scan](#test-substance-smell-scan-a-hit-is-a-prompt-to-inspect-not-a-verdict) and [agent anti-patterns](#agent--ai-generated-change-anti-patterns-a-hit-warrants-a-closer-read)
- [PR-hygiene checklist](#pr-hygiene-checklist-author--reviewer) and [self-evident changes](#self-evident-changes-reduced-justification-exemptions)
- [Finding tiers](#finding-tiers-individual-findings), [overall assessment](#overall-assessment-pr-level), [risk levels](#risk-levels-15)
- [Waiver codes](#waiver-codes)
- [PR description template](#pr-description-template) and [two-PR known-bug flow](#two-pr-known-bug-flow-satisfies-m1-without-a-same-pr-fix)
- [Worked review output](#worked-review-output-hypothetical-fixture) (evaluation fixtures: [`evals.md`](evals.md))
- [Review attestation](#review-attestation-action-2-approvals-only)

______________________________________________________________________

## MUST set (verbatim — non-overridable by overlays)

- **M1** Defect-fix PRs include a regression test, or a tracked two-PR known-bug plan.
- **M2** Product-code changes carry tests, a safe-default flag, or a written waiver.
- **M3** Never disable, skip, or weaken tests solely to green CI.
- **M4** Non-trivial PRs carry work tracking (ticket, public issue, or credible no-tracker reason).
- **M5** PRs link the artifacts they relate to (work-tracking item, the defect they fix, directly
  related PRs), and those links must resolve.

Overlays may add rules and tighten thresholds; they may not weaken M1–M5.

### Waiver-ability

| Kind              | Rules      | Exception path                                                              |
| ----------------- | ---------- | --------------------------------------------------------------------------- |
| Escape-hatch MUST | M2, M4, M5 | Waiver-able with an explicit written reason; reviewer decides, can reject   |
| Conditional MUST  | M1         | Satisfied by a regression test **or** a tracked two-PR plan; no bare waiver |
| Hard MUST         | M3         | No waiver — there is no acceptable reason                                   |

______________________________________________________________________

## Rule strengths (how overlays may change a rule)

| Strength   | Meaning         | Overlay MAY                                        | Overlay MUST NOT           |
| ---------- | --------------- | -------------------------------------------------- | -------------------------- |
| **MUST**   | ROCm-wide floor | Add new MUSTs; tighten; bind to component paths    | Remove, relax, or waive it |
| **SHOULD** | Tunable default | Tighten or loosen with an owned, written rationale | Silently drop it           |
| **MAY**    | Advisory nicety | Anything                                           | —                          |

______________________________________________________________________

## Change classes (extensible; multiple tags allowed)

A PR may carry more than one class. When ambiguous, pick the stricter one. Overlays may add
component-specific classes (and bind them to paths).

| Class                         | Typical test/flag bar                                                                                                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `new-public-api`              | Functional/API tests; compatibility check; likely flag if behavior-changing                                                                                                                                               |
| `new-op/dtype/path`           | Tests exercising the new path across applicable devices                                                                                                                                                                   |
| `heuristic/default-selection` | Tests + safe-default flag for the new default                                                                                                                                                                             |
| `kernel/tuning`               | Device/arch coverage appropriate to the kernel; perf evidence if perf-sensitive                                                                                                                                           |
| `build/ci`                    | Build/CI green on supported lanes; no behavior change expected                                                                                                                                                            |
| `docs`                        | Doc/link validation only                                                                                                                                                                                                  |
| `revert`                      | Links the PR it reverts (self-evident "why"); regression test if reverting a fix                                                                                                                                          |
| `defect-fix`                  | M1: regression test or tracked two-PR plan, **plus reproduction evidence** — the new test is shown to **fail on the unpatched build (reproducing the cited defect) and pass with the fix**, with the run linked in the PR |
| `regression-test-only`        | The test is the deliverable; must fail before / pass after the fix                                                                                                                                                        |
| `other`                       | Judgment call; default to stricter                                                                                                                                                                                        |

______________________________________________________________________

## Test-level mapping (SHOULD)

Per-repo testing docs may override this mapping. Overlays map these levels to real paths/lanes.

| Change class                  | Default minimum test level                                    |
| ----------------------------- | ------------------------------------------------------------- |
| `defect-fix`                  | Unit/lowest level that fails on the regression, plus evidence |
| `new-public-api`              | Functional/API-level test                                     |
| `new-op/dtype/path`           | Functional + device coverage for the path                     |
| `heuristic/default-selection` | Functional test of the new default + flag                     |
| `kernel/tuning`               | Integration / on-device test across affected archs            |
| `build/ci`                    | CI lane green                                                 |
| `docs`                        | Link/metadata validation                                      |

Principle: pick the **lowest** test level that would actually fail on the regression or exercise
the change; do not demand an integration test where a unit test locks the behavior.

______________________________________________________________________

## The four review questions

1. What new or changed functionality lands in this PR?
1. What test level is appropriate for it (and is it present)?
1. If tests or flags are omitted, is that acceptable — i.e. is there a valid waiver?
1. What adjacent tests should have been considered (second-order impact)?

______________________________________________________________________

## Test-substance smell scan (a hit is a prompt to inspect, not a verdict)

A smell tells you where to look. Read what the test actually asserts before grading it.

- assert-callable only / asserts only that a mock was called
- `hasattr`-guarded test bodies that silently skip
- lone `is not None` / `.called` assertions with no behavioral check
- copy-paste tests that could be parametrized
- mock-only assertions that pass against a no-op implementation
- phantom methods / attributes that do not exist in the source (AI-slop tell)
- non-obvious tolerance or parameter choices with no recorded rationale ("why")

Grading: reserve `BLOCKING` for a **demonstrated** missing guarantee: the test passes against a
no-op, asserts nothing about behavior, cannot fail, silently skips, or exercises a method that does
not exist. A mock-call assertion that is the intended contract (for example, "this function must
call the logger once") is fine. Duplicated tests that each lock distinct behavior are a
maintainability `SUGGESTION` (parametrize when touching them), not a blocker. A missing rationale
for a non-obvious tolerance is `IMPORTANT` when the value gates pass/fail, otherwise `SUGGESTION`.

Mutation question: *what single change to the source would make this test fail?* No clear answer
→ coverage padding.

### Agent / AI-generated change anti-patterns (a hit warrants a closer read)

These recur in AI-assisted PRs and frequently hide a thin or wrong change.

- **Test sprawl** — many new test files / a large patch count for a small behavioral change; new
  tests that restate the implementation rather than lock behavior. Prefer parametrizing one test
  over generating ten near-duplicates.
- **Change-narrative comments** — comments that describe the *diff* ("renamed X to Y", "now also
  handles Z", "previously this did…") instead of the code's intent. The diff already records the
  change; such comments rot immediately. Flag them.
- **"Backward-compat" framing for internal-only code** — keeping a deprecated alias / old code
  path "for compatibility" when the symbol is internal and has no external users. That is
  incomplete cleanup, not compatibility → `BLOCKING`.
- **Over-mocking** — mocking the very thing under test, or so much that the test passes against a
  no-op. The test must exercise real behavior.
- **Excessive patch count / churn** — large reformat or rename noise mixed into a functional
  change, making the real change hard to find. Ask for a split.
- **File-naming / placement drift** — new files that do not follow the repo's existing test naming
  and directory conventions; tests placed where the suite will not pick them up.

______________________________________________________________________

## PR-hygiene checklist (author & reviewer)

Quick, mostly mechanical checks that make a PR reviewable.

- **Title** — concise, follows the repo convention (and carries the tracker key if the repo
  requires it for dev-panel auto-linking).
- **Motivation answers "why"** — the description explains *why* the change is needed, not a
  restatement of the title or a file-by-file list of *what* changed. A reviewer should understand
  the problem before the diff.
- **PR size** — scoped to one logical change. Large mechanical churn (reformat, rename, generated
  code) is split from functional changes so the real change is reviewable. If it must be combined,
  call it out and isolate it in its own commit.
- **Revert vs roll-forward** — for a revert, link the PR being reverted (that is the "why"); for a
  fix-forward, say why forward is safer than reverting.
- **Reviewers / CODEOWNERS** — the right owners are requested; cross-component changes name each
  affected area's owner.
- **No leftover scaffolding** — no debug prints, commented-out code, TODOs without a tracker, or
  generated/temporary files committed by accident.

## Self-evident changes (reduced-justification exemptions)

Not every PR needs the full description, a tracker, or new tests. These classes are self-evident;
do not manufacture process for them (but M3 still always applies — never weaken tests to green CI):

- **Pure docs / comments / typo fixes** — link/metadata validation only; no tracker or tests
  required.
- **Mechanical, tool-driven changes** — auto-formatting, lint autofixes, generated-file
  regeneration, dependency-pin bumps from a bot — where the tool and command are named.
- **Reverts** — the linked reverted PR is the justification (add a regression test only if you are
  reverting a fix).
- **Trivial, obviously-correct edits** — a one-line constant/string fix, a version bump, fixing an
  obvious typo in an identifier — where the diff is its own proof.

For anything outside these classes, the normal MUST set (M1–M5) applies. When in doubt, treat the
change as non-self-evident and ask for the missing justification.

______________________________________________________________________

## Finding tiers (individual findings)

| Tier            | Meaning                                                                                                                                                                                                                                                                                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BLOCKING**    | Must fix before merge: correctness/logic error, security issue, leak/crash in normal use, ABI/API break, breaking change without a migration path, missing required tests for new functionality, an untested broad-blast-radius change to a default/shipping path, or incomplete cleanup of code this PR modifies (dead params/constants/helpers left behind). |
| **IMPORTANT**   | Should fix: real behavioral risk, missing validation for a likely edge case, meaningful test gap, missing required device/arch coverage, or a maintainability issue likely to cause defects soon.                                                                                                                                                              |
| **SUGGESTION**  | Nice to have on code already being touched: clarity, naming, a small refactor, an extra test case.                                                                                                                                                                                                                                                             |
| **FUTURE WORK** | Out of scope for this PR: improvements to code not being modified, larger refactors, follow-up features. Track separately; do not block this PR.                                                                                                                                                                                                               |

Decision framework: correctness/security issue, or incomplete cleanup of code being modified → **BLOCKING**; will cause problems for users/developers soon → **IMPORTANT**; an improvement to code being modified → **SUGGESTION**; otherwise → **FUTURE WORK**. Do not mark unrelated improvements BLOCKING, and do not soften incomplete cleanup to SUGGESTION/FUTURE WORK.

## Overall assessment (PR-level)

| Status              | When                                                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `APPROVED`          | Policy met. May carry documented, reviewer-accepted waivers and optional recommendations. A valid waiver is recorded here, not treated as a block.                                    |
| `CHANGES REQUESTED` | One or more `BLOCKING` items: missing tests/flags/waiver, bad/absent tracking, a defect without a regression plan, an unresolved `BLOCKING` finding, or a weak/again-rejected waiver. |
| `REJECTED`          | Fundamental problem with the approach; needs rework or abandonment.                                                                                                                   |

______________________________________________________________________

## Risk levels (1–5)

| Level | Meaning                                                                                                                                    |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | Docs-only, comments-only, metadata-only, or isolated non-shipping test changes.                                                            |
| 2     | Narrow change, low blast radius, good coverage, no API/schema/build impact.                                                                |
| 3     | Core subsystem, new feature path, schema/API addition, dispatch/build/test infra, or flagged change with real integration surface.         |
| 4     | Broad behavior change, compat-sensitive public API, perf-critical path, complex migration, or incomplete coverage.                         |
| 5     | Cross-project/architectural, default behavior change in a critical path, ABI break, large unproven refactor, or known unresolved failures. |

A required-but-not-yet-passed device/arch run raises the residual risk.

______________________________________________________________________

## Waiver codes

Base set (overlays may add component-specific codes and may tighten who can approve them):

| Code         | Use                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------ |
| `W-DOC`      | Docs/comments-only change; no product behavior.                                                  |
| `W-BUILD`    | Build/CI-only change with no behavior change.                                                    |
| `W-REVERT`   | Revert; the "why" is the reverted PR, which is linked.                                           |
| `W-HOTFIX`   | Emergency fix; regression test deferred to a linked follow-up (needs named approver).            |
| `W-FLAG`     | New behavior landed dark behind a safe-default flag; tests/soak before enable, with a tracker.   |
| `W-NOTICKET` | No tracker at author time; credible reason + a commitment to file & link within a stated window. |

A waiver relaxes only the rule it addresses, and the checker applies it that way. `W-DOC`, `W-BUILD`, `W-REVERT`, `W-HOTFIX`, and `W-FLAG` address the missing-tests check (M2). `W-NOTICKET`, `W-DOC`, and `W-REVERT` address the work-tracking check (M4). Any other code (for example one an overlay adds) is reported as pending the reviewer's adjudication; it does not silently suppress either check.

Every waiver needs a one-to-two-sentence "why." A bare "N/A" is not a waiver. Higher-risk waivers
(e.g. `W-HOTFIX`) require a named approver / lead sign-off.

______________________________________________________________________

## PR description template

```markdown
## Summary

<1-3 sentences: purpose, motivation, what this PR enables. Link named non-tracker references.>

## Risk Assessment

<Risk level 1-5 and a concise rationale (one short paragraph).>

## Related

- Work tracking: <ticket key and/or issue URL>
- Fixes / defect: <tracker>
- Related PRs: <#PR ...>  (dependency, two-PR flow, revert/cherry-pick source)
- Design / docs: <links>

## Device / Architecture Coverage

<Blast radius and which devices/arches must be verified before merge. State whether passing PR
CI is sufficient, a specific-arch run is required, or a full sweep is required, and why. Omit
only for docs/comments/skill-only changes with no device impact.>

## Testing Summary

- <Testing category and what it covers.>

## Testing Checklist

- [x] <Test group> - `<command>` - Status: Passed
- [x] <Hardware test group> - `<command>` - Devices: <list> - Status: Passed
- [ ] <Sweep, only if blast radius requires it> - Devices: <families> - Status: Pending
- [ ] PR CI - GitHub PR checks - Status: Pending

## Flags / Guardrails

<Any feature flag, default value, and the enable plan. "None" if not applicable.>

## Adjacent Tests Considered

<What else exercises this code path; what was run or why it was not needed.>

## Risk Acceptance / Waivers

<Any declared waiver: code + one-to-two-sentence reason + approver if required. Omit if none.>

## Technical Changes

- <Top-level technical what/why.>
```

### Checklist & body rules

- `[x]` only for passed/completed validation; `[ ]` for pending/not-run/failed/unknown.
- Include the command after the test group in backticks when known; omit if no useful command.
- Include `Devices: ...` only for hardware tests; include `Link: ...` only when a real link exists.
- Do **not** render empty placeholder fields (`Devices: N/A`, `Link: N/A`). Omit the field.
- Do not include unverified testing claims or file-by-file changelogs for large PRs.
- Tracker-key-in-title vs in-body is a per-repo convention; follow the repo's. Some repos
  (e.g. Jira-linked) require the key in the title to trigger dev-panel auto-linking — overlays
  state this.

______________________________________________________________________

## Two-PR known-bug flow (satisfies M1 without a same-PR fix)

When a defect is known but the fix is not ready, M1 may be met by a tracked two-PR plan:

1. **Test-only PR**: adds a regression test that documents the bug, quarantined/expected-fail and
   **tracked** (a tracker id + a time-box), not silently skipped.
1. **Fix PR**: lands the fix and removes the quarantine in the same PR.

The quarantine must be tracked, time-boxed, and removed by the fix — not left in place. Overlays
define the concrete mechanism (e.g. a `known_bugs` data file).

______________________________________________________________________

## Worked review output (hypothetical fixture)

This shows the shape a completed review takes. The PR is **invented** for illustration (library
`libfoo`, ticket `FOO-101`); it is not a real PR and its file names, line numbers, and run links
are placeholders. Real reviews cite real paths and runs. Evaluation cases built from the same
shape are in [`evals.md`](evals.md).

> **PR:** "[FOO-101] Fix scan overflow for 64-bit offsets", head `3f9c2a1`, base `develop`.
> **Changed:** `src/scan.cpp` (+14/-3), `tests/scan_test.cpp` (+22/-0). Class: `defect-fix`.

**Evidence checked:** the PR diff at `3f9c2a1`; the new test body (`tests/scan_test.cpp:41-62`);
the PR's CI run (placeholder link, green, finished 1 day before the review); the base branch
at review time.

**Overall assessment:** `CHANGES REQUESTED` (one `BLOCKING` item).

**The four questions**

1. *What lands:* a fix to the offset arithmetic in `scan()` so inputs above 2^31 elements no
   longer overflow a 32-bit intermediate.
1. *Appropriate test level:* a unit test that runs `scan()` over a large (or mocked-size) input
   and checks the result. A unit test is the lowest level that fails on the regression.
1. *Omitted tests or flags, and waivers:* none omitted, no waiver declared.
1. *Adjacent tests:* the 32-bit and 64-bit index instantiations of `scan()`, and `reduce()`,
   which shares the offset helper.

**Findings (most severe first)**

| Finding                                                          | Severity     | Evidence                                                                                                                                                                                              |
| ---------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The regression test cannot fail on the old code                  | `BLOCKING`   | `tests/scan_test.cpp:55` asserts only that a mocked allocator was called. It never reads the scan output, so it passes against the unpatched `scan()` (M1: the test must fail before and pass after). |
| No reproduction evidence for the defect-fix                      | `IMPORTANT`  | The PR does not link a run showing the new test failing on the unpatched build.                                                                                                                       |
| `reduce()` uses the same offset helper and is untested at 64-bit | `IMPORTANT`  | Adjacent-test question 4; `src/reduce.cpp:88` calls the helper changed in this PR.                                                                                                                    |
| Two near-identical test cases differ only in input size          | `SUGGESTION` | `scan_test.cpp:30-39` and `:41-50`; each locks a distinct size, so this is a parametrization nicety, not a gap.                                                                                       |

**Blocking items:** replace the mock-call assertion with one on the scan result for a 64-bit-sized
input, and show it failing on the unpatched build.

**Limits of this review:** reviewed commit `3f9c2a1` only; did not build or run the tests locally;
did not review `reduce()` beyond the call site; CI evidence is the one linked run.

______________________________________________________________________

## Review attestation (Action 2, approvals only)

When a review concludes `APPROVED`, the skill appends this block to the drafted approval comment
(never to `CHANGES REQUESTED` / `REJECTED`, and never to a PR description). The human posts it; the
skill never posts on its own. The `<!-- rocm-pr-quality:attestation v1 -->` marker is what a
consumer (e.g. a pre-review gate) keys on, and the `Attestation` digest ties the block to this
specific PR so a pasted template or a block copied from another PR can be detected. The digest is
deliberately loose: commits pushed after the review are allowed and do not invalidate it.

An attestation records an independent review, so the skill never produces one for the user's own PR,
whether they are authoring it or reviewing it themselves. A self-review still gives feedback, just
without the block. Compare the PR's author with the user's GitHub login; one person with several
accounts counts as one author, and if you cannot tell, ask.

Block (leaned-into default):

```markdown
<!-- rocm-pr-quality:attestation v1 -->
> **Reviewed with the ROCm PR-quality skill.**
> - Skill: `rocm-pr-quality` (base) - overlays: `<applied overlays, or "none">`
> - Action: `review` - Result: **APPROVED** (`<n>` BLOCKING; `<n>` IMPORTANT)
> - Risk: `<1-5>`
> - Reviewed commit: `<full 40-char head SHA>`
> - Attestation: `<8-hex digest>`
>
> <sub>Machine-checkable attestation. The digest ties this review to this PR; later commits are
> allowed. The reviewed commit is informational (what the review looked at).</sub>
```

**Digest.** The first 8 hex characters of the sha256 of the UTF-8 string `<owner>/<repo>#<number>`,
lowercased, with no trailing newline (for example, `rocm/therock#6653`). The repo is part of the
input because PR numbers collide across repositories. Do not compute it by hand: take it from the
`Attestation digest` line (or the `attestation` JSON field) printed by
`tools/pr_quality_check.py`. If that tool is unavailable, this works in any shell (including
PowerShell):

```text
python -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].lower().encode()).hexdigest()[:8])" "<owner>/<repo>#<number>"
```

A consumer recomputes the digest from the repo and number of the PR the comment is posted on; a
mismatch means the block was not generated for this PR. The reviewed commit is not compared to the
PR's current head. This is a checksum, not a secret: it stops pasted templates and cross-PR copies,
not a deliberate forgery.

**Checking an attestation (Action 4).** `tools/pr_quality_check.py --pr <number|url> --check-attestation` scans the PR's conversation comments and review bodies for the marker and
applies the rule above to each block it finds. A block is valid when it records an `APPROVED`
result and its digest equals the PR's own digest. Overall status is `VERIFIED` if any block is
valid, `INVALID` if blocks exist but none is valid (copied from another PR, placeholder digest,
non-approval result), and `NONE` if there is no block. Exit codes are `0`, `5`, and `4`
respectively. The report also lists each block's author, association, and time, and how many
commits were pushed after the reviewed commit (informational only).

The full head SHA is `gh pr view --json headRefOid -q .headRefOid` (or `git rev-parse HEAD`).

One-line paste fallback (same marker, for when an approval was already posted without a block):

```text
<!-- rocm-pr-quality:attestation v1 --> rocm-pr-quality review APPROVED - commit <full-sha> - attestation <8-hex digest>
```
