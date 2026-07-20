# ROCm PR Quality — Reference

Rubric tables, templates, and vocabularies for the `rocm-pr-quality` base skill. This is the
single source of truth the author/review actions and the shared checker share. Overlays add to
or tighten these; they never relax a MUST.

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

## Test-substance smell scan (a hit is `BLOCKING`, not a nit)

- assert-callable only / asserts only that a mock was called
- `hasattr`-guarded test bodies that silently skip
- lone `is not None` / `.called` assertions with no behavioral check
- copy-paste tests that should be parametrized
- mock-only assertions that pass against a no-op implementation
- phantom methods / attributes that do not exist in the source (AI-slop tell)
- non-obvious tolerance or parameter choices with no recorded rationale ("why")

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

## Review attestation (Action 2, approvals only)

When a review concludes `APPROVED`, the skill appends this block to the drafted approval comment
(never to `CHANGES REQUESTED` / `REJECTED`, and never to a PR description). The human posts it; the
skill never posts on its own. The `<!-- rocm-pr-quality:attestation v1 -->` marker is what a
consumer (e.g. a pre-review gate) keys on, and the digest binds the attestation to the exact diff
so a stale or copied block can be detected.

Block (leaned-into default):

```markdown
<!-- rocm-pr-quality:attestation v1 -->
> **Reviewed with the ROCm PR-quality skill.**
> - Skill: `rocm-pr-quality` (base) - overlays: `<applied overlays, or "none">`
> - Action: `review` - Result: **APPROVED** (`<n>` BLOCKING; `<n>` IMPORTANT)
> - Risk: `<1-5>`
> - Commit: `<full 40-char head SHA>`
> - Diff digest: `<8-hex>`
>
> <sub>Machine-checkable attestation. The digest binds this review to the exact file set at the
> commit above; pushing new commits invalidates it, at which point the skill should be re-run.</sub>
```

One-line paste fallback (same marker, for when an approval was already posted without a block):

```text
<!-- rocm-pr-quality:attestation v1 --> rocm-pr-quality review APPROVED - commit <full-sha> - digest <8-hex>
```

### Digest algorithm (pinned, so a consumer can recompute)

- **head SHA:** the full 40-char PR head SHA — `gh pr view --json headRefOid -q .headRefOid` (or
  `git rev-parse HEAD`).
- **files:** the PR's changed-file paths (`gh pr diff --name-only`) as POSIX paths, sorted by
  Unicode code point (Python `sorted()`; equivalently `LC_ALL=C sort`, since UTF-8 byte order
  matches code-point order).
- **payload:** the head SHA followed by the sorted paths, joined by a single `\n` with no trailing
  newline (so an empty file list yields just the head SHA):
  `payload = "\n".join([head_sha, *sorted_files])`.
- **digest:** `sha256(payload.encode("utf-8")).hexdigest()[:8]`.
