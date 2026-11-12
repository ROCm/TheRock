# TheRock PR Quality: review mode

Part of the `therock-pr-quality` skill. Read this file for `review` (base **Action 2, Review
assist**), after the base skill (`skills/rocm-pr-quality/SKILL.md` and `reference.md`) and this
skill's `SKILL.md`. It applies to every TheRock PR, not only those in the `SKILL.md` Scope, and works
for self-review before requesting review as well as for reviewing someone else's PR.

Review mode only adds steps and tightens how the review is gathered and presented. It never relaxes a
base MUST, the finding tiers, or the "never post without approval" principle. On any conflict, the
base wins.

______________________________________________________________________

## What review mode adds

- A diff-first review setup that pulls source only when needed and never switches the user's branch
  on its own.
- De-duplication against everything reviewers have already said, including the invoking user's own
  unsubmitted drafts.
- A title and description accuracy check, since TheRock squash-merges.
- Verification rules that prevent the most common false-positive findings.
- A findings layout ordered by file and line, so the author can walk the diff top to bottom. The
  base's severity-first lead is kept as a separate `BLOCKING` list at the top of the review.
- A "Reviewer's take": a whole-PR design pass, filtered through explicit engineering heuristics.

______________________________________________________________________

## Inputs

A PR reference: full URL, `owner/repo#N`, `#N`, or a bare number (current repo). If none is given,
ask for one before doing anything else.

______________________________________________________________________

## Workflow

Run these in addition to the base Action 2 workflow and, where the Scope applies, the `SKILL.md`
supplements.

### 1. Resolve the PR

```bash
gh pr view <ref> --json number,title,body,headRefName,headRefOid,baseRefName,author,files,url,statusCheckRollup
```

If the author is the invoking user, this is a self-review. Run the same review, phrase findings as
changes to make rather than comments to relay, and in "Already raised" mark each existing thread as
addressed by the current code or still open.

### 2. Read all prior discussion

Collect every channel; each one holds findings the others do not: the top-level conversation,
review summaries, and inline review comments. Two gaps cause silently missing discussion:

- The GitHub list endpoints return 30 items per page. Paginate, or a busy PR is truncated.
- Inline comments of the invoking user's `PENDING` (unsubmitted) review do not appear in the PR's
  inline comments list. Fetch them through that review's own comments endpoint.

Do not repeat a point that is already raised; list it under "Already raised" instead. Note author
replies that explain a non-obvious choice, so the review does not re-ask a settled question.

### 3. Get the diff; pull source only when needed

Start from `gh pr diff`. For small, self-contained changes the diff is enough. Pull source only when
the diff cannot answer a question, for example a function the diff calls but does not define. When
the review relied on the diff alone, say so in the Summary; the base rates a diff-only review as
lower confidence.

Never switch branches or otherwise modify the user's working tree without asking; they may have
work in progress. Prefer reading files at the PR head without touching the tree (for example
`git show` on a fetched PR ref).

### 4. Apply the style guides for touched languages only

Of the style guides listed under "Canonical TheRock references" in `SKILL.md`, read only those for
languages present in the diff. For Python, run the guide's "Code review checklist".

Cite the specific guide section in each style finding. Do not invent rules the guides do not contain;
a personal preference is at most a `SUGGESTION` and must be labelled as preference.

### 5. Check the title and description against the code

TheRock squash-merges PRs, and the PR title and body become the commit message on `main` verbatim.
A description that does not match the code is a permanent, misleading entry in the history. The base
PR-hygiene checklist covers title convention and whether the description explains *why*; this step
checks that title and description are **accurate**, now and after the review.

Compare the title and body against the diff and flag:

- **Claims the code does not back up.** The description says something is done (a flag added, a
  default preserved, a caller updated, a test added) and the diff does not show it.
- **Changes the description does not mention.** Behavior changes, new or changed defaults, interface
  changes to reusable workflows or scripts, and removed functionality that a reader of the commit
  message would not expect.
- **Title scope mismatch.** The title describes a narrower or broader change than the diff (for
  example, "fix typo" on a PR that also changes logic), or uses the wrong prefix for what changed.
- **Stale after earlier review rounds.** If commits landed after a previous review, check whether the
  description still describes the current code or the original version.

Then assess whether the description will **need rework once the changes requested in this review are
applied**. If a requested change alters behavior, scope, interfaces, or the approach the description
explains, name the sentences or sections that will become wrong and what they should say instead. A
description that is accurate today but will be wrong after the requested fixes still needs updating
before merge.

Tier: a description or title that misstates what the code does is `IMPORTANT`. Missing mention of a
behavior or default change is `IMPORTANT`. Wording and polish are `SUGGESTION`.

______________________________________________________________________

## Verification discipline

Each rule below exists because skipping it produced a wrong or missed finding in a real review.

- **Code comments are claims, not evidence.** A comment explaining why a permission, flag, or call
  exists may be stale or incomplete. Before writing a finding that relies on one, trace the actual
  consumers (grep every use of the permission scope, flag, or API). Example: a workflow comment said
  `actions: read` was "required for stage reuse", but a summary job also called
  `GET /actions/runs/{id}/jobs`, which needs the same scope. Removing it would have broken the job.
- **A raise is only as strong as its caller.** When a PR adds a fail-fast `raise`, check every call
  site. A caller that catches the exception and continues makes the failure silent again; that
  is the real finding.
- **Ground naming and classification findings in real data.** Before claiming a package, artifact,
  or label "falls through" a matcher, check the real published index, artifact list, or test
  fixtures. A plausible-looking name from another ecosystem is not evidence.
- **Flag coupled sites together.** When a suggested fix depends on a constant, set, or pattern defined
  elsewhere, name both locations and say they must change together, so a later edit does not leave
  them out of sync.
- **Trace the unhappy paths.** For each changed code path, check empty input, missing keys, `None`,
  and API fields that can come back `null` (a GitHub check run's `output.summary`, for example).
  Look for parsers that accept malformed input: a `[,\s]+` split turns the missing-comma typo
  `gfx94X gfx110X` into two valid tokens instead of failing.
- **Cite precedent for "do it like X".** Before suggesting a dataclass, a helper to reuse, or an
  existing pattern, confirm it exists in the repo and cite `file:line`. Without precedent, it is a
  preference.
- **Ask whether old behavior was load-bearing.** When a PR changes established behavior (a default,
  an output format, an error that used to be raised), search for consumers that rely on it: callers,
  downstream workflows, scripts that parse the output. If the search is inconclusive, ask the author.
- **Evidence over assertion** (base principle): every finding cites `file:line`, a diff hunk, or a
  CI run/step link.

______________________________________________________________________

## Output

```
## PR #N: <title>
Author: @handle · Branch: head -> base · Assessment: APPROVED | CHANGES REQUESTED | REJECTED

### Summary
<2-3 sentences: what the PR does and overall quality. If the approach itself is wrong, say so
here; do not leave it for the Reviewer's take at the bottom.>

### BLOCKING items
<Bullet list of BLOCKING findings with file:line, or "None". This is the base's required
severity-first view.>

### Base review answers
<The base's four review questions, answered briefly.>

### Findings
<Ordered as the diff lists files: by file path, then line number. Tier is a tag on each finding,
not the sort key.>

**IMPORTANT** - .github/workflows/bar.yml:88
<...>

**SUGGESTION** - CMakeLists.txt:15
<...>

**BLOCKING** - build_tools/foo.py:7
<What is wrong, why it matters, suggested fix as a code block if non-trivial.>

**SUGGESTION** - build_tools/foo.py:42
<...>

### Title and description
<Accuracy against the current diff: mismatches, unmentioned changes, title scope. Then:
"After requested changes: no rework needed" or the specific sentences/sections to update and
the suggested replacement text.>

### Already raised
<Points other reviewers (or the user's pending draft) already covered.>

### Questions for the author
<Things the diff alone cannot answer. Ask; do not guess.>

### Reviewer's take
<See below.>
```

Use the base finding tiers (`BLOCKING` / `IMPORTANT` / `SUGGESTION` / `FUTURE WORK`) and overall
assessment values exactly as defined in `rocm-pr-quality/reference.md`.

______________________________________________________________________

## Writing the "Reviewer's take"

The Findings list is exhaustive and anchored to lines. The Reviewer's take is different in kind:

1. **Distill** the findings that survive the heuristics below. Most do not; that is the point.
1. **Do a fresh whole-PR design pass.** Design problems do not live on a single line, so this section
   may raise points absent from Findings: Is this the right approach? Is the scope right? Is the
   abstraction boundary in the right place? Does this logic belong in this file, layer, or component
   (superbuild vs sub-project, workflow YAML vs `build_tools/` script)? Is there a materially simpler
   shape?

Apply these heuristics to both:

- **Scope and 80/20.** If the PR builds more than the problem needs, say so and name the smaller
  version. If fixing a finding would materially increase complexity, weigh whether it is worth it.
- **Wait for cut points before abstracting.** Do not praise a helper extracted on its first use, and
  do not demand one for two trivial lines. Two call sites sharing non-trivial logic is a valid cut
  point; call it acceptable.
- **Locality of behavior over strict DRY.** Code that reads top to bottom in one place beats the same
  logic split across files to satisfy layering. Flag indirection that makes the reader jump around.
- **Integration tests over heavy mocking.** Prefer tests that exercise real components; question
  mocks used where the real thing is cheap and available (local files, subprocesses, small fixtures).
- **Respect existing code.** Before calling odd-looking code wrong, find out why it is shaped that
  way (`git log -L`, `git blame`, linked issues). It often encodes a constraint.
- **The six-month test.** Would this be harder to understand in six months? If yes, comment. If it is
  only "not how I would write it", it does not belong here.
- **Priority when goals conflict.** Correctness, then maintainability, then readability, then
  performance. When a fix trades one for another, name the conflict and recommend a side; do not pick
  silently. Do not trade readability for a speedup nobody measured.

For each point, say whether it is a **request for change**, a **critique to note** (author's call), or
**fine as is**. End with one line: the overall assessment, plus the specific item it is pending on if
any (for example, "CHANGES REQUESTED: pending caller fix in `foo.py:42`").

Tone: terse and fact-first. State the point and the ask. No praise padding, no hedging, no lectures
on basics the author clearly knows.

______________________________________________________________________

## What not to do

- Do not check out the PR or switch branches in the user's working tree unless they ask (see step 3).
- Do not restate the diff; summarize, then give findings.
- Do not repeat findings already raised; list them under "Already raised".
- Do not mark style preferences as `BLOCKING`.
- Do not post anything to GitHub (base principle). Output the review for the user to relay.
